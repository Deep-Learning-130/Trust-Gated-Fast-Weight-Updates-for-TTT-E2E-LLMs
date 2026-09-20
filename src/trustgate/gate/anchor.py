"""Option A -- frozen-anchor consistency. THE PRIMARY GATE.

Decided in ADR-F1 as the primary signal. Hold a frozen meta-learned anchor
theta_0, keep a small rotating set of held-out probes, and ask whether
committing this update pulls probe behaviour away from the anchor's.
Behavioural, not distributional -- which is precisely the diff against MedBN,
whose signal is batch-norm *statistics*. TTT-E2E has no batch norm, so there is
nothing for a MedBN-style test to read; see `docs/patent/prior-art/medbn-diff.md`.

Cost: one extra anchor forward pass over the probe set per scored window. That
is the single largest overhead in the design and the reason the section 5.1
budget is stated as "<= ~10% of the per-window update cost". Measure it; do not
assume it -- `trustgate.eval.overhead` is what measures it.

PRE-VERDICT DEFAULTS
--------------------
Implemented before `experiments/001-attack-spike` returned a verdict, per the
dated revision in `PREREGISTERED.md` (2026-09-20, Manas Maahir). ADR-F1 and the
previous version of this docstring deferred three questions "to settle with
Phase 1 data in hand". That data does not exist. The values below are **guesses
made before the verdict, not findings**, and anything reading them as tuned
results is reading them wrong:

  - **Probe set size and rotation policy.** Defaulted to rotation on, via
    `probes.rotating`. A fixed set is cheap to overfit: an adversary who learns
    the probes can shape updates that look consistent on them and drift
    everywhere else. Rotation is the mitigation and it is not free. The *size*
    is left to the caller because it trades directly against the 10% budget.
  - **Divergence measure.** Defaulted to `symmetric_kl`, which is what ADR-F1
    named first and has no asymmetry to argue about. `kl` and `topk_agreement`
    are implemented so the choice can be swept rather than defended.
  - **theta_0 vs last-trusted checkpoint.** Defaulted to **theta_0**, the
    stronger guarantee: this gate scores against whatever `anchor_params` it was
    built with, and the caller supplies the frozen init. Scoring against the
    last trusted checkpoint would allow legitimate long-range adaptation, which
    is the central false-reject/false-accept tension of 5.6 and is exactly the
    thing Phase 1 data was supposed to inform.
"""

from __future__ import annotations

from collections.abc import Callable

import jax
import jax.numpy as jnp

from trustgate.tree_ops import tree_add
from trustgate.types import FastWeights, GateDecision, Updates, Verdict

#: Divergence measures this module implements.
DIVERGENCES = ("symmetric_kl", "kl", "topk_agreement")

#: Floor for the softmax inside the KL measures. Probe logits at bf16 can
#: produce exact zeros after exponentiation, and `log(0)` would poison the
#: whole decision with a nan -- which `jnp.where` propagates rather than
#: rejecting, so an unguarded nan reads as ACCEPT.
_EPS = 1e-8


def _log_softmax(logits: jnp.ndarray) -> jnp.ndarray:
    return jax.nn.log_softmax(logits.astype(jnp.float32), axis=-1)


def _kl(log_p: jnp.ndarray, log_q: jnp.ndarray) -> jnp.ndarray:
    """KL(p || q) averaged over every position, in nats."""
    p = jnp.exp(log_p)
    return jnp.mean(jnp.sum(p * (log_p - log_q), axis=-1))


def probe_divergence(
    updated_params: FastWeights,
    anchor_params: FastWeights,
    probes,
    divergence: str = "symmetric_kl",
    *,
    forward_fn: Callable[[FastWeights, object], jnp.ndarray],
) -> jnp.ndarray:
    """Behavioural distance between updated and anchor models on the probe set.

    `forward_fn(params, probes) -> logits` is injected rather than derived,
    because the inner-parameter subtree alone cannot be forwarded: it is a
    filtered slice of the model, and a forward pass needs the whole thing. The
    caller closes over the model; this stays a pure function of two parameter
    trees, which is what makes it testable without a vendor tree.

    Returns a float32 scalar. Larger means further from the anchor.
    """
    if divergence not in DIVERGENCES:
        raise ValueError(
            f"unknown divergence {divergence!r}; expected one of "
            f"{', '.join(DIVERGENCES)}"
        )

    updated_logits = forward_fn(updated_params, probes)
    anchor_logits = forward_fn(anchor_params, probes)

    if divergence == "topk_agreement":
        # Reported as a divergence so every measure points the same way: 0 is
        # identical behaviour, larger is further from the anchor.
        agree = jnp.argmax(updated_logits, axis=-1) == jnp.argmax(
            anchor_logits, axis=-1
        )
        return jnp.float32(1.0) - jnp.mean(agree.astype(jnp.float32))

    log_p = _log_softmax(anchor_logits)
    log_q = _log_softmax(updated_logits)
    log_p = jnp.maximum(log_p, jnp.log(_EPS))
    log_q = jnp.maximum(log_q, jnp.log(_EPS))

    if divergence == "kl":
        return _kl(log_p, log_q)
    return 0.5 * (_kl(log_p, log_q) + _kl(log_q, log_p))


def anchor_consistency_gate(
    anchor_params: FastWeights,
    probes,
    threshold: float,
    *,
    divergence: str = "symmetric_kl",
    forward_fn: Callable[[FastWeights, object], jnp.ndarray],
):
    """Build a gate scoring behavioural divergence from the frozen anchor.

    Args:
        anchor_params: frozen meta-learned init, theta_0. Never updated.
        probes: probe batch, or whatever `forward_fn` consumes. Fixed shape --
            rotating the *contents* is `probes.rotating`'s job, because changing
            shapes retriggers XLA compilation inside the scan.
        threshold: reject above this divergence. A parameter, never a constant:
            dossier 5.6 says the deliverable is the operating-point curve.
        divergence: one of `DIVERGENCES`.
        forward_fn: `(params, probes) -> logits`.

    Returns:
        A `Gate`: `(updates, fast_weights, anchor=None) -> GateDecision`.
    """
    if divergence not in DIVERGENCES:
        raise ValueError(
            f"unknown divergence {divergence!r}; expected one of "
            f"{', '.join(DIVERGENCES)}"
        )
    if threshold <= 0:
        raise ValueError(f"threshold must be positive, got {threshold}")

    threshold_arr = jnp.float32(threshold)

    def gate(
        updates: Updates,
        fast_weights: FastWeights,
        anchor: FastWeights | None = None,
    ) -> GateDecision:
        # `anchor` is part of the `Gate` protocol but `interceptor.py:100` calls
        # `gate(delta, current)` and never supplies it, so the closed-over
        # anchor is the real source. An explicitly passed one wins, which is
        # what lets a test drive both.
        theta_0 = anchor_params if anchor is None else anchor

        # The update has not committed yet, so score what it *would* produce.
        proposed = tree_add(fast_weights, updates)
        divergence_value = probe_divergence(
            proposed, theta_0, probes, divergence, forward_fn=forward_fn
        )

        # A nan divergence must not read as ACCEPT. `jnp.where` propagates nan
        # rather than choosing a branch, and `nan > threshold` is False, so an
        # unguarded nan would commit the update it failed to score.
        unscorable = jnp.isnan(divergence_value)
        over = divergence_value > threshold_arr
        verdict = jnp.where(
            jnp.logical_or(unscorable, over),
            jnp.int32(Verdict.REJECT),
            jnp.int32(Verdict.ACCEPT),
        )

        # Confidence is distance from the threshold, saturating at one
        # threshold's worth either side. A decision taken just at the bar is
        # the one most likely to be wrong, and `policy.combine` turns low
        # confidence into a conservative reject. PRE-VERDICT GUESS: the shape
        # of this curve is exactly what Phase 1 data was meant to inform.
        confidence = jnp.where(
            unscorable,
            jnp.float32(0.0),
            jnp.clip(
                jnp.abs(divergence_value - threshold_arr) / threshold_arr, 0.0, 1.0
            ),
        )

        # A rejected update never commits, so it contributes no drift.
        drift_delta = jnp.where(
            verdict == jnp.int32(Verdict.ACCEPT),
            jnp.nan_to_num(divergence_value),
            jnp.float32(0.0),
        )

        return GateDecision(
            verdict=verdict,
            drift_delta=jnp.float32(drift_delta),
            confidence=jnp.float32(confidence),
        )

    return gate
