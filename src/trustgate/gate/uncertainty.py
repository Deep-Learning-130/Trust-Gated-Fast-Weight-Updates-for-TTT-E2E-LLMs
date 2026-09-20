"""Option B -- update-uncertainty threshold. CHEAP SECONDARY.

ADR-F1 makes this the cheap secondary, run on every window alongside Option A.
Its appeal is that it costs no extra forward pass: everything it reads is
already in scope where the interceptor sits.

Known weakness, recorded before any ROC is drawn: this is weak against
low-variance stealth poison, which is exactly the threat model of dossier
section 1. A good ROC on loud attacks must not be allowed to disguise that.
Option A is the primary for this reason.

WHICH SIGNALS ACTUALLY WORK
---------------------------
The three candidates ADR-F1 lists are not equally available at the hook:

  - **`update_norm`** -- implemented. ||delta-theta|| against a reference scale
    is a pure function of the update, needs no history, and `tree_ops
    .tree_global_norm` already computes it in float32 regardless of parameter
    dtype (bf16 sum-of-squares over a 1B subtree loses too much precision to
    threshold against).
  - **`grad_norm_zscore`** and **`loss_spike`** -- NOT implemented, and not for
    want of effort. Both need running statistics across windows, and there is
    nowhere to keep them: the vendor's scan carry is fixed and cannot take a
    fourth slot (ADR-003 correction, `docs/adr/ADR-P3-1-drift-carry.md`). The
    same blocker that keeps the drift accumulator unreachable keeps an EMA
    unreachable. They raise rather than silently degrading to something
    stateless, because a z-score against a constant is not a z-score.

Raw gradients are additionally unavailable: `inner_grads` is local to the
vendor's `inner_loop_step` body (`transformer.py:625`) and never returned, so
even with a carry slot the gradient-norm signal would have to be reconstructed
from the update and the optimizer state.

PRE-VERDICT DEFAULT: `reference_norm` has no principled value without Phase 1
data. It is a caller parameter with no default for that reason -- a wrong
constant here silently rejects every update or none.
"""

from __future__ import annotations

import jax.numpy as jnp

from trustgate.tree_ops import tree_global_norm
from trustgate.types import FastWeights, GateDecision, Updates, Verdict

#: Signals this module can compute at the interceptor hook.
IMPLEMENTED_SIGNALS = ("update_norm",)

#: Signals ADR-F1 names that need cross-window state we have no slot for.
STATEFUL_SIGNALS = ("grad_norm_zscore", "loss_spike")


def uncertainty_gate(
    threshold: float,
    *,
    signal: str = "update_norm",
    reference_norm: float,
):
    """Build the cheap secondary gate.

    Args:
        threshold: reject when the signal exceeds this multiple of
            `reference_norm`. A parameter, never a constant -- the deliverable
            is the operating-point curve.
        signal: one of `IMPLEMENTED_SIGNALS`.
        reference_norm: the update norm considered ordinary. Sweep it; there is
            no defensible default before Phase 1 data.

    Returns:
        A `Gate`.
    """
    if signal in STATEFUL_SIGNALS:
        raise NotImplementedError(
            f"signal {signal!r} needs running statistics across windows, and "
            f"the vendor's scan carry is fixed at "
            f"(model, inner_opt_state, (state_all, state_suffix)) with no slot "
            f"for them -- the same blocker that keeps drift/accumulator.py "
            f"unreachable. See docs/adr/ADR-P3-1-drift-carry.md. Use "
            f"'update_norm', which needs no history."
        )
    if signal not in IMPLEMENTED_SIGNALS:
        raise ValueError(
            f"unknown signal {signal!r}; implemented: "
            f"{', '.join(IMPLEMENTED_SIGNALS)}; blocked on a carry slot: "
            f"{', '.join(STATEFUL_SIGNALS)}"
        )
    if threshold <= 0:
        raise ValueError(f"threshold must be positive, got {threshold}")
    if reference_norm <= 0:
        raise ValueError(f"reference_norm must be positive, got {reference_norm}")

    threshold_arr = jnp.float32(threshold)
    reference = jnp.float32(reference_norm)

    def gate(
        updates: Updates,
        fast_weights: FastWeights,
        anchor: FastWeights | None = None,
    ) -> GateDecision:
        norm = tree_global_norm(updates)
        ratio = norm / reference

        unscorable = jnp.isnan(ratio)
        verdict = jnp.where(
            jnp.logical_or(unscorable, ratio > threshold_arr),
            jnp.int32(Verdict.REJECT),
            jnp.int32(Verdict.ACCEPT),
        )
        confidence = jnp.where(
            unscorable,
            jnp.float32(0.0),
            jnp.clip(jnp.abs(ratio - threshold_arr) / threshold_arr, 0.0, 1.0),
        )

        # This gate measures magnitude, not behaviour. Drift is reported by the
        # primary signal, which is the behavioural one; reporting a norm here
        # would let a cheap proxy consume the behavioural budget.
        return GateDecision(
            verdict=verdict,
            drift_delta=jnp.float32(0.0),
            confidence=jnp.float32(confidence),
        )

    return gate
