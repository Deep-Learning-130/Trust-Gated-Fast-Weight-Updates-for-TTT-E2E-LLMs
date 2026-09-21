"""Phase 2 measurement: run the anchor gate against the real vendor step.

`gate/anchor.py` implements the primary signal and `eval/overhead.py` and
`metrics.clean_regression` define what to measure, but until this module
nothing connected them to a model. The missing pieces were:

  - a `forward_fn` -- `(inner params, probes) -> logits` -- for the vendor
    `MetaModel`, which `anchor_consistency_gate` needs and cannot build itself;
  - a probe batch in the vendor's `(suffix chunk, prefix output)` shape;
  - a threshold. There is no tuned one (ADR-F1 deferred it to Phase 1 data), so
    thresholds are read off the divergence distribution of a clean stream.

What this measures, per threshold: gate overhead against the ~10% budget, the
clean-accuracy cost (`clean_regression`), and whether the gate moves the
poison/control corruption the 001 spike measured.

WHAT THIS DOES NOT DO
---------------------
- **No verdict.** The PROCEED/STOP decision comes only from the 001 spike
  report. Nothing here reads or moves a frozen bar.
- **Bounded drift is not enforced.** The gate scores each update; nothing
  accumulates across windows (ADR-003 correction, ADR-P3-1).
- **The probe set is fixed, not rotating.** `anchor.py` defaults to rotation,
  but the interceptor calls `gate(delta, current)` with no window index and the
  vendor's scan carry has no slot for one -- the same blocker as the drift
  accumulator. A fixed set is the one an adaptive attacker can learn.
- **Thresholds are pre-verdict operating points**, quantiles of one clean
  stream's divergence, not tuned findings.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass, field

import equinox as eqx
import numpy as np

from trustgate.eval.metrics import CorruptionResult, clean_regression, corruption_metric
from trustgate.eval.overhead import OverheadResult, measure_overhead

#: Threshold used to calibrate: accepts every scorable update, so the gate only
#: observes. Finite, because `anchor_consistency_gate` divides by it.
OBSERVE_ONLY_THRESHOLD = 1e30

#: Floor for a calibrated threshold. A clean stream whose updates never move the
#: probe logits yields a 0 quantile, which `anchor_consistency_gate` refuses.
MIN_THRESHOLD = 1e-12


class GateInputs(eqx.Module):
    """The gate's arrays, threaded through the compiled step as an argument."""

    theta_0: object
    """Frozen anchor: the fast weights every stream starts from."""

    probe_seq: object
    """One `mini_batch_size` window of held-out tokens, as a vendor `Batch`."""

    probe_prefix: object
    """The frozen prefix blocks' output for that window."""

    state_suffix: object
    """Suffix-block state the probe forward runs with. Fresh, never adapted."""


@dataclass(frozen=True)
class AnchorGateSpec:
    """Hashable description of an anchor gate: the static half of the jit key."""

    threshold: float
    divergence: str = "symmetric_kl"

    def build(self, model_outer, spec_inner, inputs: GateInputs):
        from trustgate.gate.anchor import anchor_consistency_gate

        _, outer_params = eqx.partition(model_outer, spec_inner)

        def forward_fn(params, probes):
            probe_seq, probe_prefix = probes
            # The same recombination `vendor_bind` performs for the step itself:
            # fast weights from the argument, everything else from the frozen,
            # dtype-cast reference. `MetaModel.inner_parameters()` is an
            # `eqx.filter` on the same spec, so the trees line up.
            model = eqx.combine(params, outer_params)
            return model.language_model.suffix_call(
                prefix_outputs=probe_prefix, state=inputs.state_suffix, seq=probe_seq
            ).logits

        return anchor_consistency_gate(
            inputs.theta_0,
            (inputs.probe_seq, inputs.probe_prefix),
            self.threshold,
            divergence=self.divergence,
            forward_fn=forward_fn,
        )


def make_gate_inputs(binding, probe_tokens, *, bos_token_id: int) -> GateInputs:
    """Build the fixed probe window: chunk 0 of one full held-out sequence.

    The prefix pass runs over a whole `seq_length` sequence, the only length
    the vendor model is configured and exercised at, rather than over a lone
    mini-batch. Attention is causal, so chunk 0's prefix output is the same
    either way; this just never shows the vendor a shape it has not seen.
    """
    from trustgate.eval import vendor_bind

    need = binding.seq_length + 1
    probe_tokens = np.asarray(probe_tokens)
    if probe_tokens.shape[0] < need:
        raise ValueError(
            f"probe needs {need} tokens (one seq_length sequence plus the "
            f"target shift), got {probe_tokens.shape[0]}"
        )
    probe_seq, probe_prefix = vendor_bind.sequence_chunks(
        binding, probe_tokens[:need], bos_token_id=bos_token_id
    )[0]
    return GateInputs(
        theta_0=binding.fast_weights(),
        probe_seq=probe_seq,
        probe_prefix=probe_prefix,
        state_suffix=binding.state_suffix,
    )


def gate_series(metrics: list[dict], key: str) -> np.ndarray:
    """One `gate/*` value per inner step, as float64."""
    return np.asarray([float(np.asarray(m[key])) for m in metrics], dtype=np.float64)


# --------------------------------------------------------------- calibration --


@dataclass(frozen=True)
class Calibration:
    divergences: tuple[float, ...]
    """Per-step divergence from theta_0 on the clean calibration stream."""

    unscorable_steps: int
    """Steps whose divergence was nan. The gate rejects these."""

    quantiles: tuple[float, ...]
    thresholds: tuple[float, ...]


def calibrate(
    run_stream_fn: Callable[..., object],
    gate_inputs: GateInputs,
    quantiles: list[float],
    *,
    divergence: str,
) -> Calibration:
    """Observe the gate on a clean stream and read thresholds off its quantiles.

    `run_stream_fn(gate_spec=..., gate_inputs=..., metrics_out=...)` runs the
    clean calibration stream. With `OBSERVE_ONLY_THRESHOLD` every scorable
    update is accepted, so `gate/drift_delta` equals the divergence.

    The divergence is measured from theta_0, not per update, so it grows along a
    stream: a threshold at the q99 of a clean stream rejects the late updates of
    a longer clean one. That is the theta_0-vs-last-trusted tension `anchor.py`
    names, and it shows up in `clean_regression`.
    """
    if not quantiles or any(not 0.0 < q <= 1.0 for q in quantiles):
        raise ValueError(f"quantiles must lie in (0, 1], got {quantiles}")

    metrics: list[dict] = []
    run_stream_fn(
        gate_spec=AnchorGateSpec(OBSERVE_ONLY_THRESHOLD, divergence),
        gate_inputs=gate_inputs,
        metrics_out=metrics,
    )
    accepted = gate_series(metrics, "gate/accepted")
    div = gate_series(metrics, "gate/drift_delta")[accepted > 0.5]
    unscorable = int((accepted <= 0.5).sum())
    if div.size == 0:
        raise ValueError(
            "no scorable step on the calibration stream: every divergence was "
            "nan. The probe forward is broken; do not measure a gate on it."
        )

    thresholds = tuple(max(float(np.quantile(div, q)), MIN_THRESHOLD) for q in quantiles)
    return Calibration(
        divergences=tuple(float(d) for d in div),
        unscorable_steps=unscorable,
        quantiles=tuple(quantiles),
        thresholds=thresholds,
    )


# --------------------------------------------------------------- measurement --


@dataclass(frozen=True)
class ArmLosses:
    poisoned: tuple[float, ...]
    control: tuple[float, ...]
    poison_accept_rate: tuple[float, ...] = ()
    control_accept_rate: tuple[float, ...] = ()

    @property
    def corruption(self) -> CorruptionResult:
        return corruption_metric(np.asarray(self.poisoned), np.asarray(self.control))


@dataclass(frozen=True)
class ThresholdResult:
    quantile: float
    threshold: float
    arms: ArmLosses
    clean_regression: float
    """Gated vs ungated benign loss on the *control* arms: the false-reject cost."""


@dataclass
class GateEvalResult:
    divergence: str
    checkpoint: str
    seeds: tuple[int, ...]
    calibration: Calibration
    ungated: ArmLosses
    thresholds: list[ThresholdResult] = field(default_factory=list)
    overhead: OverheadResult | None = None


def _accept_rate(metrics: list[dict]) -> float:
    if not metrics:
        return math.nan
    return float(gate_series(metrics, "gate/accepted").mean())


def run_arms(
    seeds: list[int],
    build_arms: Callable[[int], tuple[object, object]],
    adapt_and_eval_for: Callable[..., Callable],
    condition_for: Callable[[object, int], object],
    *,
    gate_spec=None,
    gate_inputs=None,
) -> ArmLosses:
    """Adapt-and-eval both arms of every seed, with or without the gate.

    `adapt_and_eval_for(gate_spec=, gate_inputs=, metrics_out=)` returns a
    `harness.make_adapt_and_eval`-style callable; a fresh metrics list per arm
    is what lets the acceptance rate be read per arm.
    """
    poisoned, control, p_rate, c_rate = [], [], [], []
    for seed in seeds:
        poison, ctrl = build_arms(seed)
        for stream, losses, rates in ((poison, poisoned, p_rate), (ctrl, control, c_rate)):
            metrics: list[dict] = []
            fn = adapt_and_eval_for(
                gate_spec=gate_spec, gate_inputs=gate_inputs, metrics_out=metrics
            )
            losses.append(float(fn(stream, condition_for(stream, seed))))
            rates.append(_accept_rate(metrics) if gate_spec is not None else 1.0)
    return ArmLosses(
        poisoned=tuple(poisoned),
        control=tuple(control),
        poison_accept_rate=tuple(p_rate),
        control_accept_rate=tuple(c_rate),
    )


def measure_step_overhead(
    binding,
    chunk,
    gate_spec: AnchorGateSpec,
    gate_inputs: GateInputs,
    *,
    repeats: int = 20,
    warmup: int = 2,
) -> OverheadResult:
    """Gated vs ungated compiled inner step, on the same chunk and carry."""
    from trustgate.eval import vendor_bind

    carry = binding.init_carry()
    ungated = vendor_bind.make_step_fn(binding)
    gated = vendor_bind.make_step_fn(binding, gate_spec, gate_inputs)
    args = (carry.fast_weights, carry.opt_state, carry.state_tuple, chunk)
    return measure_overhead(
        lambda: ungated(*args).fast_weights,
        lambda: gated(*args).fast_weights,
        repeats=repeats,
        warmup=warmup,
    )


def run_gate_eval(
    *,
    seeds: list[int],
    build_arms,
    adapt_and_eval_for,
    condition_for,
    calibrate_stream_fn,
    gate_inputs: GateInputs,
    quantiles: list[float],
    divergence: str,
    checkpoint: str,
    overhead_fn: Callable[[AnchorGateSpec], OverheadResult] | None = None,
    log: Callable[[str], None] = print,
) -> GateEvalResult:
    """Calibrate, run the ungated arms, then each threshold, then overhead."""
    log(f"[gate] calibrating {divergence} on a clean stream")
    cal = calibrate(calibrate_stream_fn, gate_inputs, quantiles, divergence=divergence)
    log(
        f"[gate] calibration: {len(cal.divergences)} scorable steps, "
        f"{cal.unscorable_steps} unscorable; thresholds "
        + ", ".join(f"q{q:g}={t:.6g}" for q, t in zip(cal.quantiles, cal.thresholds))
    )

    log("[gate] ungated arms")
    ungated = run_arms(seeds, build_arms, adapt_and_eval_for, condition_for)
    result = GateEvalResult(
        divergence=divergence,
        checkpoint=checkpoint,
        seeds=tuple(seeds),
        calibration=cal,
        ungated=ungated,
    )

    for q, t in zip(cal.quantiles, cal.thresholds):
        log(f"[gate] gated arms at q{q:g} (threshold {t:.6g})")
        spec = AnchorGateSpec(t, divergence)
        arms = run_arms(
            seeds,
            build_arms,
            adapt_and_eval_for,
            condition_for,
            gate_spec=spec,
            gate_inputs=gate_inputs,
        )
        result.thresholds.append(
            ThresholdResult(
                quantile=q,
                threshold=t,
                arms=arms,
                clean_regression=clean_regression(
                    np.asarray(arms.control), np.asarray(ungated.control)
                ),
            )
        )

    if overhead_fn is not None:
        # The threshold does not change the work, only the verdict: the
        # divergence is computed either way.
        spec = AnchorGateSpec(cal.thresholds[-1], divergence)
        log("[gate] timing gated vs ungated step")
        result.overhead = overhead_fn(spec)
        log(f"[gate] overhead: {result.overhead.summary()}")

    return result
