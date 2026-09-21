"""The compiled inner step and the Phase 2 gate wiring, against a trace-safe fake.

Same division as `tests/test_vendor_bind.py`: the vendor is not importable here,
so this establishes the seam logic -- that the step compiles, that a gate passed
as a value actually gates, that gated and ungated graphs are not confused by the
jit cache, and that calibration and the report behave. It cannot establish that
the probe forward matches the real `MetaModel`; the random-init `--gate-eval`
smoke on the box is what checks that.
"""

import sys
import types

import equinox as eqx
import jax.numpy as jnp
import numpy as np
import pytest

from trustgate.eval.overhead import OverheadResult


class FakeBatch(eqx.Module):
    input_ids: jnp.ndarray
    target_tokens: jnp.ndarray
    loss_masks: jnp.ndarray


class FakeStepResult(eqx.Module):
    new_model: object
    new_optimizer_state: object
    new_state: object
    metrics: dict


class _FakeLM:
    def __init__(self, model):
        self.model = model

    def suffix_call(self, prefix_outputs, state, seq):
        logits = prefix_outputs[:, None] * self.model.inner[None, :] + self.model.outer
        return types.SimpleNamespace(logits=logits)


class FakeModel(eqx.Module):
    """One adapting leaf, one frozen leaf; trace-safe throughout."""

    inner: jnp.ndarray
    outer: jnp.ndarray

    def inner_parameters(self):
        return FakeModel(inner=self.inner, outer=None)

    @property
    def language_model(self):
        return _FakeLM(self)


class FakeMetaModel:
    @staticmethod
    def inner_loop_step(model, opt_state, state_tuple, seq, prefix_outputs):
        # A deterministic update that depends on the data, and a loss read
        # before it -- the vendor's predict-then-adapt order.
        loss = jnp.mean((prefix_outputs[:, None] * model.inner[None, :]) ** 2)
        # Non-uniform on purpose: softmax ignores a uniform shift, so a
        # uniform update would leave the probe distribution untouched.
        step = jnp.mean(prefix_outputs) * jnp.arange(model.inner.shape[0], dtype=jnp.float32)
        return FakeStepResult(
            new_model=FakeModel(inner=model.inner + step, outer=model.outer),
            new_optimizer_state=opt_state,
            new_state=state_tuple,
            metrics={"loss": loss},
        )


@pytest.fixture
def vb(monkeypatch):
    mods = {
        "ttt": types.ModuleType("ttt"),
        "ttt.model": types.ModuleType("ttt.model"),
        "ttt.model.data": types.ModuleType("ttt.model.data"),
        "ttt.model.transformer": types.ModuleType("ttt.model.transformer"),
        "ttt.utils": types.ModuleType("ttt.utils"),
        "ttt.utils.filter_utils": types.ModuleType("ttt.utils.filter_utils"),
        "ttt.utils.jax_utils": types.ModuleType("ttt.utils.jax_utils"),
    }
    mods["ttt.model.data"].Batch = FakeBatch
    mods["ttt.model.data"].tree_slice = lambda tree, i: tree
    mods["ttt.model.transformer"].MetaModel = FakeMetaModel
    mods["ttt.model.transformer"].BlockCollectionSplit = object
    mods["ttt.utils.filter_utils"].get_filter_spec = lambda *a, **k: None
    mods["ttt.utils.jax_utils"].clone_pytree = lambda t: t
    for name, mod in mods.items():
        monkeypatch.setitem(sys.modules, name, mod)
    monkeypatch.delenv("TRUSTGATE_NO_JIT", raising=False)

    from trustgate.eval import vendor_bind

    vendor_bind._STEP_CACHE.clear()
    yield vendor_bind
    vendor_bind._STEP_CACHE.clear()


def make_binding(vb):
    model = FakeModel(inner=jnp.zeros(3), outer=jnp.ones(3))
    return vb.VendorBinding(
        model_split=model,
        model_outer=model,
        spec_inner=FakeModel(inner=True, outer=False),
        state_all=None,
        state_prefix=None,
        state_suffix=None,
        mini_batch_size=4,
        seq_length=8,
        suffix_len=1,
        inner_lr_multiplier=1.0,
        state_dtype="fp32",
    )


def chunk(value=1.0):
    return (None, jnp.full((4,), value, dtype=jnp.float32))


def gate_inputs(binding):
    from trustgate.eval.gate_eval import GateInputs

    return GateInputs(
        theta_0=binding.fast_weights(),
        probe_seq=None,
        probe_prefix=jnp.linspace(0.5, 1.5, 4),
        state_suffix=None,
    )


def run(step_fn, binding, value=1.0):
    return step_fn(binding.fast_weights(), 0, None, chunk(value))


# ------------------------------------------------------------ compiled step --


def test_compiled_step_matches_the_eager_step(vb, monkeypatch):
    binding = make_binding(vb)
    compiled = run(vb.make_step_fn(binding), binding)

    monkeypatch.setenv("TRUSTGATE_NO_JIT", "1")
    eager = run(vb.make_step_fn(binding), binding)

    np.testing.assert_allclose(compiled.fast_weights.inner, eager.fast_weights.inner)
    np.testing.assert_allclose(compiled.metrics["loss"], eager.metrics["loss"])
    assert compiled.fast_weights.outer is None, "the carry must hold fast weights only"


def test_compiled_step_is_reused_across_calls(vb):
    binding = make_binding(vb)
    vb.make_step_fn(binding)
    vb.make_step_fn(binding)
    assert len(vb._STEP_CACHE) == 1


def test_gate_inputs_without_a_spec_are_refused(vb):
    binding = make_binding(vb)
    with pytest.raises(ValueError):
        vb.make_step_fn(binding, None, gate_inputs(binding))


# -------------------------------------------------------------- gated step --


def test_observe_only_gate_is_a_no_op_on_the_weights(vb):
    from trustgate.eval.gate_eval import OBSERVE_ONLY_THRESHOLD, AnchorGateSpec

    binding = make_binding(vb)
    ungated = run(vb.make_step_fn(binding), binding)
    gated = run(
        vb.make_step_fn(binding, AnchorGateSpec(OBSERVE_ONLY_THRESHOLD), gate_inputs(binding)),
        binding,
    )

    np.testing.assert_allclose(gated.fast_weights.inner, ungated.fast_weights.inner)
    assert float(gated.metrics["gate/accepted"]) == 1.0
    assert float(gated.metrics["gate/drift_delta"]) > 0.0, (
        "an update that moved the weights should move the probe logits"
    )


def test_tight_gate_rejects_and_keeps_the_pre_update_weights(vb):
    from trustgate.eval.gate_eval import AnchorGateSpec

    binding = make_binding(vb)
    # Compile the ungated graph first: a class-patch design would now reuse it
    # for the gated call and silently accept.
    ungated = run(vb.make_step_fn(binding), binding)
    gated = run(
        vb.make_step_fn(binding, AnchorGateSpec(1e-12), gate_inputs(binding)), binding
    )

    assert float(gated.metrics["gate/accepted"]) == 0.0
    np.testing.assert_allclose(gated.fast_weights.inner, jnp.zeros(3))
    assert not np.allclose(ungated.fast_weights.inner, gated.fast_weights.inner)


def test_each_threshold_gets_its_own_graph(vb):
    from trustgate.eval.gate_eval import AnchorGateSpec

    binding = make_binding(vb)
    inputs = gate_inputs(binding)
    loose = run(vb.make_step_fn(binding, AnchorGateSpec(1e6), inputs), binding)
    tight = run(vb.make_step_fn(binding, AnchorGateSpec(1e-12), inputs), binding)
    assert float(loose.metrics["gate/accepted"]) == 1.0
    assert float(tight.metrics["gate/accepted"]) == 0.0


# --------------------------------------------------------------- calibration --


def _fake_stream(divergences, accepted=None):
    accepted = accepted or [1.0] * len(divergences)

    def run_stream_fn(*, gate_spec, gate_inputs, metrics_out):
        for d, a in zip(divergences, accepted):
            metrics_out.append({"gate/drift_delta": d, "gate/accepted": a})

    return run_stream_fn


def test_calibration_reads_thresholds_off_the_clean_quantiles():
    from trustgate.eval.gate_eval import calibrate

    divs = list(np.linspace(0.1, 1.0, 10))
    cal = calibrate(_fake_stream(divs), None, [0.5, 0.9], divergence="symmetric_kl")
    assert cal.thresholds == pytest.approx(
        (np.quantile(divs, 0.5), np.quantile(divs, 0.9))
    )
    assert cal.unscorable_steps == 0


def test_calibration_excludes_and_counts_unscorable_steps():
    from trustgate.eval.gate_eval import calibrate

    cal = calibrate(
        _fake_stream([0.5, 0.0, 0.7], accepted=[1.0, 0.0, 1.0]),
        None,
        [1.0],
        divergence="symmetric_kl",
    )
    assert cal.unscorable_steps == 1
    assert cal.thresholds == (pytest.approx(0.7),)


def test_calibration_refuses_a_probe_that_never_scores():
    from trustgate.eval.gate_eval import calibrate

    with pytest.raises(ValueError, match="probe forward"):
        calibrate(_fake_stream([0.0], accepted=[0.0]), None, [0.9], divergence="kl")


def test_a_zero_quantile_is_floored_to_a_legal_threshold():
    from trustgate.eval.gate_eval import MIN_THRESHOLD, calibrate

    cal = calibrate(_fake_stream([0.0, 0.0, 0.0]), None, [0.5], divergence="kl")
    assert cal.thresholds == (MIN_THRESHOLD,)


# ------------------------------------------------------------ orchestration --


def test_run_gate_eval_and_report_end_to_end():
    from trustgate.eval.gate_eval import run_gate_eval
    from trustgate.eval.report import GATE_NOT_A_VERDICT, render_gate_markdown

    def adapt_and_eval_for(*, gate_spec, gate_inputs, metrics_out):
        def fn(stream, condition):
            gated = gate_spec is not None
            metrics_out.append({"gate/accepted": 0.5 if gated else 1.0})
            base = 2.0 + 0.01 * condition
            if stream == "poison":
                return base + (0.05 if gated else 0.3)
            return base + (0.02 if gated else 0.0)

        return fn

    result = run_gate_eval(
        seeds=[0, 1, 2],
        build_arms=lambda seed: ("poison", "control"),
        adapt_and_eval_for=adapt_and_eval_for,
        condition_for=lambda stream, seed: seed,
        calibrate_stream_fn=_fake_stream(list(np.linspace(0.1, 1.0, 20))),
        gate_inputs=None,
        quantiles=[0.9, 0.99],
        divergence="symmetric_kl",
        checkpoint="fake",
        overhead_fn=lambda spec: OverheadResult(1.0, 1.25, 3, (1.0,) * 3, (1.25,) * 3),
        log=lambda _msg: None,
    )

    assert len(result.thresholds) == 2
    first = result.thresholds[0]
    assert first.arms.poison_accept_rate == (0.5, 0.5, 0.5)
    assert first.clean_regression > 0, "the gated control arm is worse by construction"
    assert result.ungated.corruption.relative_degradation > first.arms.corruption.relative_degradation

    text = render_gate_markdown(result)
    assert GATE_NOT_A_VERDICT in text
    assert "+25.0%" in text, "the overhead number itself must be in the report"
    assert "PROCEED" not in text.replace("PROCEED/STOP", "")


# ----------------------------------------------------------------------- cli --


def test_cli_gate_eval_needs_a_victim(tmp_path):
    from trustgate.eval.cli import main

    with pytest.raises(SystemExit, match="needs a victim"):
        main(["--objective", "degrade", "--strategy", "select", "--gate-eval",
              "--out", str(tmp_path)])


def test_cli_gate_eval_against_a_checkpoint_needs_the_spike_arms(tmp_path):
    from trustgate.eval.cli import main

    tokens = tmp_path / "t.npy"
    np.save(tokens, np.arange(10, dtype=np.int32))
    with pytest.raises(SystemExit, match="--arms-file"):
        main(["--objective", "degrade", "--strategy", "select", "--gate-eval",
              "--checkpoint", str(tmp_path), "--corpus-file", str(tokens),
              "--eval-file", str(tokens), "--out", str(tmp_path)])


def test_cli_sequence_eval_against_a_checkpoint_needs_the_spike_arms(tmp_path):
    """The addendum ties the sequence endpoints to `corruption_metric` on the
    same streams; synthetic orderings against real weights would not be that."""
    from trustgate.eval.cli import main

    tokens = tmp_path / "t.npy"
    np.save(tokens, np.arange(10, dtype=np.int32))
    with pytest.raises(SystemExit, match="--arms-file"):
        main(["--objective", "degrade", "--strategy", "select", "--sequence-eval",
              "--checkpoint", str(tmp_path), "--corpus-file", str(tokens),
              "--eval-file", str(tokens), "--out", str(tmp_path)])
