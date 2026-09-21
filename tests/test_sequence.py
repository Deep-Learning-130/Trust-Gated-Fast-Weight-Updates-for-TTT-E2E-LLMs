"""`run_sequence_eval` end to end with fakes, plus the first `eval_benign_curve` test.

What this file establishes
--------------------------
That the sequence orchestrator threads carry where it claims to, resets it where
it claims to, and lands on exactly the number the frozen spike would have
reported. The last of those is the one that matters: two orchestrators measuring
"benign loss after the poison stream" and disagreeing would make both useless,
and nothing in the type system stops that.

What it cannot establish
------------------------
Anything about whether the attack works. Every loss here comes from a fake step
that adds arithmetic to a dict -- there is no model, no corpus semantics and no
adaptation. A curve that slopes the right way in this file is a curve the
harness computed correctly, not evidence of corruption.
"""

import sys
import types

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from trustgate.attack.corpus import TokenCorpus
from trustgate.attack.objectives import AttackSpec, Objective
from trustgate.attack.stream import build_benign_control, build_select_stream
from trustgate.eval import carry as carry_mod
from trustgate.eval.carry import InnerStepResult
from trustgate.eval.harness import RunCondition, eval_tokens_digest, run_attack_spike
from trustgate.eval.metrics import accumulation_excess, drift_floor_excess, onset_window
from trustgate.eval.report import render_sequence_markdown
from trustgate.eval.sequence import (
    CONTROL,
    CONTROL_NO_CARRY,
    POISON,
    POISON_NO_CARRY,
    run_sequence_eval,
    windows_from_chunks,
)

SPAN, MINI_BATCH, LENGTH = 8, 32, 256
N_WINDOWS = LENGTH // MINI_BATCH
DIGEST = eval_tokens_digest(np.arange(64))
SEEDS = [0, 1, 2, 3, 4]


def make_corpus(name="train"):
    rng = np.random.default_rng(999)
    return TokenCorpus(rng.integers(10, 50000, 16384, dtype=np.int32), name=name)


def make_condition(**overrides):
    base = dict(
        seed=0,
        stream_tokens=LENGTH,
        mini_batch_size=MINI_BATCH,
        seq_length=8192,
        checkpoint="1b",
        benign_eval_split="val",
        span_tokens=SPAN,
        n_chunks=N_WINDOWS,
        valid_tokens=LENGTH,
        dtype="int32",
        stream_corpus_split="train",
        eval_tokens_sha256=DIGEST,
        inner_lr_multiplier=1.0,
    )
    base.update(overrides)
    return RunCondition(**base)


def make_arm_builder(corpus):
    def build_arms(seed):
        poison = build_select_stream(
            corpus, LENGTH, seed, span_tokens=SPAN, mini_batch_size=MINI_BATCH
        )
        control = build_benign_control(
            corpus, LENGTH, seed + 500, span_tokens=SPAN, mini_batch_size=MINI_BATCH
        )
        return poison, control

    return build_arms


def windows_for(stream):
    """Chunk a stream the way the pilot does -- flat slices, no prefix pass.

    `stream.tokens` holds `length_tokens + 1` ids; the final one is the target
    of the last position and starts no chunk of its own.
    """
    return windows_from_chunks(
        carry_mod.chunk_tokens(stream.tokens[:-1], MINI_BATCH)
    )


def make_recording_step():
    """Adds each chunk's mean to the carry, and records what it was handed.

    Recording the *incoming* weights is the point, as in `tests/test_carry.py`:
    a threaded arm and a reset arm both produce a plausible final number, so
    only the sequence of inputs tells them apart.
    """
    seen = []

    def step(fast_weights, opt_state, state_tuple, chunk):
        seen.append(float(fast_weights["w"]))
        delta = float(np.asarray(chunk, dtype=np.float64).mean()) / 1e6
        return InnerStepResult(
            fast_weights={"w": fast_weights["w"] + delta},
            opt_state=(opt_state or 0) + 1,
            state_tuple=state_tuple,
            metrics={"loss": jnp.float32(1.0)},
        )

    return step, seen


def fresh_carry():
    return carry_mod.init_carry(fast_weights={"w": jnp.float32(0.0)}, opt_state=0)


def evaluate(carry, condition):
    """Benign loss as a deterministic function of the accumulated weight.

    Monotone in the carry so the direction of every curve in this file is
    predictable, which is what lets the assertions be about ordering rather
    than about magnitudes nobody chose.
    """
    return 2.0 + float(carry.fast_weights["w"])


def run(eval_every=1, seeds=SEEDS, step=None):
    corpus = make_corpus()
    step_fn = step if step is not None else make_recording_step()[0]
    return run_sequence_eval(
        make_condition(),
        seeds,
        build_arms=make_arm_builder(corpus),
        windows_for=windows_for,
        step_fn=step_fn,
        fresh_carry=fresh_carry,
        evaluate=evaluate,
        eval_every=eval_every,
    )


# ---------------------------------------------------------------------------
# The invariant that ties this module to the frozen one.
# ---------------------------------------------------------------------------


def test_final_position_equals_the_single_shot_endpoint():
    """The last point of the curve is what `run_attack_spike` reports.

    Both orchestrators claim to measure "benign loss after the poison stream".
    If they disagree, the position curve cannot be read alongside the kill-gate
    number and one of them is silently wrong. Nothing in the types prevents
    that -- only this test does.
    """
    corpus = make_corpus()
    build_arms = make_arm_builder(corpus)

    def adapt_and_eval(stream, condition):
        step_fn, _ = make_recording_step()
        carry = fresh_carry()
        for window in windows_for(stream):
            carry, _ = carry_mod.run_chunks(carry, window, step_fn)
        return evaluate(carry, condition)

    spike = run_attack_spike(
        AttackSpec(objective=Objective.DEGRADE, stream_tokens=LENGTH),
        make_condition(),
        SEEDS,
        None,
        build_arms=build_arms,
        adapt_and_eval=adapt_and_eval,
    )

    sequence = run(step=make_recording_step()[0])

    np.testing.assert_allclose(
        sequence.final_losses(POISON),
        spike.per_seed_poisoned,
        err_msg="the position curve's endpoint disagrees with the frozen spike",
    )
    np.testing.assert_allclose(
        sequence.final_losses(CONTROL), spike.per_seed_control
    )


# ---------------------------------------------------------------------------
# Carry: threaded where claimed, reset where claimed.
# ---------------------------------------------------------------------------


def test_threaded_arm_never_restarts_from_a_fresh_carry():
    step_fn, seen = make_recording_step()

    run_sequence_eval(
        make_condition(),
        [0],
        build_arms=make_arm_builder(make_corpus()),
        windows_for=windows_for,
        step_fn=step_fn,
        fresh_carry=fresh_carry,
        evaluate=evaluate,
        eval_every=1,
    )

    # Arms run in `ARM_NAMES` order, one chunk per window, so the first
    # `N_WINDOWS` recordings belong to `poison`.
    poison_inputs = seen[:N_WINDOWS]

    assert poison_inputs[0] == 0.0
    assert all(
        b > a for a, b in zip(poison_inputs, poison_inputs[1:])
    ), f"the threaded arm restarted somewhere: {poison_inputs}"


def test_no_carry_arm_starts_each_window_from_a_fresh_carry():
    """Every window of a no-carry arm must see weights at zero.

    This is the arm's entire purpose. If the reset leaked, the arm would be a
    second copy of the threaded one and `accumulation_excess` would read as a
    flat zero -- a null result that looks exactly like "accumulation does not
    matter".
    """
    step_fn, seen = make_recording_step()

    run_sequence_eval(
        make_condition(),
        [0],
        build_arms=make_arm_builder(make_corpus()),
        windows_for=windows_for,
        step_fn=step_fn,
        fresh_carry=fresh_carry,
        evaluate=evaluate,
        eval_every=1,
    )

    # `poison`, `control`, then `poison_no_carry`.
    no_carry_inputs = seen[2 * N_WINDOWS : 3 * N_WINDOWS]

    assert no_carry_inputs == [0.0] * N_WINDOWS, (
        f"a no-carry window inherited weights: {no_carry_inputs}"
    )


def test_mid_stream_measurement_does_not_contaminate_the_carry():
    """Evaluating between windows must not advance the stream's carry.

    `eval_benign_curve` discards the carry it computes (`harness.py:412`) and
    asks in its docstring that nobody optimise that away. This module is what
    makes the request load-bearing: an `evaluate` that leaked would add inner
    steps the stream never contained, and every position after the first would
    be measured at the wrong dose.
    """
    step_fn, seen = make_recording_step()

    def leak_attempt(carry, condition):
        # Advance a *local* carry, exactly as eval-time adaptation does, and
        # hand back a number. Nothing here may reach the caller's carry.
        carry_mod.run_chunks(carry, [np.arange(MINI_BATCH)], step_fn)
        return evaluate(carry, condition)

    result = run_sequence_eval(
        make_condition(),
        [0],
        build_arms=make_arm_builder(make_corpus()),
        windows_for=windows_for,
        step_fn=step_fn,
        fresh_carry=fresh_carry,
        evaluate=leak_attempt,
        eval_every=1,
    )

    threaded = result.arm(0, POISON).per_window_loss
    no_carry = result.arm(0, POISON_NO_CARRY).per_window_loss

    assert no_carry[0] == pytest.approx(threaded[0]), (
        "window 0 differs between threaded and reset arms; eval-time adaptation "
        "leaked into the stream carry"
    )


def test_floor_arm_reads_no_stream_tokens():
    result = run()
    assert result.floor_loss == pytest.approx(evaluate(fresh_carry(), None))


# ---------------------------------------------------------------------------
# Shape, stride and guards.
# ---------------------------------------------------------------------------


def test_every_seed_gets_every_arm():
    result = run()

    assert result.seeds == tuple(SEEDS)
    assert result.n_windows == N_WINDOWS
    for seed in SEEDS:
        for name in (POISON, CONTROL, POISON_NO_CARRY, CONTROL_NO_CARRY):
            assert len(result.arm(seed, name).per_window_loss) == N_WINDOWS


def test_eval_every_strides_the_measurements_but_not_the_stream():
    """A stride changes how often we look, never how much the model read."""
    dense = run(eval_every=1)
    sparse = run(eval_every=2)

    assert len(sparse.arm(0, POISON).per_window_loss) == N_WINDOWS // 2
    assert sparse.arm(0, POISON).window_index(0) == 1

    # The endpoint is a multiple of the stride, so both must land on it.
    assert sparse.arm(0, POISON).final_loss == pytest.approx(
        dense.arm(0, POISON).final_loss
    )


def test_ragged_stride_is_refused_rather_than_silently_truncated():
    with pytest.raises(ValueError, match="eval_every must be >= 1"):
        run(eval_every=0)


def test_a_stride_longer_than_the_stream_is_refused_before_any_arm_runs():
    """Otherwise every arm computes to the end with no measurement, and the
    report is the first thing to notice -- after the whole cost is paid."""
    step, calls = make_recording_step()
    with pytest.raises(ValueError, match="no window would ever be measured"):
        run(eval_every=N_WINDOWS + 1, step=step)
    assert not calls, "an arm ran before the stride was checked"


def test_no_seeds_is_refused():
    with pytest.raises(ValueError, match="no seeds"):
        run(seeds=[])


def test_unsaturated_inner_lr_fails_before_any_compute():
    """A near-frozen inner loop produces a null for the wrong reason."""

    def unsaturated():
        return carry_mod.init_carry(
            fast_weights={"w": jnp.float32(0.0)},
            opt_state=0,
            inner_lr_multiplier=0.01,
        )

    with pytest.raises(ValueError, match="inner LR multiplier"):
        run_sequence_eval(
            make_condition(),
            SEEDS,
            build_arms=make_arm_builder(make_corpus()),
            windows_for=windows_for,
            step_fn=make_recording_step()[0],
            fresh_carry=unsaturated,
            evaluate=evaluate,
        )


def test_arms_chunked_to_different_lengths_are_refused():
    """Equal token counts do not guarantee equal window counts.

    `windows_for` is injected, so `assert_streams_matched` cannot see it. A
    poison curve read position-by-position against a shorter control curve is a
    silent mismatch of exactly the kind `RunCondition` exists to prevent.
    """

    calls = []

    def lopsided(stream):
        """Full windows for the first arm, one short for the second."""
        calls.append(stream)
        windows = windows_for(stream)
        return windows if len(calls) == 1 else windows[:-1]

    with pytest.raises(ValueError, match="not comparable position by position"):
        run_sequence_eval(
            make_condition(),
            [0],
            build_arms=make_arm_builder(make_corpus()),
            windows_for=lopsided,
            step_fn=make_recording_step()[0],
            fresh_carry=fresh_carry,
            evaluate=evaluate,
        )


def test_windows_from_chunks_refuses_a_ragged_final_window():
    with pytest.raises(ValueError, match="not a multiple of chunks_per_window"):
        windows_from_chunks([1, 2, 3], chunks_per_window=2)


# ---------------------------------------------------------------------------
# Derived metrics.
# ---------------------------------------------------------------------------


def test_accumulation_excess_is_zero_when_the_stream_is_one_window():
    """With a single window there is nothing to accumulate across.

    The threaded and reset arms must then be identical, and the metric must say
    so rather than reporting float noise.
    """
    result = run_sequence_eval(
        make_condition(),
        [0],
        build_arms=make_arm_builder(make_corpus()),
        windows_for=lambda s: [carry_mod.chunk_tokens(s.tokens[:-1], LENGTH)],
        step_fn=make_recording_step()[0],
        fresh_carry=fresh_carry,
        evaluate=evaluate,
    )

    excess = accumulation_excess(
        result.arm(0, POISON).per_window_loss,
        result.arm(0, POISON_NO_CARRY).per_window_loss,
    )
    np.testing.assert_allclose(excess, [0.0], atol=1e-12)


def test_accumulation_excess_grows_once_carry_has_somewhere_to_go():
    result = run()
    excess = accumulation_excess(
        result.arm(0, POISON).per_window_loss,
        result.arm(0, POISON_NO_CARRY).per_window_loss,
    )

    assert excess[0] == pytest.approx(0.0, abs=1e-12)
    assert excess[-1] > excess[0]


def test_drift_floor_excess_measures_the_gap_to_reading_nothing():
    result = run()
    gap = drift_floor_excess(
        result.arm(0, CONTROL).per_window_loss, result.floor_loss
    )

    assert len(gap) == N_WINDOWS
    assert gap[0] == pytest.approx(
        result.arm(0, CONTROL).per_window_loss[0] - result.floor_loss
    )


def test_onset_window_returns_none_when_the_bar_is_never_cleared():
    result = run()
    assert (
        onset_window(
            result.arm(0, POISON).per_window_loss,
            result.arm(0, CONTROL).per_window_loss,
            delta=1e9,
        )
        is None
    )


def test_onset_window_finds_the_first_crossing():
    assert onset_window([1.0, 2.0, 5.0], [1.0, 1.0, 1.0], delta=0.5) == 1


def test_mismatched_curve_lengths_are_refused():
    with pytest.raises(ValueError, match="curve length mismatch"):
        accumulation_excess([1.0, 2.0], [1.0])
    with pytest.raises(ValueError, match="curve length mismatch"):
        onset_window([1.0, 2.0], [1.0], delta=0.1)


# ---------------------------------------------------------------------------
# Report: no verdict, ever.
# ---------------------------------------------------------------------------


def test_sequence_report_renders_no_verdict():
    """The secondary report must not be mistakable for the kill-gate one.

    `render_markdown` owns PROCEED/STOP. If this renderer ever grew a verdict
    line, a reader scanning `results/` for a bold verdict would find two, and
    the non-gating claim would be a matter of trust rather than structure.
    """
    text = render_sequence_markdown(run(), onset_delta=0.01)

    # The banner names PROCEED/STOP to say where the verdict *does* live, so
    # the check is for the verdict line and threshold marks themselves.
    assert "Verdict" not in text
    assert "PASS" not in text
    assert "FAIL" not in text
    assert "NON-GATING" in text
    assert "no verdict" in text


def test_sequence_report_prints_every_measured_value():
    """Standing Rule 6 -- unflattering numbers get reported, not dropped."""
    result = run()
    text = render_sequence_markdown(result, onset_delta=0.01)

    for seed in SEEDS:
        for name in (POISON, CONTROL, POISON_NO_CARRY, CONTROL_NO_CARRY):
            for value in result.arm(seed, name).per_window_loss:
                assert f"{value:.6f}" in text


def test_sequence_report_states_the_onset_delta_it_used():
    """An onset without its sensitivity is not interpretable."""
    text = render_sequence_markdown(run(), onset_delta=0.0025)
    assert "0.0025" in text


# ---------------------------------------------------------------------------
# `eval_benign_curve` against a fake vendor tree -- its first coverage.
# ---------------------------------------------------------------------------


class FakeBatch(eqx.Module):
    input_ids: jnp.ndarray
    target_tokens: jnp.ndarray
    loss_masks: jnp.ndarray


class _LossKey:
    """Stands in for `MetaModel.MetricType.loss` -- matched on `.name`."""

    name = "loss"


LOSS = _LossKey()


class FakeMetaModel:
    @staticmethod
    def inner_loop_step(model, opt_state, state_tuple, seq, prefix_outputs):
        return _FakeStepResult(
            new_model=_split_model(model.inner + 1.0, model.outer),
            new_optimizer_state=opt_state + 1,
            new_state=state_tuple,
            metrics={LOSS: jnp.asarray(float(model.inner.sum()))},
        )


class _FakeStepResult(eqx.Module):
    new_model: object
    new_optimizer_state: object
    new_state: object
    metrics: dict


@pytest.fixture
def vendor_bind(monkeypatch):
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
    mods["ttt.model.data"].tree_slice = lambda tree, i: jax.tree.map(
        lambda x: x[i], tree
    )
    mods["ttt.model.transformer"].MetaModel = FakeMetaModel
    mods["ttt.model.transformer"].BlockCollectionSplit = object
    mods["ttt.utils.filter_utils"].get_filter_spec = lambda *a, **k: None
    mods["ttt.utils.jax_utils"].clone_pytree = lambda t: t

    for name, mod in mods.items():
        monkeypatch.setitem(sys.modules, name, mod)
    # The fake step reads a concrete loss (`float(...)`), so it runs eagerly.
    monkeypatch.setenv("TRUSTGATE_NO_JIT", "1")

    from trustgate.eval import vendor_bind as vb

    return vb


class _FakePrefixResult:
    def __init__(self, last_hidden_state):
        self.last_hidden_state = last_hidden_state


class _FakeLanguageModel:
    """Enough of the vendor's language model for `prefix_pass` to run.

    The prefix is deliberately an identity map over the embedded ids: it plays
    no part in what is being tested and a non-trivial one would only add
    numbers nobody chose.
    """

    model = types.SimpleNamespace(h=types.SimpleNamespace(prefix_blocks=None))

    @staticmethod
    def wte_call(input_ids):
        return jnp.asarray(input_ids, dtype=jnp.float32)[:, None]

    @staticmethod
    def prefix_call(blocks, xt_embed, state_prefix, seq):
        return _FakePrefixResult(xt_embed)


#: Shared singletons. `eqx.field(static=True)` puts these in the pytree
#: *metadata*, which jax compares by equality -- so two separately-constructed
#: instances would make `model_split` and `spec_inner` structurally different
#: trees and every `tree_map` over the pair would fail.
_FAKE_LM = _FakeLanguageModel()
_FAKE_CONFIG = types.SimpleNamespace(
    model=types.SimpleNamespace(bos_token_id=0)
)


class _FakeSplitModel(eqx.Module):
    """Carries the leaves `spec_inner` partitions, plus static vendor attrs."""

    inner: jnp.ndarray
    outer: jnp.ndarray

    language_model: object = eqx.field(static=True)
    config: object = eqx.field(static=True)


EVAL_MINI_BATCH = 4


def _split_model(inner, outer):
    return _FakeSplitModel(
        inner=inner, outer=outer, language_model=_FAKE_LM, config=_FAKE_CONFIG
    )


def make_eval_binding(vendor_bind):
    model = _split_model(jnp.zeros(3), jnp.ones(3))
    return vendor_bind.VendorBinding(
        model_split=model,
        model_outer=model,
        spec_inner=_split_model(True, False),
        state_all=None,
        state_prefix=None,
        state_suffix=None,
        mini_batch_size=EVAL_MINI_BATCH,
        seq_length=8,
        suffix_len=1,
        inner_lr_multiplier=1.0,
        state_dtype="fp32",
    )


def test_eval_benign_curve_returns_one_loss_per_eval_chunk(vendor_bind):
    """First coverage for `eval_benign_curve`; it had none.

    The curve is returned rather than only its mean because measurement stays
    in `meta` mode, so the fast weights keep adapting across eval chunks and
    the signal dilutes along the way. A caller that only ever saw the mean
    could not tell a strong first chunk from a uniform one.
    """
    from trustgate.eval.harness import eval_benign_curve

    binding = make_eval_binding(vendor_bind)
    eval_tokens = np.arange(9, dtype=np.int32)
    condition = make_condition(
        eval_tokens_sha256=eval_tokens_digest(eval_tokens[:-1])
    )
    carry = carry_mod.init_carry(fast_weights=binding.fast_weights(), opt_state=0)

    result = eval_benign_curve(binding, carry, eval_tokens, condition)

    assert len(result.per_chunk_loss) == 8 // EVAL_MINI_BATCH
    assert result.mean_loss == pytest.approx(np.mean(result.per_chunk_loss))


def test_eval_benign_curve_does_not_mutate_the_callers_carry(vendor_bind):
    """The property `sequence.py` depends on, pinned at its source.

    `harness.py:387-391` asks that this not be "optimised" away. Until now
    nothing checked it, and the sequence orchestrator measures between every
    pair of windows -- a leak would advance the stream's carry by an inner step
    per measurement and mis-dose every position after the first.
    """
    from trustgate.eval.harness import eval_benign_curve

    binding = make_eval_binding(vendor_bind)
    eval_tokens = np.arange(9, dtype=np.int32)
    condition = make_condition(
        eval_tokens_sha256=eval_tokens_digest(eval_tokens[:-1])
    )
    carry = carry_mod.init_carry(fast_weights=binding.fast_weights(), opt_state=0)
    before = jnp.asarray(carry.fast_weights.inner)

    eval_benign_curve(binding, carry, eval_tokens, condition)

    np.testing.assert_allclose(
        carry.fast_weights.inner,
        before,
        err_msg="eval-time adaptation leaked back into the caller's carry",
    )


def test_eval_benign_curve_refuses_tokens_that_do_not_match_the_condition(
    vendor_bind,
):
    """The split *name* is not the data -- the digest is."""
    from trustgate.eval.harness import eval_benign_curve

    binding = make_eval_binding(vendor_bind)
    carry = carry_mod.init_carry(fast_weights=binding.fast_weights(), opt_state=0)

    with pytest.raises(ValueError, match="do not match"):
        eval_benign_curve(
            binding, carry, np.arange(9, dtype=np.int32), make_condition()
        )
