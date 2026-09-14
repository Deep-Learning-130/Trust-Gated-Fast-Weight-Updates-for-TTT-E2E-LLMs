"""`run_attack_spike` end to end with fakes.

The point of the injectable seams is that when hardware arrives, only the
numbers are missing -- the per-seed loop, the matching guards, the collection and
the aggregation are all exercised here with no model, no corpus and no GPU.
"""

import numpy as np
import pytest

from trustgate.attack.corpus import TokenCorpus
from trustgate.attack.objectives import AttackSpec, Objective
from trustgate.attack.stream import (
    StreamStrategy,
    build_benign_control,
    build_select_stream,
)
from trustgate.eval.harness import (
    RunCondition,
    eval_tokens_digest,
    run_attack_spike,
    run_stream,
)

SPAN, MINI_BATCH, LENGTH = 8, 32, 64
DIGEST = eval_tokens_digest(np.arange(64))


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
        n_chunks=LENGTH // MINI_BATCH,
        valid_tokens=LENGTH,
        dtype="int32",
        stream_corpus_split="train",
        eval_tokens_sha256=DIGEST,
        inner_lr_multiplier=1.0,
    )
    base.update(overrides)
    return RunCondition(**base)


def make_arm_builder(corpus, poison_offset=0):
    def build_arms(seed):
        poison = build_select_stream(
            corpus, LENGTH, seed + poison_offset, span_tokens=SPAN, mini_batch_size=MINI_BATCH
        )
        control = build_benign_control(
            corpus, LENGTH, seed + 500, span_tokens=SPAN, mini_batch_size=MINI_BATCH
        )
        return poison, control

    return build_arms


def test_five_seeds_are_collected_for_both_arms():
    corpus = make_corpus()
    calls = []

    def adapt_and_eval(stream, condition):
        calls.append(condition.seed)
        return 1.0 + 0.01 * condition.seed

    result = run_attack_spike(
        AttackSpec(objective=Objective.DEGRADE, stream_tokens=LENGTH),
        make_condition(),
        [0, 1, 2, 3, 4],
        output_dir=None,
        build_arms=make_arm_builder(corpus),
        adapt_and_eval=adapt_and_eval,
    )

    assert len(result.per_seed_poisoned) == 5
    assert len(result.per_seed_control) == 5
    assert result.corruption.n_seeds == 5
    assert calls == [0, 0, 1, 1, 2, 2, 3, 3, 4, 4]  # poison then control, per seed


def test_the_per_seed_condition_carries_the_seed():
    corpus = make_corpus()
    seeds_seen = []

    def adapt_and_eval(stream, condition):
        seeds_seen.append(condition.seed)
        return 1.0

    run_attack_spike(
        AttackSpec(objective=Objective.DEGRADE, stream_tokens=LENGTH),
        make_condition(seed=99),
        [7, 8],
        output_dir=None,
        build_arms=make_arm_builder(corpus),
        adapt_and_eval=adapt_and_eval,
    )
    # Not 99: the template seed is overridden per iteration.
    assert seeds_seen == [7, 7, 8, 8]


def test_a_mismatched_control_aborts_the_run():
    corpus = make_corpus()

    def bad_arms(seed):
        poison = build_select_stream(
            corpus, LENGTH, seed, span_tokens=SPAN, mini_batch_size=MINI_BATCH
        )
        # Half the length: a control that would make a null look positive.
        control = build_benign_control(
            corpus, 32, seed + 500, span_tokens=SPAN, mini_batch_size=MINI_BATCH
        )
        return poison, control

    with pytest.raises(ValueError, match="structurally identical"):
        run_attack_spike(
            AttackSpec(objective=Objective.DEGRADE, stream_tokens=LENGTH),
            make_condition(),
            [0],
            output_dir=None,
            build_arms=bad_arms,
            adapt_and_eval=lambda stream, condition: 1.0,
        )


def test_a_stream_drawn_from_the_eval_split_aborts_the_run():
    # Control spans from /val while the eval also reads /val is a spurious
    # positive by construction, with nothing in the numbers to reveal it.
    corpus = make_corpus(name="val")
    with pytest.raises(ValueError, match="also\\s+the benign eval split"):
        run_attack_spike(
            AttackSpec(objective=Objective.DEGRADE, stream_tokens=LENGTH),
            make_condition(stream_corpus_split="val"),
            [0],
            output_dir=None,
            build_arms=make_arm_builder(corpus),
            adapt_and_eval=lambda stream, condition: 1.0,
        )


def test_no_seeds_raises():
    with pytest.raises(ValueError, match="requires 5 per condition"):
        run_attack_spike(
            AttackSpec(objective=Objective.DEGRADE, stream_tokens=LENGTH),
            make_condition(),
            [],
            output_dir=None,
            build_arms=make_arm_builder(make_corpus()),
            adapt_and_eval=lambda stream, condition: 1.0,
        )


def test_fluency_is_nan_without_a_scorer():
    result = run_attack_spike(
        AttackSpec(objective=Objective.DEGRADE, stream_tokens=LENGTH),
        make_condition(),
        [0, 1],
        output_dir=None,
        build_arms=make_arm_builder(make_corpus()),
        adapt_and_eval=lambda stream, condition: 1.0 + 0.01 * condition.seed,
    )
    assert not result.is_fluency_scored
    assert all(np.isnan(r) for r in result.per_seed_fluency)


def test_an_injected_fluency_scorer_fills_every_seed():
    result = run_attack_spike(
        AttackSpec(objective=Objective.DEGRADE, stream_tokens=LENGTH),
        make_condition(),
        [0, 1, 2],
        output_dir=None,
        build_arms=make_arm_builder(make_corpus()),
        adapt_and_eval=lambda stream, condition: 1.0 + 0.01 * condition.seed,
        fluency_scorer=lambda poison, control: 1.1,
    )
    assert result.is_fluency_scored
    assert result.per_seed_fluency == [1.1, 1.1, 1.1]
    assert result.gating_fluency_ratio == pytest.approx(1.1)


def test_strategy_is_read_off_the_streams_that_ran():
    result = run_attack_spike(
        AttackSpec(objective=Objective.DEGRADE, stream_tokens=LENGTH),
        make_condition(),
        [0, 1],
        output_dir=None,
        build_arms=make_arm_builder(make_corpus()),
        adapt_and_eval=lambda stream, condition: 1.0 + 0.01 * condition.seed,
    )
    assert result.strategy is StreamStrategy.SELECT


def test_run_attack_spike_writes_no_report(tmp_path):
    # Its own contract: it does not decide go/no-go. render_markdown computes the
    # verdict, so a harness that wrote the report would transitively decide.
    run_attack_spike(
        AttackSpec(objective=Objective.DEGRADE, stream_tokens=LENGTH),
        make_condition(),
        [0, 1],
        output_dir=tmp_path,
        build_arms=make_arm_builder(make_corpus()),
        adapt_and_eval=lambda stream, condition: 1.0 + 0.01 * condition.seed,
    )
    assert list(tmp_path.iterdir()) == []


def test_run_stream_refuses_to_reconstruct_the_split_from_a_model_alone():
    # Was blocked on checkpoints, then on binding a step function; now bound.
    # What it will not do is guess: the block/state split in `vendor_bind.bind`
    # cannot be recovered from the model without its `equinox.nn.State`, and a
    # harness that silently proceeded here would be measuring an unsplit model.
    with pytest.raises(ValueError, match="state.*or a prebuilt .binding"):
        run_stream(object(), np.arange(LENGTH), make_condition())
