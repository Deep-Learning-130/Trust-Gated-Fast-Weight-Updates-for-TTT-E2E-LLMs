"""RunCondition matching discipline.

Poison and control must differ in *exactly one* thing: the content of the
stream. `PREREGISTERED.md` names control mismatch as an invalidating condition,
so every field gets a test in both directions -- a mismatch must raise and name
the field, and an identical pair must pass silently.
"""

import numpy as np
import pytest

from trustgate.attack.corpus import TokenCorpus
from trustgate.attack.stream import build_benign_control, build_select_stream
from trustgate.eval.harness import RunCondition, eval_tokens_digest

DIGEST = eval_tokens_digest(np.arange(64))
OTHER_DIGEST = eval_tokens_digest(np.arange(64, 128))


def condition(**overrides) -> RunCondition:
    base = dict(
        seed=0,
        stream_tokens=64,
        mini_batch_size=32,
        seq_length=8192,
        checkpoint="1b_ttt_e2e_finetune_books_8k_1x_cc",
        benign_eval_split="val",
        span_tokens=8,
        n_chunks=2,
        valid_tokens=64,
        dtype="int32",
        stream_corpus_split="train",
        eval_tokens_sha256=DIGEST,
        inner_lr_multiplier=1.0,
    )
    base.update(overrides)
    return RunCondition(**base)


def test_identical_conditions_match_silently():
    condition().assert_matches(condition())


@pytest.mark.parametrize(
    "field,value",
    [
        ("seed", 1),
        ("stream_tokens", 128),
        ("seq_length", 4096),
        ("checkpoint", "125m_ttt_e2e"),
        ("benign_eval_split", "train"),
        ("span_tokens", 16),
        ("valid_tokens", 63),
        ("dtype", "int16"),
        ("stream_corpus_split", "val"),
        ("eval_tokens_sha256", OTHER_DIGEST),
    ],
)
def test_any_single_field_mismatch_raises_and_names_the_field(field, value):
    overrides = {field: value}
    # stream_tokens and n_chunks are tied together by __post_init__.
    if field == "stream_tokens":
        overrides["n_chunks"] = value // 32

    with pytest.raises(ValueError, match=field) as excinfo:
        condition().assert_matches(condition(**overrides))

    assert "must be identical" in str(excinfo.value)


def test_mini_batch_size_mismatch_raises():
    # Its own test because n_chunks moves with it.
    with pytest.raises(ValueError, match="mini_batch_size"):
        condition().assert_matches(condition(mini_batch_size=64, n_chunks=1))


def test_chunk_count_must_follow_from_stream_and_batch_size():
    # A hand-built condition that lies about its step count would let two arms
    # take a different number of inner SGD steps while claiming to match.
    with pytest.raises(ValueError, match="n_chunks"):
        condition(n_chunks=5)


def test_span_tokens_must_divide_mini_batch_size():
    with pytest.raises(ValueError, match="must divide mini_batch_size"):
        condition(span_tokens=6)


def test_valid_tokens_cannot_exceed_stream_tokens():
    with pytest.raises(ValueError, match="exceeds stream_tokens"):
        condition(valid_tokens=65)


def test_unset_eval_digest_is_rejected_at_construction():
    # Two runs that both left it empty would *match*, which is exactly the
    # failure this field exists to prevent -- so it is caught at construction,
    # not at comparison.
    with pytest.raises(ValueError, match="full sha256 hex digest"):
        condition(eval_tokens_sha256="")


def test_truncated_eval_digest_is_rejected():
    with pytest.raises(ValueError, match="full sha256 hex digest"):
        condition(eval_tokens_sha256=DIGEST[:16])


def test_unsaturated_inner_lr_is_rejected_at_construction():
    with pytest.raises(ValueError, match="inner_lr_multiplier"):
        condition(inner_lr_multiplier=0.01)


def test_eval_tokens_digest_distinguishes_different_windows():
    # benign_eval_split is a name; two runs can evaluate different windows of the
    # same split and assert_matches would see nothing wrong without this.
    assert DIGEST != OTHER_DIGEST
    assert len(DIGEST) == 64


def test_eval_tokens_digest_is_stable_across_equal_content():
    assert eval_tokens_digest(np.arange(64)) == eval_tokens_digest(np.arange(64))


def test_from_stream_derives_matching_conditions_for_both_arms():
    rng = np.random.default_rng(7)
    corpus = TokenCorpus(rng.integers(10, 50000, 8192, dtype=np.int32), name="train")

    poison = build_select_stream(corpus, 64, 0, span_tokens=8, mini_batch_size=32)
    control = build_benign_control(corpus, 64, 1, span_tokens=8, mini_batch_size=32)

    shared = dict(
        seq_length=8192,
        checkpoint="1b",
        benign_eval_split="val",
        eval_tokens_sha256=DIGEST,
    )
    poison_condition = RunCondition.from_stream(poison, seed=0, **shared)
    control_condition = RunCondition.from_stream(control, seed=0, **shared)

    # Hand-transcribing these numbers is how an unmatched control gets built by
    # accident; derived from the streams they agree by construction.
    poison_condition.assert_matches(control_condition)
    assert poison_condition.n_chunks == 2
    assert poison_condition.valid_tokens == 64
    assert poison_condition.stream_corpus_split == "train"


def test_from_stream_surfaces_a_genuine_length_mismatch():
    rng = np.random.default_rng(7)
    corpus = TokenCorpus(rng.integers(10, 50000, 8192, dtype=np.int32), name="train")

    poison = build_select_stream(corpus, 64, 0, span_tokens=8, mini_batch_size=32)
    short = build_benign_control(corpus, 32, 1, span_tokens=8, mini_batch_size=32)

    shared = dict(
        seq_length=8192,
        checkpoint="1b",
        benign_eval_split="val",
        eval_tokens_sha256=DIGEST,
    )
    with pytest.raises(ValueError, match="stream_tokens"):
        RunCondition.from_stream(poison, seed=0, **shared).assert_matches(
            RunCondition.from_stream(short, seed=0, **shared)
        )


def test_run_attack_spike_fluency_scorer_integration():
    from trustgate.eval.harness import run_attack_spike, SpikeResult
    
    rng = np.random.default_rng(7)
    corpus = TokenCorpus(rng.integers(10, 50000, 8192, dtype=np.int32), name="train")
    
    def build_arms(seed: int):
        poison = build_select_stream(corpus, 64, seed, span_tokens=8, mini_batch_size=32)
        control = build_benign_control(corpus, 64, seed + 100, span_tokens=8, mini_batch_size=32)
        return poison, control
        
    def adapt_and_eval(stream, cond):
        return 1.5 if stream.strategy == "SELECT" else 1.0
        
    def mock_fluency_scorer(poison, control):
        return 1.25
        
    cond = condition(stream_tokens=64)
    
    result = run_attack_spike(
        spec=None,
        condition=cond,
        seeds=[0, 1, 2, 3, 4],
        output_dir=None,
        build_arms=build_arms,
        adapt_and_eval=adapt_and_eval,
        fluency_scorer=mock_fluency_scorer,
    )
    
    assert isinstance(result, SpikeResult)
    assert len(result.per_seed_fluency) == 5
    assert all(f == 1.25 for f in result.per_seed_fluency)
    
    # Also verify that when fluency_scorer is None, it outputs NaN
    result_none = run_attack_spike(
        spec=None,
        condition=cond,
        seeds=[0, 1, 2, 3, 4],
        output_dir=None,
        build_arms=build_arms,
        adapt_and_eval=adapt_and_eval,
        fluency_scorer=None,
    )
    assert all(np.isnan(f) for f in result_none.per_seed_fluency)
