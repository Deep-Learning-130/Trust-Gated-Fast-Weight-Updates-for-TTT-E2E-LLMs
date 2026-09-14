"""Crafted-stream construction.

DEFENSIVE RESEARCH. CPU only, synthetic corpus, no model.

The matching tests are the ones that carry weight. Fast weights drift on *any*
stream, so an unmatched control makes a null result look like a positive one --
and the pre-registration names control mismatch as an invalidating condition.
"""

import numpy as np
import pytest

from trustgate.attack.corpus import TokenCorpus
from trustgate.attack.stream import (
    CraftedStream,
    StreamStrategy,
    assert_arms_distinguishable,
    assert_stream_not_from_eval_split,
    assert_streams_matched,
    build_benign_control,
    build_contiguous_stream,
    build_difficulty_matched_control,
    build_paraphrase_stream,
    build_select_stream,
    build_soft_stream,
    generate_seed_pairs,
)
from trustgate.tokens import BOS_TOKEN_ID

SPAN = 8
MINI_BATCH = 32
LENGTH = 64  # 8 spans, 2 inner-loop chunks


def make_corpus(n_tokens=8192, name="train"):
    rng = np.random.default_rng(4242)
    tokens = rng.integers(10, 50000, size=n_tokens, dtype=np.int32)
    return TokenCorpus(tokens, name=name)


def build(seed=0, **kwargs):
    kwargs.setdefault("span_tokens", SPAN)
    kwargs.setdefault("mini_batch_size", MINI_BATCH)
    return build_select_stream(make_corpus(), LENGTH, seed, **kwargs)


def test_window_convention_yields_length_plus_one_tokens():
    # lm_dataset.py:22 yields seq_len + 1 and shifts in the loader, so the extra
    # token is required -- and it must be real corpus text, not padding.
    stream = build()
    assert len(stream.tokens) == LENGTH + 1
    assert stream.length_tokens == LENGTH


def test_the_lookahead_token_is_never_bos():
    # BOS padding would be masked straight out of the loss and would silently
    # break valid-token matching against the control.
    for seed in range(6):
        assert int(build(seed=seed).tokens[-1]) != BOS_TOKEN_ID


def test_stream_carries_no_bos_at_all():
    stream = build()
    assert BOS_TOKEN_ID not in set(stream.tokens.tolist())
    assert stream.valid_tokens == LENGTH


def test_chunk_count_follows_mini_batch_size():
    stream = build()
    assert stream.n_chunks == LENGTH // MINI_BATCH == 2


def test_spans_tile_the_requested_length():
    stream = build()
    assert len(stream.spans) == LENGTH // SPAN
    assert all(s.length == SPAN for s in stream.spans)


def test_same_seed_reproduces_a_byte_identical_stream():
    assert np.array_equal(build(seed=11).tokens, build(seed=11).tokens)


def test_different_seed_changes_the_stream():
    assert not np.array_equal(build(seed=1).tokens, build(seed=2).tokens)


def test_length_not_divisible_by_span_tokens_raises():
    with pytest.raises(ValueError, match="multiple of span_tokens"):
        build_select_stream(make_corpus(), 60, 0, span_tokens=8, mini_batch_size=MINI_BATCH)


def test_length_not_divisible_by_mini_batch_size_raises():
    with pytest.raises(ValueError, match="multiple of\\s+mini_batch_size"):
        build_select_stream(make_corpus(), 48, 0, span_tokens=8, mini_batch_size=32)


def test_span_tokens_not_dividing_mini_batch_size_raises():
    with pytest.raises(ValueError, match="must divide mini_batch_size"):
        build_select_stream(make_corpus(), LENGTH, 0, span_tokens=6, mini_batch_size=MINI_BATCH)


def test_poison_and_control_are_structurally_matched():
    corpus = make_corpus()
    poison = build_select_stream(corpus, LENGTH, 0, span_tokens=SPAN, mini_batch_size=MINI_BATCH)
    control = build_benign_control(corpus, LENGTH, 1, span_tokens=SPAN, mini_batch_size=MINI_BATCH)

    assert_streams_matched(poison, control)
    assert not np.array_equal(poison.tokens, control.tokens)  # content differs


def test_assert_streams_matched_rejects_a_token_count_mismatch():
    corpus = make_corpus()
    poison = build_select_stream(corpus, LENGTH, 0, span_tokens=SPAN, mini_batch_size=MINI_BATCH)
    short = build_select_stream(corpus, 32, 1, span_tokens=SPAN, mini_batch_size=MINI_BATCH)
    with pytest.raises(ValueError, match="token count"):
        assert_streams_matched(poison, short)


def test_assert_streams_matched_rejects_a_chunk_count_mismatch():
    corpus = make_corpus()
    poison = build_select_stream(corpus, LENGTH, 0, span_tokens=SPAN, mini_batch_size=MINI_BATCH)
    other = build_select_stream(corpus, LENGTH, 1, span_tokens=SPAN, mini_batch_size=64)
    with pytest.raises(ValueError, match="chunk count"):
        assert_streams_matched(poison, other)


def test_assert_streams_matched_rejects_a_valid_token_mismatch():
    # A BOS smuggled into one arm changes the loss denominator for that chunk and
    # rescales every surviving token's gradient.
    poison = build()
    tampered = CraftedStream(
        tokens=poison.tokens.copy(),
        strategy=poison.strategy,
        span_tokens=poison.span_tokens,
        mini_batch_size=poison.mini_batch_size,
    )
    tampered.tokens[5] = BOS_TOKEN_ID
    with pytest.raises(ValueError, match="valid tokens"):
        assert_streams_matched(poison, tampered)


def test_unscored_stream_fails_safe_to_stop():
    stream = build()
    assert not stream.is_scored
    assert np.isnan(stream.perplexity)
    # nan never satisfies the realism bar, so an unscored stream cannot pass it.
    assert not (stream.fluency_ratio <= 1.5)


def test_scored_stream_reports_a_ratio():
    stream = build()
    stream.perplexity = 12.0
    stream.control_perplexity = 10.0
    assert stream.is_scored
    assert stream.fluency_ratio == pytest.approx(1.2)


def test_order_fn_permutes_without_resizing():
    corpus = make_corpus()
    reversed_stream = build_select_stream(
        corpus, LENGTH, 0, span_tokens=SPAN, mini_batch_size=MINI_BATCH,
        order_fn=lambda spans: list(reversed(spans)),
    )
    plain = build_select_stream(corpus, LENGTH, 0, span_tokens=SPAN, mini_batch_size=MINI_BATCH)

    # Same spans, different order: this is exactly the attacker's lever.
    assert {s.offset for s in reversed_stream.spans} == {s.offset for s in plain.spans}
    assert not np.array_equal(reversed_stream.tokens, plain.tokens)


def test_order_fn_that_resizes_raises():
    with pytest.raises(ValueError, match="must permute, not resize"):
        build_select_stream(
            make_corpus(), LENGTH, 0, span_tokens=SPAN, mini_batch_size=MINI_BATCH,
            order_fn=lambda spans: list(spans)[:2],
        )


def test_spans_are_distinct_by_default():
    # With replacement one span could fill every chunk -- eight epochs on one
    # passage, not an attack.
    assert build().span_multiplicity == 1


def test_provenance_statistics_are_available():
    stream = build()
    assert stream.mean_offset_gap > 0
    assert stream.corpus_name == "train"


def test_contiguous_arm_is_one_unbroken_span():
    stream = build_contiguous_stream(make_corpus(), LENGTH, 0, mini_batch_size=MINI_BATCH)
    assert len(stream.spans) == 1
    assert len(stream.tokens) == LENGTH + 1
    # It is the absolute fluency anchor; no bar is expressed over it.


def test_difficulty_matched_control_tracks_the_poison_arms_scores():
    corpus = make_corpus()
    poison = build_select_stream(corpus, LENGTH, 0, span_tokens=SPAN, mini_batch_size=MINI_BATCH)

    # Injected scorer, never the victim: a deterministic stand-in for the
    # reference model's per-span NLL.
    def scorer(tokens):
        return float(np.mean(tokens)) / 1000.0

    matched = build_difficulty_matched_control(corpus, poison, seed=5, scorer=scorer)

    assert_streams_matched(poison, matched)
    poison_scores = sorted(scorer(corpus.read(s.offset, s.length)) for s in poison.spans)
    matched_scores = sorted(scorer(corpus.read(s.offset, s.length)) for s in matched.spans)
    naive = build_benign_control(corpus, LENGTH, 77, span_tokens=SPAN, mini_batch_size=MINI_BATCH)
    naive_scores = sorted(scorer(corpus.read(s.offset, s.length)) for s in naive.spans)

    matched_error = sum(abs(a - b) for a, b in zip(poison_scores, matched_scores))
    naive_error = sum(abs(a - b) for a, b in zip(poison_scores, naive_scores))
    assert matched_error <= naive_error


def test_stream_from_the_eval_split_is_rejected():
    stream = build_select_stream(
        make_corpus(name="val"), LENGTH, 0, span_tokens=SPAN, mini_batch_size=MINI_BATCH
    )
    with pytest.raises(ValueError, match="also\\s+the benign eval split"):
        assert_stream_not_from_eval_split(stream, benign_eval_split="val")


def test_stream_from_a_different_split_passes():
    assert_stream_not_from_eval_split(build(), benign_eval_split="val")


def test_strategy_is_recorded():
    assert build().strategy is StreamStrategy.SELECT


@pytest.mark.parametrize("length", [32, 64, 96, 128, 256])
def test_exact_token_counts_at_several_lengths(length):
    stream = build_select_stream(make_corpus(), length, 3, span_tokens=SPAN, mini_batch_size=MINI_BATCH)
    assert len(stream.tokens) == length + 1
    assert stream.length_tokens == length
    assert stream.valid_tokens == length
    assert stream.n_chunks == length // MINI_BATCH


@pytest.mark.parametrize("length", [32, 64, 96, 128, 256])
def test_same_seed_is_byte_identical_at_several_lengths(length):
    a = build_select_stream(make_corpus(), length, 11, span_tokens=SPAN, mini_batch_size=MINI_BATCH)
    b = build_select_stream(make_corpus(), length, 11, span_tokens=SPAN, mini_batch_size=MINI_BATCH)
    assert np.array_equal(a.tokens, b.tokens)


def test_paraphrase_raises_not_implemented():
    """T2.5: PARAPHRASE blocked on T3.2 (tokenizer). Interface is ready."""
    with pytest.raises(NotImplementedError, match="PARAPHRASE blocked on T3.2"):
        build_paraphrase_stream(
            make_corpus(), LENGTH, 0, tokenizer=None, perturbation_fn=None
        )


def test_soft_raises_not_implemented():
    """T2.5: SOFT blocked on model access. Interface is ready."""
    with pytest.raises(NotImplementedError, match="SOFT blocked on model access"):
        build_soft_stream(
            make_corpus(), LENGTH, 0, embedding_matrix=None, optimization_fn=None
        )


def test_generate_seed_pairs_returns_correct_count():
    """T2.6: five seeds -> five pairs."""
    corpus = make_corpus()
    pairs = generate_seed_pairs(corpus, LENGTH, [0, 1, 2, 3, 4], span_tokens=SPAN, mini_batch_size=MINI_BATCH)
    assert len(pairs) == 5


def test_generate_seed_pairs_each_pair_matches():
    """T2.6: every pair passes assert_streams_matched."""
    corpus = make_corpus()
    pairs = generate_seed_pairs(corpus, LENGTH, [0, 1, 2], span_tokens=SPAN, mini_batch_size=MINI_BATCH)
    for poison, control in pairs:
        assert_streams_matched(poison, control)


def test_generate_seed_pairs_reproducible_from_seed():
    """T2.6: same seed -> same pair."""
    corpus = make_corpus()
    pairs_a = generate_seed_pairs(corpus, LENGTH, [7, 13], span_tokens=SPAN, mini_batch_size=MINI_BATCH)
    pairs_b = generate_seed_pairs(corpus, LENGTH, [7, 13], span_tokens=SPAN, mini_batch_size=MINI_BATCH)
    for (poison_a, control_a), (poison_b, control_b) in zip(pairs_a, pairs_b):
        assert np.array_equal(poison_a.tokens, poison_b.tokens)
        assert np.array_equal(control_a.tokens, control_b.tokens)


def test_generate_seed_pairs_different_seeds_different_streams():
    """T2.6: different seeds -> different streams."""
    corpus = make_corpus()
    pairs = generate_seed_pairs(corpus, LENGTH, [0, 1], span_tokens=SPAN, mini_batch_size=MINI_BATCH)
    poison_0, control_0 = pairs[0]
    poison_1, control_1 = pairs[1]
    assert not np.array_equal(poison_0.tokens, poison_1.tokens)
    assert not np.array_equal(control_0.tokens, control_1.tokens)


def test_generate_seed_pairs_empty_seeds_raises():
    """T2.6: no seeds -> error."""
    with pytest.raises(ValueError, match="no seeds"):
        generate_seed_pairs(make_corpus(), LENGTH, [])


def test_generate_seed_pairs_craft_fn_called_with_seed():
    """T2.6: craft_fn receives the seed for adversarial ordering."""
    corpus = make_corpus()
    received_seeds = []

    def mock_craft_fn(seed):
        received_seeds.append(seed)
        return None  # no ordering supplied -> falls back to the shuffle

    generate_seed_pairs(corpus, LENGTH, [5, 10, 15], craft_fn=mock_craft_fn, span_tokens=SPAN, mini_batch_size=MINI_BATCH)
    assert received_seeds == [5, 10, 15]


def test_generate_seed_pairs_arms_are_never_identical():
    """T2.6: the check every other pair test passes without making.

    Both arms are built from the same seed, so they draw the same spans. If the
    poison arm is also left unordered, the two streams are byte-identical --
    which `assert_streams_matched` passes trivially, because identical streams
    match on length, chunking, span size and dtype. The comparison is then
    vacuous by construction and yields exactly zero corruption with every
    structural check still green.
    """
    corpus = make_corpus()
    pairs = generate_seed_pairs(corpus, LENGTH, [0, 1, 2, 3, 4], span_tokens=SPAN, mini_batch_size=MINI_BATCH)
    for seed, (poison, control) in enumerate(pairs):
        assert not np.array_equal(poison.tokens, control.tokens), (
            f"seed {seed}: poison and control are the same bytes"
        )


def test_generate_seed_pairs_arms_hold_the_same_spans_reordered():
    """Same seed on both arms is deliberate: it isolates ordering.

    `order_fn` is asserted to permute rather than resize, so ordering is the
    attacker's only lever in SELECT. Matching the *content* across arms removes
    span choice as a confound, which is tighter than drawing the control from a
    different seed.
    """
    corpus = make_corpus()
    poison, control = generate_seed_pairs(
        corpus, LENGTH, [3], span_tokens=SPAN, mini_batch_size=MINI_BATCH
    )[0]
    # Equal as multisets over whole spans, not equal in order. The lookahead
    # token can differ (a different span lands last), so compare the body.
    assert sorted(poison.tokens[:LENGTH].tolist()) == sorted(control.tokens[:LENGTH].tolist())


def test_assert_arms_distinguishable_rejects_an_identical_pair():
    """The failure direction, per T2.3's own convention of testing both ways."""
    corpus = make_corpus()
    stream = build_select_stream(corpus, LENGTH, 0, span_tokens=SPAN, mini_batch_size=MINI_BATCH)
    with pytest.raises(ValueError, match="byte-identical"):
        assert_arms_distinguishable(stream, stream, seed=0)


def test_generate_seed_pairs_rejects_a_craft_fn_that_does_not_reorder():
    """An identity `craft_fn` is an absent attack, not a weak one."""
    corpus = make_corpus()
    with pytest.raises(ValueError, match="byte-identical"):
        generate_seed_pairs(
            corpus,
            LENGTH,
            [0],
            craft_fn=lambda seed: (lambda spans: list(spans)),
            span_tokens=SPAN,
            mini_batch_size=MINI_BATCH,
        )
