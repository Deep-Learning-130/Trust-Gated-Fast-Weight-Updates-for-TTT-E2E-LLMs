"""Benign token corpus and span sampling.

CPU only, plain numpy backing store. `zarr` must never be imported by the
corpus: it is absent from this project's dependencies and from CI, and the real
`/val` array has to drop in by duck-typing alone.
"""

import numpy as np
import pytest

from trustgate.attack.corpus import (
    DEFAULT_SPAN_TOKENS,
    Span,
    TokenCorpus,
    mean_offset_gap,
    score_spans,
    span_multiplicity,
    validate_span_tokens,
)
from trustgate.tokens import BOS_TOKEN_ID


def make_corpus(n_tokens=4096, bos_every=None, name="val"):
    """A deterministic pseudo-corpus of real-looking ids, optionally with BOS."""
    rng = np.random.default_rng(1234)
    tokens = rng.integers(10, 50000, size=n_tokens, dtype=np.int32)
    if bos_every:
        tokens[::bos_every] = BOS_TOKEN_ID
    return TokenCorpus(tokens, name=name)


def test_corpus_accepts_a_plain_ndarray_without_zarr():
    import trustgate.attack.corpus as corpus_module

    corpus = make_corpus()
    assert len(corpus) == 4096
    assert corpus.read(0, 8).dtype == np.int32

    # The store is duck-typed, never type-checked against zarr, so the real
    # /val array drops in with no code change while CI -- which installs
    # neither zarr nor the unlicensed vendor submodule -- still runs this.
    # Asserted on the module namespace rather than sys.modules, which would
    # false-fail on the GPU box where the vendor imports zarr itself.
    assert not hasattr(corpus_module, "zarr")


def test_read_returns_the_requested_slice():
    tokens = np.arange(100, dtype=np.int32)
    corpus = TokenCorpus(tokens, name="val")
    assert list(corpus.read(10, 4)) == [10, 11, 12, 13]


def test_read_past_the_end_raises():
    corpus = make_corpus(n_tokens=64)
    with pytest.raises(ValueError, match="outside corpus"):
        corpus.read(60, 16)


def test_sampled_spans_are_exactly_span_tokens_long():
    corpus = make_corpus()
    spans = corpus.sample_spans(5, span_tokens=64, seed=0)
    assert len(spans) == 5
    assert all(s.length == 64 for s in spans)
    assert all(len(corpus.read(s.offset, s.length)) == 64 for s in spans)


def test_sampled_spans_are_chunk_aligned():
    corpus = make_corpus()
    spans = corpus.sample_spans(6, span_tokens=64, seed=0)
    assert all(s.offset % 64 == 0 for s in spans)


def test_sampled_spans_never_contain_bos():
    # Dense document boundaries: every 64th token is BOS, so a naive aligned
    # sampler would return a BOS-carrying span almost every time.
    corpus = make_corpus(bos_every=64)
    spans = corpus.sample_spans(4, span_tokens=32, seed=7)
    for span in spans:
        assert BOS_TOKEN_ID not in set(corpus.read(span.offset, span.length).tolist())


def test_sampling_raises_when_no_bos_free_span_exists():
    # Every aligned 64-token span contains a BOS. Refusing loudly matters: a
    # silent fallback to BOS-carrying spans changes the per-chunk gradient scale
    # and breaks matching against the control while looking fine.
    tokens = np.full(1024, 42, dtype=np.int32)
    tokens[::64] = BOS_TOKEN_ID
    corpus = TokenCorpus(tokens, name="val")
    with pytest.raises(ValueError, match="could not find a BOS-free span"):
        corpus.sample_spans(2, span_tokens=64, seed=0)


def test_same_seed_reproduces_identical_spans():
    corpus = make_corpus()
    a = corpus.sample_spans(8, span_tokens=64, seed=99)
    b = corpus.sample_spans(8, span_tokens=64, seed=99)
    assert [s.offset for s in a] == [s.offset for s in b]


def test_different_seed_changes_the_spans():
    corpus = make_corpus()
    a = corpus.sample_spans(8, span_tokens=64, seed=1)
    b = corpus.sample_spans(8, span_tokens=64, seed=2)
    assert [s.offset for s in a] != [s.offset for s in b]


def test_sampling_without_replacement_yields_distinct_spans():
    # With replacement, SELECT could use one span eight times -- eight identical
    # gradient steps, which would "work" for a reason a one-line dedup defends.
    corpus = make_corpus()
    spans = corpus.sample_spans(16, span_tokens=64, seed=3, replace=False)
    assert len({s.offset for s in spans}) == 16
    assert span_multiplicity(spans) == 1


def test_requesting_more_distinct_spans_than_exist_raises():
    corpus = make_corpus(n_tokens=256)
    with pytest.raises(ValueError, match="cannot draw"):
        corpus.sample_spans(10, span_tokens=64, seed=0, replace=False)


def test_offset_range_bounds_where_spans_are_drawn():
    # This is what keeps a control stream out of the split the benign eval reads.
    corpus = make_corpus()
    spans = corpus.sample_spans(6, span_tokens=64, seed=0, offset_range=(1024, 2048))
    assert all(1024 <= s.offset and s.end <= 2048 for s in spans)


def test_invalid_offset_range_raises():
    corpus = make_corpus()
    with pytest.raises(ValueError, match="not a valid sub-range"):
        corpus.sample_spans(2, span_tokens=64, seed=0, offset_range=(2048, 1024))


def test_enumerate_spans_skips_bos_carrying_positions():
    tokens = np.full(256, 42, dtype=np.int32)
    tokens[64] = BOS_TOKEN_ID  # kills exactly the second aligned span
    corpus = TokenCorpus(tokens, name="val")
    offsets = [s.offset for s in corpus.enumerate_spans(span_tokens=64)]
    assert offsets == [0, 128, 192]


def test_materialize_concatenates_in_order():
    tokens = np.arange(100, dtype=np.int32)
    corpus = TokenCorpus(tokens, name="val")
    out = corpus.materialize([Span(20, 4), Span(0, 4)])
    assert list(out) == [20, 21, 22, 23, 0, 1, 2, 3]


def test_materialize_of_no_spans_is_empty():
    assert len(make_corpus().materialize([])) == 0


def test_validate_span_tokens_accepts_a_divisor():
    validate_span_tokens(64, mini_batch_size=1024)


def test_validate_span_tokens_rejects_a_non_divisor():
    with pytest.raises(ValueError, match="must divide mini_batch_size"):
        validate_span_tokens(48, mini_batch_size=1024)


def test_default_span_tokens_divides_the_vendor_chunk():
    # ext-1b-e2e-32K.yaml:23 sets mini_batch_size 1024.
    validate_span_tokens(DEFAULT_SPAN_TOKENS, mini_batch_size=1024)
    assert DEFAULT_SPAN_TOKENS < 1024  # ADR-007: not the 8-unit lever


def test_terminator_ids_round_trip_but_are_unused_this_wave():
    corpus = TokenCorpus(np.arange(128, dtype=np.int32), name="val", terminator_ids=frozenset({13}))
    assert corpus.terminator_ids == frozenset({13})
    # Segmentation still ignores them: spans stay fixed-stride.
    assert [s.offset for s in corpus.enumerate_spans(span_tokens=64)] == [0, 64]


def test_score_spans_fills_baseline_nll_from_an_injected_scorer():
    # Injected, never the victim model: scoring with the model under attack is
    # circular and the pre-registration names it invalidating.
    corpus = make_corpus()
    spans = corpus.sample_spans(3, span_tokens=64, seed=0)
    scored = score_spans(corpus, spans, scorer=lambda tokens: float(len(tokens)))
    assert [s.baseline_nll for s in scored] == [64.0, 64.0, 64.0]
    assert all(s.baseline_nll is None for s in spans)  # originals untouched


def test_span_multiplicity_counts_repeats():
    assert span_multiplicity([Span(0, 8), Span(0, 8), Span(64, 8)]) == 2
    assert span_multiplicity([]) == 0


def test_mean_offset_gap_measures_provenance_spread():
    assert mean_offset_gap([Span(0, 8), Span(100, 8), Span(200, 8)]) == 100.0
    assert mean_offset_gap([Span(0, 8)]) == 0.0
