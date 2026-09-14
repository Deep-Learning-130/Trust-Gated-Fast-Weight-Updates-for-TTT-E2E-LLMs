"""Fluency scoring: the windowed reduction and the independence guard.

Network-free and model-free. The reference model is 548MB and CI has no network,
so everything here runs against injected fakes; the real GPT-2 path is exercised
by `scripts/acceptance_fluency_reference.py` and by the opt-in tests in
`tests/test_fluency_reference.py`.

The windowing is what gets the attention. The fluency bar is a *ratio*, so a
double-counted overlap region would weight the two arms differently -- and since
poison and control differ in content, that corrupts the ratio rather than just
the absolute.
"""

import numpy as np
import pytest

from trustgate.attack.stream import CraftedStream, StreamStrategy
from trustgate.eval.fluency import (
    PerplexityScore,
    StreamFluencyScorer,
    assert_independent_of_victim,
    sliding_window_nll,
)
from trustgate.tokens import VOCAB_SIZE

VOCAB = 16


def uniform_logits(window):
    """Every token equally likely: NLL is exactly ln(VOCAB) per position."""
    return np.zeros((len(window), VOCAB))


def make_schedule_spy(vocab=VOCAB):
    """Records each window's start offset, inferred from the token ids.

    Only works because the tests feed `arange(n)` as the token stream, which
    makes `window[0]` the window's global start.
    """
    starts = []

    def logits_fn(window):
        starts.append(int(window[0]))
        return np.zeros((len(window), vocab))

    return logits_fn, starts


def test_uniform_logits_give_log_vocab_nats():
    score = sliding_window_nll(uniform_logits, np.arange(9), max_length=16, stride=8)
    assert score.mean_nll == pytest.approx(np.log(VOCAB))
    assert score.perplexity == pytest.approx(VOCAB)


def test_perplexity_is_exp_of_mean_nll():
    score = sliding_window_nll(uniform_logits, np.arange(9), max_length=16, stride=8)
    assert score.perplexity == pytest.approx(np.exp(score.mean_nll))


def test_first_token_is_never_scored():
    # It has no context. So a sequence of n tokens yields n-1 scored targets.
    for n in (2, 5, 9, 33):
        score = sliding_window_nll(
            uniform_logits, np.arange(n) % VOCAB, max_length=16, stride=8
        )
        assert score.n_scored_tokens == n - 1


def test_stride_past_the_context_window_raises():
    # The window grid would not cover the sequence and targets in the gaps would
    # never be scored -- a partial perplexity that looks entirely plausible.
    with pytest.raises(ValueError, match="exceeds max_length"):
        sliding_window_nll(
            uniform_logits, np.arange(30) % VOCAB, max_length=10, stride=10
        )


def test_the_default_stride_is_usable_with_a_real_context_window():
    # GPT-2's 1024 positions against the 512 default. Guards against a default
    # that trips the coverage check on the actual model.
    score = sliding_window_nll(
        uniform_logits, np.arange(2600) % VOCAB, max_length=1024
    )
    assert score.n_scored_tokens == 2599


def test_single_window_matches_a_hand_computed_reference():
    rng = np.random.default_rng(0)
    tokens = rng.integers(0, VOCAB, 6)
    table = rng.normal(0, 2.0, (6, VOCAB))

    score = sliding_window_nll(lambda w: table, tokens, max_length=32)

    # Reference: target at index j is predicted by row j-1.
    shifted = table - table.max(axis=-1, keepdims=True)
    log_prob = shifted - np.log(np.exp(shifted).sum(axis=-1, keepdims=True))
    expected = -np.mean([log_prob[j - 1, tokens[j]] for j in range(1, len(tokens))])

    assert score.mean_nll == pytest.approx(expected)


def test_every_target_is_scored_exactly_once_across_windows():
    # 20 tokens, context 8, stride 4 -> heavy overlap. Total must still be 19.
    score = sliding_window_nll(
        uniform_logits, np.arange(20) % VOCAB, max_length=8, stride=4
    )
    assert score.n_scored_tokens == 19
    # Uniform logits, so a double-counted region would not change the *mean* --
    # only the count reveals it. That is why the count is asserted, not the mean.
    assert score.mean_nll == pytest.approx(np.log(VOCAB))


def test_stride_does_not_change_how_many_tokens_are_scored():
    counts = {
        stride: sliding_window_nll(
            uniform_logits, np.arange(30) % VOCAB, max_length=10, stride=stride
        ).n_scored_tokens
        for stride in (1, 3, 5, 9)
    }
    assert set(counts.values()) == {29}


def test_window_schedule_gives_each_target_maximum_left_context():
    logits_fn, starts = make_schedule_spy()
    sliding_window_nll(logits_fn, np.arange(10), max_length=6, stride=3)

    # Windows that actually scored something. Target 6 is scored by the window at
    # begin=3 (3 tokens of left context), not by begin=0, which cannot reach it.
    assert starts == [0, 3, 6]


def test_a_window_with_nothing_left_to_score_is_skipped():
    # stride 1 over a long context: most windows have no unscored targets and
    # must not trigger a forward pass. Scoring them would be pure waste, and on
    # the real model each wasted window is a full 1024-position forward.
    logits_fn, starts = make_schedule_spy()
    sliding_window_nll(logits_fn, np.arange(12), max_length=12, stride=1)
    assert starts == [0]


def test_sequence_shorter_than_two_tokens_raises():
    with pytest.raises(ValueError, match="at least 2 tokens"):
        sliding_window_nll(uniform_logits, np.arange(1), max_length=8)


def test_degenerate_max_length_raises():
    with pytest.raises(ValueError, match="max_length must be >= 2"):
        sliding_window_nll(uniform_logits, np.arange(9), max_length=1)


def test_nonpositive_stride_raises():
    with pytest.raises(ValueError, match="stride must be >= 1"):
        sliding_window_nll(uniform_logits, np.arange(9), max_length=8, stride=0)


def test_independence_guard_accepts_a_different_vocabulary():
    assert_independent_of_victim(50257)  # GPT-2


def test_independence_guard_rejects_the_victims_vocabulary():
    # The attacker optimises against the victim, so the victim rates poisoned
    # streams as maximally fluent. A bar scored with it passes every attack.
    with pytest.raises(ValueError, match="circular"):
        assert_independent_of_victim(VOCAB_SIZE)


class FakeReference:
    """Stand-in for GPT2Reference.

    Returns uniform *logits* over a per-text width, so the resulting perplexity
    is exactly that width -- which makes the expected ratio an exact integer
    quotient with no numerical slack. `config.vocab_size` stays at GPT-2's real
    50257 because that is what the independence guard reads; the logits width is
    a separate test-only dial.
    """

    def __init__(self, width_by_text, vocab_size=50257, max_length=16):
        self.width_by_text = width_by_text
        self.config = type("Config", (), {"vocab_size": vocab_size})()
        self.max_length = max_length
        self._current = None

    def tokenizer(self):
        reference = self

        class Tok:
            def encode(self, text):
                reference._current = text
                return type("Encoded", (), {"ids": list(range(6))})()

        return Tok()

    def logits(self, ids):
        return np.zeros((len(ids), self.width_by_text[self._current]))


class FakeDecoder:
    def __init__(self, text_by_first_token):
        self.text_by_first_token = text_by_first_token

    def decode(self, token_ids):
        return self.text_by_first_token[int(np.asarray(token_ids).ravel()[0])]


def make_stream(first_token):
    tokens = np.arange(first_token, first_token + 7, dtype=np.int32)
    return CraftedStream(tokens=tokens, strategy=StreamStrategy.SELECT)


def build_scorer(poison_width, control_width):
    poison, control = make_stream(100), make_stream(200)
    reference = FakeReference(
        {"poison-text": poison_width, "control-text": control_width}
    )
    decoder = FakeDecoder({100: "poison-text", 200: "control-text"})
    return StreamFluencyScorer(reference, decoder, stride=8), poison, control


def test_scorer_returns_the_perplexity_ratio():
    scorer, poison, control = build_scorer(poison_width=100, control_width=50)
    assert scorer(poison, control) == pytest.approx(2.0, rel=1e-6)


def test_scorer_records_the_numbers_on_the_streams():
    # Until this runs, CraftedStream.perplexity is nan and the realism bar fails
    # safe. This is what flips a stream from unscored to scored.
    scorer, poison, control = build_scorer(poison_width=64, control_width=64)
    assert not poison.is_scored

    scorer(poison, control)

    assert poison.is_scored
    assert poison.perplexity == pytest.approx(64.0, rel=1e-6)
    assert poison.control_perplexity == pytest.approx(64.0, rel=1e-6)
    assert poison.fluency_ratio == pytest.approx(1.0, rel=1e-6)


def test_a_less_fluent_poison_arm_pushes_the_ratio_above_one():
    scorer, poison, control = build_scorer(poison_width=120, control_width=60)
    ratio = scorer(poison, control)
    assert ratio > 1.5  # would FAIL the pre-registered realism bar


def test_identical_arms_give_a_ratio_of_one():
    scorer, poison, control = build_scorer(poison_width=32, control_width=32)
    assert scorer(poison, control) == pytest.approx(1.0, rel=1e-6)


def test_scorer_refuses_a_reference_sharing_the_victims_vocabulary():
    reference = FakeReference({"t": 32}, vocab_size=VOCAB_SIZE)
    with pytest.raises(ValueError, match="circular"):
        StreamFluencyScorer(reference, FakeDecoder({100: "t"}))


def test_perplexity_score_is_a_plain_record():
    score = PerplexityScore(mean_nll=1.0, perplexity=np.e, n_scored_tokens=10)
    assert score.n_scored_tokens == 10
    assert score.perplexity == pytest.approx(np.e)
