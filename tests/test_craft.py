"""The attacker's search, with the victim injected.

DEFENSIVE RESEARCH. CPU only, synthetic corpus, no model.

T1.9. The victim call is a seam, so everything here runs with no checkpoint, no
GPU and no spend -- the same separation `test_carry.py` uses for `step_fn` and
`test_spike_orchestration.py` uses for `adapt_and_eval`.

**What these tests do not establish.** That the search finds anything against a
real victim. The objectives below are toy functions with a known optimum, chosen
so the search's *mechanics* are checkable: that it improves a score it can
improve, spends the budget it claims, stops when told, and reports honestly when
it found nothing. Whether TTT-E2E fast weights are actually reorderable into
corruption is what the kill-gate measures, and nothing on CPU can answer it.
"""

import numpy as np
import pytest

from trustgate.attack.corpus import TokenCorpus
from trustgate.attack.craft import (
    CraftConfig,
    craft_stream,
    fluency_penalised,
    search_order,
)
from trustgate.attack.objectives import AttackSpec, Objective
from trustgate.attack.stream import StreamStrategy, build_select_stream

SPAN, MINI_BATCH, LENGTH = 8, 32, 64  # 8 spans, 2 chunks


def make_corpus(name="train"):
    rng = np.random.default_rng(4242)
    return TokenCorpus(rng.integers(10, 50000, 8192, dtype=np.int32), name=name)


def make_spans(seed=0):
    return build_select_stream(
        make_corpus(), LENGTH, seed, span_tokens=SPAN, mini_batch_size=MINI_BATCH
    ).spans


def sortedness(spans):
    """Toy objective with a known optimum: reward ascending offsets.

    Stands in for "the victim's held-out loss after adapting on this ordering".
    Deliberately something pairwise swaps can climb, so a failure here is a
    failure of the search and not of the objective.
    """
    offsets = [s.offset for s in spans]
    return float(sum(a < b for a, b in zip(offsets, offsets[1:])))


def make_config(**kw):
    base = dict(
        strategy=StreamStrategy.SELECT,
        max_iters=200,
        reference_model_path="unused-by-these-tests",
        early_stop_patience=0,
    )
    base.update(kw)
    return CraftConfig(**base)


# ---------------------------------------------------------------------------
# search_order -- the mechanics
# ---------------------------------------------------------------------------


def test_search_improves_a_climbable_objective():
    result = search_order(make_spans(), sortedness, max_iters=300, seed=0)
    assert result.best_score > result.baseline_score
    assert result.gain > 0
    assert result.search_did_anything


def test_search_is_deterministic_given_the_seed():
    a = search_order(make_spans(), sortedness, max_iters=100, seed=7)
    b = search_order(make_spans(), sortedness, max_iters=100, seed=7)
    assert [s.offset for s in a.spans] == [s.offset for s in b.spans]
    assert a.best_score == b.best_score
    assert a.n_evaluations == b.n_evaluations


def test_a_different_seed_searches_differently():
    a = search_order(make_spans(), sortedness, max_iters=50, seed=1)
    b = search_order(make_spans(), sortedness, max_iters=50, seed=2)
    assert a.history != b.history


def test_the_span_multiset_is_never_changed():
    """Pairwise swaps only. If the multiset moved, `span_multiplicity` could
    leave 1 and the fluency bar would stop meaning what it claims."""
    spans = make_spans()
    result = search_order(spans, sortedness, max_iters=200, seed=0)
    assert sorted(s.offset for s in result.spans) == sorted(s.offset for s in spans)
    assert len(result.spans) == len(spans)


def test_a_flat_objective_accepts_nothing_and_says_so():
    """The honest-null case: the search must not claim an attack it did not find."""
    result = search_order(make_spans(), lambda spans: 1.0, max_iters=50, seed=0)
    assert result.n_accepted == 0
    assert not result.search_did_anything
    assert result.gain == 0.0
    assert [s.offset for s in result.spans] == [s.offset for s in make_spans()]


def test_evaluations_are_counted_not_iterations():
    """Budget is stated in victim calls, because that is what costs money.

    A self-swap (i == j) scores nothing and must not be billed, or a run reports
    a budget it never spent.
    """
    result = search_order(make_spans(), sortedness, max_iters=100, seed=3)
    # 1 baseline + one per real proposal, and self-swaps are skipped.
    assert result.n_evaluations <= 101
    assert result.n_evaluations == len(result.history)


def test_early_stop_halts_after_patience_without_improvement():
    result = search_order(
        make_spans(), lambda spans: 1.0, max_iters=1000, seed=0, early_stop_patience=5
    )
    assert result.stopped_early
    assert result.n_evaluations <= 7  # baseline + 5 non-improving + margin


def test_zero_patience_disables_early_stopping():
    result = search_order(
        make_spans(), lambda spans: 1.0, max_iters=30, seed=0, early_stop_patience=0
    )
    assert not result.stopped_early


def test_zero_iterations_returns_the_sampled_ordering():
    spans = make_spans()
    result = search_order(spans, sortedness, max_iters=0, seed=0)
    assert result.n_evaluations == 1  # the baseline still has to be scored
    assert [s.offset for s in result.spans] == [s.offset for s in spans]


def test_fewer_than_two_spans_raises():
    with pytest.raises(ValueError, match="at least 2 spans"):
        search_order(make_spans()[:1], sortedness, max_iters=10, seed=0)


# ---------------------------------------------------------------------------
# The fluency term, which must be inside the objective
# ---------------------------------------------------------------------------


def test_fluency_penalty_is_subtracted_from_the_objective():
    base = fluency_penalised(
        lambda stream: 10.0, lambda poison, control: 2.0, 1.5, control=object()
    )
    assert base(None) == pytest.approx(10.0 - 1.5 * 2.0)


def test_zero_weight_leaves_the_objective_untouched():
    obj = lambda stream: 10.0  # noqa: E731
    assert fluency_penalised(obj, lambda p, c: 99.0, 0.0) is obj


def test_fluency_scoring_without_a_control_is_refused():
    """The bar is a ratio; it is not defined on a single stream."""
    with pytest.raises(ValueError, match="needs the control arm"):
        fluency_penalised(lambda s: 1.0, lambda p, c: 2.0, 1.0, control=None)


def test_craft_refuses_a_nonzero_fluency_weight_with_no_scorer():
    """Otherwise the fluency term is silently absent and the run is the
    unconstrained attack while claiming to be the constrained one."""
    with pytest.raises(ValueError, match="no fluency_scorer"):
        craft_stream(
            None,
            AttackSpec(objective=Objective.DEGRADE, stream_tokens=LENGTH, fluency_weight=1.0),
            make_config(),
            make_corpus(),
            seed=0,
            objective=lambda stream: 1.0,
            span_tokens=SPAN,
            mini_batch_size=MINI_BATCH,
        )


# ---------------------------------------------------------------------------
# craft_stream -- assembly, and the seams that stay closed
# ---------------------------------------------------------------------------


def recording_scorer(poison_ppl=12.0, control_ppl=10.0, calls=None):
    """Stands in for `eval.fluency.StreamFluencyScorer`, including the mutation.

    The real scorer records both perplexities onto the streams and returns the
    ratio; a stand-in that only returned a number would let the bug this
    fixture exists to catch pass again.
    """

    def score(poison, control):
        if calls is not None:
            calls.append((poison, control))
        poison.perplexity = poison_ppl
        poison.control_perplexity = control_ppl
        control.perplexity = control_ppl
        control.control_perplexity = control_ppl
        return poison.fluency_ratio

    return score


def craft(**kw):
    spec = kw.pop("spec", None) or AttackSpec(
        objective=Objective.DEGRADE, stream_tokens=LENGTH, fluency_weight=0.0
    )
    base = dict(
        objective=lambda stream: float(np.mean(stream.tokens)),
        span_tokens=SPAN,
        mini_batch_size=MINI_BATCH,
    )
    base.update(kw)
    return craft_stream(None, spec, make_config(), make_corpus(), seed=0, **base)


def test_craft_returns_a_stream_matching_the_reported_spans():
    result = craft()
    assert result.stream is not None
    assert result.spans == result.stream.spans


def test_the_crafted_stream_obeys_every_builder_constraint():
    """Built through `build_select_stream`'s order_fn seam, so the window +1
    convention, the non-BOS lookahead and the chunk arithmetic all still hold."""
    stream = craft().stream
    assert len(stream.tokens) == LENGTH + 1
    assert stream.length_tokens == LENGTH
    assert stream.valid_tokens == LENGTH
    assert stream.n_chunks == LENGTH // MINI_BATCH
    assert stream.span_multiplicity == 1
    assert stream.strategy is StreamStrategy.SELECT


def test_the_reported_score_is_the_score_of_the_returned_stream():
    """`_resolve_lookahead` runs after `order_fn` and can promote a span to the
    end, so a search that scored raw orderings could return a stream whose score
    was never observed. Everything goes through one builder; this pins it."""
    objective = lambda stream: float(np.mean(stream.tokens))  # noqa: E731
    result = craft(objective=objective)
    assert objective(result.stream) == pytest.approx(result.best_score)


def test_craft_is_deterministic():
    a, b = craft(), craft()
    assert np.array_equal(a.stream.tokens, b.stream.tokens)
    assert a.best_score == b.best_score


def test_craft_without_an_objective_says_what_to_inject():
    with pytest.raises(NotImplementedError, match="make_adapt_and_eval"):
        craft_stream(
            None,
            AttackSpec(objective=Objective.DEGRADE, stream_tokens=LENGTH, fluency_weight=0.0),
            make_config(),
            make_corpus(),
            seed=0,
            span_tokens=SPAN,
            mini_batch_size=MINI_BATCH,
        )


def test_paraphrase_and_soft_remain_blocked():
    """T2.5. Still genuinely blocked -- not silently falling back to SELECT."""
    for strategy in (StreamStrategy.PARAPHRASE, StreamStrategy.SOFT):
        with pytest.raises(NotImplementedError, match="T2.5"):
            craft_stream(
                None,
                AttackSpec(objective=Objective.DEGRADE, stream_tokens=LENGTH, fluency_weight=0.0),
                make_config(strategy=strategy),
                make_corpus(),
                seed=0,
                objective=lambda stream: 1.0,
                span_tokens=SPAN,
                mini_batch_size=MINI_BATCH,
            )


def test_trigger_crafting_remains_blocked():
    """T2.8. TRIGGER has no scalar the ordering search can hill-climb."""
    spec = AttackSpec(
        objective=Objective.TRIGGER,
        stream_tokens=LENGTH,
        fluency_weight=0.0,
        trigger_tokens=np.array([1, 2], dtype=np.int32),
        target_tokens=np.array([3, 4], dtype=np.int32),
    )
    with pytest.raises(NotImplementedError, match="T2.8"):
        craft_stream(
            None,
            spec,
            make_config(),
            make_corpus(),
            seed=0,
            objective=lambda stream: 1.0,
            span_tokens=SPAN,
            mini_batch_size=MINI_BATCH,
        )


# ---------------------------------------------------------------------------
# T1.9's fourth criterion: the returned stream must actually be scored
# ---------------------------------------------------------------------------


def test_the_returned_stream_has_both_perplexities_filled():
    """T1.9: "returns a CraftedStream with both perplexity and
    control_perplexity filled".

    The streams scored during the search are candidates that get discarded;
    the winner is rebuilt fresh and carries no scores. Without an explicit
    final scoring pass both fields stay `nan` and `is_scored` stays False --
    which fails safe, but means the crafted arm can never pass the realism bar
    at all. That is not a bar, it is a wall.
    """
    result = craft(
        spec=AttackSpec(objective=Objective.DEGRADE, stream_tokens=LENGTH, fluency_weight=1.0),
        fluency_scorer=recording_scorer(),
    )
    assert result.stream.is_scored
    assert result.stream.perplexity == pytest.approx(12.0)
    assert result.stream.control_perplexity == pytest.approx(10.0)
    assert result.stream.fluency_ratio == pytest.approx(1.2)


def test_the_control_arm_is_returned_and_scored():
    result = craft(
        spec=AttackSpec(objective=Objective.DEGRADE, stream_tokens=LENGTH, fluency_weight=1.0),
        fluency_scorer=recording_scorer(),
    )
    assert result.control is not None
    assert result.control.is_scored
    assert len(result.control.tokens) == len(result.stream.tokens)


def test_the_control_is_held_fixed_across_the_whole_search():
    """If the control moved per proposal, the attacker could lower the ratio by
    making the control look worse rather than the poison look better, and the
    realism bar would stop measuring realism."""
    calls = []
    result = craft(
        spec=AttackSpec(objective=Objective.DEGRADE, stream_tokens=LENGTH, fluency_weight=1.0),
        fluency_scorer=recording_scorer(calls=calls),
    )
    assert len(calls) > 1
    controls = {id(c) for _, c in calls}
    assert len(controls) == 1
    assert id(result.control) in controls


def test_an_unscored_craft_leaves_the_stream_failing_safe():
    """fluency_weight=0.0 is the documented diagnostic upper bound. It must not
    silently mark the stream as realistic."""
    result = craft()  # fluency_weight=0.0, no scorer
    assert not result.stream.is_scored
    assert not (result.stream.fluency_ratio <= 1.5)
