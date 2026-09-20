"""The joint between `attack.craft` and `attack.stream`, on CPU.

`tests/test_craft.py` calls `craft_stream` directly; `tests/test_stream.py`
passes a `mock_craft_fn` that never touches `craft.py`. Nothing composed the
two, which is exactly where their shapes disagree -- `generate_seed_pairs`
wants `seed -> OrderFn` and `craft_stream` returns a `CraftResult`.

What this cannot establish: that the search finds anything against a real
victim. The objective here is a toy with a known optimum, standing in for the
victim's held-out benign loss. Same division as `tests/test_craft.py`, for the
same reason.
"""

import numpy as np
import pytest

from trustgate.attack.corpus import TokenCorpus
from trustgate.attack.craft import CraftConfig
from trustgate.attack.objectives import AttackSpec, Objective
from trustgate.attack.stream import StreamStrategy, generate_seed_pairs
from trustgate.eval.attacker import NoOrderingFound, make_craft_fn

SPAN, MINI_BATCH, LENGTH = 8, 32, 64
SEEDS = [0, 1, 2]


def make_corpus():
    rng = np.random.default_rng(0)
    return TokenCorpus(
        rng.integers(10, 50_000, 8192, dtype=np.int32), name="train"
    )


def make_spec(fluency_weight=0.0):
    return AttackSpec(
        objective=Objective.DEGRADE,
        stream_tokens=LENGTH,
        fluency_weight=fluency_weight,
    )


def make_config(max_iters=12):
    return CraftConfig(
        strategy=StreamStrategy.SELECT,
        max_iters=max_iters,
        reference_model_path="unused",
    )


def mean_token_objective(stream, condition):
    """Stand-in for `make_adapt_and_eval`'s `(stream, condition) -> float`.

    Climbable and ordering-sensitive, so the search has something to find.
    """
    assert condition is not None, "the objective must receive its condition"
    weights = np.arange(len(stream.tokens), dtype=np.float64)
    return float(np.dot(stream.tokens.astype(np.float64), weights) / 1e9)


def fake_condition_for(stream, seed):
    """Stands in for `RunCondition.from_stream`, recording what it saw."""
    return {
        "seed": seed,
        "stream_tokens": stream.length_tokens,
        "n_chunks": stream.n_chunks,
        "valid_tokens": stream.valid_tokens,
        "span_tokens": stream.span_tokens,
        "mini_batch_size": stream.mini_batch_size,
        "dtype": str(stream.tokens.dtype),
        "corpus": stream.corpus_name,
    }


def build(**kwargs):
    defaults = dict(
        corpus=make_corpus(),
        spec=make_spec(),
        config=make_config(),
        adapt_and_eval=mean_token_objective,
        condition_for=fake_condition_for,
        span_tokens=SPAN,
        mini_batch_size=MINI_BATCH,
    )
    defaults.update(kwargs)
    return make_craft_fn(**defaults)


# ------------------------------------------------------------- the curry ---


def test_craft_fn_is_callable_per_seed_and_returns_an_order_fn():
    craft_fn = build()
    order_fn = craft_fn(0)
    assert callable(order_fn)


def test_order_fn_permutes_rather_than_resizes():
    """`_build` raises if the count changes (`stream.py:228-231`), so this is
    the property that lets the returned ordering be used at all."""
    corpus = make_corpus()
    craft_fn = build(corpus=corpus)
    from trustgate.attack.stream import build_select_stream

    sampled = build_select_stream(
        corpus, LENGTH, 0, span_tokens=SPAN, mini_batch_size=MINI_BATCH
    )
    ordered = craft_fn(0)(sampled.spans)
    assert len(ordered) == len(sampled.spans)
    assert sorted(s.offset for s in ordered) == sorted(
        s.offset for s in sampled.spans
    )


def test_order_fn_ignores_the_spans_it_is_handed():
    """The searched ordering is the whole point; re-sampling would discard it."""
    craft_fn = build()
    order_fn = craft_fn(0)
    assert order_fn([]) == order_fn(["nonsense"])


# ------------------------------------------- composition with the stream ---


def test_generate_seed_pairs_accepts_the_real_craft_fn():
    """The composition itself -- previously untested in either direction."""
    corpus = make_corpus()
    pairs = generate_seed_pairs(
        corpus,
        LENGTH,
        SEEDS,
        craft_fn=build(corpus=corpus),
        span_tokens=SPAN,
        mini_batch_size=MINI_BATCH,
    )
    assert len(pairs) == len(SEEDS)


def test_crafted_arms_are_distinguishable():
    """`assert_arms_distinguishable` (`stream.py:652-676`) fires when poison and
    control are byte-identical. Both arms are built at the same seed, so only a
    real reordering keeps this passing -- it is the check that makes a
    placeholder craft_fn impossible to ship by accident."""
    corpus = make_corpus()
    pairs = generate_seed_pairs(
        corpus,
        LENGTH,
        SEEDS,
        craft_fn=build(corpus=corpus),
        span_tokens=SPAN,
        mini_batch_size=MINI_BATCH,
    )
    for poison, control in pairs:
        assert not np.array_equal(poison.tokens, control.tokens)


def test_crafted_arms_stay_matched():
    """Reordering must not change any field `RunCondition` compares."""
    corpus = make_corpus()
    pairs = generate_seed_pairs(
        corpus,
        LENGTH,
        SEEDS,
        craft_fn=build(corpus=corpus),
        span_tokens=SPAN,
        mini_batch_size=MINI_BATCH,
    )
    for poison, control in pairs:
        assert poison.length_tokens == control.length_tokens
        assert poison.n_chunks == control.n_chunks
        assert poison.valid_tokens == control.valid_tokens
        assert poison.span_tokens == control.span_tokens
        assert poison.tokens.dtype == control.tokens.dtype


def test_the_run_is_deterministic_given_the_seeds():
    corpus = make_corpus()
    first = generate_seed_pairs(
        corpus, LENGTH, SEEDS, craft_fn=build(corpus=corpus),
        span_tokens=SPAN, mini_batch_size=MINI_BATCH,
    )
    second = generate_seed_pairs(
        corpus, LENGTH, SEEDS, craft_fn=build(corpus=corpus),
        span_tokens=SPAN, mini_batch_size=MINI_BATCH,
    )
    for (pa, _), (pb, _) in zip(first, second):
        assert np.array_equal(pa.tokens, pb.tokens)


# ------------------------------------------- the condition circularity ---


def test_the_condition_is_derived_once_per_seed_and_carries_that_seed():
    seen = []

    def recording_condition_for(stream, seed):
        seen.append((seed, stream.length_tokens))
        return fake_condition_for(stream, seed)

    corpus = make_corpus()
    generate_seed_pairs(
        corpus, LENGTH, SEEDS,
        craft_fn=build(corpus=corpus, condition_for=recording_condition_for),
        span_tokens=SPAN, mini_batch_size=MINI_BATCH,
    )
    assert [seed for seed, _ in seen] == SEEDS
    assert all(length == LENGTH for _, length in seen)


def test_the_sampled_condition_matches_the_crafted_stream():
    """The load-bearing claim of the whole module: a condition derived from the
    sampled ordering describes the crafted one too. If a permutation could
    change any of these, the objective would be scoring against a condition the
    victim never ran under."""
    corpus = make_corpus()
    captured = {}

    def capturing_condition_for(stream, seed):
        condition = fake_condition_for(stream, seed)
        captured[seed] = condition
        return condition

    pairs = generate_seed_pairs(
        corpus, LENGTH, SEEDS,
        craft_fn=build(corpus=corpus, condition_for=capturing_condition_for),
        span_tokens=SPAN, mini_batch_size=MINI_BATCH,
    )
    for seed, (poison, _) in zip(SEEDS, pairs):
        assert captured[seed] == fake_condition_for(poison, seed)


# --------------------------------------------------- search accounting ---


def test_the_search_result_is_observable_per_seed():
    """A null with no record of how hard the attacker searched cannot be read:
    it could be a strong negative or a search that never ran."""
    results = {}
    corpus = make_corpus()
    generate_seed_pairs(
        corpus, LENGTH, SEEDS,
        craft_fn=build(
            corpus=corpus,
            on_result=lambda seed, result: results.__setitem__(seed, result),
        ),
        span_tokens=SPAN, mini_batch_size=MINI_BATCH,
    )
    assert sorted(results) == SEEDS
    for result in results.values():
        # One baseline evaluation plus at least one proposal.
        assert result.n_evaluations >= 2
        assert result.best_score >= result.baseline_score


def test_the_search_actually_improves_a_climbable_objective():
    results = {}
    corpus = make_corpus()
    generate_seed_pairs(
        corpus, LENGTH, SEEDS,
        craft_fn=build(
            corpus=corpus,
            config=make_config(max_iters=40),
            on_result=lambda seed, result: results.__setitem__(seed, result),
        ),
        span_tokens=SPAN, mini_batch_size=MINI_BATCH,
    )
    assert any(result.gain > 0 for result in results.values())


def test_a_search_that_accepts_nothing_is_reported_as_a_null_not_a_crash():
    """Under SELECT the only lever is ordering, so a search that accepts nothing
    leaves the sampled ordering in place and the two arms byte-identical.

    `assert_arms_distinguishable` would refuse that pair and tell the caller to
    "supply a craft_fn that actually reorders" -- wrong advice here, because the
    craft_fn is present and it searched. The distinct type is what lets the
    caller record a null instead of chasing a phantom wiring bug.
    """
    corpus = make_corpus()
    craft_fn = build(corpus=corpus, config=make_config(max_iters=0))
    with pytest.raises(NoOrderingFound) as excinfo:
        craft_fn(0)
    message = str(excinfo.value)
    assert "null result" in message
    assert "max_iters" in message  # names the knob that would change it


def test_the_null_is_still_observable_before_it_raises():
    """The result has to reach `on_result` even on the null path, or the run
    loses its only evidence that the attacker searched at all."""
    seen = {}
    corpus = make_corpus()
    craft_fn = build(
        corpus=corpus,
        config=make_config(max_iters=0),
        on_result=lambda seed, result: seen.__setitem__(seed, result),
    )
    with pytest.raises(NoOrderingFound):
        craft_fn(0)
    assert seen[0].n_evaluations == 1
    assert not seen[0].search_did_anything


# ------------------------------------------------- refusals pass through ---


def test_a_non_select_strategy_is_still_refused_through_the_seam():
    """T2.5. The seam must not become a way around `craft_stream`'s guards."""
    corpus = make_corpus()
    craft_fn = build(corpus=corpus, config=CraftConfig(
        strategy=StreamStrategy.PARAPHRASE, max_iters=4,
        reference_model_path="unused",
    ))
    with pytest.raises(NotImplementedError) as excinfo:
        craft_fn(0)
    assert "T2.5" in str(excinfo.value)


def test_a_trigger_objective_is_still_refused_through_the_seam():
    """T2.8."""
    corpus = make_corpus()
    spec = AttackSpec(
        objective=Objective.TRIGGER,
        stream_tokens=LENGTH,
        target_tokens=(1, 2),
        trigger_tokens=(3, 4),
        fluency_weight=0.0,
    )
    craft_fn = build(corpus=corpus, spec=spec)
    with pytest.raises(NotImplementedError) as excinfo:
        craft_fn(0)
    assert "T2.8" in str(excinfo.value)


def test_a_nonzero_fluency_weight_without_a_scorer_is_still_refused():
    """Otherwise the realism bar is silently unconstrained."""
    corpus = make_corpus()
    craft_fn = build(corpus=corpus, spec=make_spec(fluency_weight=1.0))
    with pytest.raises(ValueError):
        craft_fn(0)
