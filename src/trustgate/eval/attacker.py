"""Build the real attacker: the seam between `attack.craft` and a live victim.

`craft.search_order` and `craft.craft_stream` have been implemented and
CPU-tested since T2.x and wired into nothing. Every CLI run passes
`craft_fn=None`, which `stream.generate_seed_pairs` documents as "not a real
attack; for testing the orchestration only". This module is the joint that
closes that gap. It is its own module rather than a helper inside `cli` so the
joint is testable: no test anywhere previously composed `craft_stream` with
`generate_seed_pairs`, and that is precisely where the two shapes disagree.

Two mismatches have to be bridged, and both are load-bearing enough to name.

**The objective is one argument too wide.** `harness.make_adapt_and_eval`
returns `(stream, condition) -> float`; `craft.StreamObjective` is
`(stream) -> float`. A `RunCondition` therefore has to exist before the search
starts -- but `RunCondition.from_stream` needs a stream, and streams are what
the search produces. The resolution is the one `craft_stream` already uses
internally (`craft.py:366`): build the *sampled* ordering first and derive the
condition from it. Every field `from_stream` reads -- `length_tokens`,
`mini_batch_size`, `span_tokens`, `n_chunks`, `valid_tokens`, `dtype`,
`corpus_name` -- is invariant under a permutation of the same span multiset, so
a condition taken from the sampled ordering is the right condition for every
proposal the search considers. `assert_matches` still guards the final arms.

**The seam wants a curry and `craft_stream` does not return one.**
`generate_seed_pairs` expects `craft_fn: seed -> OrderFn`; `craft_stream`
returns a `CraftResult`. The bridge runs the search inside the curried call and
returns the realised ordering as a constant `order_fn` -- the same shape
`craft.py:345-361` uses internally. `_resolve_lookahead` is idempotent on an
already-resolved ordering (`stream.py:169`), so replaying `result.spans`
reproduces the stream the search actually scored.

**Cost.** One objective evaluation is one full adapt-and-eval: roughly 16 inner
steps plus 2 prefix passes at 8192/1024, per proposal, per seed, plus a fluency
scoring of both arms when `fluency_weight` is nonzero. This is the dominant cost
of the whole spike. Size `max_iters` against a measured per-eval wall-clock, not
against `run_deep.py`'s CPU-pilot value of 40.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from trustgate.attack.craft import CraftConfig, CraftResult, craft_stream
from trustgate.attack.stream import DEFAULT_SPAN_TOKENS, build_select_stream


class NoOrderingFound(RuntimeError):
    """The search accepted nothing, so poison and control would be identical.

    Its own type because the caller has to tell it apart from a failure. A run
    that hits this has measured something real -- the attacker had a budget and
    found no improving order -- and the honest report is a null, not a crash and
    not a retry at a kinder threshold.
    """


def make_craft_fn(
    *,
    corpus,
    spec,
    config: CraftConfig,
    adapt_and_eval: Callable[[Any, Any], float],
    condition_for: Callable[[Any, int], Any],
    fluency_scorer=None,
    span_tokens: int = DEFAULT_SPAN_TOKENS,
    mini_batch_size: int = 1024,
    offset_range: tuple[int, int] | None = None,
    on_result: Callable[[int, CraftResult], None] | None = None,
) -> Callable[[int], Callable[[Any], list]]:
    """Return the `craft_fn` that `generate_seed_pairs` asks for.

    Args:
        corpus: the span corpus both arms draw from.
        spec: `AttackSpec`. `craft_stream` refuses anything but SELECT+DEGRADE.
        config: `CraftConfig`. `max_iters` is the search budget, in victim
            evaluations -- see the cost note in the module docstring.
        adapt_and_eval: `(stream, condition) -> float` from
            `harness.make_adapt_and_eval`. Returns benign loss, used unnegated
            because `search_order` maximises and higher benign loss is better
            for the attacker (`craft.py:214`). Do **not** route this through
            `objectives.degrade_loss`, which negates for minimising callers.
        condition_for: `(sampled_stream, seed) -> RunCondition`. Called once per
            seed on the sampled ordering; see the module docstring on why that
            is sound.
        fluency_scorer: passed through. `craft_stream` refuses a nonzero
            `spec.fluency_weight` without one, rather than running a silently
            unconstrained attack.
        on_result: observer for the per-seed `CraftResult`. The search's own
            accounting -- `n_evaluations`, `gain`, `stopped_early`,
            `search_did_anything` -- is the evidence that the attacker actually
            searched, and a run that reports a null without it cannot be read.
    """

    def craft_fn(seed: int):
        # The sampled ordering, built exactly as `generate_seed_pairs` will
        # build it and as `craft_stream` will build it again internally. Same
        # seed and same parameters, so all three agree on the span multiset.
        sampled = build_select_stream(
            corpus,
            spec.stream_tokens,
            seed,
            span_tokens=span_tokens,
            mini_batch_size=mini_batch_size,
            offset_range=offset_range,
        )
        condition = condition_for(sampled, seed)

        def objective(stream) -> float:
            return adapt_and_eval(stream, condition)

        result = craft_stream(
            None,  # the victim reaches the search through `objective`
            spec,
            config,
            corpus,
            seed,
            objective=objective,
            fluency_scorer=fluency_scorer,
            span_tokens=span_tokens,
            mini_batch_size=mini_batch_size,
            offset_range=offset_range,
        )

        if on_result is not None:
            on_result(seed, result)

        if not result.search_did_anything:
            # The sampled ordering survived, so the poison arm would be
            # byte-identical to the control and
            # `stream.assert_arms_distinguishable` would refuse the pair -- with
            # a message telling the caller to "supply a craft_fn that actually
            # reorders", which here would be wrong advice. The craft_fn is
            # present and it searched.
            #
            # This is a real null: under SELECT the attacker's only lever is
            # ordering, and no ordering beat the sampled one at this budget. It
            # needs no victim run to report, because the two arms are the same
            # stream and the effect is zero by construction. Raise it as its own
            # condition so the caller can record that rather than measure it.
            raise NoOrderingFound(
                f"the ordering search at seed {seed} accepted none of its "
                f"{result.n_evaluations - 1} proposal(s) over "
                f"{len(result.spans)} spans, so the crafted ordering is the "
                f"sampled one. That is a null result, not a broken attacker: "
                f"report it as d = 0 rather than measuring two identical arms. "
                f"Raise max_iters if the budget was too small to search."
            )

        # `result.spans` is the realised ordering, already through
        # `_resolve_lookahead` (`craft.py:407-411`). Returning it as a constant
        # discards the spans `generate_seed_pairs` sampled, which is correct:
        # they are the same multiset, and this ordering is the one that scored.
        order = list(result.spans)

        def order_fn(_spans):
            return list(order)

        return order_fn

    return craft_fn
