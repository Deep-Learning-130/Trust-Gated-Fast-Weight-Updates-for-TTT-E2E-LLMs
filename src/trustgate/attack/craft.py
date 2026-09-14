"""Optimisation loop that crafts a poison stream against a fixed objective.

DEFENSIVE RESEARCH. See DISCLOSURE.md.

Threat-model realism is a constraint, not an afterthought
--------------------------------------------------------
The optimiser searches only within whatever `StreamStrategy` allows -- SELECT
reorders real sentences, PARAPHRASE stays in a similarity ball, SOFT relaxes into
embedding space. The fluency term from `AttackSpec.fluency_weight` is inside the
objective, not a post-hoc filter, because filtering after optimisation just throws
away compute and biases toward whatever slipped through. An attack that needs the
fluency constraint switched off to work is reported as such -- it establishes a
weaker threat model, and hiding that would make the whole defense look better than
the evidence supports.

What is implemented here, and what is injected
----------------------------------------------
The **search** is here and is complete: proposal, scoring, accept/reject, the
evaluation budget, early stopping, determinism. It is pure and CPU-testable.

The **victim call** is an injected seam (`objective`), exactly as
`carry.run_chunks` injects `step_fn` and `run_attack_spike` injects
`adapt_and_eval`. That is not a placeholder -- it is the same separation the rest
of Phase 1 uses, and it is what lets the attacker's search be reviewed and tested
without a checkpoint, a GPU or a cent of spend.

Why this is not the pilot heuristic under a better name
-------------------------------------------------------
`experiments/002-pilot-tiny-ttt/README.md` deliberately left this function
raising, on the grounds that "committing a pilot-grade heuristic under that name
would misrepresent it". That objection is about *hardcoding* a toy scorer where a
victim belongs. It does not apply to injecting the victim: the objective this
searches against is supplied by the caller, so at 1B it is the real
`adapt_and_eval` and at pilot scale it is the pilot's. The search itself --
pairwise-swap hill-climbing over the ordering, budget stated in evaluations -- is
promoted from `run_deep.py:163-192`, which is the only version of it that has
ever produced a number.

Why ordering, and only ordering
-------------------------------
For SELECT the multiset of spans is fixed by the seed and `order_fn` is asserted
to permute rather than resize (`stream.py`), so ordering is the attacker's entire
lever. That is deliberately narrow, and it is what makes a positive result
credible: every span is real text, drawn without replacement, and
`span_multiplicity` stays 1, so the repetition loophole in the fluency bar
(`FLUENCY_REFERENCE.md`: a repeated passage scores 1.90 against fluent English at
19.98) is not in play.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Sequence

import numpy as np

from trustgate.attack.corpus import DEFAULT_SPAN_TOKENS, Span
from trustgate.attack.objectives import AttackSpec, Objective
from trustgate.attack.stream import (
    CraftedStream,
    StreamStrategy,
    build_benign_control,
    build_select_stream,
)

#: Scores one candidate stream. Higher is better *for the attacker*.
#:
#: The natural implementation is the victim's own held-out benign loss after
#: adapting on that stream -- the quantity `degrade_loss` targets, and what
#: `harness.make_adapt_and_eval` returns. Takes a `CraftedStream` rather than
#: raw spans for a reason that is easy to get wrong: `_resolve_lookahead`
#: (`stream.py`) runs *after* `order_fn` and may promote one span to the end to
#: secure a non-BOS lookahead token. So the ordering the search proposes is not
#: always the ordering that gets built, and scoring spans directly would score
#: something the victim never sees. Everything is scored through the builder.
StreamObjective = Callable[[CraftedStream], float]

#: The spans-level callable `search_order` hill-climbs. `craft_stream` builds it
#: by composing a `StreamObjective` with the stream builder.
SpanObjective = Callable[[Sequence[Span]], float]

#: Scores the realism bar: `(poison, control) -> perplexity ratio`.
#:
#: This is `eval.fluency.StreamFluencyScorer.__call__`'s signature, and taking
#: both arms is not incidental. The bar is a *ratio*, so a single stream cannot
#: produce it -- and the scorer records `perplexity` and `control_perplexity`
#: onto the streams by mutation, which is the only way `CraftedStream.is_scored`
#: ever becomes True. A one-argument scorer leaves both fields `nan` forever.
FluencyScorer = Callable[[CraftedStream, CraftedStream], float]


@dataclass
class CraftConfig:
    strategy: StreamStrategy
    max_iters: int
    reference_model_path: str
    """Independent model for the fluency term. NOT the victim -- scoring a
    stream's fluency with the model under attack is circular."""
    early_stop_patience: int = 50


@dataclass
class CraftResult:
    """The crafted stream plus the evidence needed to read it honestly."""

    stream: CraftedStream | None
    """None until `craft_stream` fills it in; `search_order` alone builds no
    stream, it only returns an ordering."""

    spans: tuple[Span, ...]
    """The ordering actually realised in `stream`, after `_resolve_lookahead`
    has had its say -- not necessarily the ordering the last accepted proposal
    asked for."""

    best_score: float
    """Objective value of the returned ordering, fluency term included."""

    baseline_score: float
    """Objective value of the *sampled* ordering, before any search. The
    attacker's gain is `best_score - baseline_score`; reporting only the final
    value makes an ineffective search look like a strong attack."""

    n_evaluations: int
    """Victim adapt-and-eval calls actually spent. The pilot's note applies: each
    proposal costs one, so the budget is stated in evaluations rather than left
    implicit in an iteration count."""

    n_accepted: int
    """Proposals that improved the objective. Zero means the search found
    nothing and the result is the sampled ordering -- not an attack."""

    stopped_early: bool

    history: list[float] = field(default_factory=list)
    """Best-so-far after each evaluation. A flat curve says the lever does not
    move the victim, which is a finding about the threat model rather than a
    failure of the run."""

    control: CraftedStream | None = None
    """The length-matched benign arm the fluency ratio was taken against, held
    fixed across the search. Returned so the caller measures against the same
    control the attacker was scored on rather than rebuilding its own."""

    @property
    def gain(self) -> float:
        return self.best_score - self.baseline_score

    @property
    def search_did_anything(self) -> bool:
        """False when the search returned its own starting point.

        Worth asserting on before a result is interpreted: a crafted stream that
        is just the sampled ordering is a benign stream, and comparing it to a
        benign control measures drift, not corruption.
        """
        return self.n_accepted > 0


def search_order(
    spans: Sequence[Span],
    objective: SpanObjective,
    *,
    max_iters: int,
    seed: int,
    early_stop_patience: int = 50,
) -> CraftResult:
    """Hill-climb the ordering by pairwise swaps, scored by `objective`.

    Promoted from `experiments/002-pilot-tiny-ttt/run_deep.py:163-192`, which is
    the only version of this search that has produced a measured number.

    Deliberately a local search and not something cleverer. The attacker is
    optimising a black box whose every query costs a full adapt-and-eval on an
    80GB accelerator; a method needing thousands of queries is not a threat
    model, it is a compute budget. Pairwise swaps also keep the multiset exactly
    fixed, which is what preserves `span_multiplicity == 1` and keeps the
    fluency bar meaningful.

    Returns `CraftResult` rather than just the ordering, so the caller can see
    whether the search actually moved -- see `CraftResult.search_did_anything`.
    """
    spans = list(spans)
    if len(spans) < 2:
        raise ValueError(
            f"need at least 2 spans to search an ordering, got {len(spans)}. "
            f"A single span has one ordering and the attacker has no lever."
        )
    if max_iters < 0:
        raise ValueError(f"max_iters must be non-negative, got {max_iters}")

    rng = np.random.default_rng(seed)

    baseline = float(objective(spans))
    best = baseline
    evaluations = 1
    accepted = 0
    since_improvement = 0
    stopped_early = False
    history = [best]

    for _ in range(max_iters):
        i, j = rng.integers(0, len(spans), size=2)
        if i == j:
            # Not an evaluation: nothing was scored. Counting it would let a
            # run report a spent budget it never spent.
            continue

        candidate = list(spans)
        candidate[i], candidate[j] = candidate[j], candidate[i]

        score = float(objective(candidate))
        evaluations += 1

        if score > best:
            spans, best = candidate, score
            accepted += 1
            since_improvement = 0
        else:
            since_improvement += 1

        history.append(best)

        if early_stop_patience > 0 and since_improvement >= early_stop_patience:
            stopped_early = True
            break

    return CraftResult(
        stream=None,  # filled in by craft_stream; search alone builds no stream
        spans=tuple(spans),
        best_score=best,
        baseline_score=baseline,
        n_evaluations=evaluations,
        n_accepted=accepted,
        stopped_early=stopped_early,
        history=history,
    )


def fluency_penalised(
    objective: StreamObjective,
    fluency_scorer: FluencyScorer | None,
    fluency_weight: float,
    control: CraftedStream | None = None,
) -> StreamObjective:
    """Fold the fluency term into the objective, per the module docstring.

    Inside, never after. A post-hoc filter throws away the compute spent on
    rejected candidates and biases the survivors toward whatever happened to
    slip through, which is a different (and unstated) attack than the one the
    pre-registration describes.

    `fluency_scorer` returns the stream's perplexity ratio against the
    independent reference model; higher is less benign-looking, so it is
    subtracted. With no scorer the objective is returned unchanged and the
    caller is responsible for knowing the stream is unconstrained -- which
    `craft_stream` refuses to let happen silently.
    """
    if fluency_scorer is None or fluency_weight == 0.0:
        return objective
    if control is None:
        raise ValueError(
            "fluency scoring needs the control arm: the bar is a ratio of "
            "poison perplexity to control perplexity, so it is not defined on "
            "one stream."
        )

    def penalised(stream: CraftedStream) -> float:
        return float(objective(stream)) - fluency_weight * float(
            fluency_scorer(stream, control)
        )

    return penalised


def craft_stream(
    victim_model,
    spec: AttackSpec,
    config: CraftConfig,
    corpus,
    seed: int,
    *,
    objective: StreamObjective | None = None,
    fluency_scorer: FluencyScorer | None = None,
    span_tokens: int = DEFAULT_SPAN_TOKENS,
    mini_batch_size: int = 1024,
    offset_range: tuple[int, int] | None = None,
) -> CraftResult:
    """Optimise a poison stream against `spec` within the strategy's constraints.

    Loop: propose stream (constrained) -> run through victim inner loop -> score
    objective + fluency -> keep/perturb.

    The stream is built through `build_select_stream`'s `order_fn` seam rather
    than by concatenating spans here, so every constraint that module enforces --
    the `+1` window convention, the non-BOS lookahead, span/chunk divisibility,
    sampling without replacement -- applies to the crafted arm exactly as it does
    to the control. `stream.py` documents `order_fn` as the hook "a crafting
    optimiser uses without reimplementing the constraints"; this is that caller.

    Args:
        victim_model: the model being attacked. Used only to build a default
            `objective`; pass `objective` directly and this may be None.
        objective: spans -> attacker score, higher is better. Normally the
            victim's held-out benign loss after adapting on the stream, i.e.
            `harness.make_adapt_and_eval`. Injected so the search is testable
            with no model, no checkpoint and no GPU.
        fluency_scorer: spans -> perplexity ratio under the *independent*
            reference model. Required unless `spec.fluency_weight` is zero.
    """
    if config.strategy is not StreamStrategy.SELECT:
        raise NotImplementedError(
            f"{config.strategy.name} crafting is T2.5 and is blocked: "
            f"PARAPHRASE needs a Llama-3 encode/decode round trip and cannot "
            f"work in ID space; SOFT needs the victim's embedding matrix and "
            f"gradients through the stream. SELECT is the pre-registered "
            f"headline strategy and is implemented."
        )

    if spec.objective is not Objective.DEGRADE:
        raise NotImplementedError(
            f"{spec.objective.name} crafting is T2.8 and is blocked on a victim "
            f"(T1.10). Only DEGRADE has a scalar objective the ordering search "
            f"can hill-climb; TRIGGER needs success-rate evaluation per proposal."
        )

    if objective is None:
        raise NotImplementedError(
            "craft_stream needs an `objective`. Build it from the victim with "
            "`trustgate.eval.harness.make_adapt_and_eval`, which requires an "
            "importable vendor tree and a GPU (but no checkpoint -- "
            "train.py:198 random-inits when load_part=none). The search itself "
            "is implemented and CPU-testable; see `search_order`."
        )

    if fluency_scorer is None and spec.fluency_weight != 0.0:
        raise ValueError(
            f"spec.fluency_weight is {spec.fluency_weight} but no fluency_scorer "
            f"was supplied, so the fluency term would be silently absent from "
            f"the objective. That is the unconstrained attack, which proves "
            f"nothing about the 'benign-looking' threat model in dossier "
            f"section 1. Pass a scorer, or set fluency_weight=0.0 explicitly to "
            f"run the diagnostic upper bound and report it as such."
        )

    def build(order: Sequence[Span] | None = None) -> CraftedStream:
        """Every stream in this function comes from here.

        Scoring and the final construction go through one path, so
        `_resolve_lookahead`'s span promotion is included in what the search
        measures. Building the winner separately would risk returning a stream
        whose score was never actually observed.
        """
        return build_select_stream(
            corpus,
            spec.stream_tokens,
            seed,
            span_tokens=span_tokens,
            mini_batch_size=mini_batch_size,
            offset_range=offset_range,
            order_fn=None if order is None else (lambda _spans: list(order)),
        )

    # Sample the admissible span set once. The search permutes it and never
    # resizes it, so multiplicity and provenance are fixed before optimisation
    # begins and cannot drift as a side effect of the search.
    seeded = build(None)

    # The control arm, built by the identical procedure at the same seed. It is
    # the denominator of the fluency ratio and it is held FIXED across the
    # search: if the control moved with each proposal, the attacker could lower
    # the ratio by making the control look worse instead of making the poison
    # look better, and the realism bar would stop measuring realism.
    control = build_benign_control(
        corpus,
        spec.stream_tokens,
        seed,
        span_tokens=span_tokens,
        mini_batch_size=mini_batch_size,
        offset_range=offset_range,
    )

    scored = fluency_penalised(objective, fluency_scorer, spec.fluency_weight, control)

    def score_spans(order: Sequence[Span]) -> float:
        return float(scored(build(order)))

    result = search_order(
        seeded.spans,
        score_spans,
        max_iters=config.max_iters,
        seed=seed,
        early_stop_patience=config.early_stop_patience,
    )

    crafted = build(result.spans)

    # Score the winner. The streams scored during the search were candidates
    # that have been discarded; `crafted` is rebuilt fresh and carries no
    # scores, so without this its `perplexity` and `control_perplexity` stay
    # `nan` and `is_scored` stays False. That fails safe -- `nan <= 1.5` is
    # False, so an unscored stream cannot pass the realism bar -- but it would
    # also mean the crafted arm could never pass it, which is not a bar, it is
    # a wall. One extra reference-model call, and no victim call at all.
    if fluency_scorer is not None:
        fluency_scorer(crafted, control)

    result.stream = crafted
    result.control = control
    # Report the ordering that was realised, not the one proposed:
    # `_resolve_lookahead` may have moved a span to secure the lookahead token.
    result.spans = crafted.spans
    return result
