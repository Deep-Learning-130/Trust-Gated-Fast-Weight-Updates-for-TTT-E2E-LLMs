"""Crafted-stream construction for the Phase 1 attack spike.

DEFENSIVE RESEARCH. See DISCLOSURE.md.

The hard constraint, and the reason this is not just GCG
--------------------------------------------------------
Dossier section 1 specifies "not a single adversarial token, but a slow,
benign-looking sequence". Unconstrained token optimisation produces fluent-adjacent
garbage that any perplexity filter, and any human skimming the context, would
catch. An attack that only works with visibly adversarial input does not
establish the threat model and would not justify the defense.

So the stream is constrained to stay in-distribution. Three strategies, in
increasing order of realism and difficulty:

  SELECT    -- choose and order spans of real text from a benign corpus. Never
               synthesises text, so fluency is exact by construction. Weakest
               attack, but any positive result here is immediately credible.
  PARAPHRASE -- perturb benign text within a semantic-similarity ball.
  SOFT      -- optimise in embedding space, project back to tokens. Strongest,
               least realistic; use as an upper bound on what is achievable, not
               as the headline result.

Report SELECT separately. If only SOFT works, say so plainly -- that is a much
weaker threat model and it changes whether the defense is worth building.

Spans, not sentences
--------------------
The vendor has no tokenizer, so there are no sentence boundaries to cut on
without a decoder, and the obvious substitute -- the inner-loop chunk -- would
leave the attacker only 8 units of choice per 8192-token window. `span_tokens`
is therefore an explicit parameter well below the chunk size. The reasoning, and
the confounds it creates, are in `docs/adr/ADR-007-select-span-granularity.md`.

What these builders are, and are not
------------------------------------
They are the *constrained proposal generator*: given a seed they produce one
admissible stream. They do not search. Adversarial choice and ordering is
`craft_stream`'s job (T1.9), which optimises over exactly this proposal space
with the fluency term inside the objective. Passing `order_fn` lets that
optimiser drive the ordering without reimplementing the constraints.

The window `+1` convention
--------------------------
The vendor's dataloader yields `seq_len + 1` tokens per window and shifts inside
the loader: `input_ids=tokens[:-1]`, `target_tokens=tokens[1:]`
(`lm_dataset.py:22,51-52`). So `length_tokens` here means the *model-consumed*
length -- `seq_len` -- and `CraftedStream.tokens` is one longer. That last token
is real corpus text read from the position immediately after the final span, not
padding and never BOS: BOS-padding would be masked straight out of the loss and
would break valid-token matching between the arms while looking perfectly fine.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass
from math import nan
from typing import Callable, Sequence

import numpy as np

from trustgate.attack.corpus import (
    DEFAULT_SPAN_TOKENS,
    Span,
    TokenCorpus,
    mean_offset_gap,
    span_multiplicity,
    validate_span_tokens,
)
from trustgate.eval.lm_loss import count_valid
from trustgate.tokens import BOS_TOKEN_ID


class StreamStrategy(enum.Enum):
    SELECT = "select"
    PARAPHRASE = "paraphrase"
    SOFT = "soft"


#: Ordering hook. Takes the sampled spans and returns them permuted. Defaults to
#: identity (sampled order); `craft_stream` supplies the adversarial one.
OrderFn = Callable[[Sequence[Span]], Sequence[Span]]


@dataclass
class CraftedStream:
    """A poison stream plus the evidence that it is benign-looking."""

    tokens: np.ndarray
    strategy: StreamStrategy

    perplexity: float = nan
    """Under an independent reference model -- never the victim. Scoring
    fluency with the model being attacked is circular.

    Defaults to `nan` because the reference model is T3.2 and does not exist
    yet. `nan <= max_fluency_ratio` is `False`, so an unscored stream fails safe
    to STOP rather than quietly passing the realism bar."""

    control_perplexity: float = nan
    """Same measure on the length-matched benign control. The stream is only
    'benign-looking' if these are close; report the ratio, not just the value."""

    spans: tuple[Span, ...] = ()
    """Provenance. Recorded so the structural variables below can be reported
    rather than assumed equal between the arms."""

    corpus_name: str = ""
    """Which split the spans came from. Must differ from the benign eval split:
    drawing control spans from the split the eval reads makes the control
    in-domain and the poison out-of-domain, manufacturing a positive result."""

    span_tokens: int = 0
    mini_batch_size: int = 0

    @property
    def fluency_ratio(self) -> float:
        return self.perplexity / max(self.control_perplexity, 1e-6)

    @property
    def is_scored(self) -> bool:
        """False until an independent reference model has scored the stream."""
        return not (np.isnan(self.perplexity) or np.isnan(self.control_perplexity))

    @property
    def length_tokens(self) -> int:
        """Model-consumed length, i.e. `seq_len`. One less than `tokens`."""
        return len(self.tokens) - 1

    @property
    def valid_tokens(self) -> int:
        """Loss-bearing positions -- the count that actually has to match.

        Computed over the shifted targets exactly as the vendor's loss does, not
        inferred from the length, so a BOS that slipped through is visible here
        instead of silently rescaling a chunk's gradient.
        """
        return count_valid(self.tokens[1:], BOS_TOKEN_ID)

    @property
    def n_chunks(self) -> int:
        """Inner SGD steps this stream will drive. One per `mini_batch_size`."""
        if self.mini_batch_size <= 0:
            return 0
        return self.length_tokens // self.mini_batch_size

    @property
    def span_multiplicity(self) -> int:
        return span_multiplicity(self.spans)

    @property
    def mean_offset_gap(self) -> float:
        return mean_offset_gap(self.spans)


def _resolve_lookahead(
    corpus: TokenCorpus, spans: Sequence[Span]
) -> tuple[list[Span], int]:
    """Order the spans so the last one has a real, non-BOS successor token.

    The `+1` window token has to be genuine contiguous corpus text. Rather than
    padding or resampling, promote a span that *has* a usable successor to last
    position; the multiset of spans is unchanged, and for SELECT reordering is
    within the attacker's own freedom anyway.
    """
    spans = list(spans)

    for index in range(len(spans) - 1, -1, -1):
        candidate = spans[index]
        if candidate.end >= len(corpus):
            continue
        token = int(corpus.read(candidate.end, 1)[0])
        if token == corpus.bos_token_id:
            continue
        ordered = spans[:index] + spans[index + 1 :] + [candidate]
        return ordered, token

    raise ValueError(
        f"no sampled span in corpus {corpus.name!r} has a usable successor "
        f"token for the window +1 convention (all are at the corpus end or "
        f"followed by BOS). Re-seed, or widen offset_range. Padding is not an "
        f"option: it is masked out of the loss and would break valid-token "
        f"matching against the control."
    )


def _build(
    corpus: TokenCorpus,
    length_tokens: int,
    seed: int,
    strategy: StreamStrategy,
    *,
    span_tokens: int = DEFAULT_SPAN_TOKENS,
    mini_batch_size: int = 1024,
    offset_range: tuple[int, int] | None = None,
    replace: bool = False,
    order_fn: OrderFn | None = None,
) -> CraftedStream:
    """Shared construction. Poison and control differ only in seed and ordering."""
    validate_span_tokens(span_tokens, mini_batch_size)

    if length_tokens % span_tokens != 0:
        raise ValueError(
            f"length_tokens {length_tokens} must be a multiple of span_tokens "
            f"{span_tokens}; a partial span would have to be padded, and "
            f"padding is masked out of the loss"
        )
    if length_tokens % mini_batch_size != 0:
        raise ValueError(
            f"length_tokens {length_tokens} must be a multiple of "
            f"mini_batch_size {mini_batch_size}, or the final inner-loop chunk "
            f"is ragged and the two arms take a different number of steps"
        )

    n_spans = length_tokens // span_tokens
    spans = corpus.sample_spans(
        n_spans,
        span_tokens,
        seed=seed,
        replace=replace,
        offset_range=offset_range,
    )

    if order_fn is not None:
        spans = list(order_fn(spans))
        if len(spans) != n_spans:
            raise ValueError(
                f"order_fn must permute, not resize: got {len(spans)} spans, "
                f"expected {n_spans}"
            )

    spans, lookahead = _resolve_lookahead(corpus, spans)

    tokens = np.concatenate(
        [corpus.materialize(spans), np.array([lookahead], dtype=np.int32)]
    )

    return CraftedStream(
        tokens=tokens,
        strategy=strategy,
        spans=tuple(spans),
        corpus_name=corpus.name,
        span_tokens=span_tokens,
        mini_batch_size=mini_batch_size,
    )


def build_select_stream(
    corpus: TokenCorpus,
    length_tokens: int,
    seed: int,
    *,
    span_tokens: int = DEFAULT_SPAN_TOKENS,
    mini_batch_size: int = 1024,
    offset_range: tuple[int, int] | None = None,
    replace: bool = False,
    order_fn: OrderFn | None = None,
) -> CraftedStream:
    """SELECT strategy: order existing benign spans adversarially.

    Fluency is exact -- every span is real text. The only attacker freedom is
    choice and ordering, which is a genuinely weak lever and therefore a
    genuinely strong result if it works.

    `replace=False` by default: with replacement the attacker could use one span
    for every chunk, which is eight epochs on one passage rather than an attack,
    and a one-line dedup would defend against it.
    """
    return _build(
        corpus,
        length_tokens,
        seed,
        StreamStrategy.SELECT,
        span_tokens=span_tokens,
        mini_batch_size=mini_batch_size,
        offset_range=offset_range,
        replace=replace,
        order_fn=order_fn,
    )


def build_benign_control(
    corpus: TokenCorpus,
    length_tokens: int,
    seed: int,
    *,
    span_tokens: int = DEFAULT_SPAN_TOKENS,
    mini_batch_size: int = 1024,
    offset_range: tuple[int, int] | None = None,
) -> CraftedStream:
    """Length-matched benign control stream.

    Required by the pre-registration. Fast weights drift on *any* stream; without
    this control, benign adaptation is indistinguishable from poisoning and the
    attack result is uninterpretable.

    Built by the identical span-concatenation procedure as the poison arm, so the
    baseline cost of the seams is common to both and largely divides out of the
    fluency ratio. It does **not** simply cancel: the poison arm's adjacencies
    are adversarially chosen, so its excess NLL at the seams is larger, and that
    residual is the signal the fluency bar exists to catch.
    """
    return _build(
        corpus,
        length_tokens,
        seed,
        StreamStrategy.SELECT,
        span_tokens=span_tokens,
        mini_batch_size=mini_batch_size,
        offset_range=offset_range,
        replace=False,
        order_fn=None,
    )


def build_contiguous_stream(
    corpus: TokenCorpus,
    length_tokens: int,
    seed: int,
    *,
    mini_batch_size: int = 1024,
    offset_range: tuple[int, int] | None = None,
) -> CraftedStream:
    """One unbroken corpus slice. **Reported, never gating.**

    An absolute fluency anchor. If the shared concatenation procedure dominates
    perplexity in both arms, the poison/control *ratio* tends to 1.0 regardless
    of how unnatural the poison ordering is -- and smaller spans make that
    worse. This arm quantifies what the procedure itself costs, so the ratio can
    be read honestly instead of assumed clean.

    No pre-registered bar is expressed over it and none moves because of it.

    Built directly rather than through `_build`, because a single span the whole
    length of the window does not have to divide `mini_batch_size` -- there are
    no seams to align.
    """
    spans = corpus.sample_spans(
        1, length_tokens, seed=seed, replace=False, offset_range=offset_range
    )
    spans, lookahead = _resolve_lookahead(corpus, spans)
    tokens = np.concatenate(
        [corpus.materialize(spans), np.array([lookahead], dtype=np.int32)]
    )
    return CraftedStream(
        tokens=tokens,
        strategy=StreamStrategy.SELECT,
        spans=tuple(spans),
        corpus_name=corpus.name,
        span_tokens=length_tokens,
        mini_batch_size=mini_batch_size,
    )


def build_difficulty_matched_control(
    corpus: TokenCorpus,
    poison: CraftedStream,
    seed: int,
    scorer: Callable[[np.ndarray], float],
    *,
    candidate_pool: int = 512,
    offset_range: tuple[int, int] | None = None,
) -> CraftedStream:
    """A control matched on intrinsic difficulty. **Reported, never gating.**

    Addresses the sharpest confound in the design. Adversarial selection will
    prefer high-entropy, rare-token spans because those produce the largest
    inner-loop gradients, so the poison arm can differ from a uniformly-drawn
    control in intrinsic *difficulty* rather than in adversarial ordering -- a
    distribution shift wearing an attack's clothes.

    Greedy nearest-NLL matching, span by span, without replacement. `scorer` is
    injected and must not be the victim model: scoring with the model under
    attack is circular and the pre-registration names it invalidating.

    The pre-registered length-matched control (`build_benign_control`) remains
    the one that gates. This is evidence reported alongside it.
    """
    if poison.span_tokens <= 0:
        raise ValueError("poison stream carries no span_tokens; cannot match difficulty")

    targets = [
        scorer(corpus.read(span.offset, span.length)) for span in poison.spans
    ]

    pool = corpus.sample_spans(
        min(candidate_pool, max(len(poison.spans) * 4, len(poison.spans))),
        poison.span_tokens,
        seed=seed,
        replace=False,
        offset_range=offset_range,
    )
    pool_scores = {span.offset: scorer(corpus.read(span.offset, span.length)) for span in pool}

    chosen: list[Span] = []
    remaining = {span.offset: span for span in pool}

    for target in targets:
        if not remaining:
            raise ValueError(
                "candidate pool exhausted before the control was full; raise "
                "candidate_pool or widen offset_range"
            )
        best = min(remaining, key=lambda offset: abs(pool_scores[offset] - target))
        chosen.append(remaining.pop(best))

    chosen, lookahead = _resolve_lookahead(corpus, chosen)
    tokens = np.concatenate(
        [corpus.materialize(chosen), np.array([lookahead], dtype=np.int32)]
    )

    return CraftedStream(
        tokens=tokens,
        strategy=poison.strategy,
        spans=tuple(chosen),
        corpus_name=corpus.name,
        span_tokens=poison.span_tokens,
        mini_batch_size=poison.mini_batch_size,
    )


def assert_streams_matched(poison: CraftedStream, control: CraftedStream) -> None:
    """Fail loudly on any structural difference between the two arms.

    Poison and control must differ in *exactly one* thing: the content of the
    stream. Everything checked here is a plausible confound, and fast weights
    drift on any input at all -- so an unmatched control would make a null
    result look like a positive one.
    """
    mismatches = []

    checks = {
        "token count": (len(poison.tokens), len(control.tokens)),
        "valid tokens": (poison.valid_tokens, control.valid_tokens),
        "chunk count": (poison.n_chunks, control.n_chunks),
        "span tokens": (poison.span_tokens, control.span_tokens),
        "mini batch size": (poison.mini_batch_size, control.mini_batch_size),
        "dtype": (poison.tokens.dtype, control.tokens.dtype),
    }
    for label, (a, b) in checks.items():
        if a != b:
            mismatches.append(f"{label}: {a!r} != {b!r}")

    if mismatches:
        raise ValueError(
            "poison and control streams must be structurally identical; "
            "mismatched: " + "; ".join(mismatches)
        )


def assert_stream_not_from_eval_split(stream: CraftedStream, benign_eval_split: str) -> None:
    """Streams must not be drawn from the split the benign eval reads.

    The vendor has only `/train` and `/val` (`lm_dataset.py:16`). If control
    spans come from `/val` while the benign eval also reads `/val`, the control
    is in-domain to the measurement and the poison is not -- a spurious positive
    by construction, with nothing in the numbers to reveal it.
    """
    if stream.corpus_name and stream.corpus_name == benign_eval_split:
        raise ValueError(
            f"stream was drawn from split {stream.corpus_name!r}, which is also "
            f"the benign eval split. The control would be in-domain to the "
            f"measurement and the poison would not."
        )


def build_paraphrase_stream(
    corpus: TokenCorpus,
    length_tokens: int,
    seed: int,
    *,
    tokenizer,
    perturbation_fn: Callable[[np.ndarray], np.ndarray],
    span_tokens: int = DEFAULT_SPAN_TOKENS,
    mini_batch_size: int = 1024,
    offset_range: tuple[int, int] | None = None,
) -> CraftedStream:
    """PARAPHRASE strategy: perturb spans within a semantic-similarity ball.

    T2.5 implementation. Blocked on T3.2 (tokenizer) for actual execution.

    Takes a base stream (via SELECT), decodes each span to text, applies a
    perturbation function that stays within a semantic-similarity ball, then
    re-encodes. The perturbation must preserve length (or be padded/truncated
    to match).

    Args:
        corpus: token-ID stream to draw spans from
        length_tokens: target stream length (must divide span_tokens and mini_batch_size)
        seed: for reproducible span sampling
        tokenizer: injected tokenizer with decode() and encode() methods
        perturbation_fn: takes decoded text, returns perturbed text within similarity ball
        span_tokens: span size (default 64)
        mini_batch_size: vendor's inner-loop chunk size (default 1024)
        offset_range: restrict sampling to this corpus region
    """
    raise NotImplementedError(
        "PARAPHRASE blocked on T3.2 (independent fluency reference model with tokenizer). "
        "The perturbation_fn must edit text and cannot work in ID space. "
        "Interface is ready; implementation requires tokenizer decode/encode."
    )


def build_soft_stream(
    corpus: TokenCorpus,
    length_tokens: int,
    seed: int,
    *,
    embedding_matrix: np.ndarray,
    optimization_fn: Callable[[np.ndarray], np.ndarray],
    span_tokens: int = DEFAULT_SPAN_TOKENS,
    mini_batch_size: int = 1024,
    offset_range: tuple[int, int] | None = None,
) -> CraftedStream:
    """SOFT strategy: optimize in embedding space, project back to tokens.

    T2.5 implementation. Blocked on model access for embedding matrix and
    gradient computation.

    Starts with continuous embeddings, optimizes via gradient descent, projects
    back to discrete tokens each step (argmax over vocab). Strongest attack,
    least realistic -- report as upper bound, never headline.

    Args:
        corpus: token-ID stream to initialize from
        length_tokens: target stream length
        seed: for reproducible initialization
        embedding_matrix: vocab_size x embed_dim matrix from the model
        optimization_fn: takes embeddings, returns optimized embeddings
        span_tokens: span size (default 64)
        mini_batch_size: vendor's inner-loop chunk size (default 1024)
        offset_range: restrict sampling to this corpus region
    """
    raise NotImplementedError(
        "SOFT blocked on model access for embedding matrix and gradient computation. "
        "Interface is ready; implementation requires forward passes and projection."
    )


def generate_seed_pairs(
    corpus: TokenCorpus,
    length_tokens: int,
    seeds: list[int],
    *,
    craft_fn: Callable[[int], OrderFn] | None = None,
    span_tokens: int = DEFAULT_SPAN_TOKENS,
    mini_batch_size: int = 1024,
    offset_range: tuple[int, int] | None = None,
) -> list[tuple[CraftedStream, CraftedStream]]:
    """Generate per-seed poison/control pairs for the attack spike.

    T2.6 implementation. For each seed, builds a poison stream (with adversarial
    ordering from craft_fn when available) and a length-matched control (random
    ordering). Asserts structural matching.

    Args:
        corpus: token-ID stream to draw spans from
        length_tokens: target stream length for each stream
        seeds: list of seeds (typically [0, 1, 2, 3, 4] for 5 seeds)
        craft_fn: seed -> order_fn for adversarial ordering. None = random ordering
            (not a real attack; for testing the orchestration only).
        span_tokens: span size (default 64)
        mini_batch_size: vendor's inner-loop chunk size (default 1024)
        offset_range: restrict sampling to this corpus region

    Returns:
        List of (poison, control) pairs, one per seed. Each pair passes
        assert_streams_matched.
    """
    if not seeds:
        raise ValueError("no seeds: the pre-registration requires 5 per condition")

    pairs: list[tuple[CraftedStream, CraftedStream]] = []

    for seed in seeds:
        order_fn = craft_fn(seed) if craft_fn is not None else None

        poison = build_select_stream(
            corpus,
            length_tokens,
            seed,
            span_tokens=span_tokens,
            mini_batch_size=mini_batch_size,
            offset_range=offset_range,
            order_fn=order_fn,
        )

        control = build_benign_control(
            corpus,
            length_tokens,
            seed,
            span_tokens=span_tokens,
            mini_batch_size=mini_batch_size,
            offset_range=offset_range,
        )

        assert_streams_matched(poison, control)

        pairs.append((poison, control))

    return pairs
