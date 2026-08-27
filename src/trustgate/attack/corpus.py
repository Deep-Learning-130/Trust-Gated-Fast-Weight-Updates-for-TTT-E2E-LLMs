"""Benign token corpus and span sampling for crafted streams.

DEFENSIVE RESEARCH. See DISCLOSURE.md.

Why this works in token-ID space
--------------------------------
The vendor has no tokenizer (see `trustgate.tokens`), and `/train` and `/val`
are single flat 1-D token-ID streams (`lm_dataset.py:11-27`). That is a gift
rather than an obstacle: a span of this corpus *is* real text by construction,
so the SELECT strategy's fluency-is-exact property holds without decoding
anything, and length matching between poison and control is exact in the units
the model actually consumes. No tokenizer enters the pipeline at all. One is
needed only to *decode* for fluency scoring against an independent reference
model (T3.2), which is deliberately out of this wave.

Duck-typed backing store
------------------------
`store` is anything supporting `len()` and slicing. Today that is a numpy array;
on the GPU box it is `zarr.open_array(...)` over the real `/val`, with no code
change. **`zarr` is deliberately not imported here** -- it is absent from this
project's dependencies and from CI, and the CPU test suite has to run anywhere.

Span granularity
----------------
`span_tokens` is a parameter and is *not* pinned to `mini_batch_size`. See
`docs/adr/ADR-007-select-span-granularity.md`: chunk-sized spans would give the
attacker only 8 units of choice per 8192-token window, and a STOP under an
8-unit lever could not be distinguished from "SELECT cannot corrupt fast
weights". The kill-gate is asymmetric -- a false STOP ends the project -- so the
default is well below the chunk size.

BOS-free spans
--------------
Any span containing BOS is rejected. The reason is stronger than length
matching: the inner objective's denominator is *that chunk's* valid count
(`loss.py:15,26`), so a chunk holding fewer valid tokens up-weights every
surviving token's gradient. BOS is a per-inner-step gradient-scale confound, not
merely a sequence-length one. The cost is a selection bias worth stating --
BOS delimits documents, so "BOS-free" means "intra-document", and a real
attacker submitting a document would start at a boundary. Both arms are equally
biased, so the comparison holds; the §1 story is slightly undercut.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Callable, Sequence

import numpy as np

from trustgate.tokens import BOS_TOKEN_ID

#: Default span length in tokens. Divides every `mini_batch_size` the vendor
#: configs use (1024 for the 1B extension configs), leaving 16 spans per inner
#: step and 128 per 8192-token window -- a real search space rather than 8.
DEFAULT_SPAN_TOKENS = 64


@dataclass(frozen=True)
class Span:
    """A contiguous run of real corpus tokens, identified by where it came from.

    Provenance is recorded, not incidental. Poison spans clustered in one region
    of the corpus are topically coherent in a way spans spread across it are
    not, which can make the poison arm *more* fluent and even lower-loss than
    its control for reasons that have nothing to do with the attack.
    """

    offset: int
    length: int

    baseline_nll: float | None = None
    """Filled by `score_spans`. Adversarial selection will prefer high-entropy,
    rare-token spans because those produce the largest inner-loop gradients, so
    without this the poison arm differs from the control in intrinsic
    *difficulty* and not only in adversarial ordering. Needed to build the
    difficulty-matched control."""

    @property
    def end(self) -> int:
        return self.offset + self.length


class TokenCorpus:
    """A flat token-ID stream, sampled in BOS-free spans."""

    def __init__(
        self,
        store,
        *,
        name: str,
        bos_token_id: int = BOS_TOKEN_ID,
        terminator_ids: frozenset[int] | None = None,
    ):
        """
        Args:
            store: anything with `len()` and slicing -- a numpy array now, the
                real zarr `/val` array later. Never imported or type-checked
                against `zarr`.
            name: the split this came from (`"train"` / `"val"`). Carried so a
                stream can assert it was not drawn from the split the benign
                eval reads; drawing control spans from the eval split makes the
                control in-domain and the poison out-of-domain, which
                manufactures a positive result.
            terminator_ids: sentence-terminator token ids. **Unused in this
                wave** -- identifying them requires a decoder, and guessing
                would be worse than not cutting on sentences at all. Enables
                sentence-level segmentation once T3.2 brings a tokenizer.
        """
        self.store = store
        self.name = name
        self.bos_token_id = bos_token_id
        self.terminator_ids = terminator_ids

    def __len__(self) -> int:
        return len(self.store)

    def read(self, offset: int, length: int) -> np.ndarray:
        """Materialise a slice as int32 -- the dtype the vendor casts to.

        `transformer.py:513` does `seq.input_ids.astype(jnp.int32)`, and
        `vocab_size=128256` rules out uint16, so int32 is the honest choice
        rather than whatever the offline tokenizer happened to write.
        """
        if offset < 0 or length <= 0 or offset + length > len(self):
            raise ValueError(
                f"span [{offset}, {offset + length}) is outside corpus "
                f"{self.name!r} of length {len(self)}"
            )
        return np.asarray(self.store[offset : offset + length], dtype=np.int32)

    def is_bos_free(self, offset: int, length: int) -> bool:
        return not bool(np.any(self.read(offset, length) == self.bos_token_id))

    def materialize(self, spans: Sequence[Span]) -> np.ndarray:
        """Concatenate spans into one token stream, in the given order."""
        if not spans:
            return np.empty(0, dtype=np.int32)
        return np.concatenate([self.read(s.offset, s.length) for s in spans])

    def enumerate_spans(
        self,
        span_tokens: int = DEFAULT_SPAN_TOKENS,
        *,
        offset_range: tuple[int, int] | None = None,
    ) -> list[Span]:
        """Every aligned, BOS-free span in `offset_range`.

        Exhaustive, so only appropriate for a bounded region or a small corpus.
        `sample_spans` is the route for the real `/val`.
        """
        lo, hi = self._resolve_range(offset_range, span_tokens)
        return [
            Span(offset=offset, length=span_tokens)
            for offset in range(lo, hi - span_tokens + 1, span_tokens)
            if self.is_bos_free(offset, span_tokens)
        ]

    def sample_spans(
        self,
        n_spans: int,
        span_tokens: int = DEFAULT_SPAN_TOKENS,
        *,
        seed: int,
        replace: bool = False,
        offset_range: tuple[int, int] | None = None,
        max_attempts_per_span: int = 64,
    ) -> list[Span]:
        """Draw `n_spans` aligned BOS-free spans, reproducibly from `seed`.

        Rejection sampling rather than an exhaustive scan, so this stays usable
        against a multi-million-token `/val` without reading all of it.

        `replace=False` by default. With replacement, SELECT could use one span
        eight times -- eight identical gradient steps, which is effectively
        eight epochs on one passage and would "work" for a reason a one-line
        dedup defends against. The control never repeats, so multiplicity is an
        unmatched structural variable unless both arms are drawn the same way.
        """
        if n_spans <= 0:
            raise ValueError(f"n_spans must be positive, got {n_spans}")

        lo, hi = self._resolve_range(offset_range, span_tokens)
        n_positions = (hi - lo) // span_tokens
        if n_positions <= 0:
            raise ValueError(
                f"corpus {self.name!r} range [{lo}, {hi}) holds no aligned span "
                f"of {span_tokens} tokens"
            )
        if not replace and n_spans > n_positions:
            raise ValueError(
                f"cannot draw {n_spans} distinct spans from {n_positions} "
                f"aligned positions in corpus {self.name!r}"
            )

        rng = np.random.default_rng(seed)
        spans: list[Span] = []
        taken: set[int] = set()

        for _ in range(n_spans):
            for _attempt in range(max_attempts_per_span):
                index = int(rng.integers(n_positions))
                offset = lo + index * span_tokens
                if not replace and offset in taken:
                    continue
                if not self.is_bos_free(offset, span_tokens):
                    continue
                taken.add(offset)
                spans.append(Span(offset=offset, length=span_tokens))
                break
            else:
                raise ValueError(
                    f"could not find a BOS-free span of {span_tokens} tokens in "
                    f"corpus {self.name!r} after {max_attempts_per_span} "
                    f"attempts (found {len(spans)} of {n_spans}). The corpus "
                    f"region may be too dense in document boundaries, or too "
                    f"small. Refusing to fall back to BOS-containing spans: "
                    f"they change the per-chunk gradient scale and would break "
                    f"matching against the control while looking fine."
                )

        return spans

    def _resolve_range(
        self, offset_range: tuple[int, int] | None, span_tokens: int
    ) -> tuple[int, int]:
        if span_tokens <= 0:
            raise ValueError(f"span_tokens must be positive, got {span_tokens}")

        lo, hi = (0, len(self)) if offset_range is None else offset_range
        if lo < 0 or hi > len(self) or lo >= hi:
            raise ValueError(
                f"offset_range {(lo, hi)} is not a valid sub-range of corpus "
                f"{self.name!r} of length {len(self)}"
            )
        return lo, hi


def validate_span_tokens(span_tokens: int, mini_batch_size: int) -> None:
    """Spans must tile the vendor's inner-loop chunk exactly.

    One inner SGD step happens per `mini_batch_size` tokens
    (`transformer.py:709-718`). A span size that does not divide it would
    straddle chunk boundaries unevenly, so poison and control would place their
    seams at different points within the gradient and the arms would differ by
    more than content.
    """
    if span_tokens <= 0:
        raise ValueError(f"span_tokens must be positive, got {span_tokens}")
    if mini_batch_size % span_tokens != 0:
        raise ValueError(
            f"span_tokens {span_tokens} must divide mini_batch_size "
            f"{mini_batch_size}; otherwise spans straddle inner-loop chunk "
            f"boundaries unevenly between the two arms"
        )


def score_spans(
    corpus: TokenCorpus,
    spans: Sequence[Span],
    scorer: Callable[[np.ndarray], float],
) -> list[Span]:
    """Fill `baseline_nll` on each span using an injected scorer.

    The scorer is injected rather than imported because the independent
    reference model is T3.2 and does not exist yet. Nothing here may score with
    the victim model -- that is circular, and the pre-registration names it as
    an invalidating condition.
    """
    return [
        replace(span, baseline_nll=float(scorer(corpus.read(span.offset, span.length))))
        for span in spans
    ]


def span_multiplicity(spans: Sequence[Span]) -> int:
    """Largest number of times any single span appears. 1 means all distinct."""
    if not spans:
        return 0
    counts: dict[int, int] = {}
    for span in spans:
        counts[span.offset] = counts.get(span.offset, 0) + 1
    return max(counts.values())


def mean_offset_gap(spans: Sequence[Span]) -> float:
    """Mean absolute distance between consecutive spans' source offsets.

    A concentration statistic. Poison spans drawn from one region are topically
    coherent; control spans spread across the corpus are not. Reported for both
    arms so the difference is visible rather than latent.
    """
    if len(spans) < 2:
        return 0.0
    gaps = [abs(b.offset - a.offset) for a, b in zip(spans, spans[1:])]
    return float(np.mean(gaps))
