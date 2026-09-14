"""Fluency scoring under an independent reference model.

T3.2 / TEAM_PLAN P1-8. This closes the third pre-registered bar: "Fluency ratio
(poison perplexity / control perplexity, under an independent reference model)
**≤ 1.5**". Until it exists, no verdict can be PROCEED -- an unscored stream
fails the realism bar safe, so `report.md` says STOP and calls itself
incomplete, whatever the corruption numbers do.

Independence is the whole point
-------------------------------
`PREREGISTERED.md` names "fluency scored with the victim model" as an
invalidating condition, and it is not a technicality: the attacker optimises
against the victim, so the victim assigns *low* perplexity to precisely the
streams the attack has shaped. Scoring with it would report the poison stream as
maximally fluent by construction. `assert_independent_of_victim` makes the check
structural rather than a matter of care -- it compares vocabularies, because a
model sharing the victim's 128256-token Llama-3 vocabulary is the one thing a
GPT-2-shaped mistake would look like.

Why a ratio and not an absolute
-------------------------------
The bar is poison perplexity over *control* perplexity. Both arms are built by
the same span-concatenation procedure (ADR-007), so the baseline cost of the
seams is common to both and largely divides out. It does not fully cancel: the
poison arm's adjacencies are adversarially chosen, so its excess at the seams is
larger, and that residual is exactly the signal this bar exists to catch. A
ratio is only trustworthy if both arms are scored with identical tokenization,
windowing and masking, which is why this module owns all three.

The decode step
---------------
Crafted streams are Llama-3 token ids -- the vendor ships no tokenizer, so the
whole pipeline works in id space (`trustgate.tokens`, ADR-007). The reference
model reads text, so ids are decoded on the way in. This is the one place a
tokenizer is genuinely required, and ADR-007 anticipated it.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np

from trustgate.attack.stream import CraftedStream
from trustgate.tokens import VOCAB_SIZE as VICTIM_VOCAB_SIZE

#: Windowing defaults. `stride` well below the model's context means most tokens
#: are scored with substantial left context; scoring every token in
#: non-overlapping blocks would charge each block's first tokens a
#: no-context penalty and inflate both arms' perplexity unevenly.
DEFAULT_STRIDE = 512


def reference_cache_root() -> Path:
    """Where `scripts/fetch_reference_model.py` put the weights."""
    return Path(
        os.environ.get(
            "TRUSTGATE_REFERENCE_CACHE",
            Path.home() / ".cache" / "trustgate" / "reference",
        )
    )


@dataclass(frozen=True)
class PerplexityScore:
    mean_nll: float
    """Nats per token, the same unit as the victim's reported loss."""

    perplexity: float
    """`exp(mean_nll)`."""

    n_scored_tokens: int
    """Positions that actually contributed. The first token of a sequence can
    never be scored -- it has no context -- so this is at most `len(ids) - 1`."""


def _log_softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - logits.max(axis=-1, keepdims=True)
    return shifted - np.log(np.exp(shifted).sum(axis=-1, keepdims=True))


def sliding_window_nll(
    logits_fn: Callable[[np.ndarray], np.ndarray],
    token_ids,
    max_length: int,
    stride: int = DEFAULT_STRIDE,
) -> PerplexityScore:
    """Mean NLL over a sequence longer than the model's context window.

    Every target position is scored **exactly once**, by the window that gives
    it the most left context. Double-counting would silently weight the overlap
    region twice, and since poison and control streams differ in content the
    two arms would be weighted differently -- which corrupts the ratio rather
    than just the absolute.

    `logits_fn` takes a 1-D id array and returns `[len(ids), vocab]`. Injected so
    the reduction is testable with no model at all.
    """
    tokens = np.asarray(token_ids, dtype=np.int64).ravel()
    n = tokens.size

    if n < 2:
        raise ValueError(
            f"need at least 2 tokens to score one target, got {n}; the first "
            f"token has no context and is never scored"
        )
    if max_length < 2:
        raise ValueError(f"max_length must be >= 2, got {max_length}")
    if stride < 1:
        raise ValueError(f"stride must be >= 1, got {stride}")
    if n > max_length and stride > max_length - 1:
        # Only a multi-window sequence can be under-covered. With a stride past
        # the context window the grid leaves gaps, and targets in them are
        # silently never scored -- a partial perplexity that looks entirely
        # plausible. A sequence that fits in one window is fine at any stride,
        # so this deliberately does not fire there: crafted streams shorter than
        # the reference model's context are ordinary, not an error.
        raise ValueError(
            f"stride {stride} exceeds max_length - 1 ({max_length - 1}) for a "
            f"{n}-token sequence; windows would not cover it and targets would "
            f"be silently dropped"
        )

    total_nll = 0.0
    total_count = 0
    next_target = 1  # global index of the next unscored target

    for begin in range(0, n, stride):
        end = min(begin + max_length, n)

        # Targets this window can score, minus those already scored.
        lo = max(next_target, begin + 1)
        if lo < end:
            window = tokens[begin:end]
            log_prob = _log_softmax(np.asarray(logits_fn(window), dtype=np.float64))

            # Target at global index j is predicted by the logits row j-1.
            rows = np.arange(lo, end) - begin - 1
            targets = tokens[lo:end]
            nll = -log_prob[rows, targets]

            total_nll += float(nll.sum())
            total_count += int(nll.size)
            next_target = end

        if end == n:
            break

    # Internal invariant, not a user error: every target from 1..n-1 must have
    # been scored exactly once. Both halves matter -- scoring too few understates
    # coverage, scoring twice weights the overlap region, and because poison and
    # control differ in content either fault biases the *ratio* rather than
    # cancelling out of it.
    if total_count != n - 1:
        raise RuntimeError(
            f"windowing scored {total_count} targets, expected {n - 1} "
            f"(n={n}, max_length={max_length}, stride={stride}). This is a bug "
            f"in sliding_window_nll, not bad input."
        )

    mean_nll = total_nll / total_count
    return PerplexityScore(
        mean_nll=mean_nll,
        perplexity=float(np.exp(mean_nll)),
        n_scored_tokens=total_count,
    )


def assert_independent_of_victim(reference_vocab_size: int) -> None:
    """Refuse a reference model that shares the victim's vocabulary.

    A structural check, not a proof of independence -- but the failure it guards
    is the one that matters. The attacker optimises against the victim, so the
    victim rates poisoned streams as maximally fluent, and a fluency bar scored
    with it would pass every attack by construction.
    """
    if reference_vocab_size == VICTIM_VOCAB_SIZE:
        raise ValueError(
            f"reference model vocabulary is {reference_vocab_size}, identical to "
            f"the victim's Llama-3 vocabulary. Scoring fluency with the model "
            f"under attack is circular and PREREGISTERED.md names it an "
            f"invalidating condition."
        )


class Llama3Decoder:
    """Decodes crafted-stream token ids back to text, for scoring only.

    Loaded from an ungated Llama-3 tokenizer mirror. The vocabulary size is
    asserted against the vendor's model configs rather than trusted: a tokenizer
    with the wrong vocab would decode to plausible-looking but wrong text, and
    the resulting perplexities would be wrong in a way nothing downstream could
    detect.
    """

    def __init__(self, tokenizer_path: Path):
        from tokenizers import Tokenizer

        self._tokenizer = Tokenizer.from_file(str(Path(tokenizer_path)))
        actual = self._tokenizer.get_vocab_size(with_added_tokens=True)
        if actual != VICTIM_VOCAB_SIZE:
            raise ValueError(
                f"decoder vocabulary is {actual}, expected {VICTIM_VOCAB_SIZE} "
                f"(configs/model/1b.yaml:5). Wrong tokenizer -- it would decode "
                f"to plausible but wrong text."
            )

    def decode(self, token_ids) -> str:
        ids = [int(i) for i in np.asarray(token_ids).ravel()]
        return self._tokenizer.decode(ids, skip_special_tokens=True)


class StreamFluencyScorer:
    """Scores crafted streams under an independent reference model.

    Callable as `(poison, control) -> ratio`, which is the signature
    `run_attack_spike`'s `fluency_scorer` hook expects and the one
    `build_select_stream` was left injectable for.
    """

    def __init__(
        self,
        reference,
        decoder: Llama3Decoder,
        *,
        stride: int = DEFAULT_STRIDE,
    ):
        assert_independent_of_victim(reference.config.vocab_size)
        self.reference = reference
        self.decoder = decoder
        self.stride = stride
        self._tokenizer = reference.tokenizer()

    def score_text(self, text: str) -> PerplexityScore:
        encoded = self._tokenizer.encode(text)
        return sliding_window_nll(
            lambda ids: np.asarray(self.reference.logits(ids)),
            encoded.ids,
            max_length=self.reference.max_length,
            stride=self.stride,
        )

    def score_stream(self, stream: CraftedStream) -> PerplexityScore:
        return self.score_text(self.decoder.decode(stream.tokens))

    def __call__(self, poison: CraftedStream, control: CraftedStream) -> float:
        """Score both arms and record the numbers on the streams themselves.

        Mutates `poison` (and `control`) so `CraftedStream.is_scored` and
        `fluency_ratio` stop reporting `nan`. The returned ratio is what the
        harness collects per seed.
        """
        poison_score = self.score_stream(poison)
        control_score = self.score_stream(control)

        poison.perplexity = poison_score.perplexity
        poison.control_perplexity = control_score.perplexity
        control.perplexity = control_score.perplexity
        control.control_perplexity = control_score.perplexity

        return poison.fluency_ratio


def load_default_scorer(
    cache_root: Path | None = None, *, stride: int = DEFAULT_STRIDE
) -> StreamFluencyScorer:
    """Build the scorer from the fetched cache.

    Raises with instructions rather than downloading anything itself: a scoring
    call that silently pulls 548MB is a scoring call that behaves differently in
    CI than on a box.
    """
    from trustgate.eval.gpt2_reference import GPT2Reference

    root = reference_cache_root() if cache_root is None else Path(cache_root)
    gpt2_dir = root / "gpt2"
    llama3_tokenizer = root / "llama3" / "tokenizer.json"

    missing = [
        str(p)
        for p in (gpt2_dir / "config.json", gpt2_dir / "model.safetensors", llama3_tokenizer)
        if not p.exists()
    ]
    if missing:
        raise FileNotFoundError(
            "reference model not fetched; missing "
            + ", ".join(missing)
            + ". Run: python scripts/fetch_reference_model.py"
        )

    return StreamFluencyScorer(
        GPT2Reference(gpt2_dir), Llama3Decoder(llama3_tokenizer), stride=stride
    )
