"""Phase 1 evaluation harness: run poison and control streams under matched conditions.

Status: orchestration is specified; the model-touching steps raise until the
checkpoints are fetched (`scripts/fetch_checkpoints.sh`) and the baseline
reproduction in `experiments/000-repro-baseline/` has passed.

Matching discipline
-------------------
Poison and control runs must differ in *exactly one* thing: the content of the
stream. Same seed, same length in tokens, same chunking, same dtype, same
benign eval set, same model revision. Every one of these is a plausible
confound, and fast weights drift on any input at all -- so an unmatched control
would make a null result look like a positive one. `RunCondition` exists to make
the matching explicit and assertable rather than a matter of care.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from trustgate.attack.objectives import AttackSpec
from trustgate.attack.stream import CraftedStream, StreamStrategy
from trustgate.eval.carry import SATURATED_INNER_LR_MULTIPLIER
from trustgate.eval.metrics import CorruptionResult, corruption_metric


def eval_tokens_digest(tokens) -> str:
    """sha256 over the benign eval tokens.

    `benign_eval_split` is a *name*, not the data: two runs can evaluate
    different windows of the same split under the same string and
    `assert_matches` would see nothing wrong. The vendor's own eval loader is
    `seed=0, shuffle=False, repeat=False` (`loop.py:63-74`), so its windows are
    deterministic -- but a hand-rolled harness loses that guarantee, and the
    poison/control comparison is only meaningful if both arms were measured on
    identical held-out tokens.
    """
    return hashlib.sha256(np.ascontiguousarray(tokens, dtype=np.int32).tobytes()).hexdigest()


@dataclass(frozen=True)
class RunCondition:
    """Everything that must match between poison and control runs.

    Every field here is a plausible confound, and fast weights drift on any
    input at all -- so an unmatched control would make a null result look like a
    positive one. `PREREGISTERED.md` names control mismatch as an invalidating
    condition, which is why this is assertable rather than a matter of care.
    """

    seed: int
    stream_tokens: int
    mini_batch_size: int
    seq_length: int
    checkpoint: str
    benign_eval_split: str

    span_tokens: int
    """ADR-007. A STOP verdict is scoped to the granularity that produced it, so
    the granularity has to travel with the result."""

    n_chunks: int
    """Inner SGD steps the stream drives. `inner_opt_state` is re-initialised per
    call (`transformer.py:686`), so the first chunks run with an unwarmed
    optimiser -- identical across arms only if the chunk counts match."""

    valid_tokens: int
    """Loss-bearing positions, not raw length. The inner objective's denominator
    is that chunk's valid count (`loss.py:15,26`), so a difference here rescales
    every surviving token's gradient."""

    dtype: str
    """Pre-registered as guarded and previously absent. `PREREGISTERED.md`
    lists "control not length/seed/**dtype**-matched" as invalidating and says
    `assert_matches` guards it; it did not. `transformer.py:685` casts the whole
    model to `state_dtype` per call and the outer params come from that cast
    copy, so dtype is load-bearing. Adding this field satisfies a pre-registered
    claim rather than moving a bar."""

    stream_corpus_split: str
    """Which split the stream was drawn from. Must differ from
    `benign_eval_split` -- see `assert_stream_not_from_eval_split`."""

    eval_tokens_sha256: str
    """Identity of the benign eval tokens themselves. See `eval_tokens_digest`."""

    inner_lr_multiplier: float
    """`get_ilr_multiplier` (`transformer.py:564-573`) ramps the inner LR off
    `state[step_index]`; `train.py:224` pins it high in eval mode to saturate
    the ramp. A harness leaving `step_index` at 0 runs a near-frozen inner loop
    and returns a null for a reason unrelated to the attack."""

    def __post_init__(self) -> None:
        """Reject a condition that cannot be a valid measurement.

        Validated at construction rather than at comparison time: two runs that
        both left a field unset would *match* each other, which is precisely the
        failure this class exists to prevent.
        """
        if self.mini_batch_size <= 0 or self.span_tokens <= 0:
            raise ValueError(
                f"mini_batch_size and span_tokens must be positive, got "
                f"{self.mini_batch_size} and {self.span_tokens}"
            )
        if self.mini_batch_size % self.span_tokens != 0:
            raise ValueError(
                f"span_tokens {self.span_tokens} must divide mini_batch_size "
                f"{self.mini_batch_size} (ADR-007)"
            )
        expected_chunks = self.stream_tokens // self.mini_batch_size
        if self.n_chunks != expected_chunks:
            raise ValueError(
                f"n_chunks {self.n_chunks} does not match "
                f"stream_tokens // mini_batch_size = {expected_chunks}"
            )
        if self.valid_tokens > self.stream_tokens:
            raise ValueError(
                f"valid_tokens {self.valid_tokens} exceeds stream_tokens "
                f"{self.stream_tokens}"
            )
        if len(self.eval_tokens_sha256) != 64:
            raise ValueError(
                "eval_tokens_sha256 must be a full sha256 hex digest; an unset "
                "or truncated value would match between two runs and defeat the "
                "eval-identity check. Use `eval_tokens_digest`."
            )
        if abs(self.inner_lr_multiplier - SATURATED_INNER_LR_MULTIPLIER) > 1e-6:
            raise ValueError(
                f"inner_lr_multiplier is {self.inner_lr_multiplier}, expected "
                f"{SATURATED_INNER_LR_MULTIPLIER}. An unsaturated inner LR means "
                f"a near-frozen inner loop (transformer.py:564-573) and a null "
                f"result for the wrong reason."
            )

    @classmethod
    def from_stream(
        cls,
        stream: CraftedStream,
        *,
        seed: int,
        seq_length: int,
        checkpoint: str,
        benign_eval_split: str,
        eval_tokens_sha256: str,
        inner_lr_multiplier: float = SATURATED_INNER_LR_MULTIPLIER,
    ) -> RunCondition:
        """Derive the stream-side fields from the stream itself.

        Hand-transcribing seven numbers into two conditions is exactly how an
        unmatched control gets built by accident.
        """
        return cls(
            seed=seed,
            stream_tokens=stream.length_tokens,
            mini_batch_size=stream.mini_batch_size,
            seq_length=seq_length,
            checkpoint=checkpoint,
            benign_eval_split=benign_eval_split,
            span_tokens=stream.span_tokens,
            n_chunks=stream.n_chunks,
            valid_tokens=stream.valid_tokens,
            dtype=str(stream.tokens.dtype),
            stream_corpus_split=stream.corpus_name,
            eval_tokens_sha256=eval_tokens_sha256,
            inner_lr_multiplier=inner_lr_multiplier,
        )

    def assert_matches(self, other: RunCondition) -> None:
        """Fail loudly on any mismatch except the stream itself."""
        mismatches = [
            f"{f.name}: {getattr(self, f.name)!r} != {getattr(other, f.name)!r}"
            for f in self.__dataclass_fields__.values()
            if getattr(self, f.name) != getattr(other, f.name)
        ]
        if mismatches:
            raise ValueError(
                "poison and control conditions must be identical; mismatched: "
                + "; ".join(mismatches)
            )


@dataclass
class SpikeResult:
    """Outcome of a full attack spike across seeds."""

    corruption: CorruptionResult
    fluency_ratio: float
    strategy: StreamStrategy
    attack_success_rate: float | None = None
    per_seed_poisoned: list[float] = field(default_factory=list)
    per_seed_control: list[float] = field(default_factory=list)


def run_stream(model, stream_tokens, condition: RunCondition):
    """Feed one stream through TTT-E2E, returning the adapted fast weights.

    Runs with `train_mode="meta"` -- the only mode with an inner loop. In
    `"pretrain"` mode there are no fast weights to poison and the attack is
    vacuously null.
    """
    raise NotImplementedError(
        "Blocked on checkpoints + experiments/000-repro-baseline passing."
    )


def eval_benign(model, condition: RunCondition) -> float:
    """Benign-task loss for a model with adapted fast weights."""
    raise NotImplementedError("Blocked on checkpoints.")


def run_attack_spike(
    spec: AttackSpec,
    condition: RunCondition,
    seeds: list[int],
    output_dir: Path,
) -> SpikeResult:
    """Execute the full pre-registered comparison.

    For each seed: build a poison stream and a length-matched control, run both,
    evaluate benign loss after each. Then compute the corruption metric across
    seeds and compare against the pre-registered thresholds.

    Does **not** decide go/no-go. It reports; the threshold lives in
    PREREGISTERED.md and the comparison is made explicitly by the caller, so the
    bar cannot be quietly relaxed inside the code that produces the number.
    """
    raise NotImplementedError("Blocked on checkpoints.")


def summarize(poisoned: list[float], control: list[float]) -> CorruptionResult:
    """Aggregate per-seed losses. Pure -- unit-tested without a model."""
    return corruption_metric(np.asarray(poisoned), np.asarray(control))
