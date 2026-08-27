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
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Callable

import numpy as np

from trustgate.attack.objectives import AttackSpec
from trustgate.attack.stream import (
    CraftedStream,
    StreamStrategy,
    assert_stream_not_from_eval_split,
    assert_streams_matched,
)
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
    strategy: StreamStrategy

    per_seed_fluency: list[float] = field(default_factory=list)
    """One ratio per seed, not a single aggregate.

    The pre-registration states the fluency bar as one number, but there are
    five streams and T3.6 requires a ratio for each. The two corruption rows are
    explicitly aggregate ("across seeds") because Cohen's *d* cannot be
    per-seed; the fluency rationale is a property of an *artifact*, and there
    are five artifacts. See `gating_fluency_ratio`."""

    attack_success_rate: float | None = None
    per_seed_poisoned: list[float] = field(default_factory=list)
    per_seed_control: list[float] = field(default_factory=list)

    @property
    def gating_fluency_ratio(self) -> float:
        """The **max** across seeds. This is what the verdict turns on.

        Under a mean, one seed at 3.0 is laundered by four at 1.1 while that
        seed's loss still pools into the gated Cohen's *d* -- so the gating
        fluency statistic has to cover every seed contributing to the gated
        corruption statistic. Max is the strictest defensible reading of an
        ambiguous pre-registration, and Rule 5 exists to stop bars being
        *loosened*, so choosing it cannot be self-serving.

        `experiments/001-attack-spike/README.md` pre-commits the escalation
        path: if the verdict turns on this choice, that is a pre-registration
        ambiguity requiring a dated revision note *before* the verdict is
        recorded.
        """
        if not self.per_seed_fluency:
            return float("nan")
        return float(np.max(self.per_seed_fluency))

    @property
    def mean_fluency_ratio(self) -> float:
        """Reported alongside the max, never gating."""
        if not self.per_seed_fluency:
            return float("nan")
        return float(np.mean(self.per_seed_fluency))

    @property
    def worst_fluency_seed_index(self) -> int | None:
        if not self.per_seed_fluency:
            return None
        return int(np.argmax(self.per_seed_fluency))

    @property
    def is_fluency_scored(self) -> bool:
        return bool(self.per_seed_fluency) and not any(
            np.isnan(r) for r in self.per_seed_fluency
        )


def run_stream(model, stream_tokens, condition: RunCondition):
    """Feed one stream through TTT-E2E, returning the adapted fast weights.

    Runs with `train_mode="meta"` -- the only mode with an inner loop. In
    `"pretrain"` mode there are no fast weights to poison and the attack is
    vacuously null.

    **Not blocked on checkpoints.** It is blocked on binding a step function to
    the real `MetaModel.inner_loop_step`, because the vendor discards its scan
    carry and never returns adapted fast weights at all (ADR-006). The threading
    itself is implemented and tested in `trustgate.eval.carry`; use
    `carry.run_sequences` and pass the result to `eval_benign`.
    """
    raise NotImplementedError(
        "Bind step_fn to MetaModel.inner_loop_step and use "
        "trustgate.eval.carry.run_sequences -- see ADR-006. The vendor cannot "
        "return adapted fast weights on its own."
    )


def eval_benign(model, condition: RunCondition) -> float:
    """Benign-task loss for a model with adapted fast weights."""
    raise NotImplementedError("Blocked on checkpoints.")


def run_attack_spike(
    spec: AttackSpec,
    condition: RunCondition,
    seeds: list[int],
    output_dir: Path,
    *,
    build_arms: Callable[[int], tuple[CraftedStream, CraftedStream]],
    adapt_and_eval: Callable[[CraftedStream, RunCondition], float],
    fluency_scorer: Callable[[CraftedStream, CraftedStream], float] | None = None,
) -> SpikeResult:
    """Execute the full pre-registered comparison.

    For each seed: build a poison stream and a length-matched control, run both,
    evaluate benign loss after each. Then compute the corruption metric across
    seeds and compare against the pre-registered thresholds.

    Does **not** decide go/no-go, and does **not** write a report. It reports;
    the threshold lives in PREREGISTERED.md and the comparison is made
    explicitly by the caller, so the bar cannot be quietly relaxed inside the
    code that produces the number. Report generation belongs to the CLI layer.

    Args:
        build_arms: seed -> (poison, control). Injected so the orchestration is
            testable without a corpus, and so `craft_stream` can supply an
            adversarially-ordered poison arm later.
        adapt_and_eval: stream, condition -> benign loss after adapting on that
            stream. The one genuinely model-dependent step. Build it from
            `trustgate.eval.carry.run_sequences` plus the vendor inner step.
        fluency_scorer: poison, control -> perplexity ratio under an
            *independent* reference model. `None` until T3.2 lands, which leaves
            every per-seed ratio `nan` and fails the realism bar safe.
    """
    if not seeds:
        raise ValueError("no seeds: the pre-registration requires 5 per condition")

    poisoned_losses: list[float] = []
    control_losses: list[float] = []
    fluency: list[float] = []
    strategy: StreamStrategy | None = None

    for seed in seeds:
        poison, control = build_arms(seed)
        strategy = poison.strategy

        assert_streams_matched(poison, control)
        assert_stream_not_from_eval_split(poison, condition.benign_eval_split)
        assert_stream_not_from_eval_split(control, condition.benign_eval_split)

        seed_condition = replace(condition, seed=seed)
        poison_condition = RunCondition.from_stream(
            poison,
            seed=seed,
            seq_length=seed_condition.seq_length,
            checkpoint=seed_condition.checkpoint,
            benign_eval_split=seed_condition.benign_eval_split,
            eval_tokens_sha256=seed_condition.eval_tokens_sha256,
            inner_lr_multiplier=seed_condition.inner_lr_multiplier,
        )
        control_condition = replace(poison_condition, stream_corpus_split=control.corpus_name)
        poison_condition.assert_matches(control_condition)

        poisoned_losses.append(float(adapt_and_eval(poison, poison_condition)))
        control_losses.append(float(adapt_and_eval(control, control_condition)))
        fluency.append(
            float("nan") if fluency_scorer is None else float(fluency_scorer(poison, control))
        )

    return SpikeResult(
        corruption=summarize(poisoned_losses, control_losses),
        # Read off the streams that actually ran rather than off the spec.
        # `PREREGISTERED.md` fixes SELECT as the headline and says PARAPHRASE and
        # SOFT "do not substitute for" it, so a report must name what ran.
        strategy=strategy or StreamStrategy.SELECT,
        per_seed_fluency=fluency,
        per_seed_poisoned=poisoned_losses,
        per_seed_control=control_losses,
    )


def summarize(poisoned: list[float], control: list[float]) -> CorruptionResult:
    """Aggregate per-seed losses. Pure -- unit-tested without a model."""
    return corruption_metric(np.asarray(poisoned), np.asarray(control))


if __name__ == "__main__":
    # `python -m trustgate.eval.harness` is the invocation
    # `experiments/001-attack-spike/README.md` documents. The implementation
    # lives in `cli.py` so that threshold loading and report writing stay out of
    # this module -- `run_attack_spike` must not transitively decide the verdict.
    from trustgate.eval.cli import main

    raise SystemExit(main())
