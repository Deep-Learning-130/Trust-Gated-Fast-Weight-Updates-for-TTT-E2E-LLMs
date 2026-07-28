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

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from trustgate.attack.objectives import AttackSpec
from trustgate.attack.stream import StreamStrategy
from trustgate.eval.metrics import CorruptionResult, corruption_metric


@dataclass(frozen=True)
class RunCondition:
    """Everything that must match between poison and control runs."""

    seed: int
    stream_tokens: int
    mini_batch_size: int
    seq_length: int
    checkpoint: str
    benign_eval_split: str

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
