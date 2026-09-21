"""Phase 1 evaluation: corruption metrics, attack harness, go/no-go report."""

from trustgate.eval.metrics import (
    CorruptionResult,
    accumulation_excess,
    attack_success_rate,
    clean_regression,
    cohens_d,
    corruption_metric,
    drift_floor_excess,
    onset_window,
)

__all__ = [
    "CorruptionResult",
    "accumulation_excess",
    "attack_success_rate",
    "clean_regression",
    "cohens_d",
    "corruption_metric",
    "drift_floor_excess",
    "onset_window",
]
