"""Metrics for the Phase 1 kill-gate decision.

These are pure functions over arrays -- no model, no GPU -- so they are fully
unit-tested on CPU and their behaviour is fixed *before* any attack result
exists. That ordering is the point: the pre-registration in
`experiments/001-attack-spike/PREREGISTERED.md` commits to thresholds expressed
in terms of these functions, so they must not be tunable after the fact.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class CorruptionResult:
    """Outcome of one poisoned-vs-control comparison."""

    poisoned_loss: float
    control_loss: float
    effect_size: float
    """Cohen's d, poisoned vs control, across seeds."""
    relative_degradation: float
    """(poisoned - control) / control."""
    n_seeds: int

    def exceeds(self, min_effect_size: float, min_relative: float) -> bool:
        """Whether this clears the pre-registered bar. Both conditions, not either.

        Requiring both stops a large-but-trivial effect (tight variance, tiny
        absolute change) or a large-but-noisy one from passing alone.
        """
        return self.effect_size >= min_effect_size and self.relative_degradation >= min_relative


def cohens_d(treatment: np.ndarray, control: np.ndarray) -> float:
    """Standardised mean difference with pooled SD.

    Returns 0.0 when both groups are constant and equal -- the degenerate case
    where the effect is genuinely absent rather than undefined.
    """
    treatment = np.asarray(treatment, dtype=np.float64).ravel()
    control = np.asarray(control, dtype=np.float64).ravel()

    if treatment.size < 2 or control.size < 2:
        raise ValueError(
            f"need >= 2 seeds per group for a pooled SD, got "
            f"{treatment.size} treatment / {control.size} control"
        )

    nt, nc = treatment.size, control.size
    pooled_var = (
        (nt - 1) * treatment.var(ddof=1) + (nc - 1) * control.var(ddof=1)
    ) / (nt + nc - 2)

    mean_diff = treatment.mean() - control.mean()

    if pooled_var <= 0.0:
        return 0.0 if np.isclose(mean_diff, 0.0) else float(np.inf * np.sign(mean_diff))

    return float(mean_diff / np.sqrt(pooled_var))


def corruption_metric(
    poisoned_losses: np.ndarray,
    control_losses: np.ndarray,
) -> CorruptionResult:
    """THE kill-gate metric.

    Benign-task loss after a poison stream vs. after a length-matched benign
    control stream. Higher poisoned loss means the fast weights were corrupted.

    Both arrays are per-seed benign-task losses measured *after* the respective
    stream. Measuring during the stream would conflate corruption with the
    model's ordinary adaptation to whatever it is currently reading.
    """
    poisoned = np.asarray(poisoned_losses, dtype=np.float64).ravel()
    control = np.asarray(control_losses, dtype=np.float64).ravel()

    control_mean = control.mean()
    if control_mean <= 0.0:
        raise ValueError(f"control loss must be positive, got mean {control_mean}")

    return CorruptionResult(
        poisoned_loss=float(poisoned.mean()),
        control_loss=float(control_mean),
        effect_size=cohens_d(poisoned, control),
        relative_degradation=float((poisoned.mean() - control_mean) / control_mean),
        n_seeds=int(min(poisoned.size, control.size)),
    )


def attack_success_rate(
    triggered_outputs: np.ndarray,
    target_outputs: np.ndarray,
) -> float:
    """TRIGGER objective: fraction of trigger presentations yielding the target.

    Exact-match on token sequences. Deliberately strict -- a partial match is
    not an implanted trigger, and a lenient criterion here would be the easiest
    place in the whole pipeline to fool ourselves.
    """
    triggered = np.asarray(triggered_outputs)
    target = np.asarray(target_outputs)

    if triggered.shape != target.shape:
        raise ValueError(f"shape mismatch: {triggered.shape} vs {target.shape}")
    if triggered.size == 0:
        raise ValueError("no trigger presentations to score")

    matches = np.all(triggered == target, axis=tuple(range(1, triggered.ndim)))
    return float(matches.mean())


def clean_regression(gated_losses: np.ndarray, ungated_losses: np.ndarray) -> float:
    """Phase 2 metric: clean-accuracy cost of running the gate.

    Positive means the gate hurt benign performance -- the false-reject cost of
    dossier 5.6. Kept here rather than in Phase 2 code so the definition is
    fixed before we have an incentive to prefer a flattering one.
    """
    gated = np.asarray(gated_losses, dtype=np.float64).ravel()
    ungated = np.asarray(ungated_losses, dtype=np.float64).ravel()

    ungated_mean = ungated.mean()
    if ungated_mean <= 0.0:
        raise ValueError(f"ungated loss must be positive, got mean {ungated_mean}")

    return float((gated.mean() - ungated_mean) / ungated_mean)
