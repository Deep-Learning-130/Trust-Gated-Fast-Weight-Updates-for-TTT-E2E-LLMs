import numpy as np
import pytest

from trustgate.eval.metrics import (
    attack_success_rate,
    clean_regression,
    cohens_d,
    corruption_metric,
)


def test_cohens_d_zero_when_identical():
    x = np.array([1.0, 1.1, 0.9, 1.0])
    assert abs(cohens_d(x, x.copy())) < 1e-9


def test_cohens_d_sign_follows_direction():
    hi = np.array([2.0, 2.1, 1.9, 2.0])
    lo = np.array([1.0, 1.1, 0.9, 1.0])
    assert cohens_d(hi, lo) > 0
    assert cohens_d(lo, hi) < 0


def test_cohens_d_requires_two_per_group():
    with pytest.raises(ValueError):
        cohens_d(np.array([1.0]), np.array([1.0, 2.0]))


def test_corruption_metric_detects_degradation():
    poisoned = np.array([1.5, 1.6, 1.4, 1.55])
    control = np.array([1.0, 1.05, 0.95, 1.0])
    res = corruption_metric(poisoned, control)
    assert res.relative_degradation > 0.4
    assert res.effect_size > 0.8
    assert res.exceeds(min_effect_size=0.8, min_relative=0.3)


def test_corruption_metric_null_result_does_not_exceed():
    rng = np.random.default_rng(0)
    poisoned = 1.0 + rng.normal(0, 0.05, 8)
    control = 1.0 + rng.normal(0, 0.05, 8)
    res = corruption_metric(poisoned, control)
    assert not res.exceeds(min_effect_size=0.8, min_relative=0.3)


def test_corruption_requires_both_conditions():
    # Large relative change but the caller demands a huge effect size too.
    poisoned = np.array([1.30, 1.32, 1.28, 1.31])
    control = np.array([1.00, 1.01, 0.99, 1.00])
    res = corruption_metric(poisoned, control)
    assert res.relative_degradation >= 0.3
    assert not res.exceeds(min_effect_size=50.0, min_relative=0.3)


def test_attack_success_rate_exact_match():
    triggered = np.array([[1, 2, 3], [1, 2, 3], [9, 9, 9]])
    target = np.array([[1, 2, 3], [1, 2, 3], [1, 2, 3]])
    assert attack_success_rate(triggered, target) == pytest.approx(2 / 3)


def test_clean_regression_positive_when_gate_hurts():
    assert clean_regression(np.array([1.1, 1.1]), np.array([1.0, 1.0])) == pytest.approx(0.1)
