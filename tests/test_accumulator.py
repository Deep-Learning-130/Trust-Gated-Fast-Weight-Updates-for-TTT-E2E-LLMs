import jax.numpy as jnp

from trustgate.drift.accumulator import (
    accumulate,
    init_drift,
    reset_window,
    would_breach,
)


def test_accumulates_and_reports_remaining():
    s = init_drift(budget=1.0)
    s = accumulate(s, jnp.float32(0.3))
    s = accumulate(s, jnp.float32(0.4))
    assert abs(float(s.cumulative) - 0.7) < 1e-6
    assert abs(float(s.remaining) - 0.3) < 1e-6
    assert not bool(s.breached)


def test_breach_fires_exactly_at_budget_crossing():
    s = init_drift(budget=1.0)
    s = accumulate(s, jnp.float32(0.9))
    assert not bool(s.breached)
    s = accumulate(s, jnp.float32(0.2))  # 1.1 > 1.0
    assert bool(s.breached)


def test_breach_latches_even_if_drift_would_decrease():
    # The adversarial case: a large update trips the budget, a later negative
    # update pulls cumulative back under it. The latch must NOT clear.
    s = init_drift(budget=1.0)
    s = accumulate(s, jnp.float32(1.5))
    assert bool(s.breached)
    s = accumulate(s, jnp.float32(-1.0))  # cumulative back to 0.5
    assert float(s.cumulative) < float(s.budget)
    assert bool(s.breached), "breach latch cleared -- bounded-drift guarantee is void"


def test_reset_window_clears_latch_and_cumulative():
    s = init_drift(budget=1.0)
    s = accumulate(s, jnp.float32(2.0))
    assert bool(s.breached)
    s = reset_window(s)
    assert float(s.cumulative) == 0.0
    assert not bool(s.breached)
    assert float(s.budget) == 1.0


def test_would_breach_is_predictive_not_mutating():
    s = init_drift(budget=1.0)
    s = accumulate(s, jnp.float32(0.8))
    assert bool(would_breach(s, jnp.float32(0.3)))
    assert not bool(would_breach(s, jnp.float32(0.1)))
    # querying must not change state
    assert abs(float(s.cumulative) - 0.8) < 1e-6
