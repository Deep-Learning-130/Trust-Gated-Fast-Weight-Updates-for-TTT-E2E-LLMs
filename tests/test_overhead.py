"""The gate's compute cost, against ADR-F1 section 5.1's ~10% budget.

CPU only. These cover the measurement's own discipline -- that compilation is
warmed out, that both arms are blocked, that the arithmetic is right -- not the
real overhead, which needs an accelerator and the vendor tree.
"""

import jax.numpy as jnp
import pytest

from trustgate.eval.overhead import BUDGET_FRACTION, OverheadResult, measure_overhead


def result(ungated, gated, repeats=3):
    return OverheadResult(
        ungated_seconds=ungated,
        gated_seconds=gated,
        repeats=repeats,
        ungated_samples=(ungated,) * repeats,
        gated_samples=(gated,) * repeats,
    )


def test_the_budget_matches_the_documented_target():
    """Dossier 5.1: "<= ~10% of the per-window update cost"."""
    assert BUDGET_FRACTION == 0.10


def test_overhead_is_a_fraction_of_the_ungated_cost():
    assert result(1.0, 1.1).overhead_fraction == pytest.approx(0.10)


def test_a_gate_inside_the_budget_passes():
    assert result(1.0, 1.05).within_budget()


def test_a_gate_over_the_budget_fails():
    assert not result(1.0, 1.5).within_budget()


def test_exactly_at_the_budget_passes():
    """The target is "<=", so the boundary is inside it.

    Exactly-representable values deliberately: (1.1 - 1.0) / 1.0 is
    0.1000000000000000888 in binary float and would fail a `<= 0.10` on
    representation alone. That is a fact about floats, not about the gate, and
    a measurement landing that close to the bar should be read as "at the bar"
    rather than adjudicated by the last mantissa bit.
    """
    assert result(2.0, 2.25).within_budget(0.125)


def test_a_custom_budget_is_honoured():
    """The 10% is a target that a measurement can justify moving -- unlike the
    three frozen bars, which cannot move without a dated revision."""
    assert result(1.0, 1.2).within_budget(0.25)


def test_a_zero_denominator_is_refused_rather_than_dividing():
    """A timer whose resolution exceeds the work would otherwise report an
    infinite or nonsensical overhead."""
    with pytest.raises(ValueError) as excinfo:
        result(0.0, 0.1).overhead_fraction
    assert "denominator" in str(excinfo.value)


def test_the_summary_names_the_verdict_and_the_budget():
    text = result(1.0, 1.5).summary()
    assert "OVER" in text
    assert "10%" in text


def test_the_summary_reports_a_pass_as_within():
    assert "within" in result(1.0, 1.01).summary()


# ------------------------------------------------------- the measurement ---


def test_measure_runs_both_arms_the_requested_number_of_times():
    calls = {"ungated": 0, "gated": 0}

    def ungated():
        calls["ungated"] += 1
        return jnp.float32(1.0)

    def gated():
        calls["gated"] += 1
        return jnp.float32(1.0)

    measured = measure_overhead(ungated, gated, repeats=4, warmup=2)
    # Warmup calls are made but not timed.
    assert calls["ungated"] == calls["gated"] == 6
    assert measured.repeats == 4
    assert len(measured.ungated_samples) == 4


def test_warmup_is_mandatory_and_says_why():
    """Timing the first call charges the gate for a one-time compilation no
    served model would ever pay."""
    with pytest.raises(ValueError) as excinfo:
        measure_overhead(lambda: None, lambda: None, warmup=0)
    assert "compil" in str(excinfo.value)


def test_repeats_must_be_positive():
    with pytest.raises(ValueError):
        measure_overhead(lambda: None, lambda: None, repeats=0)


def test_a_measurably_slower_gated_arm_reports_positive_overhead():
    """The one end-to-end property: real work, real clock, right sign."""
    big = jnp.ones((256, 256), dtype=jnp.float32)

    def ungated():
        return big @ big

    def gated():
        out = big @ big
        for _ in range(4):
            out = out @ big
        return out

    measured = measure_overhead(ungated, gated, repeats=5, warmup=2)
    assert measured.overhead_fraction > 0.0
