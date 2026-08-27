"""Verdict rendering.

Three failure modes are asserted here because each of them would put a wrong
word next to "Verdict:" in the only artefact anyone reads: a laundered bad
fluency seed, a suppressed unflattering per-seed number (Rule 6), and a
degenerate infinite effect size passing the bar.
"""

import numpy as np
import pytest

from trustgate.attack.stream import StreamStrategy
from trustgate.eval.harness import SpikeResult, summarize
from trustgate.eval.prereg import FROZEN
from trustgate.eval.report import render_markdown, write_report

BARS = dict(
    min_effect_size=FROZEN.min_effect_size,
    min_relative_degradation=FROZEN.min_relative_degradation,
    max_fluency_ratio=FROZEN.max_fluency_ratio,
)


def make_result(poisoned=None, control=None, fluency=None, **kwargs):
    poisoned = [1.20, 1.22, 1.18, 1.21, 1.19] if poisoned is None else poisoned
    control = [1.00, 1.01, 0.99, 1.00, 1.00] if control is None else control
    return SpikeResult(
        corruption=summarize(poisoned, control),
        strategy=kwargs.pop("strategy", StreamStrategy.SELECT),
        per_seed_fluency=[] if fluency is None else fluency,
        per_seed_poisoned=poisoned,
        per_seed_control=control,
        **kwargs,
    )


def test_a_clean_pass_renders_proceed():
    result = make_result(fluency=[1.05, 1.10, 1.02, 1.08, 1.06])
    report = render_markdown(result, **BARS)
    assert "**Verdict: PROCEED**" in report


def test_max_fluency_gates_not_mean():
    # Four good seeds and one at 3.0. The mean is 1.43 and would pass; the max
    # is 3.0 and must not. That bad seed's loss still pools into the gated
    # Cohen's d, so the fluency statistic has to cover it.
    result = make_result(fluency=[1.05, 1.10, 1.02, 1.08, 3.00])
    report = render_markdown(result, **BARS)

    assert result.mean_fluency_ratio <= FROZEN.max_fluency_ratio
    assert "**Verdict: STOP**" in report
    assert "(worst)" in report


def test_a_verdict_that_turns_on_aggregation_is_flagged():
    result = make_result(fluency=[1.05, 1.10, 1.02, 1.08, 3.00])
    report = render_markdown(result, **BARS)
    assert "The verdict turns on this aggregation choice" in report
    assert "dated revision note" in report


def test_both_aggregates_are_shown_side_by_side():
    result = make_result(fluency=[1.1, 1.2, 1.3, 1.4, 1.5])
    report = render_markdown(result, **BARS)
    assert "max across seeds" in report
    assert "mean across seeds" in report
    assert "**(gating)**" in report


def test_every_per_seed_value_appears():
    # Rule 6. SpikeResult has always carried these and the renderer never
    # printed them, so a seed where poison helped was invisible.
    poisoned = [1.20, 1.22, 1.18, 1.21, 1.19]
    control = [1.00, 1.01, 0.99, 1.00, 1.00]
    report = render_markdown(make_result(poisoned, control), **BARS)

    assert "## Per-seed values" in report
    for value in poisoned + control:
        assert f"{value:.4f}" in report


def test_a_seed_where_poison_helped_is_called_out():
    poisoned = [1.20, 1.22, 0.90, 1.21, 1.19]  # seed 2 got better
    control = [1.00, 1.01, 0.99, 1.00, 1.00]
    report = render_markdown(make_result(poisoned, control), **BARS)
    assert "helped* the victim: [2]" in report
    assert "Rule 6" in report


def test_infinite_effect_size_does_not_pass():
    # cohens_d returns inf when pooled variance is zero and means differ, and
    # exceeds() accepts inf >= 0.8. A stubbed harness, or fast weights that
    # never moved, would otherwise render PROCEED.
    result = make_result(
        poisoned=[1.2] * 5, control=[1.0] * 5, fluency=[1.0] * 5
    )
    assert np.isinf(result.corruption.effect_size)
    assert result.corruption.exceeds(FROZEN.min_effect_size, FROZEN.min_relative_degradation)

    report = render_markdown(result, **BARS)
    assert "**Verdict: STOP**" in report
    assert "Effect size is not finite" in report


def test_unscored_fluency_fails_safe_and_says_so():
    report = render_markdown(make_result(fluency=[]), **BARS)
    assert "**Verdict: STOP**" in report
    assert "not yet scored" in report
    assert "Fluency is unscored" in report
    assert "it is incomplete" in report


def test_nan_fluency_fails_safe():
    report = render_markdown(make_result(fluency=[float("nan")] * 5), **BARS)
    assert "**Verdict: STOP**" in report
    assert "not yet scored" in report


def test_a_null_corruption_result_renders_stop():
    rng = np.random.default_rng(0)
    control = list(1.0 + rng.normal(0, 0.05, 5))
    poisoned = list(1.0 + rng.normal(0, 0.05, 5))
    report = render_markdown(make_result(poisoned, control, fluency=[1.0] * 5), **BARS)
    assert "**Verdict: STOP**" in report
    assert "no demonstrated attack means no defense to build" in report


def test_observed_and_preregistered_appear_together():
    report = render_markdown(make_result(fluency=[1.0] * 5), **BARS)
    assert "| Criterion | Observed | Pre-registered | |" in report
    assert f">= {FROZEN.min_effect_size}" in report
    assert f"<= {FROZEN.max_fluency_ratio}" in report


def test_strategy_and_seed_count_are_named():
    report = render_markdown(make_result(fluency=[1.0] * 5), **BARS)
    assert "`select`" in report
    assert "Seeds: 5" in report


def test_attack_success_rate_is_marked_non_gating():
    result = make_result(fluency=[1.0] * 5, attack_success_rate=0.42)
    report = render_markdown(result, **BARS)
    assert "42.00%" in report
    assert "non-gating" in report


def test_write_report_creates_the_file_and_parent(tmp_path):
    path = write_report(
        make_result(fluency=[1.0] * 5), tmp_path / "nested" / "report.md", **BARS
    )
    assert path.is_file()
    assert "Verdict" in path.read_text(encoding="utf-8")
