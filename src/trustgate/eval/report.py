"""Render a Phase 1 spike result against its pre-registration.

The report always states the pre-registered threshold next to the observed
value, and always names the verdict PROCEED or STOP. A result that has to be
interpreted is a result that can be rationalised, and dossier section 7 makes
this a hard kill-gate.

Every per-seed number is printed
--------------------------------
Standing Rule 6: unflattering numbers get reported, not dropped. `SpikeResult`
has carried `per_seed_poisoned` and `per_seed_control` from the start and this
renderer never printed them, so a seed where the poison stream *helped* the
victim was invisible in the only artefact anyone reads. Fixed here.

Degenerate statistics cannot pass
---------------------------------
`cohens_d` returns `±inf` when pooled variance is zero and the means differ
(`metrics.py:60-61`), and `CorruptionResult.exceeds` accepts `inf >= 0.8`. So a
constant-valued result -- a stubbed harness, a model that never adapted, an
`adapt_and_eval` returning the same number every seed -- would render **PROCEED**
with an infinite effect size. A non-finite effect size is now a hard FAIL.
"""

from __future__ import annotations

import math
from pathlib import Path

from trustgate.eval.harness import SpikeResult


def render_markdown(
    result: SpikeResult,
    *,
    min_effect_size: float,
    min_relative_degradation: float,
    max_fluency_ratio: float,
) -> str:
    """Produce the go/no-go report."""
    corruption = result.corruption

    effect_finite = math.isfinite(corruption.effect_size)
    effect_pass = effect_finite and corruption.effect_size >= min_effect_size
    degradation_pass = corruption.relative_degradation >= min_relative_degradation

    gating_fluency = result.gating_fluency_ratio
    fluency_pass = result.is_fluency_scored and gating_fluency <= max_fluency_ratio

    verdict = "PROCEED" if (effect_pass and degradation_pass and fluency_pass) else "STOP"

    def mark(ok: bool) -> str:
        return "PASS" if ok else "FAIL"

    lines = [
        "# Phase 1 Attack Spike -- Result",
        "",
        f"**Verdict: {verdict}**",
        "",
        f"Strategy: `{result.strategy.value}` | Seeds: {corruption.n_seeds}",
        "",
        "| Criterion | Observed | Pre-registered | |",
        "|---|---|---|---|",
        f"| Effect size (Cohen's d) | {corruption.effect_size:.3f} | >= {min_effect_size} "
        f"| {mark(effect_pass)} |",
        f"| Relative degradation | {corruption.relative_degradation:+.2%} "
        f"| >= {min_relative_degradation:.2%} "
        f"| {mark(degradation_pass)} |",
        f"| Fluency ratio vs control (max across seeds) | "
        f"{'not yet scored' if not result.is_fluency_scored else f'{gating_fluency:.3f}'} "
        f"| <= {max_fluency_ratio} | {mark(fluency_pass)} |",
        "",
        f"Benign loss after poison stream: {corruption.poisoned_loss:.4f}",
        f"Benign loss after control stream: {corruption.control_loss:.4f}",
    ]

    lines += _per_seed_section(result)
    lines += _fluency_aggregate_section(result, max_fluency_ratio)

    if not effect_finite:
        lines += [
            "",
            "**Effect size is not finite.** Pooled variance across seeds was zero",
            "with differing means, which `cohens_d` reports as infinity and",
            "`exceeds` would otherwise accept. That is the signature of a result",
            "that did not really vary -- a stubbed harness, or fast weights that",
            "never moved. Treated as FAIL rather than as an unbounded effect.",
        ]

    if result.attack_success_rate is not None:
        lines += [
            "",
            f"Trigger attack success rate: {result.attack_success_rate:.2%}",
            "",
            "Secondary and non-gating: DEGRADE alone suffices to PROCEED",
            "(`PREREGISTERED.md` Secondary).",
        ]

    if verdict == "STOP":
        lines += [
            "",
            "---",
            "",
            "Per dossier section 7: no demonstrated attack means no defense to build.",
            "Do not proceed to the gate prototype. Either revise the attack with a",
            "documented reason, or kill F1.",
        ]

    if not result.is_fluency_scored:
        lines += [
            "",
            "**Fluency is unscored.** No independent reference model has run yet",
            "(T3.2), so the realism bar cannot be evaluated and fails safe. This",
            "verdict is therefore not a kill-gate result -- it is incomplete.",
            "Scoring fluency with the victim model would be circular and",
            "`PREREGISTERED.md` names it an invalidating condition.",
        ]
    elif not fluency_pass:
        lines += [
            "",
            "**Note:** the corruption result may be real but the stream is not",
            "benign-looking, so it does not establish the section 1 threat model.",
            "A stream a perplexity filter would catch does not motivate this defense.",
        ]

    return "\n".join(lines) + "\n"


def _per_seed_section(result: SpikeResult) -> list[str]:
    """Rule 6: every seed is visible, including the ones that went the wrong way."""
    if not (result.per_seed_poisoned and result.per_seed_control):
        return []

    lines = [
        "",
        "## Per-seed values",
        "",
        "| Seed | Poisoned | Control | Delta | Fluency ratio |",
        "|---|---|---|---|---|",
    ]

    worst = result.worst_fluency_seed_index
    for index, (poisoned, control) in enumerate(
        zip(result.per_seed_poisoned, result.per_seed_control)
    ):
        ratio = (
            result.per_seed_fluency[index]
            if index < len(result.per_seed_fluency)
            else float("nan")
        )
        ratio_cell = "--" if math.isnan(ratio) else f"{ratio:.3f}"
        if index == worst and not math.isnan(ratio):
            ratio_cell += " (worst)"
        lines.append(
            f"| {index} | {poisoned:.4f} | {control:.4f} "
            f"| {poisoned - control:+.4f} | {ratio_cell} |"
        )

    helped = [
        index
        for index, (p, c) in enumerate(
            zip(result.per_seed_poisoned, result.per_seed_control)
        )
        if p < c
    ]
    if helped:
        lines += [
            "",
            f"Seeds where the poison stream *helped* the victim: {helped}. "
            "Reported, not dropped (Rule 6).",
        ]

    return lines


def _fluency_aggregate_section(result: SpikeResult, max_fluency_ratio: float) -> list[str]:
    """Show max and mean side by side, and say which one gates."""
    if not result.per_seed_fluency:
        return []

    lines = [
        "",
        "## Fluency aggregation",
        "",
        f"- max across seeds: "
        f"{'--' if math.isnan(result.gating_fluency_ratio) else f'{result.gating_fluency_ratio:.3f}'} "
        f"**(gating)**",
        f"- mean across seeds: "
        f"{'--' if math.isnan(result.mean_fluency_ratio) else f'{result.mean_fluency_ratio:.3f}'} "
        f"(reported)",
        "",
        f"The pre-registration states the bar as one number (<= {max_fluency_ratio}) but",
        "there are five streams. The **max** gates: under a mean, one bad seed is",
        "laundered by four good ones while its loss still pools into the gated",
        "Cohen's d. Max is the strictest defensible reading, and Rule 5 exists to",
        "stop bars being loosened.",
    ]

    if result.is_fluency_scored:
        mean_passes = result.mean_fluency_ratio <= max_fluency_ratio
        max_passes = result.gating_fluency_ratio <= max_fluency_ratio
        if mean_passes and not max_passes:
            lines += [
                "",
                "**The verdict turns on this aggregation choice.** Mean passes, max",
                "fails. Per `experiments/001-attack-spike/README.md` this is a",
                "pre-registration ambiguity: it requires a dated revision note",
                "signed by the Lead *before* the verdict is recorded, not a free",
                "choice at report time.",
            ]

    return lines


def write_report(result: SpikeResult, path: Path, **thresholds) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(render_markdown(result, **thresholds), encoding="utf-8")
    return path
