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

Two renderers, one verdict
--------------------------
`render_markdown` decides PROCEED/STOP. `render_sequence_markdown` renders the
secondary sequence-position arms and deliberately emits no verdict line, no
threshold table and no PASS/FAIL column. Keeping them as separate functions --
rather than one renderer with a flag -- is what makes "the non-gating arms
cannot move the verdict" a structural fact instead of a promise.
"""

from __future__ import annotations

import math
from pathlib import Path

from trustgate.eval.harness import SpikeResult
from trustgate.eval.metrics import accumulation_excess, onset_window
from trustgate.eval.sequence import (
    ARM_NAMES,
    CONTROL,
    POISON,
    POISON_NO_CARRY,
    SequenceResult,
)


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


# --------------------------------------------------------------------------
# Sequence-position report. Separate renderer, no verdict.
# --------------------------------------------------------------------------

NO_VERDICT_BANNER = (
    # No inner `**`: the banner is emitted inside a bold blockquote and nested
    # emphasis renders as literal asterisks.
    "SECONDARY, NON-GATING. This report renders no verdict and touches no "
    "pre-registered bar. The PROCEED/STOP decision is made only by "
    "`render_markdown` against the three thresholds frozen on 2026-07-28. See "
    "`PREREGISTERED.md`, Addendum 2026-09-20."
)


def render_sequence_markdown(result: SequenceResult, *, onset_delta: float) -> str:
    """Render the position curves.

    Emits no verdict line at all -- not a STOP, not a PROCEED, not a PASS/FAIL
    column. `render_markdown`'s verdict conjunction is untouched by
    construction rather than by convention, which is the whole reason this is a
    second function instead of another section in the first one.

    `onset_delta` is a reported sensitivity for `onset_window`, printed next to
    every value it produced. An onset without its delta is not interpretable.
    """
    lines = [
        f"> **{NO_VERDICT_BANNER}**",
        "",
        "# Sequence-position arms -- secondary",
        "",
        f"Windows: {result.n_windows} | Measured every: {result.eval_every} | "
        f"Seeds: {len(result.seeds)}",
        "",
        f"Floor (read nothing at all): {result.floor_loss:.6f}",
        "",
    ]

    lines += _arm_summary_section(result)
    lines += _derived_section(result, onset_delta)
    lines += _per_seed_curve_section(result)

    return "\n".join(lines) + "\n"


def _arm_summary_section(result: SequenceResult) -> list[str]:
    """End-of-stream mean per arm, plus the gap to the floor."""
    lines = ["## End of stream", "", "| Arm | Mean final loss | Above floor |", "|---|---|---|"]

    for name in ARM_NAMES:
        finals = result.final_losses(name)
        mean_final = sum(finals) / len(finals)
        lines.append(
            f"| `{name}` | {mean_final:.6f} | {mean_final - result.floor_loss:+.6f} |"
        )

    lines += [
        "",
        "`poison` and `control` here are the same quantity the frozen spike "
        "measures; the two should agree seed for seed.",
        "",
    ]
    return lines


def _derived_section(result: SequenceResult, onset_delta: float) -> list[str]:
    """Accumulation excess and onset, per seed. All non-gating."""
    lines = [
        "## Derived",
        "",
        f"Onset delta (reported sensitivity, not a bar): {onset_delta:g}",
        "",
        "| Seed | Final accumulation excess | Onset window |",
        "|---|---|---|",
    ]

    for seed in result.seeds:
        poison = result.arm(seed, POISON)
        control = result.arm(seed, CONTROL)
        poison_nc = result.arm(seed, POISON_NO_CARRY)

        excess = accumulation_excess(poison.per_window_loss, poison_nc.per_window_loss)
        onset = onset_window(
            poison.per_window_loss, control.per_window_loss, onset_delta
        )
        onset_cell = "never" if onset is None else str(poison.window_index(onset))

        lines.append(f"| {seed} | {float(excess[-1]):+.6f} | {onset_cell} |")

    lines += [
        "",
        "Accumulation excess is `poison` minus `poison_no_carry`. Near zero "
        "means the harm did not need carry to build up, and fast-weight "
        "accumulation is not the vector.",
        "",
        "Onset is the first window where `poison` exceeds `control` by the "
        "delta above -- the 'attack slowness' axis `PREREGISTERED.md` lists as "
        "secondary.",
        "",
    ]
    return lines


def _per_seed_curve_section(result: SequenceResult) -> list[str]:
    """Every measured value, for every arm, for every seed.

    Standing Rule 6 again: `render_markdown` had to be fixed once already for
    hiding per-seed numbers a reader would have wanted. Not repeating that.
    """
    lines = ["## Per-seed curves", ""]

    for seed in result.seeds:
        lines += [f"### Seed {seed}", "", "| Window | " + " | ".join(f"`{n}`" for n in ARM_NAMES) + " |"]
        lines.append("|---" * (len(ARM_NAMES) + 1) + "|")

        traces = [result.arm(seed, name) for name in ARM_NAMES]
        for i in range(len(traces[0].per_window_loss)):
            cells = " | ".join(f"{t.per_window_loss[i]:.6f}" for t in traces)
            lines.append(f"| {traces[0].window_index(i)} | {cells} |")

        lines.append("")

    return lines


def write_sequence_report(
    result: SequenceResult, path: Path, *, onset_delta: float
) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        render_sequence_markdown(result, onset_delta=onset_delta), encoding="utf-8"
    )
    return path
