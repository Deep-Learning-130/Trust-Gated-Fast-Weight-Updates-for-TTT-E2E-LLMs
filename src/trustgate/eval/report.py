"""Render a Phase 1 spike result against its pre-registration.

The report always states the pre-registered threshold next to the observed
value, and always names the verdict PROCEED or STOP. A result that has to be
interpreted is a result that can be rationalised, and dossier section 7 makes
this a hard kill-gate.
"""

from __future__ import annotations

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

    corruption_pass = corruption.exceeds(min_effect_size, min_relative_degradation)
    fluency_pass = result.fluency_ratio <= max_fluency_ratio
    verdict = "PROCEED" if (corruption_pass and fluency_pass) else "STOP"

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
        f"| {mark(corruption.effect_size >= min_effect_size)} |",
        f"| Relative degradation | {corruption.relative_degradation:+.2%} "
        f"| >= {min_relative_degradation:.2%} "
        f"| {mark(corruption.relative_degradation >= min_relative_degradation)} |",
        f"| Fluency ratio vs control | {result.fluency_ratio:.3f} | <= {max_fluency_ratio} "
        f"| {mark(fluency_pass)} |",
        "",
        f"Benign loss after poison stream: {corruption.poisoned_loss:.4f}",
        f"Benign loss after control stream: {corruption.control_loss:.4f}",
    ]

    if result.attack_success_rate is not None:
        lines += ["", f"Trigger attack success rate: {result.attack_success_rate:.2%}"]

    if verdict == "STOP":
        lines += [
            "",
            "---",
            "",
            "Per dossier section 7: no demonstrated attack means no defense to build.",
            "Do not proceed to the gate prototype. Either revise the attack with a",
            "documented reason, or kill F1.",
        ]

    if not fluency_pass:
        lines += [
            "",
            "**Note:** the corruption result may be real but the stream is not",
            "benign-looking, so it does not establish the section 1 threat model.",
            "A stream a perplexity filter would catch does not motivate this defense.",
        ]

    return "\n".join(lines) + "\n"


def write_report(result: SpikeResult, path: Path, **thresholds) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(render_markdown(result, **thresholds), encoding="utf-8")
    return path
