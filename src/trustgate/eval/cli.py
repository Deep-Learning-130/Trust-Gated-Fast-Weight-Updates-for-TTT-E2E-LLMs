"""CLI behind `python -m trustgate.eval.harness` -- the Phase 1 spike entry point.

`experiments/001-attack-spike/README.md` has documented these flags since the
scaffold; no such entry point existed. The flags are exactly as documented:
`--objective`, `--strategy`, `--checkpoint`, `--seeds`, `--out`.

Where the verdict is decided
----------------------------
Here, not in `run_attack_spike`. That function's own contract says it "does not
decide go/no-go ... the comparison is made explicitly by the caller, so the bar
cannot be quietly relaxed inside the code that produces the number" -- and
`report.render_markdown` computes the verdict, so a harness that generated the
report would transitively decide. Threshold loading and `write_report` therefore
live in this layer.

Thresholds come from `trustgate.eval.prereg.FROZEN`, which CI pins to
`PREREGISTERED.md` itself. They are never spelled out in this file.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from trustgate.attack.objectives import Objective
from trustgate.attack.stream import StreamStrategy
from trustgate.eval.prereg import FROZEN
from trustgate.eval.report import write_report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m trustgate.eval.harness",
        description="Run the pre-registered Phase 1 attack spike and emit a verdict report.",
    )
    parser.add_argument(
        "--objective",
        required=True,
        choices=[o.value for o in Objective],
        help="DEGRADE is the gating objective; TRIGGER is secondary.",
    )
    parser.add_argument(
        "--strategy",
        required=True,
        choices=[s.value for s in StreamStrategy],
        help="SELECT is the headline; PARAPHRASE and SOFT do not substitute for it.",
    )
    parser.add_argument("--checkpoint", default=None, help="Path or gs:// URI.")
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=[0, 1, 2, 3, 4],
        help="Five per condition, per the pre-registration.",
    )
    parser.add_argument("--out", required=True, type=Path, help="Output directory.")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help=(
            "Prove the wiring end to end with synthetic per-seed losses and no "
            "model. Emits a report clearly marked as a smoke test, never a result."
        ),
    )
    return parser


def _synthetic_result(seeds, strategy):
    """A deterministic fake result, for proving the report pipeline only.

    Losses vary per seed on purpose: constant values would give `cohens_d` an
    infinite effect size, which is the degenerate case the report now rejects,
    and a smoke test should exercise the normal path.
    """
    import numpy as np

    from trustgate.eval.harness import SpikeResult, summarize

    rng = np.random.default_rng(0)
    control = [1.00 + float(rng.normal(0, 0.01)) for _ in seeds]
    poisoned = [c + 0.15 + float(rng.normal(0, 0.01)) for c in control]

    return SpikeResult(
        corruption=summarize(poisoned, control),
        strategy=StreamStrategy(strategy),
        per_seed_fluency=[1.25 + float(rng.normal(0, 0.1)) for _ in seeds],
        per_seed_poisoned=poisoned,
        per_seed_control=control,
    )


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    if len(args.seeds) < 5:
        print(
            f"warning: {len(args.seeds)} seeds given; the pre-registration "
            f"requires 5 usable seeds per condition and names fewer as an "
            f"invalidating condition.",
            file=sys.stderr,
        )

    if not args.dry_run:
        from trustgate.eval.fluency import fluency_scorer
        
        # When the model loop is implemented (T1.8), run_attack_spike will be called here:
        # result = run_attack_spike(..., fluency_scorer=fluency_scorer)
        
        raise SystemExit(
            "A real run needs a victim model bound to the inner step. The "
            "carry overlay is implemented (trustgate.eval.carry, ADR-006) but "
            "`eval_benign` still needs a checkpoint, so there is nothing to "
            "measure yet. Re-run with --dry-run to exercise the report "
            "pipeline, or fetch a checkpoint first "
            "(scripts/fetch_checkpoints.sh)."
        )

    result = _synthetic_result(args.seeds, args.strategy)

    report_path = write_report(
        result,
        Path(args.out) / "report.md",
        min_effect_size=FROZEN.min_effect_size,
        min_relative_degradation=FROZEN.min_relative_degradation,
        max_fluency_ratio=FROZEN.max_fluency_ratio,
    )

    banner = (
        "WIRING SMOKE TEST -- SYNTHETIC LOSSES, NOT A RESULT. "
        "Do not copy this out of results/ and do not cite it."
    )
    existing = report_path.read_text(encoding="utf-8")
    report_path.write_text(f"> **{banner}**\n\n{existing}", encoding="utf-8")

    print(f"[dry run] {banner}")
    print(report_path.read_text(encoding="utf-8"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
