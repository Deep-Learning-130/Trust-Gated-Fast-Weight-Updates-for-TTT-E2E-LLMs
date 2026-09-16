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
        choices=[o.value for o in Objective],
        help="DEGRADE is the gating objective; TRIGGER is secondary.",
    )
    parser.add_argument(
        "--strategy",
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
    parser.add_argument("--out", type=Path, help="Output directory.")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help=(
            "Prove the wiring end to end with synthetic per-seed losses and no "
            "model. Emits a report clearly marked as a smoke test, never a result."
        ),
    )
    parser.add_argument(
        "--random-init",
        action="store_true",
        help=(
            "Build a RANDOM-INIT victim instead of loading --checkpoint, and run "
            "the full spike against it. Instrument validation only: a model that "
            "has learned nothing has nothing worth corrupting, so this renders NO "
            "verdict and the report says so. Its value is that every seam the real "
            "run needs -- crafting, the seed loop, carry threading, fluency "
            "scoring, report generation -- is proven to execute."
        ),
    )
    parser.add_argument(
        "--size",
        default="125m",
        help="Model size for --random-init. See trustgate.eval.model_build.SIZES.",
    )
    parser.add_argument(
        "--seq-length",
        type=int,
        default=8192,
        help=(
            "Must be a multiple of 1024. 8192 gives 8 inner steps -- the "
            "granularity PREREGISTERED.md is scoped to."
        ),
    )
    parser.add_argument(
        "--stream-tokens",
        type=int,
        default=8192,
        help="Length of each crafted stream, in tokens.",
    )
    parser.add_argument(
        "--corpus-tokens",
        type=int,
        default=1 << 17,
        help="Size of the synthetic span corpus for --random-init.",
    )
    parser.add_argument(
        "--fluency-selftest",
        action="store_true",
        help=(
            "Check the independent reference model is installed and actually "
            "discriminating: score fluent English against a scrambled version of "
            "itself and print the ratio. Writes no report. Run this on a fresh "
            "box before trusting any fluency number."
        ),
    )
    return parser


def _fluency_selftest() -> int:
    """Score fluent text against a scrambled copy and report the ratio.

    A reference model that does not separate these two cannot support the
    pre-registered realism bar, and the failure would otherwise only show up as
    an implausible fluency ratio buried in a report.
    """
    import numpy as np

    from trustgate.eval.fluency import load_default_scorer

    fluent = (
        "It is a truth universally acknowledged, that a single man in possession "
        "of a good fortune, must be in want of a wife."
    )
    words = fluent.split()
    np.random.default_rng(0).shuffle(words)
    scrambled = " ".join(words)

    try:
        scorer = load_default_scorer()
    except FileNotFoundError as exc:
        print(f"fluency self-test: {exc}", file=sys.stderr)
        return 1

    fluent_score = scorer.score_text(fluent)
    scrambled_score = scorer.score_text(scrambled)
    ratio = scrambled_score.perplexity / fluent_score.perplexity

    print("fluency reference self-test")
    print(f"  fluent    : ppl {fluent_score.perplexity:9.2f}  "
          f"({fluent_score.mean_nll:.4f} nats/token, "
          f"{fluent_score.n_scored_tokens} tokens)")
    print(f"  scrambled : ppl {scrambled_score.perplexity:9.2f}  "
          f"({scrambled_score.mean_nll:.4f} nats/token, "
          f"{scrambled_score.n_scored_tokens} tokens)")
    print(f"  ratio     : {ratio:.2f}x")
    print()

    if ratio < 2.0:
        print(
            "  FAIL: the reference model barely separates fluent from scrambled "
            "text. It cannot support the pre-registered realism bar.",
            file=sys.stderr,
        )
        return 1

    print("  PASS: the reference model discriminates.")
    print()
    print("  Note: perplexity punishes scrambling but *rewards* repetition, so a")
    print("  repetitive poison stream can score LOW. The realism bar cannot")
    print("  detect that on its own -- see FLUENCY_REFERENCE.md and ADR-007.")
    return 0


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
        per_seed_fluency=[],  # unscored: no reference model exists yet (T3.2)
        per_seed_poisoned=poisoned,
        per_seed_control=control,
    )


NOT_A_RESULT = (
    "INSTRUMENT VALIDATION -- RANDOM-INIT VICTIM, NOT A RESULT. "
    "This model has learned nothing, so it has nothing worth corrupting. "
    "Every number below is noise on noise. Do not copy this out of results/, "
    "do not cite it, and do not read its verdict line as a verdict."
)


def _banner_report(report_path: Path, banner: str) -> None:
    """Prepend a banner so a number lifted out of the file carries its caveat."""
    existing = report_path.read_text(encoding="utf-8")
    report_path.write_text(f"> **{banner}**\n\n{existing}", encoding="utf-8")


def _random_init_run(args) -> int:
    """Drive the whole spike against a random-init victim.

    This is the run that makes the real one cheap: afterwards the only missing
    input is weights. It proves the seams execute against a real vendor
    `MetaModel` -- which no CPU test can, because the vendor is not importable
    in CI by design and `tests/test_vendor_bind.py` therefore runs against a
    fake.

    It proves nothing about the attack. See `NOT_A_RESULT`.
    """
    from trustgate.attack.corpus import TokenCorpus
    from trustgate.attack.objectives import AttackSpec
    from trustgate.attack.stream import generate_seed_pairs
    from trustgate.eval import vendor_bind
    from trustgate.eval.fluency import load_default_scorer
    from trustgate.eval.harness import (
        RunCondition,
        eval_tokens_digest,
        make_adapt_and_eval,
        run_attack_spike,
    )
    from trustgate.eval.model_build import build_random_init, dummy_tokens

    print(f"[random-init] building a {args.size} victim at seq_length={args.seq_length}")
    try:
        cfg, model, state, _mesh = build_random_init(
            size=args.size, seq_length=args.seq_length
        )
    except (ImportError, ValueError) as exc:
        # `model_build` already explains which case this is -- a missing vendor
        # tree, an unknown size, or a seq_length the vendor would assert on.
        # None of them deserves a traceback.
        raise SystemExit(str(exc)) from exc
    mini_batch = int(cfg.model.mini_batch_size)
    binding = vendor_bind.bind(model, state)
    print(f"[random-init] bound; mini_batch_size={mini_batch}")

    # Span corpus. Ids span a wide range rather than DummyDataset's [0, 20) so
    # the Llama-3 decode behind the fluency scorer produces varied text instead
    # of twenty tokens repeated -- perplexity *rewards* repetition, so a
    # degenerate corpus would make the realism path look like it passes.
    # BOS is 128000 and is excluded by construction.
    corpus = TokenCorpus(
        dummy_tokens(args.corpus_tokens, seed=7, vocab_lo=10, vocab_hi=50_000),
        name="synthetic",
    )
    eval_tokens = dummy_tokens(args.seq_length + 1, seed=2)

    adapt_and_eval = make_adapt_and_eval(binding, eval_tokens, model=model)

    # craft_fn=None means random ordering, which `generate_seed_pairs` documents
    # as "not a real attack; for testing the orchestration only". That is exactly
    # what this is. Running the real SELECT hill-climb here would be optimising
    # against a random-init victim's noise -- expensive and uninformative -- so
    # `craft.search_order` is the one seam this path deliberately leaves
    # unexercised. It gets its first real run against trained weights.
    pairs = generate_seed_pairs(
        corpus,
        args.stream_tokens,
        list(args.seeds),
        craft_fn=None,
        mini_batch_size=mini_batch,
    )
    arms = dict(zip(args.seeds, pairs))

    try:
        fluency_scorer = load_default_scorer()
        print("[random-init] fluency reference loaded")
    except FileNotFoundError as exc:
        # Leaves every per-seed ratio nan, which fails the realism bar SAFE.
        print(f"[random-init] no fluency reference ({exc}); ratios stay nan", file=sys.stderr)
        fluency_scorer = None

    spec = AttackSpec(
        objective=Objective(args.objective), stream_tokens=args.stream_tokens
    )
    first_poison, _ = arms[args.seeds[0]]
    condition = RunCondition.from_stream(
        first_poison,
        seed=args.seeds[0],
        seq_length=args.seq_length,
        checkpoint=f"random-init-{args.size} (NO CHECKPOINT)",
        benign_eval_split="synthetic",
        eval_tokens_sha256=eval_tokens_digest(eval_tokens),
    )

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    result = run_attack_spike(
        spec,
        condition,
        list(args.seeds),
        out,
        build_arms=arms.__getitem__,
        adapt_and_eval=adapt_and_eval,
        fluency_scorer=fluency_scorer,
    )

    report_path = write_report(
        result,
        out / "report.md",
        min_effect_size=FROZEN.min_effect_size,
        min_relative_degradation=FROZEN.min_relative_degradation,
        max_fluency_ratio=FROZEN.max_fluency_ratio,
    )
    _banner_report(report_path, NOT_A_RESULT)

    print(f"\n[random-init] {NOT_A_RESULT}")
    print(report_path.read_text(encoding="utf-8"))
    return 0


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    if args.fluency_selftest:
        return _fluency_selftest()

    # Required for a run, but not for the self-test -- so validated here rather
    # than by argparse.
    missing = [
        flag
        for flag, value in (
            ("--objective", args.objective),
            ("--strategy", args.strategy),
            ("--out", args.out),
        )
        if value is None
    ]
    if missing:
        raise SystemExit(f"missing required argument(s): {', '.join(missing)}")

    if len(args.seeds) < 5:
        print(
            f"warning: {len(args.seeds)} seeds given; the pre-registration "
            f"requires 5 usable seeds per condition and names fewer as an "
            f"invalidating condition.",
            file=sys.stderr,
        )

    if args.random_init:
        if args.checkpoint:
            raise SystemExit(
                "--random-init and --checkpoint are mutually exclusive: one builds "
                "a victim with no weights, the other loads weights. Pick one."
            )
        if args.strategy != StreamStrategy.SELECT.value:
            print(
                f"warning: --strategy {args.strategy} is recorded on the report but "
                f"the orderings here are random; see the note in _random_init_run.",
                file=sys.stderr,
            )
        return _random_init_run(args)

    if not args.dry_run:
        raise SystemExit(
            "A real run needs a victim model bound to the inner step. The carry "
            "overlay is implemented (trustgate.eval.carry, ADR-006) and "
            "`vendor_bind` binds it to the real MetaModel, but `eval_benign` "
            "still needs weights.\n"
            "  --random-init  builds a weightless victim and exercises every seam "
            "(instrument validation, renders NO verdict)\n"
            "  --dry-run      exercises the report pipeline alone, with synthetic "
            "losses and no model"
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
    _banner_report(report_path, banner)

    print(f"[dry run] {banner}")
    print(report_path.read_text(encoding="utf-8"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
