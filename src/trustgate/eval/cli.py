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
import contextlib
import pickle
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
    parser.add_argument(
        "--checkpoint",
        default=None,
        help=(
            "Path or gs:// URI of the directory CONTAINING the step "
            "directories. Turns on the real run: trained weights, a searching "
            "attacker, and a report with no banner."
        ),
    )
    parser.add_argument(
        "--checkpoint-step",
        type=int,
        default=None,
        help="Step to restore. Default: the latest in the directory.",
    )
    parser.add_argument(
        "--checkpoint-manifest",
        default=None,
        help=(
            "Output of scripts/fingerprint_checkpoint.sh. Its manifest_sha256 "
            "travels into the report as the identity of the weights; without "
            "it the run is recorded as UNFINGERPRINTED."
        ),
    )
    parser.add_argument(
        "--allow-checkpoint-layout-mismatch",
        action="store_true",
        help=(
            "Skip the integer-step-directory check. Orbax reports 'No "
            "checkpoints found' for a wrong path, which reads like a missing "
            "checkpoint; the check exists to tell those apart."
        ),
    )
    parser.add_argument(
        "--corpus-file",
        default=None,
        help=(
            ".npy of int32 token ids the attacker draws spans from. Required "
            "for a checkpoint run: against trained weights, dummy_tokens noise "
            "measures the reaction to gibberish, not the threat model."
        ),
    )
    parser.add_argument(
        "--eval-file",
        default=None,
        help=".npy of int32 token ids for the held-out benign eval.",
    )
    parser.add_argument(
        "--corpus-split",
        default="train",
        help="Name of the split the stream is drawn from; recorded in the condition.",
    )
    parser.add_argument(
        "--eval-split",
        default="val",
        help=(
            "Name of the benign eval split. Must differ from --corpus-split; "
            "`assert_stream_not_from_eval_split` enforces it."
        ),
    )
    parser.add_argument(
        "--max-iters",
        type=int,
        default=40,
        help=(
            "Ordering-search budget, in proposals per seed. THE COST KNOB: one "
            "proposal is one full adapt-and-eval. Measure a single evaluation "
            "before choosing this."
        ),
    )
    parser.add_argument(
        "--early-stop-patience",
        type=int,
        default=50,
        help="Stop after this many proposals with no improvement. 0 disables.",
    )
    parser.add_argument(
        "--fluency-weight",
        type=float,
        default=1.0,
        help=(
            "Penalty on the fluency ratio inside the search objective. 0 lets "
            "the attacker ignore realism, which the pre-registration does not."
        ),
    )
    parser.add_argument(
        "--span-tokens",
        type=int,
        default=64,
        help="Span granularity (ADR-007). Travels with the verdict.",
    )
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
        "--sequence-eval",
        action="store_true",
        help=(
            "Run the SECONDARY sequence-position arms instead of the spike: "
            "poison and control measured after every window, plus no-carry "
            "counterparts and a no-stream floor. Renders NO verdict and touches "
            "no pre-registered bar (PREREGISTERED.md, Addendum 2026-09-20). "
            "Requires --random-init or --checkpoint for a victim."
        ),
    )
    parser.add_argument(
        "--windows",
        type=int,
        default=50,
        help=(
            "Target number of windows for --sequence-eval. One window is one "
            "inner step, so --stream-tokens must be this many mini-batches."
        ),
    )
    parser.add_argument(
        "--eval-every",
        type=int,
        default=1,
        help=(
            "Measure after every Nth window. The cost knob: four arms times the "
            "window count is not free. Pick it on the 002 pilot first."
        ),
    )
    parser.add_argument(
        "--onset-delta",
        type=float,
        default=0.01,
        help=(
            "Reported sensitivity for the onset metric, in nats/token. Not a "
            "bar -- it is printed next to every onset it produces."
        ),
    )
    parser.add_argument(
        "--gate-eval",
        action="store_true",
        help=(
            "Run the PHASE 2 gate measurement instead of the spike: calibrate "
            "the anchor gate on a clean stream, then measure overhead, "
            "clean_regression and gated vs ungated corruption on the 001 arms. "
            "Renders NO verdict. With --checkpoint it needs --arms-file (the "
            "arms.pkl the spike run writes) and --probe-file."
        ),
    )
    parser.add_argument(
        "--arms-file",
        type=Path,
        default=None,
        help="arms.pkl written by the checkpoint spike run: the crafted streams, by seed.",
    )
    parser.add_argument(
        "--uncrafted-arms",
        action="store_true",
        help=(
            "Fallback for --gate-eval when the spike returned NULL RESULT and "
            "wrote no arms.pkl: build uncrafted (random-order) arms from the "
            "real corpus. Overhead and clean_regression stay meaningful; the "
            "corruption columns do not measure an attack, and gate.md says so."
        ),
    )
    parser.add_argument(
        "--probe-file",
        type=Path,
        default=None,
        help=(
            ".npy of held-out int32 token ids for the gate's probe window. "
            "Must not overlap --eval-file; scripts/dump_tokens.py writes both."
        ),
    )
    parser.add_argument(
        "--gate-quantiles",
        type=float,
        nargs="+",
        default=[0.9, 0.99],
        help="Clean-stream divergence quantiles to use as thresholds. Pre-verdict guesses.",
    )
    parser.add_argument(
        "--divergence",
        default="symmetric_kl",
        help="Anchor-gate divergence. See trustgate.gate.anchor.DIVERGENCES.",
    )
    parser.add_argument(
        "--overhead-repeats",
        type=int,
        default=20,
        help="Timed repeats per arm for the gate overhead. 0 skips it.",
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


#: Split names for the synthetic random-init runs. They must differ: the spike
#: and sequence harnesses refuse a stream drawn from the benign eval split
#: (`assert_stream_not_from_eval_split`), and naming both "synthetic" made every
#: random-init spike and sequence run refuse itself.
SYNTHETIC_CORPUS = "synthetic-corpus"
SYNTHETIC_EVAL = "synthetic-eval"

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
    from trustgate.eval.fluency import load_default_scorer
    from trustgate.eval.harness import (
        RunCondition,
        benign_eval_digest,
        make_adapt_and_eval,
        run_attack_spike,
    )
    from trustgate.eval.model_build import dummy_tokens

    cfg, model, binding, mini_batch, _label = _build_victim(args, tag="random-init")

    # Span corpus. Ids span a wide range rather than DummyDataset's [0, 20) so
    # the Llama-3 decode behind the fluency scorer produces varied text instead
    # of twenty tokens repeated -- perplexity *rewards* repetition, so a
    # degenerate corpus would make the realism path look like it passes.
    # BOS is 128000 and is excluded by construction.
    corpus = TokenCorpus(
        dummy_tokens(args.corpus_tokens, seed=7, vocab_lo=10, vocab_hi=50_000),
        name=SYNTHETIC_CORPUS,
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
        benign_eval_split=SYNTHETIC_EVAL,
        eval_tokens_sha256=benign_eval_digest(eval_tokens),
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


SEQUENCE_NOT_GATING = (
    "SECONDARY, NON-GATING -- NOT A VERDICT. These arms were added as a dated "
    "addendum to PREREGISTERED.md on 2026-09-20 and move none of the three "
    "bars frozen on 2026-07-28. The PROCEED/STOP decision comes only from the "
    "spike report."
)


def _load_arms(args) -> dict:
    """The 001 spike's crafted streams, by seed, from the arms.pkl it wrote."""
    if args.arms_file is None:
        raise SystemExit(
            "against a checkpoint this needs --arms-file: the spike run's "
            "arms.pkl (in its --out). Its streams are the ones the attacker "
            "found; any other ordering measures a different experiment."
        )
    with Path(args.arms_file).open("rb") as fh:
        arms = pickle.load(fh)
    absent = [s for s in args.seeds if s not in arms]
    if absent:
        raise SystemExit(f"--arms-file has no arms for seeds {absent}")
    return arms


def _sequence_run(args) -> int:
    """Drive the secondary sequence-position arms.

    Against --random-init: random orderings over synthetic tokens, instrument
    validation only, bannered. Against --checkpoint: the spike's own crafted
    streams from --arms-file and the real --eval-file, because the addendum
    holds the end-of-stream values to the same quantity `corruption_metric`
    consumes -- which is only true on the same streams and the same eval.
    """
    import numpy as np

    from trustgate.attack.corpus import TokenCorpus
    from trustgate.attack.stream import generate_seed_pairs
    from trustgate.eval import vendor_bind
    from trustgate.eval.harness import (
        RunCondition,
        eval_benign,
        benign_eval_digest,
        model_bos_token_id,
    )
    from trustgate.eval.model_build import dummy_tokens
    from trustgate.eval.report import write_sequence_report
    from trustgate.eval.sequence import make_vendor_windows, run_sequence_eval

    if args.checkpoint:
        arms = _load_arms(args)
        eval_tokens = _load_tokens(Path(args.eval_file), "--eval-file")
        eval_split = args.eval_split
    else:
        arms = None
        eval_tokens = dummy_tokens(args.seq_length + 1, seed=2)
        eval_split = SYNTHETIC_EVAL

    cfg, _model, binding, mini_batch, label = _build_victim(args, tag="sequence")

    if arms is None:
        expected = args.windows * mini_batch
        if args.stream_tokens != expected:
            raise SystemExit(
                f"--windows {args.windows} at mini_batch_size {mini_batch} needs "
                f"--stream-tokens {expected}, got {args.stream_tokens}. One window "
                f"is one inner step; a stream that does not divide evenly would "
                f"measure the last window at a different dose than the rest."
            )
        corpus = TokenCorpus(
            dummy_tokens(args.corpus_tokens, seed=7, vocab_lo=10, vocab_hi=50_000),
            name=SYNTHETIC_CORPUS,
        )
        pairs = generate_seed_pairs(
            corpus,
            args.stream_tokens,
            list(args.seeds),
            craft_fn=None,
            mini_batch_size=mini_batch,
        )
        arms = dict(zip(args.seeds, pairs))
        windows = args.windows
    else:
        # The window count is whatever the spike's streams hold; --windows
        # sizes synthetic streams only.
        n = int(np.asarray(arms[args.seeds[0]][0].tokens).shape[0])
        windows = n // mini_batch
    print(f"[sequence] {windows} windows of {mini_batch} tokens")

    first_poison, _ = arms[args.seeds[0]]
    condition = RunCondition.from_stream(
        first_poison,
        seed=args.seeds[0],
        seq_length=args.seq_length,
        checkpoint=label,
        benign_eval_split=eval_split,
        eval_tokens_sha256=benign_eval_digest(eval_tokens),
    )

    result = run_sequence_eval(
        condition,
        list(args.seeds),
        build_arms=arms.__getitem__,
        windows_for=make_vendor_windows(
            binding, condition, bos_token_id=model_bos_token_id(binding)
        ),
        step_fn=vendor_bind.make_step_fn(binding),
        fresh_carry=binding.init_carry,
        evaluate=lambda carry, cond: eval_benign(binding, carry, eval_tokens, cond),
        eval_every=args.eval_every,
    )

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    report_path = write_sequence_report(
        result, out / "sequence.md", onset_delta=args.onset_delta
    )
    if not args.checkpoint:
        _banner_report(report_path, NOT_A_RESULT)

    print(f"\n[sequence] {SEQUENCE_NOT_GATING}")
    print(report_path.read_text(encoding="utf-8"))
    return 0


def _gate_run(args) -> int:
    """Drive the Phase 2 gate measurement. See `trustgate.eval.gate_eval`.

    Against a checkpoint it re-runs the 001 spike's own crafted streams, loaded
    from `--arms-file`, so gated and ungated are measured on exactly the
    orderings the attacker found. Against --random-init it uses random
    orderings and is instrument validation only.
    """
    import numpy as np

    from trustgate.attack.corpus import TokenCorpus
    from trustgate.attack.stream import generate_seed_pairs
    from trustgate.eval import gate_eval, vendor_bind
    from trustgate.eval.harness import (
        RunCondition,
        benign_eval_digest,
        make_adapt_and_eval,
        model_bos_token_id,
        run_stream,
    )
    from trustgate.eval.model_build import dummy_tokens
    from trustgate.eval.report import GATE_NOT_A_VERDICT, write_gate_report
    from trustgate.gate.anchor import DIVERGENCES

    if args.divergence not in DIVERGENCES:
        raise SystemExit(f"--divergence must be one of {', '.join(DIVERGENCES)}")

    if args.checkpoint:
        if args.probe_file is None:
            raise SystemExit(
                "--gate-eval against a checkpoint needs --probe-file: the probe "
                "window must be held out from the eval (scripts/dump_tokens.py)."
            )
        if args.arms_file is not None and args.uncrafted_arms:
            raise SystemExit("--arms-file and --uncrafted-arms are exclusive; pick one.")
        if args.arms_file is None and not args.uncrafted_arms:
            raise SystemExit(
                "--gate-eval against a checkpoint needs --arms-file (the spike's "
                "arms.pkl). If the spike returned NULL RESULT and wrote none, "
                "pass --uncrafted-arms to measure overhead and clean_regression "
                "on uncrafted orderings instead; the report labels it."
            )
        if Path(args.probe_file).resolve() == Path(args.eval_file).resolve():
            raise SystemExit("--probe-file and --eval-file are the same file")
        arms = None if args.uncrafted_arms else _load_arms(args)
        eval_tokens = _load_tokens(Path(args.eval_file), "--eval-file")
        probe_tokens = _load_tokens(Path(args.probe_file), "--probe-file")
        corpus_tokens = _load_tokens(Path(args.corpus_file), "--corpus-file")
    else:
        eval_tokens = dummy_tokens(args.seq_length + 1, seed=2)
        probe_tokens = dummy_tokens(args.seq_length + 1, seed=3)
        corpus_tokens = dummy_tokens(args.corpus_tokens, seed=7, vocab_lo=10, vocab_hi=50_000)
        arms = None

    cfg, model, binding, mini_batch, label = _build_victim(args, tag="gate")
    bos = model_bos_token_id(binding)

    if arms is None:
        corpus = TokenCorpus(
            corpus_tokens, name=args.corpus_split if args.checkpoint else SYNTHETIC_CORPUS
        )
        pairs = generate_seed_pairs(
            corpus, args.stream_tokens, list(args.seeds), craft_fn=None,
            span_tokens=args.span_tokens, mini_batch_size=mini_batch,
        )
        arms = dict(zip(args.seeds, pairs))
        arms_source = gate_eval.UNCRAFTED_ARMS
    else:
        arms_source = f"crafted by the 001 search ({args.arms_file})"

    eval_digest = benign_eval_digest(eval_tokens)
    eval_split = args.eval_split if args.checkpoint else SYNTHETIC_EVAL

    def condition_for(stream, seed):
        return RunCondition.from_stream(
            stream,
            seed=seed,
            seq_length=args.seq_length,
            checkpoint=label,
            benign_eval_split=eval_split,
            eval_tokens_sha256=eval_digest,
        )

    # The clean calibration stream: a contiguous, uncrafted slice of the corpus,
    # the same length as the arms, so its step count matches theirs.
    first_poison, first_control = arms[args.seeds[0]]
    n = int(np.asarray(first_poison.tokens).shape[0])
    if corpus_tokens.shape[0] < n:
        raise SystemExit(f"corpus has {corpus_tokens.shape[0]} tokens; calibration needs {n}")
    calib_tokens = np.asarray(corpus_tokens[-n:], dtype="int32")
    calib_condition = condition_for(first_control, args.seeds[0])

    def calibrate_stream_fn(**gate_kwargs):
        return run_stream(model, calib_tokens, calib_condition, binding=binding, **gate_kwargs)

    def adapt_and_eval_for(**gate_kwargs):
        return make_adapt_and_eval(binding, eval_tokens, model=model, **gate_kwargs)

    gate_inputs = gate_eval.make_gate_inputs(binding, probe_tokens, bos_token_id=bos)

    overhead_fn = None
    if args.overhead_repeats > 0:
        chunk = vendor_bind.sequence_chunks(binding, eval_tokens, bos_token_id=bos)[0]

        def overhead_fn(spec):
            return gate_eval.measure_step_overhead(
                binding, chunk, spec, gate_inputs, repeats=args.overhead_repeats
            )

    result = gate_eval.run_gate_eval(
        seeds=list(args.seeds),
        build_arms=arms.__getitem__,
        adapt_and_eval_for=adapt_and_eval_for,
        condition_for=condition_for,
        calibrate_stream_fn=calibrate_stream_fn,
        gate_inputs=gate_inputs,
        quantiles=list(args.gate_quantiles),
        divergence=args.divergence,
        checkpoint=label,
        overhead_fn=overhead_fn,
        arms_source=arms_source,
    )

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    with (out / "gate-result.pkl").open("wb") as fh:
        pickle.dump(result, fh)
    report_path = write_gate_report(result, out / "gate.md")
    if not args.checkpoint:
        _banner_report(report_path, NOT_A_RESULT)

    print(f"\n[gate] {GATE_NOT_A_VERDICT}")
    print(report_path.read_text(encoding="utf-8"))
    return 0


def _load_tokens(path: Path, what: str):
    """Load a saved token array, refusing anything that is not one.

    A real run measures language, so its corpus and its eval set have to be real
    tokens. `dummy_tokens` is uniform noise over a vocab range: against trained
    weights it measures the model's reaction to gibberish, which is not the
    threat model and not a result.
    """
    import numpy as np

    if not path.is_file():
        raise SystemExit(
            f"{what} file {path} does not exist. A checkpoint-backed run needs "
            f"real tokens, not `dummy_tokens` noise -- dump a held-out slice of "
            f"the eval corpus to .npy (int32 token ids) and pass it here."
        )
    tokens = np.load(path)
    if tokens.ndim != 1:
        raise SystemExit(f"{what} file {path} must be 1-D, got shape {tokens.shape}")
    if not np.issubdtype(tokens.dtype, np.integer):
        raise SystemExit(f"{what} file {path} must hold integer ids, got {tokens.dtype}")
    return np.asarray(tokens, dtype="int32")


def _checkpoint_label(args) -> str:
    """What travels into `RunCondition.checkpoint`.

    A bare path names a directory that may have changed since; the manifest hash
    names the bytes. `scripts/fingerprint_checkpoint.sh` writes that hash, and a
    run that cannot quote one says so in the report rather than implying a
    provenance it does not have.
    """
    if not args.checkpoint_manifest:
        return f"{args.checkpoint} (UNFINGERPRINTED)"

    text = Path(args.checkpoint_manifest).read_text(encoding="utf-8")
    hexdigits = set("0123456789abcdef")
    for line in text.splitlines():
        if "manifest_sha256" not in line:
            continue
        for token in line.replace(":", " ").replace("=", " ").split():
            candidate = token.strip().lower()
            if len(candidate) == 64 and set(candidate) <= hexdigits:
                return f"{args.checkpoint} sha256:{candidate}"
    raise SystemExit(
        f"--checkpoint-manifest {args.checkpoint_manifest} carries no "
        f"manifest_sha256 line. `scripts/fingerprint_checkpoint.sh` writes one; "
        f"a manifest without it cannot identify the weights that were run."
    )


#: Holds the victim's device mesh open for the rest of the run. The vendor's
#: sliding-window attention calls `with_sharding_constraint` with a bare
#: `PartitionSpec` (`attention.py:308-314`, unconditionally), which only resolves
#: inside an active `with mesh:` -- `train.py:222` runs everything under one.
#: `model_build` builds the victim under the mesh and returns it; every forward
#: pass after that has to stay inside it too, or the first inner step fails.
#: `main` closes this on the way out.
_VICTIM_SCOPE = contextlib.ExitStack()


def _build_victim(args, *, tag: str):
    """Build the victim once, whichever path was asked for.

    All three runners need the same four things and previously built them
    separately. The label is the only place the two paths differ visibly, which
    is the point: `vendor_bind.bind` reads its config off `model.config` and
    consumes only `(MetaModel, eqx.nn.State)`, so everything downstream is
    indifferent to how the weights arrived.

    Returns `(cfg, model, binding, mini_batch, label)`.
    """
    from trustgate.eval import vendor_bind
    from trustgate.eval.model_build import build_from_checkpoint, build_random_init

    if args.checkpoint:
        print(f"[{tag}] loading {args.size} weights from {args.checkpoint}")
        try:
            cfg, model, state, mesh = build_from_checkpoint(
                checkpoint=args.checkpoint,
                size=args.size,
                seq_length=args.seq_length,
                step=args.checkpoint_step,
                allow_layout_mismatch=args.allow_checkpoint_layout_mismatch,
            )
        except (ImportError, ValueError, FileNotFoundError) as exc:
            # `model_build` already says which case this is -- absent vendor,
            # bad size, bad seq_length, missing checkpoint, or a partial
            # restore. None of them deserves a traceback.
            raise SystemExit(str(exc)) from exc
        label = _checkpoint_label(args)
    else:
        print(f"[{tag}] building a {args.size} victim at seq_length={args.seq_length}")
        try:
            cfg, model, state, mesh = build_random_init(
                size=args.size, seq_length=args.seq_length
            )
        except (ImportError, ValueError) as exc:
            raise SystemExit(str(exc)) from exc
        label = f"random-init-{args.size} (NO CHECKPOINT)"

    _VICTIM_SCOPE.enter_context(mesh)
    mini_batch = int(cfg.model.mini_batch_size)
    binding = vendor_bind.bind(model, state)
    print(f"[{tag}] bound; mini_batch_size={mini_batch}")
    return cfg, model, binding, mini_batch, label


def _timed(adapt_and_eval, tag: str):
    """Print the wall-clock of every adapt-and-eval as it happens.

    This is the number runbook C2 exists to measure: one ordering-search
    proposal is one adapt-and-eval, and `--max-iters` multiplies it. The first
    call includes compilation and says so; the rest are the per-proposal cost.
    `adapt_and_eval` returns a Python float, so the device is already done.
    """
    import time

    count = 0

    def timed(stream, condition):
        nonlocal count
        count += 1
        start = time.perf_counter()
        value = adapt_and_eval(stream, condition)
        note = "  (includes compile)" if count == 1 else ""
        print(f"{tag} adapt-and-eval #{count}: {time.perf_counter() - start:.1f} s{note}", flush=True)
        return value

    return timed


def _checkpoint_run(args) -> int:
    """The real run: trained weights, a searching attacker, a real verdict.

    Everything `_random_init_run` proves about the seams, plus the two things it
    deliberately does not do -- load weights, and let the attacker search. There
    is no banner at the end of this one, which is the whole point of it.

    Cost warning. The search is the dominant expense: one proposal is one full
    adapt-and-eval, and `--max-iters` multiplies that by the seed count. Size it
    against a measured per-evaluation wall-clock before starting a run that
    bills.
    """
    from trustgate.attack.corpus import TokenCorpus
    from trustgate.attack.craft import CraftConfig
    from trustgate.attack.objectives import AttackSpec
    from trustgate.attack.stream import generate_seed_pairs
    from trustgate.eval.attacker import NoOrderingFound, make_craft_fn
    from trustgate.eval.fluency import load_default_scorer
    from trustgate.eval.harness import (
        RunCondition,
        benign_eval_digest,
        make_adapt_and_eval,
        run_attack_spike,
    )

    if args.strategy != StreamStrategy.SELECT.value:
        raise SystemExit(
            f"--strategy {args.strategy} is not wired: `craft_stream` refuses "
            f"anything but SELECT (T2.5). SELECT is the pre-registered headline "
            f"and the only strategy that must work."
        )

    corpus_tokens = _load_tokens(Path(args.corpus_file), "--corpus-file")
    eval_tokens = _load_tokens(Path(args.eval_file), "--eval-file")

    cfg, model, binding, mini_batch, label = _build_victim(args, tag="run")

    corpus = TokenCorpus(corpus_tokens, name=args.corpus_split)
    adapt_and_eval = _timed(make_adapt_and_eval(binding, eval_tokens, model=model), "[run]")
    eval_digest = benign_eval_digest(eval_tokens)

    # A real run must not silently skip the realism bar: a corruption that only
    # appears with a non-fluent stream does not pass, and an absent scorer
    # leaves every ratio nan rather than failing loudly.
    try:
        fluency_scorer = load_default_scorer()
        print("[run] fluency reference loaded")
    except FileNotFoundError as exc:
        raise SystemExit(
            f"the fluency reference model is required for a verdict run: the "
            f"realism bar is one of the three frozen criteria and cannot be "
            f"scored without an independent reference. {exc}"
        ) from exc

    spec = AttackSpec(
        objective=Objective(args.objective),
        stream_tokens=args.stream_tokens,
        fluency_weight=args.fluency_weight,
    )
    config = CraftConfig(
        strategy=StreamStrategy.SELECT,
        max_iters=args.max_iters,
        reference_model_path="(injected scorer)",
        early_stop_patience=args.early_stop_patience,
    )

    def condition_for(stream, seed):
        return RunCondition.from_stream(
            stream,
            seed=seed,
            seq_length=args.seq_length,
            checkpoint=label,
            benign_eval_split=args.eval_split,
            eval_tokens_sha256=eval_digest,
        )

    searches: dict[int, object] = {}

    def record(seed, result):
        searches[seed] = result
        print(
            f"[run] seed {seed}: {result.n_evaluations} evaluations, "
            f"gain {result.gain:+.6f}, accepted {result.n_accepted}"
            + (", stopped early" if result.stopped_early else "")
        )

    print(
        f"[run] searching: {len(args.seeds)} seeds x up to {args.max_iters} "
        f"proposals, one adapt-and-eval each"
    )
    try:
        pairs = generate_seed_pairs(
            corpus,
            args.stream_tokens,
            list(args.seeds),
            craft_fn=make_craft_fn(
                corpus=corpus,
                spec=spec,
                config=config,
                adapt_and_eval=adapt_and_eval,
                condition_for=condition_for,
                fluency_scorer=fluency_scorer,
                span_tokens=args.span_tokens,
                mini_batch_size=mini_batch,
                on_result=record,
            ),
            span_tokens=args.span_tokens,
            mini_batch_size=mini_batch,
        )
    except NoOrderingFound as exc:
        # A real null, and the honest report is that the attacker searched and
        # found nothing -- not a crash, and not a rerun at a kinder setting.
        raise SystemExit(
            f"NULL RESULT (not an error): {exc}\n"
            f"Under SELECT the attacker's only lever is ordering. Record this "
            f"as the outcome; PREREGISTERED.md says publishing a STOP is the "
            f"honest result."
        ) from exc

    arms = dict(zip(args.seeds, pairs))
    first_poison, _ = arms[args.seeds[0]]
    condition = condition_for(first_poison, args.seeds[0])

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    # Written the moment the search ends, before anything else can crash: the
    # search is the session's dominant cost, and the Phase 2 gate measurement
    # (--gate-eval --arms-file) re-runs these exact streams rather than
    # searching again.
    arms_path = out / "arms.pkl"
    with arms_path.open("wb") as fh:
        pickle.dump(arms, fh)
    print(f"[run] crafted arms saved to {arms_path}")

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
    print(report_path.read_text(encoding="utf-8"))
    return 0


def main(argv: list[str] | None = None) -> int:
    with _VICTIM_SCOPE:
        return _main(argv)


def _main(argv: list[str] | None = None) -> int:
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

    if args.gate_eval and args.sequence_eval:
        raise SystemExit("--gate-eval and --sequence-eval are separate runs; pick one.")

    if args.gate_eval and not (args.random_init or args.checkpoint):
        raise SystemExit(
            "--gate-eval needs a victim: --random-init for instrument "
            "validation, or --checkpoint with --arms-file for the real measurement."
        )

    if args.sequence_eval and not (args.random_init or args.checkpoint):
        raise SystemExit(
            "--sequence-eval needs a victim. Pair it with --random-init for "
            "instrument validation, or with --checkpoint for the real arms."
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
        if args.gate_eval:
            return _gate_run(args)
        return _sequence_run(args) if args.sequence_eval else _random_init_run(args)

    if args.checkpoint:
        missing = [
            flag
            for flag, value in (
                ("--corpus-file", args.corpus_file),
                ("--eval-file", args.eval_file),
            )
            if value is None
        ]
        if missing:
            raise SystemExit(
                f"a checkpoint run needs real tokens: missing "
                f"{', '.join(missing)}. Against trained weights, dummy_tokens "
                f"noise measures the reaction to gibberish rather than the "
                f"threat model, so there is no synthetic fallback here."
            )
        if args.corpus_split == args.eval_split:
            raise SystemExit(
                f"--corpus-split and --eval-split are both {args.corpus_split!r}. "
                f"Measuring degradation on the split the stream was drawn from "
                f"is named as an invalidating condition in PREREGISTERED.md."
            )
        if args.gate_eval:
            return _gate_run(args)
        return _sequence_run(args) if args.sequence_eval else _checkpoint_run(args)

    if not args.dry_run:
        raise SystemExit(
            "A real run needs a victim model bound to the inner step. The carry "
            "overlay is implemented (trustgate.eval.carry, ADR-006) and "
            "`vendor_bind` binds it to the real MetaModel, but `eval_benign` "
            "still needs weights.\n"
            "  --checkpoint    loads trained weights and runs the real attacker "
            "(this is the one that renders a verdict)\n"
            "  --random-init   builds a weightless victim and exercises every seam "
            "(instrument validation, renders NO verdict)\n"
            "  --sequence-eval with --random-init, runs the secondary "
            "sequence-position arms (also NO verdict)\n"
            "  --dry-run       exercises the report pipeline alone, with synthetic "
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
