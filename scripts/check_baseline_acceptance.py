#!/usr/bin/env python
"""Score a collected baseline session against TOLERANCE.md, bar by bar.

Reads the bars as TOLERANCE.md writes them and moves none of them (Standing
Rule 5). It exists so the verdict is not assembled by eye on a billing clock.

    Band  (s4.1)  2.314 < train_holdout/loss < 2.805            run eval-1
    S1    (s5)    mean NLL over the last 1024 positions is strictly lower
                  than the mean over positions 128..1152         eval-1 .npy
    S2    (s5)    run 2 reproduces run 1 to 4 decimals           eval-1, eval-2
    S3    (s5)    dummy_dataset=true lands far above the band    control
    S4    (s5)    resolved config says dataset_name books3, and
                  no trustgate module appears anywhere in the run   all logs
    s4.3          2.60..2.70 -- non-binding expectation, reported only

Two readings are made explicit here rather than left to whoever runs it:

* S2 "reproduce to at least 4 decimals" is checked as |run1 - run2| < 5e-5,
  i.e. the two values agree when rounded to 4 decimals up to rounding-boundary
  luck. Both raw values are printed.
* S3 "far above the band, of order 10-12 nats" has no numeric threshold in
  TOLERANCE.md. This script FAILs S3 only if the control is not above the band
  at all; otherwise it reports the value next to the 10-12 reference for a human
  to read. It does not invent a cut-off.

Layout expected under --collected (what run_gpu_session.sh produces):

    logs/eval-1.log  logs/eval-2.log  logs/control.log  logs/smoke.log
    eval-1/train_holdout_token_nll_loss.npy

Exit: 0 PASS, 1 FAIL, 2 incomplete (a bar could not be evaluated).
"""

from __future__ import annotations

import argparse
import math
import re
import sys
from pathlib import Path

BAND_LOW, BAND_HIGH = 2.314, 2.805
EXPECT_LOW, EXPECT_HIGH = 2.60, 2.70
S2_TOLERANCE = 5e-5
CONTROL_REFERENCE = "order 10-12 nats (TOLERANCE.md s5; uniform over 128256 tokens = 11.762)"

LOSS_LINE = re.compile(r"Eval -- train_holdout/loss:\s*(?P<v>[-+]?(?:\d+\.?\d*(?:[eE][-+]?\d+)?|nan|inf))")
DATASET_NAME = re.compile(r"""['"]dataset_name['"]\s*:\s*['"](?P<v>[^'"]+)['"]""")
DATASET_PATH = re.compile(r"""['"]dataset_path['"]\s*:\s*['"](?P<v>[^'"]+)['"]""")
TRUSTGATE = re.compile(r"\btrustgate\b")

PASS, FAIL, UNVERIFIED, INFO = "PASS", "FAIL", "UNVERIFIED", "INFO"


def read_log(collected: Path, name: str) -> str | None:
    path = collected / "logs" / f"{name}.log"
    return path.read_text(encoding="utf-8", errors="replace") if path.exists() else None


def loss_of(log: str | None) -> float | None:
    if log is None:
        return None
    matches = LOSS_LINE.findall(log)
    return float(matches[-1]) if matches else None


def s1_means(npy_path: Path) -> tuple[float, float]:
    import numpy as np

    curve = np.load(npy_path)
    curve = np.asarray(curve, dtype=np.float64)
    while curve.ndim > 1:  # [.., T] -> [T]
        curve = curve.mean(axis=0)
    if curve.shape[0] < 1152:
        raise ValueError(f"token NLL curve has {curve.shape[0]} positions; S1 needs at least 1152")
    early = float(curve[128:1152].mean())
    late = float(curve[-1024:].mean())
    return early, late


def evaluate(collected: Path) -> list[tuple[str, str, str]]:
    rows: list[tuple[str, str, str]] = []
    eval1, eval2, control = (read_log(collected, n) for n in ("eval-1", "eval-2", "control"))
    l1, l2, lc = loss_of(eval1), loss_of(eval2), loss_of(control)

    # Band
    if l1 is None:
        rows.append(("Band", UNVERIFIED, "no 'Eval -- train_holdout/loss' in logs/eval-1.log"))
    elif not math.isfinite(l1):
        rows.append(("Band", FAIL, f"loss is {l1}"))
    else:
        ok = BAND_LOW < l1 < BAND_HIGH
        rows.append(("Band", PASS if ok else FAIL, f"{l1:.6f} nats/token; bar {BAND_LOW} < loss < {BAND_HIGH}"))

    # S1
    npy = collected / "eval-1" / "train_holdout_token_nll_loss.npy"
    if not npy.exists():
        rows.append(("S1 monotonicity", UNVERIFIED, f"{npy.relative_to(collected)} not found"))
    else:
        try:
            early, late = s1_means(npy)
            ok = late < early
            rows.append(("S1 monotonicity", PASS if ok else FAIL,
                         f"mean NLL pos 128..1152 = {early:.4f}; last 1024 = {late:.4f}; need last < early"))
        except Exception as exc:  # a malformed curve is a finding, not a crash
            rows.append(("S1 monotonicity", UNVERIFIED, f"could not read curve: {exc}"))

    # S2
    if l1 is None or l2 is None:
        rows.append(("S2 determinism", UNVERIFIED, "needs both eval-1 and eval-2 losses"
                     + ("" if eval2 is not None else " (eval-2 did not run)")))
    else:
        diff = abs(l1 - l2)
        ok = diff < S2_TOLERANCE
        rows.append(("S2 determinism", PASS if ok else FAIL,
                     f"run1 {l1:.6f}, run2 {l2:.6f}, |diff| {diff:.2e}; bar < {S2_TOLERANCE:g} (4 decimals)"))

    # S3
    if lc is None:
        rows.append(("S3 negative control", UNVERIFIED, "no loss in logs/control.log"))
    elif not (math.isfinite(lc) and lc > BAND_HIGH):
        rows.append(("S3 negative control", FAIL, f"control loss {lc} is not above the band (> {BAND_HIGH})"))
    else:
        rows.append(("S3 negative control", PASS,
                     f"control loss {lc:.4f} is above the band; read it against {CONTROL_REFERENCE}"))

    # S4
    logs = {n: t for n, t in (("eval-1", eval1), ("eval-2", eval2), ("control", control)) if t is not None}
    if eval1 is None:
        rows.append(("S4 resolved config", UNVERIFIED, "no eval-1 log"))
    else:
        names = DATASET_NAME.findall(eval1)
        paths = DATASET_PATH.findall(eval1)
        if not names:
            rows.append(("S4 resolved config", UNVERIFIED, "no 'dataset_name' in the config echo of eval-1"))
        else:
            ok = all(n == "books3" for n in names)
            rows.append(("S4 resolved config", PASS if ok else FAIL,
                         f"dataset_name={sorted(set(names))}; dataset_path={sorted(set(paths))}"))
    all_logs = dict(logs)
    smoke = read_log(collected, "smoke")
    if smoke is not None:
        all_logs["smoke"] = smoke
    hits = [n for n, t in all_logs.items() if TRUSTGATE.search(t)]
    if not all_logs:
        rows.append(("S4 gate absent", UNVERIFIED, "no logs"))
    else:
        rows.append(("S4 gate absent", FAIL if hits else PASS,
                     f"'trustgate' found in: {hits}" if hits else f"no 'trustgate' in {sorted(all_logs)}"))

    # s4.3, non-binding
    if l1 is not None and math.isfinite(l1):
        inside = EXPECT_LOW <= l1 <= EXPECT_HIGH
        rows.append(("s4.3 expectation", INFO,
                     f"{l1:.4f} {'inside' if inside else 'OUTSIDE'} {EXPECT_LOW}-{EXPECT_HIGH} (non-binding"
                     + ("" if inside else "; investigate dataset_name, seq_length, inner loop before recording")
                     + ")"))
    return rows


def verdict(rows: list[tuple[str, str, str]]) -> tuple[str, int]:
    statuses = [s for _, s, _ in rows if s != INFO]
    if FAIL in statuses:
        return "FAIL -- stop. Nothing downstream is attributable (TOLERANCE.md s4.1).", 1
    if UNVERIFIED in statuses:
        missing = [n for n, s, _ in rows if s == UNVERIFIED]
        return f"NOT YET PASS -- unverified: {', '.join(missing)}. Record each gap in the Outcome.", 2
    return "PASS -- all bars in TOLERANCE.md s4.1 and s5 met.", 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Score a baseline session against TOLERANCE.md.")
    parser.add_argument("--collected", type=Path, required=True)
    parser.add_argument("--out", type=Path, help="also write the report here")
    args = parser.parse_args(argv)

    rows = evaluate(args.collected)
    text, code = verdict(rows)
    width = max(len(n) for n, _, _ in rows)
    lines = ["Baseline acceptance -- experiments/000-repro-baseline/TOLERANCE.md", ""]
    lines += [f"  {n:<{width}}  {s:<10}  {d}" for n, s, d in rows]
    lines += ["", f"VERDICT: {text}"]
    report = "\n".join(lines) + "\n"
    sys.stdout.write(report)
    if args.out:
        args.out.write_text(report, encoding="utf-8")
    return code


if __name__ == "__main__":
    raise SystemExit(main())
