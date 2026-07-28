# Experiment 001 — Attack Spike (kill-gate)

**Read `PREREGISTERED.md` before running anything here.** The go/no-go thresholds
are frozen there.

## Goal

Answer the one question that gates the entire project (dossier §4, §7): can a
benign-looking input stream measurably corrupt TTT-E2E fast weights?

- **PROCEED** → the threat model is real; build the gate (Phase 2).
- **STOP** → no attack surface; F1 stops. This is a real, acceptable outcome.

## Prerequisites

1. `../000-repro-baseline/` has reproduced the vendor's published numbers.
2. Checkpoints fetched (`scripts/fetch_checkpoints.sh`).

## How to run *(pending — model code blocked on checkpoints)*

```
# 1B meta-mode, 5 seeds, SELECT strategy
python -m trustgate.eval.harness \
    --objective degrade --strategy select \
    --checkpoint <path> --seeds 0 1 2 3 4 \
    --out results/
```

Produces `results/report.md` via `trustgate.eval.report`, which prints observed
vs. pre-registered values and a PROCEED/STOP verdict.

## Outputs (git-ignored)

`results/` — per-seed losses, crafted streams, the go/no-go report. Commit only
the final `report.md` (copy it out of `results/`), not the raw runs.
