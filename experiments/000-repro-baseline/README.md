# Experiment 000 — Reproduce the TTT-E2E baseline

**This is a gate, not a formality.** If we cannot reproduce the vendor's published
numbers on our hardware, then any "corruption" measured in experiment 001 is
unattributable — it could be our misconfiguration rather than an attack. Do not
start 001 until this passes.

## Goal

Run the **unmodified** vendor eval on the 1B DCLM+Books @8K checkpoint and match
the paper's reported long-context numbers within tolerance.

## Steps

1. `scripts/setup_vendor.sh` — init the submodule at the pinned SHA, install the
   vendor's own environment (its `uv.lock`, CUDA 12.8 / cuDNN 9.8).
2. `scripts/fetch_checkpoints.sh` — pull the 1B checkpoint (GCS requester-pays;
   needs a billing project).
3. Run the vendor's eval entry point (`uv run --exact` per vendor README), no
   `trustgate` imports, gate NOT installed.

## Pass criterion

Reported eval loss / perplexity matches the paper within a documented tolerance.
Record exact numbers, checkpoint hash, and vendor SHA in `results/` for the record.

## Note

The gate is installed via `trustgate.vendor_patch.install_gate()`. For THIS
experiment it must **not** be installed — we are validating the untouched baseline.
`is_installed()` should return `False` throughout.
