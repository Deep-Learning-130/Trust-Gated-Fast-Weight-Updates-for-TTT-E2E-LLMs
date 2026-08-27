# Experiment 000 — Reproduce the TTT-E2E baseline

**This is a gate, not a formality.** If we cannot reproduce the vendor's published
numbers on our hardware, then any "corruption" measured in experiment 001 is
unattributable — it could be our misconfiguration rather than an attack. Do not
start 001 until this passes.

## Goal

Run the **unmodified** vendor eval on the 1B DCLM+Books @8K checkpoint and confirm the
result is consistent with the paper.

> **Not a reproduction of a published number — see [`ADR-005`](../../docs/adr/ADR-005-baseline-comparison-basis.md).**
> The paper reports **no** number for this checkpoint: every printed absolute loss is for the
> 760M model and every plotted one is for the 3B at 3× ([`TOLERANCE.md`](TOLERANCE.md) §2).
> So the bar is a two-sided bracket derived from signed monotonicity, plus structural checks
> that carry the real discriminating power. What this gate establishes is that **our
> environment is sound** — which is what makes a later corruption measurement attributable,
> and was always its actual purpose.

## Before booking the box

Three prerequisites are **procurement, not configuration** — none of them is fixable
once the instance is billing:

- **A Weights & Biases entity, project, and API key.** Mandatory. `training.log_wandb=false`
  does not avoid them. See ADR-004 §3.
- **A local copy of the eval dataset.** The loader has no GCS path — `gs://llama3-books3`
  `/val` must be on disk. The checkpoint may stay on `gs://`.
- **`gcloud` authed against a billing project.** Both buckets are requester-pays.

## Steps

0. **Rehearse on the 125M checkpoint first**, in the same session. It costs minutes and
   exercises every failure mode except memory. See [`COST_MODEL.md`](COST_MODEL.md) §4.
1. `scripts/bootstrap_gpu_box.sh` — does steps 1–2 and writes out the exact eval command
   with every path filled in. Idempotent; safe to re-run after a partial failure. It has
   **never been executed** — read it before trusting it.
   (`scripts/setup_vendor.sh` alone still works for just the submodule + environment.)
2. `PROBE_ONLY=1 scripts/fetch_checkpoints.sh` — metadata-only byte count **before** any
   egress; then drop `PROBE_ONLY` to fetch. Records sha256 into `results/` (P0-5).
   Also copy `gs://llama3-books3` `/val` **plus both arrays' metadata** to local disk —
   `/train`'s `zarr.json` is required or the run dies before eval ([`COST_MODEL.md`](COST_MODEL.md) §1.1).
3. Run the vendor eval **unmodified**: `uv run --exact train … training.eval_mode=true`.
   The exact command with every override and its source citation is in
   [`EVAL_ENTRYPOINT.md`](EVAL_ENTRYPOINT.md); the reasoning is in
   [`../../docs/adr/ADR-004-baseline-eval-invocation.md`](../../docs/adr/ADR-004-baseline-eval-invocation.md).
   No `trustgate` imports, gate NOT installed. Set `training.exp_dir` **outside this repo**.

## Pass criterion

**Pre-registered in [`TOLERANCE.md`](TOLERANCE.md) as of 2026-08-08, before any hardware
exists. Read the bar before reading the number** (Rule 5). All four must hold:

- **Band:** `2.314 < train_holdout/loss < 2.805` nats/token (TOLERANCE.md §4.1).
- **Falling per-token NLL curve**, from `train_holdout_token_nll_loss.npy` (§5 S1).
- **Determinism** across a rerun, and a `dummy_dataset` control far above the band (§5 S2–S3).
- **`is_installed()` is False** throughout — this is the ungated baseline (§5 S4).

Also verify the **resolved** `training.dataset_name == books3` from the run's own config
echo. The band is wide enough that evaluating on DCLM by mistake would pass it; this check
is what catches that (§4.2).

Units: the paper's "Loss (log perplexity)" is mean CE in nats/token — the same quantity the
vendor emits, so **no conversion is needed**. `ppl = exp(loss)` is available but never
required; the vendor computes no perplexity anywhere.

Record exact numbers, checkpoint hash, vendor SHA, and env versions in `results/`. Note
that `results/` and `*.npy` are both git-ignored: `train_holdout_token_nll_loss.npy` must
be deliberately copied out and its numbers transcribed into a tracked file, or the run
leaves no record.

## Note

The gate is installed via `trustgate.vendor_patch.install_gate()`. For THIS
experiment it must **not** be installed — we are validating the untouched baseline.
`is_installed()` should return `False` throughout.
