# ADR-004 — Baseline eval is a mode on `train`, invoked unmodified

**Status:** Accepted **Date:** 2026-08-08 **Deciders:** project owner (Lead)

## Context

`experiments/000-repro-baseline/README.md` step 3 said "run the vendor's eval entry
point" without naming it. Nobody had established what that command was. Deliverable D1
of `docs/superpowers/specs/2026-08-03-phase0-lead-prep-design.md` was to resolve it from
source before renting a GPU, because the answer could have turned P1-1 from "run a
command" into "write an eval harness" — a critical-path scope change.

Three prior observations suggested no eval path existed: `pyproject.toml` declares only
`train`, `configs/experiment/1b/` holds only `extension/` and `pretrain/`, and the vendor
README documents only `uv run --exact train`. Each was accurate and each was insufficient.

The vendor tree was read at the pinned SHA `a4fc4788ace38e29b5067916d4f4be33da894085`.
Nothing under `vendor/ttt-e2e/` was modified (ADR-002).

## Decision

1. **Eval is `training.eval_mode=true` on the `train` entry point.** It short-circuits at
   `ttt/train.py:222-227` and `return`s before the training loop at `:230` — zero optimizer
   steps, no checkpoint write. It is the same `evaluator.eval_fn` the vendor's own published
   numbers came through (`:265`). **P1-1 does not change scope**; experiment 000 step 3 stands.

2. **Base config: `1b/extension/ext-1b-e2e-32K` with `training.seq_length=8192`.** The
   released checkpoint is Books @8K and no `ext-1b-e2e-8K.yaml` exists, so one override is
   unavoidable. This base inherits `dataset_name: books3`, which is the corpus a Books
   checkpoint should be evaluated on. The alternative (`1b/pretrain/pretrain-1b-e2e` +
   `dataset_name=books3`) is defensible but no better: the `model:` blocks are identical,
   and the differing inner-loop LR settings both resolve to `ilr_multiplier == 1.0` at eval
   because `train.py:224` forces `step_index` to `INT32_MAX - 100`. The effective delta is
   only `dataset_name`, `seq_length`, and `global_batch_size` — all three set explicitly.

3. **Accept mandatory Weights & Biases rather than patch the vendor.** `wandb.login()` and
   an authenticated `api.runs()` query execute under an `is_master` guard only, so
   `training.log_wandb=false` does not avoid them, and the query result drives the resume
   decision at `train.py:145-147` — it is load-bearing control flow, not telemetry. An
   air-gapped eval would require editing `wandb_utils.py`, which ADR-002 forbids. A real
   W&B entity/project/key is therefore a **procurement item before the first booking**.

4. **Do not force eval with `training.total_steps=0`.** `assert total_steps >= 1`
   (`train.py:219`) fires before the eval branch. Leave the inherited `total_steps` alone.

The exact command, every override with its file-and-line citation, and the first-launch
gotchas live in `experiments/000-repro-baseline/EVAL_ENTRYPOINT.md`. That document is the
operational reference; this ADR records why the shape was chosen.

## Consequences

- **The command has never been executed.** It is derived from source, not from a
  successful run. First launch is still where it gets tested.
- **The eval batch is 128, and `training.eval_batch_size` cannot lower it.** `train.py:210-216`
  takes `max(eval_batch_size, global_batch_size // accum_steps * 4)`, and the second term
  dominates. Lower `training.global_batch_size` instead (`=2` gives an eval batch of 8).
- **`training.exp_dir` must point outside this repo.** It defaults to `./experiments`, which
  would write `experiments/demo/` into our tracked tree — `.gitignore` covers
  `experiments/*/results/` but not that path.
- **The eval dataset must be on local disk.** The loader opens `zarr.storage.LocalStore`
  (`lm_dataset.py:14`); there is no GCS store. Checkpoints are asymmetric — those *do*
  accept a `gs://` path (`checkpoint.py:83-84`). This changes the D4 cost model: budget a
  full `gs://llama3-books3` `/val` transfer, and only `/val` — the DCLM bucket is not
  touched for this checkpoint. **⚠ `/val` alone is not sufficient — see the 2026-08-08
  correction at the end of this file.**
- **Eval duration is not config-boundable.** No `num_eval_batches`, and `repeat=False`, so
  eval runs the entire val split. Wall-clock is set by val-split size. The only cheaper path
  is `training.dummy_dataset=true`, which is a plumbing smoke test on random tokens, not a
  measurement.
- **No perplexity is computed anywhere in the vendor tree.** Eval emits
  `train_holdout/loss` (mean CE in nats/token) and writes `train_holdout_token_nll_loss.npy`
  into `log_dir`. ~~If the paper reports perplexity, D2's tolerance must convert via
  `ppl = exp(loss)`.~~ **No conversion is needed — see the correction below.** Both `.npy`
  and `results/` are git-ignored, so the per-token array must be deliberately copied out and
  its numbers transcribed, or the run leaves no record.
- The metric key is `train_holdout` even though it reads the `val` split. Do not misread it
  as a train-set number.

## Open — inference, not confirmed

- ~~`_make_train_iterator` … unbudgeted egress …~~ **Superseded — see the correction below.**
- `WANDB_MODE=offline` is unlikely to stub the live `api.runs()` query, but this was not
  verified.

---

## Correction — 2026-08-08 (from D4)

Two statements above are wrong in kind, not degree. Left visible rather than edited away.

**1. The `/train` array is a crash risk, not an egress risk.** The open item framed grain's
`prefetch_buffer_size=500` as "bounded upside risk for D4 … not a blocker". The risk is the
other way round. `_make_train_iterator` (`train.py:125`) reaches
`zarr.open_array(store, path="/train")` **before** the eval branch returns at `:222-227`, and
`zarr.open_array` raises if the node is absent. So a store containing only `/val` — which is
exactly what the Consequences section below told D4 to budget — **fails during setup, after
the instance has started billing.**

The fix is free. The vendor pins `zarr>=3.0.4` (3.0.7 in `uv.lock`); in zarr v3 each array
carries its own `zarr.json` and **absent chunks read as the fill value**. Copying the group
metadata, all of `/val`, and *only* `/train/zarr.json` gives a store where `/train` opens at
the right shape with no data behind it. `next()` is never called on that iterator in eval
mode, and if grain does prefetch eagerly it reads zeros rather than crashing — which also
retires the egress question. Commands and caveats: `COST_MODEL.md` §1.1;
`scripts/bootstrap_gpu_box.sh` step 5 implements it. **Still unverified against the live
bucket** — no path has been listed.

**2. "If the paper reports perplexity, D2's tolerance must convert via `ppl = exp(loss)`."**
No conversion is needed. The paper's y-axis is labelled "Loss (log perplexity)" and §3.4.1
defines it as the mean per-token next-token-prediction loss — the *same* quantity as
`train_holdout/loss`, in nats/token. Compare directly. (`TOLERANCE.md` §2.)

A third consequence of D2 does not contradict this ADR but changes what it is for: the paper
reports **no** number for the 1B Books @8K checkpoint at all, so the command specified here
produces an environment check rather than a reproduction. See **ADR-005**.
