# GPU session 1: runbook

> **One page, in order.** Written 2026-09-16 before the first booking, and revised the same
> day after the scripts were rehearsed end to end on a fake box.
> Target: **the 1B Books @8K baseline in one session on E2E Networks.** Tasks `P0-1` / `T1.4`–`T1.7`.
> Supporting detail: `COST_MODEL.md` §4/§9, `TOLERANCE.md` §4–§5, `EVAL_ENTRYPOINT.md`.

---

## 0. Before you rent: none of this can be fixed while the box bills

| | Item | How to prove it |
|---|---|---|
| ☐ | **Checkpoint and val subset are on a disk you control** | See §0.1. This is the blocker. Without these files a booking produces nothing |
| ☐ | **W&B credentials work the way the vendor uses them** | `WANDB_ENTITY=… WANDB_PROJECT=… WANDB_KEY=… uv run --no-project --with wandb==0.19.9 python scripts/preflight_wandb.py` exits 0 |
| ☐ | **This branch is pushed** | The box clones it. `git log origin/infra/gpu-session-1 -1` shows the commit you mean to run |
| ☐ | **`TOLERANCE.md` §4.1 and §5 read.** You know the bar before any number exists | Standing Rule 5 |
| ☐ | **Booking row claimed** in `gpu-bookings.md` | The record is the lock |

### 0.1 The data, and why there is no Google Cloud step

The 1B checkpoint (`1b_ttt_e2e_finetune_books_8k_1x_cc`, 5.35 GB) and books3 `/val` exist **only**
in requester-pays GCS buckets, and there is no mirror (COST_MODEL §9.2). **No Google Cloud
account is used in this plan.** Someone who has access has to hand the files over. Anyone
with a GCP billing project can run `fetch_checkpoints.sh` and `make_val_subset.py fetch
--tokens 200000000` on their own machine. Either way, you need two directories:

```
1b_ttt_e2e_finetune_books_8k_1x_cc/        <- must contain an integer step dir, e.g. 1250/model_weights/
llama3-books3/                              <- the store ROOT
  zarr.json
  train/zarr.json                           <- metadata only; train.py:125 opens /train
  val/zarr.json
  val/c/0  val/c/1                          <- 2 chunks x 100M tokens = 0.8 GB
  val-subset-manifest.json
```

Also ask for `checkpoint-sha256-1b_ttt_e2e_finetune_books_8k_1x_cc.txt` (written by the fetch).
With it, the bootstrap **verifies** every byte you received. Without it, the bootstrap
only fingerprints what arrived.

If both directories are present, the bootstrap never touches GCS, gsutil or gcloud. That was
rehearsed: stub `gsutil`/`gcloud` binaries were never called.

---

## 1. What to rent: E2E Networks

**Why E2E:** it takes UPI (cards are not an option), bills per minute, offers 80 GB cards,
and gives a real Linux shell with tmux and a persistent disk.
**Why not Colab:** its A100 is 40 GB, which sits inside the unmeasured 36–49 GB estimate
(COST_MODEL §9.4.1). Its runtime disconnects and wipes the disk mid-eval, and there is no tmux.

| Choose | Why |
|---|---|
| **H100 80 GB** (else A100 80 GB) | 36–49 GB estimated peak, never measured. Do not gamble on 40 GB |
| Ubuntu 22.04 image whose `nvidia-smi` shows **CUDA Version ≥ 12.8** (driver ≥ 570) | The vendor lock pins CUDA 12.8 / cuDNN 9.8 pip wheels; the host supplies only the driver. The bootstrap refuses < 525 and warns < 570 |
| **≥ 150 GB disk** | vendor env ~10 GB, checkpoint 5.4 GB, val 0.8 GB, XLA cache, logs |
| Prepaid credits via UPI for **~6 hours** | Session plan below is ~4 h. Check the live hourly rate at booking and write it in the booking row |

---

## 2. On the box: four commands

```bash
tmux new -s ttt                                  # ALWAYS. A dropped SSH kills an eval otherwise
git clone --recursive https://github.com/Deep-Learning-130/Trust-Gated-Fast-Weight-Updates-for-TTT-E2E-LLMs.git TTT
cd TTT && git checkout infra/gpu-session-1

# from your laptop, in another terminal: side-load the data (see §0.1)
#   rsync -avP 1b_ttt_e2e_finetune_books_8k_1x_cc llama3-books3 <user>@<box-ip>:/mnt/data/

export WANDB_ENTITY=…  WANDB_PROJECT=…  WANDB_KEY=…
export DATA_ROOT=/mnt/data  EXP_DIR=/mnt/runs
export CKPT_DIR=/mnt/data/1b_ttt_e2e_finetune_books_8k_1x_cc
export CKPT_SHA_MANIFEST=/mnt/data/checkpoint-sha256-1b_ttt_e2e_finetune_books_8k_1x_cc.txt   # if you have it

bash scripts/bootstrap_gpu_box.sh               # ~30-50 min, mostly `uv sync`
DEADLINE_HOURS=3 bash scripts/run_gpu_session.sh   # unattended; set to the hours LEFT in the booking
```

The org repo is private, so cloning needs a credential. Use a **fine-grained read-only token
scoped to this one repo**, and revoke it after the session. Or clone the public `origin`,
which holds the same history.

**Bootstrap, and what it stops on:** a driver too old; JAX unable to see or compute on the GPU
(this is the "kill inside 30 minutes" rule, automated); a checkpoint without an orbax step
directory; a checksum mismatch; a store missing `zarr.json` or `train/zarr.json`; a dirty vendor tree.
Every step is idempotent. Re-running after a fix is the intended repair.

**`run_gpu_session.sh` runs, in order:**

| Phase | What | Nominal (H100) |
|---|---|---|
| smoke | reshape to 131,073 tokens (2 batches), run the real command | compile 5–20 min |
| eval-1 | reshape to 150M tokens, baseline | ~35 min + compile |
| eval-2 | identical command, for bar S2 | ~35 min + compile |
| control | `dummy_dataset=true`, for bar S3 | ~10 min + compile |
| collect | redact secrets, copy into `results/session-<utc>/`, score against TOLERANCE.md | 1 min |

**Every run compiles from scratch.** Each eval is a new process. The XLA cache directory is
wired through (`backend.compilation_cache_dir`, because `train.py:278` ignores the env var),
but it does not save the trace-and-compile of the model. Budget it per run.

**What it handles on its own:**
- **OOM on the smoke pass:** retries once at eval batch 4 (`global_batch_size=1 eval_batch_size=4`). Every later run keeps that batch, and the summary records `RUN CONDITION CHANGED`.
- **Overrunning:** the runbook's drop order (§4) applied against `DEADLINE_HOURS`. Run 2 goes first, then the control, and each is recorded as UNVERIFIED, never as passed.
- **A crash:** stops, prints the log tail, and tells you how to resume, e.g. `START_AT=control bash scripts/run_gpu_session.sh`.
- **Peak GPU memory:** sampled every 5 s into the summary. This is the COST_MODEL §9.4.1 measurement.

---

## 3. Stop rules, decided in advance so nothing is a judgement call at 2am

- **Bootstrap Step 3 fails** (JAX cannot run on the GPU): release the box. Do not debug a broken image while it bills.
- **12 GPU-hours without a completed eval:** release it, write down what broke, re-book.
- **Cumulative Phase 0.5 spend past $325:** raising a cap is a written decision by the Lead.
- **A FAIL verdict stops the session.** Nothing downstream is attributable until the baseline passes.

## 4. Drop order, if overrunning

1. **Baseline run 2.** S2 is recorded as unverified.
2. **The negative control.** S3 is recorded as unverified.
3. **Token count.** Reshape smaller with `bash $EXP_DIR/bootstrap/reshape-to-real.sh <tokens>` and use `START_AT=eval-1`.

**Never drop the copy-out.**

---

## 5. Acceptance: printed for you, but read the bar first (Rule 5)

`scripts/check_baseline_acceptance.py` runs at the end and writes `ACCEPTANCE.txt`:

| Check | Pass condition (TOLERANCE.md) |
|---|---|
| Band | `2.314 < train_holdout/loss < 2.805` |
| S1 | mean NLL over the last 1024 positions < mean over positions 128–1152 |
| S2 | run 2 matches run 1 to 4 decimals (|diff| < 5e-5) |
| S3 | the control lands above the band. **Read the value yourself** against "order 10–12 nats"; the script invents no cut-off |
| S4 | config echo says `dataset_name: books3`; no `trustgate` in any log |
| §4.3 | 2.60–2.70, **non-binding**. Outside it means investigate, not FAIL |

Exit 0 = PASS, 1 = FAIL, 2 = NOT YET PASS (something unverified).

---

## 6. Copy-out, then release

The vendor logs **every environment variable and the W&B key** (`train.py:80-81`), and this
repo is public. `collect_results.py` redacts them and then re-scans the copy, failing if any
secret survives. **Only copy the `results/session-<utc>/` directory off the box**, never the raw
`$EXP_DIR`. The copy is pulled from the laptop, so no GitHub credentials go on the box:

```bash
# on the laptop
scp -r <user>@<box-ip>:~/TTT/experiments/000-repro-baseline/results/session-<utc>  experiments/000-repro-baseline/results/
```

Then **release the box**, and write the tracked record. `results/` is git-ignored, so one file
must be added deliberately. It records: the number, the bar, the verdict, the checkpoint
`manifest_sha256`, the vendor and overlay SHAs, env versions, peak GPU memory, and any run
condition change. Fill in the Outcome column in `gpu-bookings.md`.

---

## 7. What this session is not

The **001 kill gate** does not run here. `cli.py` runs the spike only against a random-init
victim (`--random-init`); nothing yet joins it to a real `--checkpoint`, and `PREREGISTERED.md`
requires a passing baseline first regardless.
