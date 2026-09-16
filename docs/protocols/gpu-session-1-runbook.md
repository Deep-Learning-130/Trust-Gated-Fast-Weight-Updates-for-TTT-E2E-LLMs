# GPU session 1 — runbook

> **One page, in order.** Everything here is also in `gpu-queue.md`, `COST_MODEL.md` §4/§6/§9,
> `TOLERANCE.md` §5, `EVAL_ENTRYPOINT.md` §5 and `task-allocation.md` T1.4–T1.7. It is repeated
> rather than referenced because on a billing clock you should be reading one file.
>
> Written 2026-09-16, before the first booking. Target: **the 1B Books @8K baseline, in one
> 3–4 hour session.** Tasks `P0-1` / `T1.4`, `T1.5`, `T1.6`, `T1.7`.

---

## 0. Before you rent — none of this is fixable on the clock

| | Item | Where it comes from |
|---|---|---|
| ☐ | **W&B entity, project and API key**, proven with `wandb login` | T1.2 / P0-10. Mandatory: the vendor login runs *before* the eval branch and `training.log_wandb=false` does not avoid it (ADR-004 §3) |
| ☐ | **`gcloud` authenticated** against a billing project | Both buckets are requester-pays. COST_MODEL §9.1 — this is the blocker that has stopped every previous attempt |
| ☐ | Image verified **CUDA 12.8 / cuDNN 9.8** | COST_MODEL §6.2. A mismatch eats days |
| ☐ | Byte counts re-probed: `GCP_BILLING_PROJECT=… PROBE_ONLY=1 bash scripts/fetch_checkpoints.sh` | Free, metadata only |
| ☐ | **`TOLERANCE.md` read** — the bar, before any number exists | Standing Rule 5 |
| ☐ | **Booking row claimed** in `gpu-bookings.md` | The record is the lock. Claim it *before* the instance starts |

A booking with no task ID, no owner or no estimate is not a booking.

---

## 1. What to rent

**An 80 GB card. Not 40 GB.**

COST_MODEL §9.4 estimated 1B at ≈25 GB via unmodified `train.py` and suggested a 40 GB card
might do. That estimate assumed `global_batch_size=1`. It is wrong about the batch: the
Evaluator floors the eval batch independently of `global_batch_size`,

```python
ttt/train.py:211
global_batch_size=max(cfg.training.eval_batch_size,          # default 8
                      cfg.training.global_batch_size // cfg.training.accum_steps * 4)
```

so the eval runs at **batch 8** whether `global_batch_size` is 1 or 2, and the two
batch-dependent terms below are 8× what §9.4 assumed.

| Term | Size | Source |
|---|---|---|
| Params, fp32 | 5.9 GB | 1.467B × 4 B (COST_MODEL §2.1) |
| **Outer AdamW state** | **11.7 GB** | `train.py:180` runs `optimizer_outer_loop.init(...)` even in eval mode — `load_part="params"` means `opt_state` is absent from the restore, so it is built from scratch and then never used |
| Dtype-cast model copy | 0–5.9 GB | `transformer.py:685`, at `state_dtype=fp32`; XLA may elide it |
| Adapted fast weights, ×8 batch | 5.1 GB | 160.4M × 4 B × 8, vmapped at `loop.py:29` |
| Per-chunk logits + fp32 log-softmax | ~10.5 GB | `[8, 1024, 128256]`; `loss.py:18` casts to fp32 before `log_softmax` |
| Prefix output + embeddings | ~1.1 GB | `[8, 8192, 2048]` |
| Remat'd chunk activations | 2–4 GB | bounded by `scan_remat_chunk` + `eqx.filter_checkpoint` |
| **Total** | **≈ 36–49 GB** | |

A 40 GB card sits inside that range, and the range has never been measured. **Record the
actual peak this session** (`nvidia-smi --query-gpu=memory.used`) and §9.4 can be settled with
evidence, in the right direction: downsize the *next* booking, do not gamble this one.

**Recommendation, at COST_MODEL §9.3's JarvisLabs rates:**

| | Rate | 4 h | Verdict |
|---|---|---|---|
| **H100 80 GB** | ₹255/hr | **≈ ₹1,020 (~$12)** | **Rent this.** COST_MODEL §3's throughput anchors are H100 prefill latencies, so the time estimates apply directly instead of being extrapolated — you get roughly 2× the tokens in a fixed window for 1.8× the rate |
| A100 80 GB | ₹141/hr | ≈ ₹564 (~$7) | Fine, but expect ~2× the wall-clock per pass. Halve `VAL_TOKENS` |
| A100 40 GB | ₹84/hr | ≈ ₹336 (~$4) | **No.** Saves ~₹460 and risks the whole booking on an unmeasured estimate |

All three are far inside the **$150** Phase 0.5 cap. The binding constraint on this session is
the **12-GPU-hour stop rule**, not money.

---

## 2. Bootstrap

```bash
git clone --recursive <org remote> TTT && cd TTT
git checkout infra/gpu-session-1        # or main, once this has merged

export GCP_BILLING_PROJECT=...  WANDB_ENTITY=...  WANDB_PROJECT=...  WANDB_KEY=...
export DATA_ROOT=/path/to/persistent/data
export EXP_DIR=/path/to/persistent/runs   # MUST be outside the repo; the script enforces it

bash scripts/bootstrap_gpu_box.sh
```

Idempotent — re-running after a partial failure is the intended repair. It installs `uv` and
the gcloud SDK if missing, pins the vendor submodule, syncs the vendor env, fetches the
checkpoint, builds a truncated `/val`, and writes five scripts into `$EXP_DIR/bootstrap/`.

**Set `EXP_DIR` and `DATA_ROOT` to whichever JarvisLabs path actually persists.** The default
is `$HOME`, which on some images is ephemeral.

---

## 3. The session

Times are nominal on an H100. The smoke pass is where the four known first-launch failures
happen; budget for them there, not in the real run.

| | Step | Command | Nominal |
|---|---|---|---|
| 0:00 | Claim booking row; `nvidia-smi`; check CUDA/cuDNN | | 0:15 |
| 0:15 | Bootstrap (installs, `uv sync --frozen`, checkpoint, val subset) | `bash scripts/bootstrap_gpu_box.sh` | 0:50 |
| 1:05 | **Smoke pass** — 2 eval batches, same command as the real run | `bash $EXP_DIR/bootstrap/1-smoke-*.sh` | 0:30 |
| 1:35 | Read tokens/sec off the smoke pass; size the real run | `bash $EXP_DIR/bootstrap/reshape-to-real.sh <tokens>` | 0:05 |
| 1:40 | Baseline, run 1 | `bash $EXP_DIR/bootstrap/2-eval-*.sh` | 0:35 |
| 2:15 | Baseline, run 2 — **unchanged**, for bar S2 | same command again | 0:35 |
| 2:50 | Negative control, bar S3 | `bash $EXP_DIR/bootstrap/3-dummy-control-*.sh` | 0:10 |
| 3:00 | Copy results out; fill Outcome; release | §6 below | 0:15 |

### Sizing the real run

`VAL_TOKENS` is the only knob that sets eval wall-clock, and it is not a config value — it is
the zarr array's declared shape, which `lm_dataset.py:27` divides by `seq_len` to get the
batch count. Change it with a reshape; **nothing is refetched and the XLA cache still hits**,
because batch size and sequence length do not change.

```bash
python3 scripts/make_val_subset.py --dest $DATA_ROOT/llama3-books3 reshape --tokens 150000000
python3 scripts/make_val_subset.py --dest $DATA_ROOT/llama3-books3 status
```

Pick the number so **two** passes fit the time remaining. Default 150M ≈ 18,310 sequences.
`reshape` refuses a shape larger than the chunks on disk — absent zarr chunks read as fill
value 0, which would evaluate the model on padding and return a confident, meaningless loss
with no error at all.

### If the smoke pass OOMs

Lower the eval batch from 8 to 4 — it needs **both** overrides, because of the `max()`:

```
training.global_batch_size=1 training.eval_batch_size=4
```

That halves the two batch-dependent memory terms (~10.5 GB → ~5.2 GB, and 5.1 GB → 2.6 GB).
Record it in the Outcome: it changes the run condition, though not the quantity being
estimated.

---

## 4. Stop rules — already decided, so they are not judgement calls at 2am

- **12 GPU-hours without a completed eval.** Release, write down what broke, re-book. Session
  2 is always cheaper than hour 13.
- **Wrong environment** (CUDA mismatch, driver failure): kill it **inside 30 minutes**. Do not
  debug a broken image while it bills. Pick a different image.
- **Cumulative Phase 0.5 spend past $325.** Raising a cap is a written decision by the Lead.

### Drop order if you are overrunning

In this order, and record each omission in the Outcome:

1. **Baseline run 2** — record bar S2 as *unverified*. An honest recorded gap beats a lost session.
2. **The negative control** (bar S3).
3. **Token count** — reshape smaller and rerun.

**Never drop the copy-out.** `results/`, `*.npy` and `wandb/` are all git-ignored; a run that
is not deliberately transcribed leaves no record, and re-running costs another booking.

---

## 5. Acceptance — read the bar before the number (Rule 5)

`TOLERANCE.md` §4.1 and §5. All five, and the structural ones do most of the real work:

| | Check | Pass condition |
|---|---|---|
| ☐ | **Band** | `2.314 < train_holdout/loss < 2.805` nats/token |
| ☐ | **S1 — monotonicity** | per-token NLL falls monotonically across the sequence |
| ☐ | **S2 — determinism** | the identical command, rerun, agrees to **4 decimals** |
| ☐ | **S3 — negative control** | `dummy_dataset=true` lands **far above** the band |
| ☐ | **S4 — resolved config** | the run's own echo shows `dataset_name == books3` |
| ☐ | Gate absent | `trustgate.vendor_patch.is_installed()` is `False`; no `trustgate` import anywhere in the run (T1.6) |

Inside the band but outside **2.60–2.70** is a non-binding expectation miss (§4.3) —
investigate before recording, but it is not a FAIL.

**A FAIL stops the session.** Nothing measured downstream is attributable until the baseline
passes; that is the entire reason experiment 000 exists.

---

## 6. Copy-out, then release

```bash
R=experiments/000-repro-baseline/results
cp $EXP_DIR/demo/eval-<ckpt>/train_holdout_token_nll_loss.npy  $R/
cp $DATA_ROOT/llama3-books3/val-subset-manifest.json           $R/
cp $EXP_DIR/bootstrap/env-record.txt                           $R/
cp $EXP_DIR/bootstrap/bootstrap-*.log                          $R/
```

Then write a single tracked file recording: **our number, the bar, PASS/FAIL**, the checkpoint
sha256, the vendor SHA, the overlay SHA, the env versions, the peak GPU memory, and the exact
commands if they differed from the generated ones. `results/` is git-ignored — this file has
to be added deliberately.

The manifest is not optional bookkeeping. The baseline was measured over a **subset** of
`/val`; without the manifest the number cannot name the tokens it was computed over, and
`TOLERANCE.md` §8's dated note is the other half of that record.

**Fill in the Outcome column in `gpu-bookings.md`.** What ran, what it produced, what broke,
how far it got, and anything the next holder must know. Two honest sentences beat a clean
blank. **The box is not free until that is done.**

---

## 7. What this session is not

The **001 kill gate** does not run here. `cli.py main()` still raises `SystemExit` on any
non-`--dry-run` invocation — every piece exists (`vendor_bind.bind`,
`harness.make_adapt_and_eval`, `craft.craft_stream` for SELECT+DEGRADE, `run_attack_spike`,
`report.write_report`) but nothing joins them to `--checkpoint`. That wiring is a separate
branch with CPU tests, done cold. `PREREGISTERED.md` requires a passing baseline first
regardless.
