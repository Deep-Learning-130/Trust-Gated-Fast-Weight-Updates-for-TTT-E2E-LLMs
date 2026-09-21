# GPU session 1: runbook

> **One page, in order.** Rewritten 2026-09-20. The previous version routed around a
> Google Cloud lockout and a UPI-only payment constraint: a teammate issued a
> **service-account key**, the box downloaded with it, and the compute ran on **E2E
> Networks**. Both constraints are gone. Jaykay has a working, KYC-upgraded GCP account,
> and the run happens **inside their project** — in-project GCS reads, ambient credentials,
> GCP accelerator, and **no key on rented hardware**.
>
> **Scope changed too.** This is no longer a baseline-only session. It is
> **one-and-done**: 000 baseline, then the 001 kill gate, then the sequence-position arms,
> then the Phase 2 gate measurement. The expensive instance starts only after everything
> free has already passed.
>
> Detail: `COST_MODEL.md` §4/§9, `TOLERANCE.md` §4, §5, §8, `PREREGISTERED.md`.

## Cost plan

| Item | Estimate | Note |
|---|---|---|
| Checkpoint + `/val` download | ~5.75 GB, **$0** | In-project, same-region. The old plan's ₹60–120 egress line does not exist any more. |
| **A100 80 GB** (`a2-ultragpu-1g`), on-demand | ~$4–5/h | Confirm at the console; it varies by region. |
| 000 baseline | ~1 h | 50M tokens, two runs plus the control |
| 001 kill gate | **see the warning below** | The one number nobody has measured |
| Sequence arms + Phase 2 | ~1–2 h | |
| **Session total** | well under the credit | Money is not the binding constraint. Quota is. |

**Why these choices:**
- **A100 80 GB, not H100.** `PREREGISTERED.md`'s frozen Setup names "single A100/H100
  80GB", so either is in spec. Much of this session is install and XLA compile, which do
  not get faster on a pricier card.
- **On-demand, not Spot.** Spot is roughly a third of the price and a preemption mid-run
  breaks the one-and-done requirement. Do not.
- **50M eval tokens, not 150M** (`TOLERANCE.md` §8, second note). Saves about 1.5 GPU-hours
  and no bar moves.
- **Download on the box, in parallel with `uv sync`.** It adds no GPU minutes.

> **THE COST YOU HAVE NOT MEASURED.** One ordering-search proposal is one full
> adapt-and-eval: roughly **16 inner steps plus 2 prefix passes** at 8192/1024. With
> `--max-iters 40` and 5 seeds that is ~3,280 inner steps at 1B, plus a GPT-2 scoring of
> both arms per proposal. **This is the dominant cost of the session and nothing has ever
> timed it.** Step C2 measures one evaluation and sets the budget from that number. Do not
> start the search on a guess.

---

## Part A: before renting (all free)

Every item must pass. If one fails, nothing gets rented.

### A0. The GCP account is actually able to rent a GPU (Jaykay, ~10 min, possibly ~1 day)

**This is the schedule risk, not the money.** Upgrading past the free trial makes the
account *eligible* to ask for GPUs; it does not grant any.

1. **Billing → Credits.** Record the **actual remaining credit and its expiry date**. The
   90-day trial clock does not reset on upgrade. That number, not "$300", is the budget.
2. **IAM & Admin → Quotas.** Filter for `NVIDIA A100 80GB GPUs` in the intended region.
   The default is **0**. Request an increase now — it can take a day and it can be refused.
3. Confirm `a2-ultragpu-1g` capacity in a region that **also** holds the checkpoint bucket.
   Same-region reads are free; cross-region reads are not.
4. Note that GST is billed to the payment method separately and is not offset by credit.

**If step 2 is not granted, nothing below can run.** Say so immediately rather than
working around it.

### A1. In-project access works (Jaykay, ~2 min)

No service-account key. No key on rented hardware. From Cloud Shell or any box with
ambient application-default credentials in the project:

```bash
gcloud config get-value project
gcloud storage du -s gs://ttt-e2e-checkpoints/1b_ttt_e2e_finetune_books_8k_1x_cc
gcloud storage ls gs://llama3-books3/val | head
gcloud storage ls gs://llama3-books3/train/zarr.json
```

The first must print `5347020507`. **The last one matters more than it looks:**
`_make_train_iterator` runs *before* the eval branch in `train.py`, so a missing
`train/zarr.json` kills the 000 baseline during setup, long after billing starts.

`scripts/probe_gcs_access.sh` is **stale in premise** — it exists so a third party need not
clone a private repo, and that is no longer the situation. Its `train/zarr.json` check is
the part worth keeping.

### A2. The CPU suite passes on the laptop

```bash
PYTHONPATH=src JAX_PLATFORMS=cpu .venv/Scripts/python.exe -m pytest
```

As of 2026-09-20: **473 passed** with the reference model fetched, **462 passed / 11
skipped** without (what CI sees). Quote both, per `branch-and-review.md` item 2, and update
that line if the counts change. **Use `.venv/Scripts/python.exe`** — the system interpreter
has no `equinox` and every test errors at collection.

### A3. The frozen bars still match the document

```bash
PYTHONPATH=src JAX_PLATFORMS=cpu .venv/Scripts/python.exe -m pytest tests/test_thresholds.py
```

Ten tests. This is the pin proving the three bars survived the 2026-09-20 revision.

### A4. Hygiene

```bash
git -C vendor/ttt-e2e status        # must be clean; ADR-002, any diff blocks merge
```

Then run the `hygiene` job's checks locally with the updated `ci.yml`.

### A5. Rehearse everything that is free

```bash
python experiments/002-pilot-tiny-ttt/run_pilot.py
python experiments/002-pilot-tiny-ttt/run_deep.py
python experiments/002-pilot-tiny-ttt/run_null.py
python experiments/002-pilot-tiny-ttt/run_sequence.py
```

All CPU, all $0, ~40 s each. Pick `--eval-every` here: four arms × 50 windows × 5 seeds is
roughly an order of magnitude more measurement than the spike's single endpoint.

### A6. Rehearse the real path against a dummy checkpoint

```bash
# the weightless victim, end to end
python -m trustgate.eval.cli --objective degrade --strategy select \
  --random-init --size 125m --out /tmp/spike-smoke

# the checkpoint code path, against a locally written orbax checkpoint
python scripts/make_dummy_checkpoint.py /tmp/dummy-ckpt   # if absent, write it
python -m trustgate.eval.cli --objective degrade --strategy select \
  --checkpoint /tmp/dummy-ckpt --corpus-file /tmp/c.npy --eval-file /tmp/e.npy \
  --size 125m --out /tmp/ckpt-smoke
```

**This is what catches the silent partial restore on the ground rather than at $5/h.**
`build_from_checkpoint` refuses a restore that left weights at their random-init values;
prove that refusal fires before you trust it with real weights.

### A7. Add a claim row to `gpu-bookings.md`, and confirm Jaykay is available

Jaykay must be reachable **for the whole session**, not only the download.

---

## Part B: on the box (billing starts)

### B1. Create the instance (0:00)

```
Machine type : a2-ultragpu-1g   (1 x A100 80GB)
Provisioning : on-demand, NOT Spot
Image        : Deep Learning VM with CUDA >= 12.8, or Ubuntu 22.04 + CUDA >= 12.8
Disk         : >= 200 GB   (checkpoint, dataset, and two runs of artifacts)
Scopes       : default service account with storage read
Region       : the one A0 step 3 settled on
```

### B2. Log in and clone (0:05)

```bash
gcloud compute ssh <instance> --zone <zone>
tmux new -s ttt                       # always; reattach with: tmux attach -t ttt
nvidia-smi                            # CUDA Version must be >= 12.8, else delete the instance now
git clone --recursive <repo-url> TTT
cd TTT && git checkout infra/gpu-session-1
```

**No `scp` of any credential.** The instance's own service account reads the bucket.

### B3. Bootstrap (0:10, ~35 min)

```bash
export WANDB_ENTITY=…  WANDB_PROJECT=…  WANDB_KEY=…
export GCP_BILLING_PROJECT=<Jaykay's project id>   # the bootstrap requires it for the requester-pays reads
bash scripts/bootstrap_gpu_box.sh
```

**Leave `GCP_SA_KEY` unset.** On a GCE VM the instance's default service account is
already the active gcloud account, so the script's active-account check at `:176` passes
without a key. The key path is left over from the old route and is dead in-project. **Keep its check at `:329-344`**: the checkpoint directory
must contain an integer-named step directory, or orbax reports "No checkpoints found",
which reads like a missing checkpoint rather than a wrong path.

It must end with `Bootstrap complete`.

**If it stops:**
- **At the GCS check:** wrong project, wrong region, or missing storage scope on the
  instance. Fix and re-run.
- **At the JAX-on-GPU step:** **delete the instance.**
- **Anywhere else:** fix and re-run. Every step resumes.

### B4. Fingerprint before trusting the weights

```bash
bash scripts/fingerprint_checkpoint.sh
```

Pass the resulting `manifest_sha256` to every later run via `--checkpoint-manifest`. Without
it the report records the run as `UNFINGERPRINTED`, which is honest but weaker: a bare path
names a directory that may have changed, and the hash names the bytes.

---

## Part C: the run, in order

### C1. 000 baseline — a hard gate, not a formality

```bash
DEADLINE_HOURS=2.5 bash scripts/run_gpu_session.sh
```

**If the unmodified upstream eval does not reproduce on this hardware, every corruption
measured afterwards is unattributable** — it might be our own misconfiguration. Do not
proceed past a FAIL. Bar and structural checks are in the Acceptance table below.

### C1b. Pull the gate code and dump the token files (after 000 PASS)

```bash
git pull                                            # the gate wiring may have landed during 000
source "$EXP_DIR/bootstrap/session.env"             # BOOKS3_LOCAL, CKPT_DEST, CKPT_MANIFEST
( cd vendor/ttt-e2e && uv run python ../../scripts/dump_tokens.py     --books3 "$BOOKS3_LOCAL" --billing "$GCP_BILLING_PROJECT" --out "$EXP_DIR/tokens" )
T=$EXP_DIR/tokens
```

Writes `train.npy` (attacker corpus, /train chunk 0), `val.npy` (the benign eval, one
8193-token sequence), `probe.npy` (the gate's probe window, disjoint from the eval) and
`tokens-manifest.json`. It refuses a constant slice, which is the zarr fill-value trap.

**Then prove the gate compiles on the real vendor, for free, before any real run:**

```bash
python -m trustgate.eval.cli --objective degrade --strategy select --gate-eval   --random-init --size 125m --stream-tokens 8192 --seeds 0 1   --overhead-repeats 3 --out /tmp/gate-smoke
```

Banner-marked instrument validation. It has to finish with `gate.md` and a
**non-zero** divergence range under Calibration. All-zero or all-unscorable means the
probe forward is wrong, so stop and fix it before C5. The inner step is now compiled
with `eqx.filter_jit`. If a trace fails, `TRUSTGATE_NO_JIT=1` runs it eagerly for
diagnosis only; never time anything under it.

### C2. Measure one adapt-and-eval before budgeting the search

Run the C3 command below with `--seeds 0 --max-iters 1` and time it. Multiply by
seeds × proposals. **Choose `--max-iters` from that number**, not from `run_deep.py`'s
40, which was picked on CPU against a tiny model and does not transfer.

### C3. 001 kill gate

```bash
python -m trustgate.eval.cli \
  --objective degrade --strategy select \
  --checkpoint <dir> --checkpoint-manifest <manifest> \
  --corpus-file $T/train.npy --eval-file $T/val.npy \
  --corpus-split train --eval-split val \
  --size 1b --seq-length 8192 --stream-tokens 8192 \
  --seeds 0 1 2 3 4 --max-iters <measured> \
  --out experiments/001-attack-spike/results
```

Five seeds per condition, SELECT headline, DEGRADE objective, `meta` mode only (enforced in
`vendor_bind.bind`, no flag). All three bars required: Cohen's *d* ≥ **0.8**, relative
degradation ≥ **10%**, fluency ratio ≤ **1.5** under the independent reference model. **The
fluency scorer must not be the victim.**

If the search accepts nothing, the CLI exits with `NULL RESULT (not an error)`. That is a
real finding — under SELECT the attacker's only lever is ordering — and it is reported as a
null, not retried at a kinder setting.

### C4. Sequence-position arms — secondary, non-gating

Add `--sequence-eval`. Renders no verdict line at all, by design, so the two artifacts
cannot be confused.

C3 writes `arms.pkl` (the crafted streams, by seed) into its `--out` **the moment the
search ends**, before the spike itself runs. C5 re-runs exactly those streams.

### C5. Phase 2 gate measurement

```bash
python -m trustgate.eval.cli   --objective degrade --strategy select --gate-eval   --checkpoint <dir> --checkpoint-manifest <manifest>   --corpus-file $T/train.npy --eval-file $T/val.npy --probe-file $T/probe.npy   --arms-file experiments/001-attack-spike/results/arms.pkl   --size 1b --seq-length 8192 --seeds 0 1 2 3 4   --gate-quantiles 0.9 0.99   --out experiments/001-attack-spike/results/gate
```

This calibrates the anchor gate on a clean, uncrafted corpus slice. It reads thresholds
off that slice's divergence quantiles, and those are **pre-verdict operating points, not
tuned values**. Then it measures, into `gate.md`:
- gate overhead against the ~10% budget (median of 20, compiled, warm);
- `clean_regression` on the control arms at each threshold;
- gated vs ungated corruption (d, relative degradation) and acceptance rates on the 001
  arms.

Cost: seeds × 2 arms × (1 + number of thresholds) adapt-and-evals, with no search. Budget
it from the C2 timing. **Report the overhead number, not just the pass/fail bool**: a
marginal pass is inside the noise of any wall-clock measurement. The probe set is fixed,
not rotating, and `gate.md` says so.

Bounded drift is **not** measured here and must not be reported as enforced: there is no
carry slot for the accumulator (ADR-003 correction, ADR-P3-1).

Write every artifact to disk as it is produced. A crash at hour two must not restart hour
one.

---

## Part D: after

```bash
python scripts/collect_results.py     # scrubs secrets
```

Copy **only** `results/session-*`. The raw `~/ttt-runs` logs contain the W&B key.

**Delete the instance.** Stopping is not enough — a stopped instance still bills for its
disk.

Then update:
- `ORIENTATION.md` §9's status table and the spend figure;
- `gpu-bookings.md`: verdict, loss, `peak_gpu_mib`, any `RUN CONDITION CHANGED` or
  `SKIPPED` line, the checkpoint `manifest_sha256`, and what the session actually cost.

`results/` is git-ignored, so add one tracked file recording those deliberately.

---

## Stop rules, decided in advance

- **A100 quota not granted:** the session does not happen. Do not improvise onto a smaller
  card; `PREREGISTERED.md`'s frozen Setup names A100/H100 80GB, and deviating is itself a
  documented deviation.
- **`nvidia-smi` shows CUDA < 12.8, or JAX cannot use the GPU:** delete the instance within
  30 minutes.
- **000 returns FAIL:** stop. Nothing downstream is attributable.
- **The session passes its deadline without a verdict:** copy out what exists, delete the
  instance, write down what broke.

## Acceptance for 000 (read the bar before the number: Rule 5)

`ACCEPTANCE.txt`, per `TOLERANCE.md`:

| Check | Pass condition |
|---|---|
| Band | `2.314 < train_holdout/loss < 2.805` |
| S1 | mean NLL over the last 1024 positions < mean over positions 128–1152 |
| S2 | run 2 matches run 1 to 4 decimals |
| S3 | the control is above the band. **Read the value yourself** against "order 10–12 nats" |
| S4 | `dataset_name: books3`, and no `trustgate` in any log |
| §4.3 | 2.60–2.70, **non-binding** |

Exit 0 = PASS, 1 = FAIL, 2 = NOT YET PASS (something unverified).

## Report the verdict as it came out

`PREREGISTERED.md` says publishing a STOP is the honest outcome. With Phase 2 already
built, a STOP also writes off that work, so **the temptation to soften one is higher than
it has ever been.** That is the reason this paragraph is in the runbook and not only in the
pre-registration.
