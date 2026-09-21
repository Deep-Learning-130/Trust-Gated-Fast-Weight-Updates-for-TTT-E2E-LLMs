# GPU session 1: runbook

> **Revised 2026-09-21 (evening): E2E Networks, Manas operating.** Jaykay's A100 quota
> request was refused on submission, so the GPU is rented from **E2E Networks** (₹189/h
> + GST, paid by UPI). The data still comes from Google Cloud: both buckets are
> requester-pays, and the downloads (~6.5 GB, ~$1-2) are billed to **Jaykay's GCP
> project** through a service-account key that holds one role, Service Usage Consumer.
> Part B below is the E2E version. Parts C and D run unchanged, apart from how files come
> off the box. Everything written about the in-project GCP route is superseded where Part
> B says so. The step-by-step operator page is the "Session 1 Field Protocol"
> (https://claude.ai/artifact/GeBm35yqY6eMbqRZTwM8Tt).

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
| Checkpoint + `/val` + `/train` chunk 0 download | ~6.5 GB, **~$1–2** | Internet egress to India, billed to Jaykay's project via the service-account key. |
| **A100 80 GB** on E2E Networks, on-demand hourly | ₹189/h + 18% GST ≈ ₹223/h | UPI prepaid balance. A powered-off node still bills; only deleting stops it. |
| 000 baseline | ~1 h | 50M tokens, two runs plus the control |
| 001 kill gate | **see the warning below** | The one number nobody has measured |
| Sequence arms + Phase 2 | ~1–2 h | |
| **Session total** | ≈ ₹1,400–1,800 for 6–8 h | Top up the E2E balance for 8 h plus a buffer before starting. |

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

> **Superseded 2026-09-21:** the A100 quota request was refused, and the GPU now comes
> from E2E Networks (Part B). Jaykay's project only pays for the downloads.

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

> **Now done with the service-account key.** Jaykay runs the same checks in Cloud Shell
> with the key activated in a throwaway `CLOUDSDK_CONFIG` before sending it (the steps
> Manas forwarded). The expected output below is unchanged.

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

As of 2026-09-21: **510 passed** with the reference model fetched (the laptop). Without it
the fluency tests skip, which is what CI sees. Quote both, per `branch-and-review.md` item 2, and update
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

### B0. Before renting (Manas, on the laptop, free)

You need all of these before creating the node:
- **`sa-key.json` from Jaykay**, saved **outside the repo**, e.g. `C:\Users\manas\sa-key.json`
  (the repo is public, and the bootstrap refuses a key inside it). Jaykay has already
  proved it reads both buckets from his Cloud Shell. Also get **his project ID**.
- **W&B:** `preflight_wandb.py` printed `PREFLIGHT OK` for entity
  `manasmaahir27-vellore-institute-of-technology`, project `ttt-trustgate-session1`.
- **An SSH key** registered with E2E. In PowerShell, `ssh-keygen -t ed25519` if
  `~\.ssh\id_ed25519.pub` does not exist yet. Then add the `.pub` file's contents under
  MyAccount → SSH Keys.
- **E2E balance** topped up by UPI for about 8 hours (≈ ₹1,800) plus a buffer.
- A claim row in `gpu-bookings.md` (A7).

### B1. Create the node (0:00)

```
Console      : MyAccount -> Compute -> Nodes -> Create -> GPU tab -> NVIDIA A100 80GB
Plan         : 1x A100 80GB, hourly on-demand. NOT spot, NOT committed.
               The plan label reads <vCPU>-<RAM>-<GPU mem>-<CUDA>-<disk>-<IOPS>:
               pick one showing CUDA >= 12.8 and disk >= 200 GB.
OS           : Ubuntu 22.04 (24.04 also fine)
SSH key      : the one from B0
Backup (CDP) : off (extra cost, and nothing on the box needs it)
Public IP    : yes (uv sync, the GCS download and the GPT-2 fetch all need internet)
Security     : default group (SSH, port 22)
Location     : whichever of Delhi NCR / Chennai has an A100 80GB free
```

**A powered-off node still bills.** Only deleting it stops the meter (E2E's own docs).

### B2. Key onto the box, log in, clone (0:05)

On the laptop, in PowerShell. `root` is E2E's default login, and the node's page shows
the IP and the user:

```powershell
scp $HOME\sa-key.json root@<IP>:~/sa-key.json
ssh root@<IP>
```

On the box:

```bash
apt-get update -y && apt-get install -y git tmux    # prefix sudo if not root
tmux new -s ttt                        # always; reattach with: tmux attach -t ttt
nvidia-smi                             # A100 80GB, CUDA Version >= 12.8, else delete the node now
chmod 600 ~/sa-key.json
cd ~ && git clone --recursive https://github.com/Manas-Maahir/Trust-Gated-Fast-Weight-Updates-for-TTT-E2E-LLMs.git TTT
cd TTT && git checkout infra/gpu-session-1 && git log --oneline -1
```

The key sits at `~/sa-key.json`, outside `~/TTT`. That is the only credential file on
the box.

### B3. Bootstrap (0:10, ~35-45 min)

```bash
export WANDB_ENTITY=manasmaahir27-vellore-institute-of-technology WANDB_PROJECT=ttt-trustgate-session1
read -rs -p "W&B key: " WANDB_KEY; echo; export WANDB_KEY     # typed hidden, not in history
export GCP_BILLING_PROJECT=<Jaykay's project id> GCP_SA_KEY=$HOME/sa-key.json
bash scripts/bootstrap_gpu_box.sh
```

It installs `uv` and the Google Cloud SDK itself (neither is on an E2E image), activates
the key, and checks GCS access **before** anything slow. The download runs over the
public internet, so it is slower than in-region GCP but still minutes for ~6 GB. The W&B
key reaches the vendor through the environment and a private netrc; see commit
`a78c7c7` for why.

It must end with `Bootstrap complete`. Look for `gcloud : service-account key
activated` and `gcs access: checkpoint and /val readable, billed to <project>`.
**Keep its check at `:329-344`:** the checkpoint directory must contain an integer-named
step directory, or orbax reports "No checkpoints found", which reads like a missing
checkpoint rather than a wrong path.

**If it stops:**
- **At the GCS check:** the key, the project ID or Jaykay's role binding is wrong. Ask
  him to rerun the test lines he used before sending the key. Fix, then re-run.
- **At the JAX-on-GPU step (Step 3):** **delete the node.**
- **Anywhere else:** fix and re-run. Every step resumes.

`WANDB_KEY` and `GCP_BILLING_PROJECT` live only in this shell. C1 and C1b need them, so
run those in the same tmux window, or export them again first.

### B4. Fingerprint before trusting the weights

The bootstrap already fingerprinted the checkpoint (its Step 4). **Do not run
`fingerprint_checkpoint.sh` bare**: without `DEST` and `CKPT` it exits at once. Check the
manifest instead:

```bash
grep manifest_sha256 experiments/000-repro-baseline/results/checkpoint-sha256-1b_ttt_e2e_finetune_books_8k_1x_cc.txt
```

That must print one line with a 64-hex-digit hash. `phase1.env` passes that file to every
later run as `--checkpoint-manifest $CKPT_MANIFEST`. Without
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

### C1b. Prepare for 001 and the gate (after 000 PASS, ~10 min, no verdict)

```bash
git pull                                   # the gate wiring may have landed during 000
export GCP_BILLING_PROJECT=<project>       # same shell as the bootstrap
bash scripts/prepare_phase1.sh
source "${EXP_DIR:-$HOME/ttt-runs}/phase1.env"   # defines $TG $T $R $EXP_DIR $CKPT_DEST $CKPT_MANIFEST
```

**Before this, no interpreter on the box could run the checkpoint CLI.** `trustgate`
runs from `PYTHONPATH=src` and is installed nowhere. The vendor env lacks the fluency
scorer's `tokenizers` and `safetensors`. The reference model was never fetched, and the
`.npy` token files did not exist. The script fixes all four. Then it proves one
interpreter sees `ttt`, `trustgate` and the GPU, runs the fluency self-test, and writes
`$T/{train,val,probe}.npy`, refusing a constant slice (the zarr fill-value trap). Last,
it runs a **random-init 125m `--gate-eval` smoke** through the real vendor forward and
checks that its calibration divergence is non-zero and finite. If that check fails, the
probe forward is broken: **do not run C5**.

Always invoke the CLI as `$TG ...`. A bare `python -m trustgate.eval.cli` picks the
wrong interpreter. `$TG` lives only in the shell that sourced `phase1.env`. **In a new tmux
window, or after reattaching to a new shell, source it again first.**

When a checkpoint loads, orbax prints `ERROR:absl: ... metrics not found` and
`Missing metrics for step ...`. Both are harmless: the checkpoint has no metrics item,
and only `model_weights` is restored. The line that matters is `[run] bound;` (or
`[gate] bound;`), which follows it.

The inner step is compiled with `eqx.filter_jit`. If a trace fails,
`TRUSTGATE_NO_JIT=1 $TG ...` runs it eagerly, for diagnosis only; never time anything
under it.

**Everything is saved as it is produced, and every run resumes.** Each checkpoint run
appends every finished evaluation to `<out>/ledger.jsonl`. Each record is fsynced and
keyed by the stream's tokens, the run condition (which carries the checkpoint fingerprint
and the eval digest) and the gate. **If C3, C4 or C5 dies (OOM, a dropped SSH session,
Ctrl-C), rerun the identical command with the same `--out`.** It replays what finished
and continues from the crash. For C3 that includes a search interrupted mid-seed, because
the search is seeded. If any input changes, the recorded values simply don't match, so a
wrong number is never reused. C3 also writes `search-log.jsonl` (one line per finished
seed's search) and, on a null, `NULL-RESULT.md`. Every command below tees its output to
`$R/logs/`.

### C2. Measure one adapt-and-eval before budgeting the search

```bash
time $TG --objective degrade --strategy select \
  --checkpoint $CKPT_DEST --checkpoint-manifest $CKPT_MANIFEST \
  --corpus-file $T/train.npy --eval-file $T/val.npy \
  --size 1b --seq-length 8192 --stream-tokens 8192 \
  --seeds 0 1 --max-iters 1 --early-stop-patience 0 --no-resume \
  --out $EXP_DIR/c2-timing 2>&1 | tee -a $R/logs/c2-timing.log
```

Read the `[run] adapt-and-eval #k: … s` lines. The first includes compilation; the
second and later ones are the per-proposal cost. A seed costs `max_iters + 1` of those
(the search scores the sampled ordering first), plus a GPT-2 fluency scoring per
proposal, plus 2 more for the spike itself. **Choose `--max-iters` so that
5 × (max_iters + 3) × per-eval time fits the time left.**

**This run can end in `NULL RESULT (not an error)`, and here that means nothing.** With
one proposal per seed, a rejected proposal is the likeliest outcome. C2 is a timing run:
its output is not the 001 result and must not be recorded as one. It uses two seeds so
that if the proposals are accepted it still finishes cleanly (`cohens_d` needs two per
group). **Choose `--max-iters` from that number**, not from `run_deep.py`'s
40, which was picked on CPU against a tiny model and does not transfer. Its `--out` is
deliberately not the C3 directory, and its report is not a result.

### C3. 001 kill gate

```bash
$TG --objective degrade --strategy select \
  --checkpoint $CKPT_DEST --checkpoint-manifest $CKPT_MANIFEST \
  --corpus-file $T/train.npy --eval-file $T/val.npy \
  --corpus-split train --eval-split val \
  --size 1b --seq-length 8192 --stream-tokens 8192 \
  --seeds 0 1 2 3 4 --max-iters <measured> \
  --out $R 2>&1 | tee -a $R/logs/c3-spike.log
```

Five seeds per condition, SELECT headline, DEGRADE objective, `meta` mode only (enforced in
`vendor_bind.bind`, no flag). All three bars required: Cohen's *d* ≥ **0.8**, relative
degradation ≥ **10%**, fluency ratio ≤ **1.5** under the independent reference model. **The
fluency scorer must not be the victim.**

If the search accepts nothing, the CLI exits with `NULL RESULT (not an error)`. That is a
real finding — under SELECT the attacker's only lever is ordering — and it is reported as a
null, not retried at a kinder setting.

C3 writes `arms.pkl` (the crafted streams, by seed) into its `--out` **the moment the
search ends**, before the spike itself runs. C4 and C5 re-run exactly those streams.
**Copy `$R` off the box now** (Part D's commands), before starting C4. It takes seconds,
and the search is the one result nothing else can recreate.

### C4. Sequence-position arms — secondary, non-gating

```bash
$TG --objective degrade --strategy select --sequence-eval \
  --checkpoint $CKPT_DEST --checkpoint-manifest $CKPT_MANIFEST \
  --corpus-file $T/train.npy --eval-file $T/val.npy \
  --arms-file $R/arms.pkl \
  --size 1b --seq-length 8192 --seeds 0 1 2 3 4 \
  --out $R/sequence 2>&1 | tee -a $R/logs/c4-sequence.log
```

This renders no verdict line at all, by design, so the two artifacts cannot be confused.
Against a checkpoint it requires `--arms-file`: the addendum holds the end-of-stream values
to the same quantity `corruption_metric` consumes, and that is only true on the spike's own
streams. The window count comes from those streams (8 at 8192 tokens), not from
`--windows`.

**Cost:** `1 + seeds × 4 arms × (windows / eval_every)` benign evaluations, each about
half an adapt-and-eval. At 8 windows and 5 seeds that is 161. It is non-gating, so if
time is short, run it **after** C5 or with `--eval-every 2`, which halves the cost.

### C5. Phase 2 gate measurement

```bash
$TG --objective degrade --strategy select --gate-eval \
  --checkpoint $CKPT_DEST --checkpoint-manifest $CKPT_MANIFEST \
  --corpus-file $T/train.npy --eval-file $T/val.npy --probe-file $T/probe.npy \
  --arms-file $R/arms.pkl \
  --size 1b --seq-length 8192 --seeds 0 1 2 3 4 \
  --gate-quantiles 0.9 0.99 \
  --out $R/gate 2>&1 | tee -a $R/logs/c5-gate.log
```

This calibrates the anchor gate on a clean, uncrafted corpus slice. It reads thresholds
off that slice's divergence quantiles, and those are **pre-verdict operating points, not
tuned values**. Then it measures, into `gate.md` (and `gate-result.pkl`):
- gate overhead against the ~10% budget (median of 20, compiled, warm);
- `clean_regression` on the control arms at each threshold;
- gated vs ungated corruption (d, relative degradation) and acceptance rates on the 001
  arms.

Cost: seeds × 2 arms × (1 + number of thresholds) adapt-and-evals, with no search, plus
one compile per threshold. Budget it from the C2 timing. **Report the overhead number, not
just the pass/fail bool**: a marginal pass is inside the noise of any wall-clock
measurement. The probe set is fixed, not rotating, and `gate.md` says so.

**If C3 returned `NULL RESULT`**, it wrote no `arms.pkl`: the search stops at the first
seed that accepts nothing. Skip C4, since sequence arms on uncrafted streams measure no
attack. For C5, replace `--arms-file …` with `--uncrafted-arms`. That builds uncrafted
orderings from `$T/train.npy`, so overhead and `clean_regression` are still real
measurements, and `gate.md` labels the corruption columns as not an attack.

Bounded drift is **not** measured here and must not be reported as enforced: there is no
carry slot for the accumulator (ADR-003 correction, ADR-P3-1).

Every artifact is written to disk as it is produced (the ledger, above). A crash at hour
two does not restart hour one: rerun the same command.

---

## Part D: after

```bash
python scripts/collect_results.py     # scrubs secrets
```

Copy **only** `results/session-*`. The raw `~/ttt-runs` logs contain the W&B key.

Also copy `experiments/001-attack-spike/results/`. It holds:
- `report.md` (the verdict) and `spike-result.pkl`;
- `arms.pkl`, `search-log.jsonl` and `ledger.jsonl`;
- `logs/`;
- `sequence/` and `gate/`, each with its `.md`, `-result.pkl` and `ledger.jsonl`.

These CLI runs set `training.log_wandb=false` and write no key, so they need no scrubbing.
Copy `$T/tokens-manifest.json` with them: it records which tokens were measured. On the
box:

```bash
cp $T/tokens-manifest.json $R/
tar czf ~/001-results.tgz -C "$(dirname $R)" results
```

The 000 results, also on the box, are only the scrubbed session folders and the
checkpoint fingerprint:

```bash
cd ~/TTT/experiments/000-repro-baseline/results && tar czf ~/000-results.tgz session-* checkpoint-sha256-*.txt
```

Then **on the laptop**, in PowerShell from the repo root:

```powershell
scp root@<IP>:~/000-results.tgz root@<IP>:~/001-results.tgz .
tar xzf 000-results.tgz -C experiments/000-repro-baseline/results/
tar xzf 001-results.tgz -C experiments/001-attack-spike/
```

Run that after C3, after C5, and once more at the end. Each copy takes seconds, and a
deleted or broken instance takes everything not yet copied with it. Open `report.md` and
`gate/gate.md` **on the laptop** before deleting the instance: a copy you have not opened
is not a copy. Both copies must happen **before the instance is deleted**.

**Delete the node:** MyAccount → Nodes → the node → Actions → Delete. **Powering off is
not enough**: E2E keeps billing a powered-off node, because its disk, CPU, RAM and IP stay
reserved. Refresh the node list and confirm it is gone.

**Then tell Jaykay to delete the service account.** That also kills the key:
`gcloud iam service-accounts delete ttt-data-reader@<project>.iam.gserviceaccount.com`.
Delete `sa-key.json` from the laptop too.

Then update:
- `ORIENTATION.md` §9's status table and the spend figure;
- `gpu-bookings.md`: verdict, loss, `peak_gpu_mib`, any `RUN CONDITION CHANGED` or
  `SKIPPED` line, the checkpoint `manifest_sha256`, and what the session actually cost.

`results/` is git-ignored, so add one tracked file recording those deliberately.

---

## Stop rules, decided in advance

- **No A100 80GB node available on E2E:** wait, or try the other location. Do not
  improvise onto a 40 GB or another card: `PREREGISTERED.md`'s frozen Setup names A100/H100
  80GB, and deviating is itself a documented deviation. An H100 80GB is in spec.
- **`nvidia-smi` shows CUDA < 12.8, or JAX cannot use the GPU:** delete the node within
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
