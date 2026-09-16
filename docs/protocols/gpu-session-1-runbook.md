# GPU session 1: runbook

> **One page, in order.** Revised 2026-09-16 for the lowest-cost route. Manas takes over the
> data download (Jaykay is on leave; his 2026-09-14 probe was metadata only, so no files exist
> anywhere yet). Target: **the 1B Books @8K baseline in one E2E session of about 3 hours.**
> Tasks `P0-1` / `T1.4`–`T1.7`. Detail: `COST_MODEL.md` §4/§9, `TOLERANCE.md` §4, §5 and §8.

## Cost plan

| Item | Estimate | Note |
|---|---|---|
| GCP UPI prepayment | ₹500–1,000, **once** | Sits as credit on the billing account. Egress uses about ₹60–120 of it |
| Download egress | ~5.75 GB | 5.35 GB checkpoint plus **one** 0.4 GB `/val` chunk |
| **E2E A100 80 GB**, ~3 h | $2.10/h ≈ **₹650 incl. 18% GST** | Bad day, 4 h: ≈ ₹870 |

**Why these choices:**
- **A100 80 GB, not H100 ($2.69/h).** Much of this session is install and XLA compile, which don't get faster on a pricier card.
- **Not the L40S ($1.20/h, 48 GB).** It would save about ₹200, but 48 GB sits inside the 36–49 GB memory estimate. It could force the batch-4 fallback or fail outright.
- **Not Colab.** 40 GB, and it wipes the disk on disconnect.
- **50M eval tokens, not 150M** (`TOLERANCE.md` §8, second note). This saves about 1.5 GPU-hours and no bar moves.
- **The download happens on the box, in parallel with `uv sync`.** It adds no GPU minutes. Uploading 6 GB from home would bill 20–45 minutes.

---

## Part A: before renting (free, laptop)

### A1. Google Cloud billing, paid by UPI (one time)
1. Go to <https://console.cloud.google.com> and create a **project**, e.g. `ttt-egress`.
2. Go to **Billing → Create billing account**, country India, and choose **UPI** as the payment method. Google asks for a **prepayment (typically ₹500–1,000)**; pay it by scanning the QR code with your UPI app. Docs: [Use UPI for Google Cloud payments](https://docs.cloud.google.com/billing/docs/resources/upi-payment-india).
3. Link the billing account to the project, and note the **project ID** (not the name).
4. An India billing account must complete **identity verification within 30 days**, or Google suspends it. Do it now, while you remember.

### A2. Prove access for free: Google Cloud Shell (no install)
Open Cloud Shell (the `>_` icon, top right of the console). Upload `scripts/probe_gcs_access.sh`, then:
```bash
GCP_BILLING_PROJECT=<project-id> bash probe_gcs_access.sh
```
It must print `ckpt_1b_bytes: 5347020507`. This is metadata only (under $0.05). **If it fails, do not rent.** Fix billing first.

### A3. Prove W&B works (laptop, WSL or Git Bash)
```bash
WANDB_ENTITY=<you> WANDB_PROJECT=<project> WANDB_KEY=<key> \
  uv run --no-project --with wandb==0.19.9 python scripts/preflight_wandb.py
```
It must end with `PREFLIGHT OK`.

### A4. Prepare for SSH and the booking
- If you have no key yet, run `ssh-keygen -t ed25519`. You'll paste `~/.ssh/id_ed25519.pub` into E2E.
- Top up about **₹1,000** of E2E credit via UPI (the extra is headroom).
- Add a claim row to `gpu-bookings.md`.

---

## Part B: on the box (billing starts)

### B1. Rent (0:00)
In the E2E console:
- **GPU:** A100 80 GB, **on-demand, not spot**. A pre-empted spot box loses the session.
- **Image:** Ubuntu 22.04 with CUDA ≥ 12.8.
- **Disk:** ≥ 100 GB.
- **SSH key:** yours, added.

### B2. Log in, install gcloud, authenticate (0:05, ~10 min)
```bash
ssh root@<box-ip>
tmux new -s ttt                       # always; reattach later with: tmux attach -t ttt
nvidia-smi                            # "CUDA Version" must be >= 12.8, else destroy the box now
git clone --recursive https://github.com/Manas-Maahir/Trust-Gated-Fast-Weight-Updates-for-TTT-E2E-LLMs.git TTT
cd TTT && git checkout infra/gpu-session-1
curl -sSL https://sdk.cloud.google.com | bash -s -- --disable-prompts && export PATH="$HOME/google-cloud-sdk/bin:$PATH"
gcloud auth login --no-launch-browser   # open the printed link on your laptop, paste the code back
```

### B3. Bootstrap (0:15, ~35 min)
```bash
export GCP_BILLING_PROJECT=<project-id>
export WANDB_ENTITY=…  WANDB_PROJECT=…  WANDB_KEY=…
bash scripts/bootstrap_gpu_box.sh
```
What it does, in order:
- **Checks GCS access in seconds** before anything slow starts.
- **Downloads in the background** while `uv sync` installs the environment.
- **Proves JAX can compute on the GPU.**
- **Waits for the download, then fingerprints the checkpoint.**
- **Writes the eval commands.**

It must end with `Bootstrap complete`. To watch the download, open a second tmux window (`Ctrl-b c`) and run `tail -f ~/ttt-runs/bootstrap/fetch-*.log`.

**If it stops:**
- **At `gcs access`:** your project ID or billing is wrong. Fix it and re-run.
- **At Step 3** (JAX can't use the GPU): **destroy the box.** Don't debug on the clock.
- **Anywhere else:** fix it and re-run. Every step resumes.

### B4. The session, unattended (0:50, ~2 h)
```bash
DEADLINE_HOURS=2.5 bash scripts/run_gpu_session.sh
```
It runs the smoke test, baseline run 1, baseline run 2, then the negative control, and prints the verdict. Each run recompiles (5–20 min), and each 50M-token pass takes roughly 20–25 min on an A100.

You can detach with `Ctrl-b d` and close the laptop. **What it handles on its own:**
- **Out of memory:** retries once at eval batch 4.
- **Running late:** drops run 2, then the control, and records each drop.
- **A crash:** stops and prints a resume command, e.g. `START_AT=eval-2 bash scripts/run_gpu_session.sh`.

### B5. Copy out, sign out of Google, destroy the box (~2:50)
```bash
# on the box: the copy-out folder has the W&B key removed; revoke Google credentials
gcloud auth revoke --all
```
```bash
# on the laptop
scp -r root@<box-ip>:/root/TTT/experiments/000-repro-baseline/results/session-*  experiments/000-repro-baseline/results/
```
Check that `ACCEPTANCE.txt` and `SESSION_SUMMARY.txt` arrived, then **destroy the box in the E2E console**. Stopping it is not enough, because a stopped box still bills for storage. Copy **only** `results/session-*`, never the raw `~/ttt-runs`, whose logs contain the key.

---

## Stop rules, decided in advance

- **Step 3 fails, or `nvidia-smi` shows CUDA < 12.8:** destroy the box within 30 minutes.
- **The session passes 4 hours without a verdict:** copy out what exists, destroy the box, write down what broke.
- **The verdict is FAIL:** stop. Nothing downstream is attributable until the baseline passes.

## Acceptance (read the bar before the number: Rule 5)

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

## After

Fill in the Outcome column in `gpu-bookings.md`:
- the verdict and the loss;
- `peak_gpu_mib` from the summary;
- any `RUN CONDITION CHANGED` or `SKIPPED` line;
- the checkpoint `manifest_sha256`;
- what the session actually cost.

`results/` is git-ignored, so add one tracked file recording those deliberately.

## What this session is not

The **001 kill gate** does not run here. `cli.py` runs the spike only with `--random-init`, and `PREREGISTERED.md` requires a passing baseline first.
