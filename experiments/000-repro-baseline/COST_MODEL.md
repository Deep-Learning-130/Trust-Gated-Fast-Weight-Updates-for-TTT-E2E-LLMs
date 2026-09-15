# Cost model and budget cap — first GPU booking

> Deliverable **D4** of `docs/superpowers/specs/2026-08-03-phase0-lead-prep-design.md`.
> Written 2026-08-08. **Nothing has been rented, probed, transferred or spent.**
> Depends on D1 (`EVAL_ENTRYPOINT.md` — which data the eval path touches) and
> D3 (`scripts/fetch_checkpoints.sh` — the probe that will replace §2's estimates).

**Headline:** the 1B baseline is **cheap** — order **$10–35**, of which egress is under
**$1**. The budget risk in this booking is not the bill, it is **burning a paid session on a
command that has never been executed**. §7's cap is sized accordingly. The 3B/128K run is a
different animal: roughly **10×** the cost, because it needs a multi-GPU node rather than a
single card (§5).

---

## 1. What the run actually touches

From ADR-004 / `EVAL_ENTRYPOINT.md`, settled and not re-derived here:

- **Checkpoint** — `gs://ttt-e2e-checkpoints/1b_ttt_e2e_finetune_books_8k_1x_cc`. May be read
  directly from `gs://` (`ttt/infra/checkpoint.py:83-84`), but we copy it locally anyway so
  the sha256 in `results/` describes bytes we actually hold.
- **Dataset** — `gs://llama3-books3` only, and only the **`/val`** array. **A local copy is
  required**: the loader opens `zarr.storage.LocalStore` (`ttt/dataloader/lm_dataset.py:14`)
  and has no GCS backend. No streaming is possible.
- **`gs://llama3-dclm-filter-8k` is NOT touched.** The extension config sets
  `dataset_name: books3`, and `dataset_path` resolves to exactly one bucket. Budgeting the
  DCLM bucket would roughly double a number we do not owe.

Both buckets are **requester-pays**: we pay egress, billed to `GCP_BILLING_PROJECT`.

> **Verified 2026-09-14, at zero cost and with no `gcloud` auth.** This was an
> assumption read off the vendor README; it is now measured. Anonymous
> `GET https://storage.googleapis.com/storage/v1/b/<bucket>` returns **HTTP 400,
> "Bucket is a requester pays bucket but no user project provided"** for all
> three of `ttt-e2e-checkpoints`, `llama3-books3` and `llama3-dclm-filter-8k`.
>
> Two things follow, and they point in opposite directions:
>
> - The requester-pays assumption underpinning this whole document is **correct**.
>   No free anonymous path to the checkpoint or the dataset exists.
> - A 400 rather than a 403 means the bucket *exists* and the endpoint is
>   reachable; what is missing is a billing project, not permission. Combined
>   with §4's estimate, the blocker on T1.3/T1.5 is **having any GCP billing
>   account at all — not the size of the bill**, which stays under $2.
>
> Still unverified, and still needing auth: the object *paths* inside the
> buckets, and the `/val` byte count (§2.2), which remains the one number that
> could move the cost model.

### 1.1 `/val` alone is not quite enough — and the reason is cheap, not expensive

`_make_train_iterator` runs at `ttt/train.py:125`, **before** the eval branch returns at
`:222-227`. It calls `zarr.open_array(store, path="/train")`. So the local store must contain
the **`/train` array's metadata**, or the run dies during setup — before eval, after billing
starts. A naive `/val`-only copy fails.

The fix costs nothing. The vendor pins `zarr>=3.0.4` (resolved 3.0.7 in `uv.lock`), and in
zarr v3 each array carries its own `zarr.json` while **absent chunks read as the fill
value**. So copying the group metadata, all of `/val`, and *only* `/train/zarr.json` gives a
store where `/train` opens with the correct shape and holds no data. `next()` is never called
on that iterator in eval mode, so nothing is read; and if grain's `prefetch_buffer_size=500`
does read eagerly, it reads zeros for free rather than crashing.

```
gsutil -u $P cp      gs://llama3-books3/zarr.json        $LOCAL/
gsutil -u $P cp      gs://llama3-books3/train/zarr.json  $LOCAL/train/
gsutil -u $P -m cp -r gs://llama3-books3/val             $LOCAL/
```

This supersedes ADR-004's framing of the grain prefetch as an egress risk. It is not a
billing risk; it is a **crash risk**, and the mitigation above removes it. D5 implements it.
**Unverified** — no bucket has been listed. If the layout differs (zarr v2, or a nested
store), fall back to the vendor README's full `gcloud storage cp -r`, and re-probe first,
because that is a much larger transfer.

---

## 2. Bytes — estimated, pending the probe

**None of this has been measured.** `gsutil du -s` is metadata-only and effectively free, so
every number in §2 is replaced by one free command as soon as `gcloud` auth exists. Run
`PROBE_ONLY=1 bash scripts/fetch_checkpoints.sh` first, always.

### 2.1 Checkpoint

Parameters counted analytically from the config the checkpoint was trained under
(`configs/model/1b.yaml` + `configs/experiment/1b/extension/ext-1b-e2e-32K.yaml`), stored at
`param_dtype = fp32` (`ttt/config.py:108`) with tied embeddings and no optimizer state
(vendor README: "Optimizer state isn't included").

| | embed | attn | MLP | prime (fast weights) | total | fp32 on disk |
|---|---|---|---|---|---|---|
| 760M | 197.0M | 226.5M | 368.1M | 92.0M | 0.884B | 3.53 GB |
| **1B** | 262.7M | 402.7M | 641.7M | **160.4M** | **1.467B** | **≈ 5.9 GB** |
| 3B | 328.3M | 838.9M | 1384.1M | 346.0M | 2.897B | ≈ 11.6 GB |

*Confidence.* The same formula puts the 760M model's fast-weight ("prime") parameters at
92.0M against the paper's stated **88M** (§2.3.1) — **+4.6%**. Treat every figure above as
±10%. Note also that `suffix_len` is 6 of 24 layers for 1B and 8 of 32 for 3B: exactly the
last quarter, as the paper's ablation concluded.

### 2.2 Dataset — the one genuinely unknown quantity

The `/val` token count is **not derivable** from the paper or the code. Tokens are int32
(4 B) inside a zarr array compressed with `BloscCodec(cname="zstd", clevel=3, shuffle)`;
`du` returns the compressed size, which is both the egress and the on-disk size. Llama-3
token IDs (< 128256) leave the top byte of each int32 zero, which byte-shuffle plus zstd
compress well — assume **≈ 2.5×**, so ≈ 1.6 B stored per token.

| `/val` tokens | stored ≈ | egress @ $0.12/GB | same-region GCP |
|---|---|---|---|
| 50 M | 0.08 GB | $0.01 | $0 |
| 250 M | 0.40 GB | $0.05 | $0 |
| 1 B | 1.60 GB | $0.19 | $0 |
| 2 B | 3.20 GB | $0.38 | $0 |

**Even at the top of this range the dataset costs under a dollar.** The `/val` size matters
for *wall-clock* (§3), not for egress. Plausible range is 50 M–1 B tokens: Figure 6 averages
per-token-index curves at 128 K context, which needs at least a few hundred sequences.

### 2.3 Total egress

| Item | GB | @ $0.12/GB |
|---|---|---|
| 1B checkpoint | 5.9 | $0.70 |
| books3 `/val` (250 M tokens) | 0.4 | $0.05 |
| Class A/B operations (a few thousand objects) | — | < $0.05 |
| **Total, 1B baseline** | **≈ 6.3** | **≈ $0.80** |

Zero if the GPU box is in GCP in the same region as the buckets. **Egress is not a
meaningful part of this budget** — which is worth knowing, because it means bucket-region
shopping is not worth anyone's time.

---

## 3. GPU hours for a full val pass

**The eval pass cannot be shortened by configuration.** There is no `num_eval_batches`, and
`repeat=False`, so it runs the entire `/val` split (`ttt/model/loop.py:98-109`). Wall-clock
is set by val-split size. The only cheaper path is `training.dummy_dataset=true`, which is a
plumbing smoke test on random tokens, not a measurement.

**Throughput anchors, both from the paper, both on an H100:** 760M TTT-E2E prefill =
**0.0086** sec per 1K tokens (§2.3.1, printed); 3B TTT-E2E = **0.0252** at 8K rising to
**0.0274** at 128K (Figure 1 right, read from the figure's vector data — the same extraction
that reproduces the abstract's "2.7× faster than full attention at 128K" as 0.0734/0.0274 =
2.67, which is what validates it). Interpolating linearly in parameter count puts the **1B at
≈ 0.0134** sec per 1K tokens.

| `/val` tokens | 1B @ 8K | 3B @ 8K | 3B @ 128K |
|---|---|---|---|
| 50 M | 0.19 h | 0.35 h | 0.38 h |
| 250 M | 0.93 h | 1.75 h | 1.90 h |
| 500 M | 1.86 h | 3.50 h | 3.81 h |
| 1 B | 3.73 h | 7.00 h | 7.61 h |
| 2 B | 7.45 h | 14.00 h | 15.22 h |

*(GPU-hours on one card. Multiply by devices for a multi-GPU node.)*

**Accuracy: ±2×, not ±10%.** These are the authors' prefill latencies, probably at batch 1,
measured on their setup; our eval runs batch 8, which amortises better, but also computes a
full-vocabulary loss. Directionally right, precisely wrong. The first real run replaces them.

---

## 4. The session, not the eval, is the cost

Eval is a fraction of a paid session. This is the table that actually sets the cap.

| Phase | Nominal | Bad day | Note |
|---|---|---|---|
| Provision, boot, drivers | 0.25 h | 1.0 h | CUDA 12.8 / cuDNN 9.8 mismatch eats days, not hours — verify the image *before* booking |
| `uv sync --frozen` | 0.5 h | 1.0 h | multi-GB JAX CUDA wheels |
| Checkpoint fetch (5.9 GB) | 0.2 h | 0.5 h | requester-pays, off-GCP is slower |
| Dataset fetch | 0.2 h | 0.5 h | §1.1 layout may need a retry |
| **First-launch debugging** | **1.5 h** | **4.0 h** | **the dominant term — see below** |
| XLA compilation | 0.3 h | 1.0 h | first trace of a 24-layer scan |
| Eval pass | 0.9 h | 3.7 h | §3, at 250 M / 1 B tokens |
| Copy results out | 0.15 h | 0.3 h | `.npy` and `results/` are git-ignored |
| **Total** | **≈ 4 h** | **≈ 12 h** | |

**Why first-launch debugging dominates.** The command in `EVAL_ENTRYPOINT.md` was derived
from source and **has never been executed**. Its three known failure modes each cost a cycle:
W&B credentials (`log_wandb=false` does not avoid the login — ADR-004 §3), the eval batch
silently being 128 unless `global_batch_size` is lowered, and `exp_dir` defaulting into our
own tracked tree. §1.1 adds a fourth. All four are written down, which is the entire point of
this prep package — but written down is not the same as tested.

**The cheapest possible de-risking is not on this list because it is not a GPU cost:**
rehearse on `125m_ttt_e2e_finetune_books_8k_1x_cc` (0.5 GB, minutes to eval). It exercises
W&B, Hydra overrides, the zarr layout, `exp_dir`, and the results copy-out — every failure
mode above except memory. **Do the 125M rehearsal first, in the same session, before the 1B
run.** It has no pre-registered numeric bar (TOLERANCE.md §6) and needs none; it is plumbing.

---

### 4.1 The total above is understated, because `/val` turned out to be 2 B tokens

**Added 2026-09-15.** §4's table was built when §2.2 assumed 50 M – 1 B tokens, and
its eval-pass line reads 0.9 h nominal at "250 M / 1 B tokens". The probe settled
the number: `/val` is **2,000,168,321 tokens** (§2.2). §3's own table prices that
row at **7.45 h**, not 0.9 h.

| | §4 as written | with the measured `/val` |
|---|---|---|
| eval pass | 0.9 h | **7.45 h** |
| everything else | 3.1 h | 3.1 h |
| **nominal total** | **≈ 4 h** | **≈ 10.5 h** |
| **bad day** | ≈ 12 h | **≈ 23 h** |

The original table is left as written rather than edited, because what it got
wrong is the interesting part: every line except the eval pass held.

**This collides with the 12 GPU-hour hard stop.** A full-split 1B run is inside
it nominally, with 1.5 h of margin, and outside it on any bad day. The hard stop
is not a budget to be spent down to — it exists so that a session that is going
wrong gets abandoned rather than extended.

#### The fix is the fetch, not the eval

`loop.py:98-109` runs the entire split and there is no `num_eval_batches` (§3).
But **what is on disk is what gets evaluated**, and `/val` is uncompressed zarr
in 21 chunks of 100,000,000 tokens (§2.2). Fetching fewer chunks is therefore
the subsample knob the vendor does not expose:

| fetched | tokens | sequences @8K | 1B eval | SE of mean CE |
|---|---|---|---|---|
| 1 chunk | 100 M | 12,207 | **0.37 h** | 0.0027 nats |
| 3 chunks | 300 M | 36,621 | 1.12 h | 0.0015 nats |
| all 21 | 2,000 M | 244,161 | 7.45 h | 0.0006 nats |

`TOLERANCE.md`'s band is **0.491 nats wide**. One chunk's standard error is
**185× smaller than the band**; the full split buys 0.0021 nats of precision for
7 hours of GPU time. Against a bar that wide, that is not a measurement
improvement, it is a rounding error bought at the hard stop's expense.

**Stated before any result is seen, so it cannot be a post-hoc choice.** Which
fraction of `/val` is evaluated is a sampling decision, not a bar: `TOLERANCE.md`
is untouched and Standing Rule 5 is not in play. What *would* be in play is
choosing the fraction after seeing where the number landed, so the fraction is
fixed here, now, in writing: **three chunks**, which keeps the eval pass near §4's
original 0.9 h assumption and holds the session inside the hard stop with room to
spare.

---

## 5. The 3B runs, priced alongside

Priced now so a later scope change is costed rather than discovered.

**3B @ 128K does not fit on one card, and that is the whole story.** Its stock config sets
`global_batch_size: 16` across 8 devices, which makes the eval batch
`max(8, 16//1*4) = 64` sequences of 131,072 tokens — 8.4 M tokens in flight. Even forced to
`global_batch_size=2` (eval batch 8) it is 1 M tokens of activations on top of 11.6 GB of
fp32 parameters. Assume an **8×H100 node**. That multiplies the hourly rate by 8 and is why
this run is ~10× the 1B, not ~2×.

**3B @ 8K is cheap and is the one with a published number.** 11.6 GB of parameters plus batch
8 × 8192 tokens fits one 80 GB card comfortably. TOLERANCE.md §6 pre-registers its bar:
**2.314 ± 0.010** nats/token. This is the cheapest route to a *quantitative* baseline, because
the 1B checkpoint has no published number at all (TOLERANCE.md §1).

| Scenario | Hours | Devices | @ $2/h | @ $3/h | @ $4/h |
|---|---|---|---|---|---|
| 1B @ 8K, nominal | 5 | 1 | $10 | $15 | $20 |
| 1B @ 8K, bad first session | 8 | 1 | $16 | $24 | $32 |
| **3B @ 8K, nominal** | 6 | 1 | **$12** | **$18** | **$24** |
| 3B @ 128K, nominal | 6 | **8** | $96 | $144 | $192 |
| 3B @ 128K, bad session | 10 | **8** | $160 | $240 | $320 |

Add ~$1.40 egress for a 3B checkpoint. Rounding error either way.

**Rates are an assumption with a shelf life.** $2–4/GPU-hour is the single-H100 neocloud band
as last known to us; A100 80 GB sits lower, first-party hyperscalers 2–4× higher. **Re-check
the actual quoted rate at booking and, if it falls outside $2–4, redo §5 before renting** —
the whole table is linear in it, so that is a one-minute job.

---

## 6. What would blow this model up

Ranked by expected damage. Each has a named early-warning signal.

1. **`/val` is far larger than assumed.** At 5 B tokens the 1B eval alone is 18 h. *Warning:*
   the `du -s` probe. *Response:* re-plan before renting; this is knowable for free, today.
2. **The CUDA/cuDNN stack does not match** (12.8 / 9.8). *Warning:* `uv sync --frozen` or the
   first JAX device init. *Response:* kill the instance inside the first 30 minutes. Do not
   debug a broken image on a billing clock — pick a different image.
3. **The 1B does not fit the way we expect**, forcing a multi-GPU node and an 8× rate.
   *Warning:* OOM on the first eval batch. *Response:* lower `global_batch_size` to 2 first;
   only then consider more devices.
4. **W&B blocks the launch.** *Warning:* it fails in the first minutes. *Response:* none
   available on the box — this is why P0-10 procures the key *before* booking.
5. **Repeated short sessions.** Three 2-hour failures cost more than one clean 5-hour run and
   produce nothing. *Response:* the queue protocol's hour estimate and its stop rule (D6).

---

## 7. The cap

**Written before any spend. Binding until revised in writing, dated, in §8.**

| Bucket | Cap | Expected | Authority to exceed |
|---|---|---|---|
| **Phase 0.5 — 1B @ 8K baseline (incl. 125M rehearsal)** | **$150** | $10–35 | Lead, in writing, with the reason |
| GCS egress, all of Phase 0.5 | **$25** | < $1 | Lead |
| Optional 3B @ 8K quantitative anchor | **$150** | $12–25 | **Separate decision** — see ADR-005 |
| 3B @ 128K | **$600** | $100–320 | **NOT AUTHORISED.** Needs its own written decision, and Phase 1 PROCEED first |
| **Phase 0.5 total** | **$325** | | |

**Hard stops — no discussion required, any one of them ends the session:**

- Any single session passing **12 GPU-hours** without a completed eval. Stop, release the
  box, write down what broke, re-book with a plan. Session 2 is always cheaper than hour 13.
- The `du -s` probe returning **more than 40 GB** for the checkpoint. `MAX_BYTES` in
  `scripts/fetch_checkpoints.sh` enforces this and exits non-zero; raising it is a decision,
  not a reflex.
- Cumulative Phase 0.5 spend passing **$325**.

### Assumptions the cap rests on

Every one of these is stated so that a later overrun is diagnosable rather than mysterious:

1. Single H100 80 GB (or A100 80 GB) at **$2–4/GPU-hour**, billed hourly, on-demand. §5.
2. `/val` is between **50 M and 1 B tokens**. Unmeasured. The probe settles it for free, and
   it is the first thing to check. §2.2.
3. Checkpoint ≈ **5.9 GB** — analytic, ±10%, validated to +4.6% against the paper's own 88M
   figure. §2.1.
4. Throughput within **2×** of the paper's H100 prefill latencies. §3.
5. Egress at **$0.12/GB**, or **$0** same-region GCP. Either way under $1. §2.3.
6. **The 125M rehearsal happens first**, in the same session. Removing it does not save money;
   it converts a cheap failure into an expensive one. §4.
7. W&B credentials, `gcloud` auth and a verified CUDA 12.8 / cuDNN 9.8 image all exist
   **before** the instance starts billing. None is fixable on the clock.
8. The 1B eval runs on **one** GPU. If it needs a node, §5's 8× multiplier applies and the
   cap must be re-derived, not quietly consumed.

### What this cap buys

One clean 1B baseline plus roughly two full failed attempts. That ratio is deliberate: the
command has never run, so the first session is as likely to end in a stack trace as a number,
and a cap that only funds the happy path would force a bad decision at exactly the wrong
moment.

---

## 8. Revision log

| Date | Change | Reason |
|---|---|---|
| 2026-08-08 | Initial. All figures estimated; nothing probed, rented or spent. | — |
