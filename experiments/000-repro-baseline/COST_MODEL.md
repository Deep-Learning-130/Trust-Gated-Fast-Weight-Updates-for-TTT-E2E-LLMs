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
gsutil -u $P cp      gs://llama3-books3/data.zarr/zarr.json        $LOCAL/
gsutil -u $P cp      gs://llama3-books3/data.zarr/train/zarr.json  $LOCAL/train/
gsutil -u $P -m cp -r gs://llama3-books3/data.zarr/val             $LOCAL/
```

This supersedes ADR-004's framing of the grain prefetch as an egress risk. It is not a
billing risk; it is a **crash risk**, and the mitigation above removes it. D5 implements it.
**Unverified** — no bucket has been listed. If the layout differs (zarr v2, or a nested
store), fall back to the vendor README's full `gcloud storage cp -r`, and re-probe first,
because that is a much larger transfer.

> **Probed 2026-09-14 with a real billing account (P3's). The layout above is WRONG.**
>
> - `gsutil du -s gs://llama3-books3/val` returned **no bytes** — the command succeeded,
>   the prefix is empty. Not a permissions or billing failure: the checkpoint probes in the
>   same run, against the same project, succeeded.
> - `gsutil ls gs://llama3-books3/train/zarr.json` returned **"One or more URLs matched no
>   objects."**
>
> So all three `gsutil` lines above would fail. The paragraph's own escape hatch — *"if the
> layout differs, fall back to the vendor README's full `gcloud storage cp -r`, and re-probe
> first"* — is now the live path, and the re-probe is a **bucket listing**, still pending.
>
> What is *not* in doubt: the bucket name. `vendor/ttt-e2e/README.md:37` gives
> `gs://llama3-books3` verbatim, the August anonymous probe got HTTP 400 (requester-pays,
> bucket exists), and P3's account reaches it. The array names are not in doubt either —
> `ttt/config.py:158-159` sets `data_split: "train"` and `eval_split: "val"`, and
> `dataloader/lm_dataset.py:16` opens `zarr.open_array(store, path=f"/{split}")`. **What is
> wrong is where those arrays sit inside the bucket.** The vendor README's own instruction
> (`gcloud storage cp -r gs://llama3-books3/ llama3-books3`) copies the *whole* bucket into
> a directory and points `deploy_paths` at it, which is consistent with the arrays being
> nested a level down rather than at the root.
>
> **This is the probe earning its keep.** Every one of these failures would otherwise have
> happened on a rented box, after billing started, which is the exact scenario this section
> was written to prevent.
>
> **Resolved the same day, by listing.** The store is nested one level down:
>
> ```
> gs://llama3-books3/data.zarr/zarr.json      <- group metadata
> gs://llama3-books3/data.zarr/train/
> gs://llama3-books3/data.zarr/val/
> ```
>
> The array names were right all along; only the root was wrong. **`data.zarr/` is the zarr
> store root**, so it is what `LocalStore(path)` must be pointed at — which makes it the
> value of `deploy_paths.data.books3`, *not* the directory the vendor README's `cp -r`
> creates. Getting that one level wrong produces a `LocalStore` that opens and then fails to
> find `/val`, on the box, after billing starts.
>
> The recipe above and `scripts/bootstrap_gpu_box.sh` are corrected. The full `cp -r`
> fallback is **not** needed: the selective copy works, it just needed the right prefix.
> Still outstanding: the `/val` byte count itself (§2.2), now a single `du -s` away.

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

> **Measured 2026-09-14** (P3's billing account, metadata only, nothing transferred):
>
> | checkpoint | estimated | **measured** | error | egress @ $0.12/GB |
> |---|---|---|---|---|
> | 1B books@8K | 5.9 GB | **5,347,020,507 B = 5.35 GB** | **−9.4%** | $0.64 |
> | 125M books@8K | — | **683,046,332 B = 0.68 GB** | — | $0.08 |
>
> The analytic estimate holds, but **only just** — −9.4% against a stated ±10% band. Treat
> the band as real rather than conservative; the 3B figure above (≈11.6 GB) carries the same
> formula and the same uncertainty, and it has not been measured.
>
> Consequences: `MAX_BYTES=40000000000` in `scripts/fetch_checkpoints.sh` is ~7.5× the
> actual size, so the guard will not fire spuriously. And the 125M checkpoint is real,
> reachable and costs **$0.08** — which makes a full end-to-end rehearsal on a *real*
> checkpoint, rather than the random-init one in `experiments/003-smoke-125m/`, affordable
> to the point of being free.

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

> ### Measured 2026-09-14 — **8.40 GB, and the band above was wrong**
>
> `gs://llama3-books3/data.zarr/val` = **8,400,000,517 bytes**. Carrying this section's own
> 2.5× compression assumption, that is **≈ 5.25 B tokens — 5.25× the top of the 50 M–1 B
> range guessed in August.** §6's first listed way this model blows up has fired.
>
> The token count is still an *estimate*, because it inherits the compression guess:
>
> | if compression is | tokens | sequences @ 8K |
> |---|---|---|
> | 2.0× | 4.20 B | 512,695 |
> | **2.5× (assumed)** | **5.25 B** | **640,869** |
> | 4.0× | 8.40 B | 1,025,391 |
>
> **Get the exact number before deciding anything** — it is free. zarr v3 stores the array
> shape in its metadata, so `gsutil -u $P cat gs://llama3-books3/data.zarr/val/zarr.json`
> returns a few hundred bytes and settles it exactly, with no compression assumption at all.
>
> **Egress is still a non-issue**, exactly as this section predicted: 8.40 GB is ~$1.01
> off-GCP, $0 same-region, and $1.65 including the checkpoint. The cap's egress line is fine.
>
> **Wall-clock is not.** Scaling §3's own figures (1 B tokens = 0.9 h nominal, 3.7 h bad day)
> to 5.25 B gives **≈ 4.7 h nominal and ≈ 19.4 h on a bad day** for a single 1B eval pass.
> §6 predicted "at 5 B tokens the 1B eval alone is 18 h". That was accurate.
>
> **This collides with §7.** The hard stop reads *"any single session passing 12 GPU-hours
> without a completed eval"*. A bad-day full pass is ~19 h, so the session would trip its own
> hard stop **before producing a number** — and the stop is correct; it is the plan that no
> longer fits. §4's "≈ 4 h nominal / ≈ 12 h bad day" total is likewise no longer reachable.
>
> **The full pass buys nothing measurable.** `TOLERANCE.md`'s bar is
> `2.314 < loss < 2.805`, a window **0.491 nats wide**, taken from the paper's published
> table — it is not defined against a split size. Standard error of the mean CE:
>
> | sequences evaluated | SE (nats) | as % of the band |
> |---|---|---|
> | 1,000 | 0.0095 | 1.9% |
> | 10,000 | 0.0030 | 0.6% |
> | 30,000 (≈ 250 M tokens) | 0.0017 | 0.35% |
> | 640,869 (the whole split) | 0.0004 | 0.08% |
>
> Going from 30,000 sequences to all 640,869 tightens the estimate by 0.0013 nats against a
> 0.491-nat bar, for roughly 20× the GPU time. **Running the full split is not rigour, it is
> waste** — and on these numbers it is waste that breaks the budget.
>
> **There is no vendor knob for this.** `Evaluator.__init__` (`ttt/model/loop.py:63-72`)
> builds the holdout loader with `repeat=False, shuffle=False` over `eval_split="val"`, and
> `eval_fn` iterates `total=len(ds)`. There is no `eval_steps`, no `max_eval_batches`. The
> only lever that respects ADR-002 is **what we put in the local store**: copy a prefix of
> `/val`'s chunks and write a local `val/zarr.json` whose shape matches exactly what was
> copied, so `len(ds)` is honest and no absent chunk is ever read as fill value.
>
> **That is a pre-registration decision, not an implementation detail.** It changes what the
> Phase 0.5 number is computed over. Standing Rule 5 permits it — no result has been seen —
> but only as a **dated, written revision made before the run**, in `PREREGISTERED.md` and
> `TOLERANCE.md`. Made afterwards it is indistinguishable from moving the bar. **The Lead
> decides; this document records the options and does not pick one.**

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
| 2026-09-14 | §1 requester-pays verified anonymously at zero cost, all three buckets. | Settled a standing assumption without auth or spend. |
| 2026-09-14 | Added §9 (access routes). No cap, bar or estimate above it changed. | The first GCP billing signup was denied; the routes needed recording where the next person looks. |
| 2026-09-14 | §2.1 checkpoint sizes **measured** (1B = 5.35 GB, −9.4% vs estimate; 125M = 0.68 GB). §1.1's store layout **falsified**. | First probe with a real billing account, supplied by P3. |
| 2026-09-14 | Store root corrected to `gs://llama3-books3/data.zarr` in §1.1, `EVAL_ENTRYPOINT.md`, `bootstrap_gpu_box.sh` and `probe_gcs_access.sh`. | Bucket listing. The selective-copy recipe stands; only the prefix was wrong. |
| 2026-09-14 | §2.2 `/val` **measured at 8.40 GB (≈5.25 B tokens)** — 5.25× over the assumed band. §6's first blow-up condition fired; §7's 12 h hard stop and §4's 12 h bad-day total are both unreachable with a full pass. **No cap or bar changed.** | T1.3 complete. The decision it forces belongs to the Lead, in writing, before any run. |

---

## 9. Access routes — how we actually get a checkpoint and a GPU

**Added 2026-09-14, after the first attempt to open a GCP billing account failed.**
Everything above prices a run we can reach. This section records that we currently
cannot reach it, what was ruled out, and what the live routes are. It changes no
cap and no bar.

### 9.1 The blocker is account access, not money

The Phase 0.5 bill is under $2 of egress plus a few GPU-hours. The obstacle is
that a Google Cloud billing account does not exist.

- **2026-09-14.** A Google Cloud free-trial signup (P1) was **denied** at the
  payment step. Google Payments returned the generic *"For your protection, a
  Google Cloud signup was denied"* template. That template is sent for both
  fraud and ordinary risk declines and asserts nothing about account compromise.
- The attempt used the **automatic-payments** profile. Google's own
  documentation states that Indian **debit** cards cannot be accepted for
  automatic payments at all (a 3D Secure issue on their side), and that certain
  Indian credit cards also fail, because RBI rules require a registered
  e-mandate for recurring card payments. A one-off charge is a different
  mechanism and is not implicated.
- **Therefore the failure is specific to recurring-mandate registration, not to
  the card.** Routes that bill one-off or prepaid are unaffected. This matters
  for §9.3.

**Untried, in order of cost:** the manual-payments profile at signup where the
country's flow offers it; a co-worker's billing account (P2 or P3 — the
checkpoint may be read directly from `gs://` per `ttt/infra/checkpoint.py:83-84`,
so whoever holds the billing project can stream it on the run box without anyone
copying or redistributing weights); an institutional account.

### 9.2 There is no mirror — do not go looking again

Checked 2026-09-14, so nobody spends another afternoon on it:

- `vendor/ttt-e2e/README.md:95-119` lists all six released checkpoints, every
  one of them `gs://ttt-e2e-checkpoints/...`, and flags Requester Pays
  explicitly. No alternative download is offered anywhere in the vendor tree.
- The HuggingFace `Test-Time-Training` organisation **does** publish weights,
  and they are **not substitutes**. Those are TTT-Linear and TTT-MLP from the
  earlier paper: a different architecture with no `feed_forward_prime`, no meta
  branch, and no inner-loop fast weights of the kind ADR-006 is about. Loading
  one would not be a cheaper version of this experiment; it would be a different
  experiment on a different model.

So the checkpoint is reachable only through a GCP billing account. Every route in
§9.1 is about obtaining one.

### 9.3 Compute routes, priced

§7's assumption 1 budgets **$2–4/GPU-hour**. That band was written for US
on-demand H100/A100 and is now the pessimistic end. Indian providers bill in INR
via UPI/NEFT — which, per §9.1, sidesteps the exact mechanism that failed:

| Route | A100 80GB | H100 | Billing | Notes |
|---|---|---|---|---|
| **Jarvislabs** | ₹141/hr | ₹255/hr | INR, per-minute, no commitment | A100 40GB at ₹84/hr. Best shape for 3h sessions. |
| **E2E Networks** | ₹179–189/hr | from $1.80/hr | INR, UPI/NEFT | NSE-listed. Spot 80GB at ₹66/hr. |
| Yotta Shakti | enterprise quote | quote | — | Not worth pursuing at this scale. |
| **Kaggle TPU v3-8** | — | — | **free**, weekly quota | 128 GB HBM, JAX-native, native bf16. Needs no checkpoint, so it is the right host for `experiments/003-smoke-125m`. |

At Jarvislabs' A100 80GB rate a 3-hour session is about **₹425**, which is below
the $12/session §4 assumed. The cap in §7 therefore has *more* headroom than it
was written with, not less. **The cap does not move on that basis** — a cheaper
hour is not authority to buy more of them.

**`vast.ai` was considered and is not recommended here.** It is a marketplace of
third-party machines. Running there means placing a PAT for a private repository
whose value is its dated commit timeline, plus a W&B key and eventually GCS
credentials, on hardware we do not control. Its verified/datacenter tier
mitigates this; community hosts do not. The saving does not pay for the exposure.

### 9.4 A 40GB card may be enough — verify before booking 80GB

§7's assumption 1 and §5 both presume an 80 GB card. That figure is the vendor's
**training** footprint at the vendor's batch sizes. At the settings this project
actually runs — `global_batch_size=1`, `seq_length=8192`, `mini_batch_size=1024` —
a parameter-count reconstruction of §2.1's formula (revalidated against its three
published rows) puts 1B at roughly:

| Path | Estimate | What it includes |
|---|---|---|
| via `trustgate.eval.vendor_bind` | **≈ 13 GB** fp32, ≈ 7.5 GB bf16 | params + the `transformer.py:685` state copy + per-chunk logits |
| via unmodified `train.py` | **≈ 25 GB** fp32 | the above plus the outer AdamW state `train.py:200` allocates before the eval branch returns and never uses |

**This is an estimate, not a measurement, and it has never run.** If it holds, an
A100 40GB at ₹84/hr covers both T1.6 and T1.10 and roughly halves the compute
line. Confirm it on the box with one memory reading before booking anything
larger — and if it is wrong, say so here rather than quietly renting up.

### 9.5 IndiaAI Mission — the structurally right route, on a slow clock

India's IndiaAI Mission operates a common compute facility offering subsidised
GPU access at roughly **₹65–100/GPU-hour**, with 38,000+ GPUs onboarded.
Students, academic institutions and early-stage researchers are explicitly named
eligible categories. Applications go through the IndiaAI portal and are assessed
by committee.

Worth an application, with two caveats stated up front so nobody schedules
against it: allocations favour declared national priorities (healthcare AI,
agri-tech, vernacular language models), which a test-time-training security study
may not match; and committee assessment is not a timeline anyone can plan around.
**Apply in parallel. Do not make it the critical path.**

### 9.6 What this section does not change

Stated explicitly because the temptation runs the other way when a project is
blocked:

- **No pre-registered bar moves.** `PREREGISTERED.md` and `TOLERANCE.md` are
  untouched. Standing Rule 5 forbids moving a bar after a result is seen; no
  result has been seen, and none of the above is a result.
- **The $325 cap stands**, at §7's figures, cheaper hours notwithstanding.
- **"Blocked on access" is not a STOP.** A STOP is *we ran it and the attack did
  not clear the bar*. If the 1B run never happens, the paper says which of the
  two it was, in those words.
