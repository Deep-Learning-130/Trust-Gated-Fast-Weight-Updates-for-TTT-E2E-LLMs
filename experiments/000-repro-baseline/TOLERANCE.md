# Pre-registered tolerance for the baseline reproduction (P1-2)

> Deliverable **D2** of `docs/superpowers/specs/2026-08-03-phase0-lead-prep-design.md`.
> **Written 2026-08-08. No eval has been run. No number has been observed.**
> Source: *End-to-End Test-Time Training for Long Context*, Tandon, Dalal, Li et al.,
> retrieved 2026-08-08 from `https://test-time-training.github.io/e2e.pdf`.
>
> **TEAM_PLAN.md Rule 5 applies to this file.** Every bar below is fixed as of this date.
> If a bar is missed, the outcome is STOP-and-debug or a **dated, written revision** in
> §8 explaining what changed. Never a quiet edit.

---

## 1. Verdict, stated plainly

**The paper reports no number — absolute or plotted — for the checkpoint we plan to
evaluate.**

Our plan of record (ADR-004) is `gs://ttt-e2e-checkpoints/1b_ttt_e2e_finetune_books_8k_1x_cc`:
the **1B** model, extension fine-tuned on **Books** at **8K** context, at **1×** Chinchilla
compute. That configuration appears in no table and on no curve in the paper. §2 is the
complete inventory that establishes this, so the claim is checkable rather than asserted.

Two consequences, and they are the substance of this document:

1. **P1-2 cannot be "match the published number" for this checkpoint.** §4 defines the
   closest defensible comparison instead — a two-sided bracket built only from printed
   values and sign-determined monotonicity — plus the structural checks in §5 that carry
   most of the real discriminating power. No other configuration's number is substituted
   for ours.
2. **A directly comparable alternative exists and is also released.** The 3B Books @8K and
   @128K checkpoints *are* plotted points in Figure 9. §6 pre-registers exact bars for them
   now, so that switching to them later is a costed decision rather than a post-hoc
   rationalisation. The choice is recorded in `docs/adr/ADR-005-baseline-comparison-basis.md`.

---

## 2. Every quantitative loss the paper reports

Compiled by extracting all text and all figure vector data from the PDF and sweeping for
loss-valued numbers. "Printed" means the digits appear in the text; "plotted" means the
value exists only as figure geometry (see §3 for how those were read).

| # | Config | Dataset | Ctx | Value(s) | Where | Form |
|---|---|---|---|---|---|---|
| 1 | 760M, 1×, pre-train only | DCLM | 8K | SWA (k=8K) **2.827**, TTT-KVB **2.818**, TTT-KVB simplified **2.819**, TTT-E2E all-layers-MH **2.806**, **TTT-E2E 2.805** | Table 1 | printed |
| 2 | 760M, 1×, pre-train, TTT disabled (b=8K) | DCLM | 8K | TTT-E2E **2.825**, TTT-KVB **2.826**, full attention **2.827** | §3.2 prose (Figure 4 discussion) | printed |
| 3 | 760M, 1×, GatedDeltaNet | DCLM | 8K | **2.814** → **2.809** with QK-norm | Appendix C | printed |
| 4 | 760M, 1×, GatedDeltaNet | Books | 32K | **2.691** → **2.683** with QK-norm | Appendix C | printed |
| 5 | 3B, 3×, all 7 methods | Books | 8K–128K | full table in §3 | Figure 9 (= Figure 1 left, as absolutes) | plotted |
| 6 | 3B, 3×, all 7 methods | Books | 32K, 128K | loss vs token index | Figure 6 | plotted |
| 7 | 125M/350M/760M/**1B**/3B, 1× | DCLM 8K (a,c) and **Books 32K** (b,d) | — | **differences vs full attention only**, range 0.00–0.06 | Figure 5 | plotted, difference |
| 8 | 760M, 1×, hyper-parameter ablations | DCLM | 8K | window-size and mini-batch sweeps | Figure 4 | plotted |
| 9 | 3B, 3× | Books | 8K–128K | S-NIAH **accuracy**, not loss | Table 2 | printed |

**Why none of these is our checkpoint.** The only place a **1B** model appears at all is
Figure 5 (row 7), and there it is (a) at **32K**, not 8K, and (b) reported as a *difference*
from full attention, which cannot be turned into an absolute without a full-attention
number at the same size — which the paper does not print and for which no checkpoint is
released. Rows 1–4 and 8 are **760M**. Rows 5, 6 and 9 are **3B at 3×**. Nothing is
1B / Books / 8K / 1×.

### Units — no conversion is needed

The paper's y-axis is labelled verbatim **"Loss (log perplexity)"**, and §3.4.1 defines the
plotted quantity as the average over token positions of the per-token next-token-prediction
loss. That is mean cross-entropy in **nats per token** — exactly what the vendor's eval
emits as `train_holdout/loss` (`ttt/model/loss.py:26-27`, per `EVAL_ENTRYPOINT.md` §6).

**The paper reports no perplexity, and neither does the vendor code.** `ppl = exp(loss)` is
therefore never required. Perplexities are shown below only as a readability aid; every bar
in this document is stated in nats/token, which is the quantity actually measured.

---

## 3. The Figure 9 values, and how they were obtained

Figure 9 is the appendix version of Figure 1 that "directly plots the loss values instead
of the loss [differences]". It is the only place in the paper with absolute losses for a
Books-fine-tuned TTT-E2E model, and it covers 8K — so it is worth reading precisely.

The figure is vector art, so the values are recoverable exactly rather than estimated by
eye. The embedded `figs/flagship_app.pdf` XObject was decompressed; the nine y-axis tick
stubs and their labels (2.25 … 2.33) were paired inside that same coordinate space, giving
a linear calibration with a **maximum residual of 4×10⁻¹⁶ nats** (2458.458 pt per nat); the
seven data polylines were mapped to method names using the figure's own embedded legend.

**Config:** 3B (Table 3's 2.7B-parameter row), 3× Chinchilla — 162B pre-training tokens on
DCLM plus ~8B extension fine-tuning tokens on Books, matching the abstract's "164B tokens".
Pre-trained once on DCLM, then fine-tuned separately at each context length on Books, and
evaluated on held-out Books. Mean CE, nats/token.

| Method | 8K | 16K | 32K | 64K | 128K |
|---|---|---|---|---|---|
| Transformer, full attention | 2.328 | 2.301 | 2.283 | 2.270 | 2.261 |
| SWA (k=8K) | 2.328 | 2.307 | 2.299 | 2.301 | 2.309 |
| Hybrid SWA and full | 2.328 | 2.304 | 2.288 | 2.280 | 2.276 |
| Mamba 2 | 2.312 | 2.291 | 2.283 | 2.285 | 2.293 |
| Gated DeltaNet | 2.322 | 2.301 | 2.291 | 2.289 | 2.295 |
| TTT-KVB | 2.319 | 2.298 | 2.291 | 2.293 | 2.305 |
| **TTT-E2E (ours ← the vendor's)** | **2.314** | **2.288** | **2.270** | **2.258** | **2.249** |

Perplexity equivalents for the TTT-E2E row: 10.115, 9.855, 9.679, 9.564, 9.478.

### Why this reading is trustworthy

Four independent checks, all of which pass. This matters because a mis-read axis would
silently corrupt every bar below.

1. **Cross-validated against a second figure.** Figure 1 (left) plots the same runs as
   differences from full attention, drawn separately with its own axis. All **35** values
   (7 methods × 5 context lengths) agree with the differences derived from the table above,
   to 4 decimal places.
2. **A constraint theory forces.** At 8K context with window k=8K, sliding-window attention
   *is* full attention, and the hybrid is too. All three curves read **2.328** — identical
   to 4 decimals at the one point where they are required to be identical.
3. **Every value lands on a 0.001 grid**, matching the 3-decimal precision the paper prints
   in Table 1. Nothing was rounded into place by hand.
4. **Every qualitative claim in the captions is reproduced**: TTT-E2E is lowest at every
   context length; SWA, Mamba 2, Gated DeltaNet and TTT-KVB all worsen after 32K while full
   attention and the hybrid keep improving; and TTT-E2E "turns the worst line into the best
   at 128K" (SWA 2.309 → TTT-E2E 2.249).

**Residual risk.** These are values read from a figure, not digits printed in the paper. If
the authors' plotting pipeline rounded before drawing, we inherit that rounding — bounded
by the 0.001 grid, an order of magnitude below every tolerance below, so it does not change
any bar. Recorded because a reader should not have to guess our provenance.

---

## 4. The bar for the 1B Books @8K checkpoint — plan of record

No published number exists, so the bar is a **two-sided bracket**. Each bound uses only a
value printed in the paper plus an adjustment whose **sign** is certain, never a magnitude
that had to be guessed.

### 4.1 Hard bound — PASS band

> ### **PASS iff 2.314 < `train_holdout/loss` < 2.805 nats/token**
> (perplexity 10.12 to 16.53)

**Lower bound 2.314** — the 3B/3× TTT-E2E Books @8K value (§3). Same method, same corpus,
same context length, same eval metric. Our checkpoint has **2.1× fewer parameters** (1.3B
vs 2.7B, Table 3) and **3× fewer training tokens**. Less compute cannot produce a lower
loss on the same task, so our value must exceed it. The only assumption is that loss
decreases with training compute — which the paper's own Figure 5 is an entire section
devoted to demonstrating.

**Upper bound 2.805** — the 760M/1× TTT-E2E DCLM @8K value (Table 1). Three differences
separate it from our run and **all three push loss down**: more parameters (1.3B vs 760M),
more training tokens (26B vs 15B, Table 3), and evaluation on Books — a corpus that is both
easier than filtered CommonCrawl and one our checkpoint was explicitly fine-tuned on, so
in-distribution. Direction is certain; magnitude is not needed.

**FAIL ⇒ stop and debug the environment.** Do not proceed to experiment 001. A value
outside this band means the setup is wrong, not that the model is interesting.

### 4.2 What the band does and does not catch

Stated because a bar whose limits are unknown is not a bar. Reference point: uniform
prediction over the Llama-3 vocabulary is ln(128256) = **11.762** nats/token.

Caught by the band:
- Parameters not actually loaded, or loaded into the wrong shape → near 11.762, or wildly off.
- Tokenizer or vocabulary mismatch → inflated by ~1 nat or more.
- Sharding, dtype or masking bugs of any real size.
- Loss aggregated over padding or without the BOS mask.

**Not caught by the band — and this is its main weakness:**
- **Evaluating on DCLM instead of Books.** A 1.3B DCLM number would plausibly land inside
  the band (extrapolating from 760M's 2.805), so the band alone cannot detect the mix-up.
  *Mitigation, mandatory:* record the **resolved** `training.dataset_name` and
  `training.dataset_path` from the run's own config echo into `results/`. Verify
  `dataset_name == books3`. This is a config check, not a numeric one.
- **A modest inner-loop misconfiguration.** At 8K the whole TTT mechanism is worth only
  0.014 nats (SWA 2.328 vs TTT-E2E 2.314 at 3B), which is far narrower than the band. A run
  with TTT effectively disabled would still PASS. §7 draws the consequence.

### 4.3 Non-binding expectation, recorded before the fact

**Not a bar.** Recorded so that our prediction is on the timeline, and so a value inside
the band but far from expectation prompts a look rather than a shrug. It needs deltas
transferred across model scales, which is why it cannot be a pass/fail criterion.

Chain, from printed values only:
- 760M/1× Gated DeltaNet, Books @32K = **2.683** (Appendix C). *Assumed 1×:* Appendix C does
  not state a token budget, and Figure 5 varies 760M's budget up to 5×, so this could be an
  over-trained run. If it is, the gap below is overstated and the true expectation is lower.
  Another reason §4.3 is not a bar.
- 3B/3× Gated DeltaNet, Books @32K = **2.291** (§3). So the 760M/1× → 3B/3× compute gap on
  this corpus is **+0.392** nats.
- Applying that gap to 3B/3× TTT-E2E Books @32K (2.270) ⇒ 760M/1× TTT-E2E @32K ≈ **2.66**.
  Sanity: 0.02 better than GDN at the same size, consistent with Figure 5's 0.00–0.06 range.
- 32K → 8K costs TTT-E2E **+0.044** at 3B (2.314 − 2.270) ⇒ 760M/1× @8K ≈ **2.71**.
- 760M → 1.3B at fixed 1× recipe removes an unquantified amount, order 0.05–0.10.

⇒ **expect roughly 2.60–2.70 nats/token** (perplexity ≈ 13.5–14.9).

If the observed value is inside §4.1's band but outside 2.60–2.70, **investigate before
proceeding** — check `dataset_name`, `seq_length`, and that the inner loop ran — but this
does **not** by itself constitute a FAIL. Only §4.1 and §5 decide PASS/FAIL.

---

## 5. Structural bars — these do the real work

Independent of the paper, so they are unaffected by everything in §1–4. All are computed
from artefacts the run already produces. **All four are required for PASS.**

**S1 — the per-token NLL curve must fall with token index.**
`train_holdout_token_nll_loss.npy` (written to `log_dir`, `ttt/model/loop.py:125-128`) is
the array Figure 6 plots. Require: the mean over the last 1024 positions is **strictly
lower** than the mean over positions 128–1152 (skipping the first 128, which are dominated
by having no context at all). This is the signature of context actually being used, and it
is the check that would catch a checkpoint loaded but not adapting.

**S2 — determinism.** Eval sets `shuffle=False, repeat=False` and takes no optimizer step,
so a rerun of the identical command must reproduce `train_holdout/loss` to at least 4
decimals. A larger spread means uncontrolled nondeterminism, which must be explained before
any effect size in experiment 001 can be believed.

**S3 — negative control.** One run with `training.dummy_dataset=true` (random tokens,
`ttt/dataloader/lm_dataset.py:30-40`) must yield a loss **far above** the band, of order
10–12 nats. This proves the metric is wired to the data and fixes the top of the scale.
Cheap, and it needs no dataset download.

**S4 — the gate is absent.** `trustgate.vendor_patch.is_installed()` is `False` and no
`trustgate` module is imported for the whole run. This is the baseline; the overlay must
not be in it.

---

## 6. Bars for the checkpoints that *do* have published numbers

Pre-registered now, at zero cost, so that a later switch is a decision and not an
improvisation. If either run happens, these are its bars — fixed as of 2026-08-08.

| Checkpoint | Published (§3) | Pre-registered bar |
|---|---|---|
| `3b_ttt_e2e_finetune_books_8k_3x_cc` | **2.314** | PASS iff \|observed − 2.314\| ≤ **0.010** |
| `3b_ttt_e2e_finetune_books_128k_3x_cc` | **2.249** | PASS iff \|observed − 2.249\| ≤ **0.010** |
| `125m_ttt_e2e_finetune_books_8k_1x_cc` | none | **No numeric bar.** Figure 5 has a 125M point but at 32K and as a difference. Plumbing rehearsal only — §5 S1–S4 apply, §4.1 does not. |

**Why ±0.010 nats.** Three reasons, and the middle one is the binding one:

1. **10× the paper's own noise floor.** Table 1's caption: "We consider a difference of
   0.001 below the threshold of statistical significance." Ten times that leaves room for
   hardware, CUDA/JAX version and eval-batch differences without admitting a real regression.
2. **It preserves the decision the number has to support.** At 8K the TTT-E2E-vs-SWA gap is
   0.014, so ±0.010 still distinguishes "we are on the TTT-E2E curve" from "we are on the
   full-attention curve" — marginally. At 128K the same gap is 0.060, so the margin is
   comfortable there. **This is why 128K is the better reproduction target if a 3B run is
   ever funded**, despite costing more.
3. It is one to two orders of magnitude below every configuration-confusion failure mode in
   §4.2, so a PASS at this tolerance is informative rather than merely permissive.

Assumption behind both bars: the `/val` split inside `gs://llama3-books3` is the same
held-out Books partition the paper evaluated on. The vendor states all experiments are
reproducible from the released code and datasets (§3 of the paper), so this is presumed —
but it is an assumption, not a verified fact, and it is the first thing to question if a 3B
run misses its bar by a small margin.

---

## 7. What a PASS here does *not* establish

Recorded now so the Phase 0.5 exit is not over-read later.

**Reproducing an 8K number does not validate the long-context TTT machinery.** At 8K the
sliding window (k=8K) spans the entire sequence, so the architecture is equivalent to full
attention and the entire TTT contribution is worth 0.014 nats at 3B — smaller than our band
by more than an order of magnitude. At 8192 tokens with `mini_batch_size=1024`, the fast
weights receive only **8** inner updates.

This matters for experiment 001, which attacks precisely those updates. Eight update steps
is enough to mount an attack spike, but it exercises the mechanism far less than the 128
steps a 128K run would. So:

- A PASS is evidence that **our environment is sound** — the right weights, the right
  tokenizer, the right data, a sane metric. That is what makes a later corruption
  measurement attributable, and it is the gate's actual purpose.
- A PASS is **not** evidence that our numbers agree with the paper's, because for this
  checkpoint there is no paper number to agree with.
- Any claim in the invention disclosure of the form "we reproduced the published TTT-E2E
  baseline" must **cite §6**, not §4. Under §4 the honest claim is narrower: "our setup
  produces a loss consistent with the published compute-scaling behaviour."

---

## 8. Revision log

Rule 5: bars above do not move after a number is observed. Any change lands here, dated,
with the reason, and leaves the superseded value visible.

| Date | Change | Reason |
|---|---|---|
| 2026-08-08 | Initial pre-registration. Nothing observed. | — |
