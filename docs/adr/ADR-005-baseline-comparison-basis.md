# ADR-005 — The 1B baseline is a soundness check, not a reproduction

**Status:** Accepted **Date:** 2026-08-08 **Deciders:** project owner (Lead)
**Supersedes in part:** the framing in `experiments/000-repro-baseline/README.md`
("match the paper's reported long-context numbers within tolerance")

## Context

`ROADMAP.md` Phase 0.5 and TEAM_PLAN P1-2 both assume the same thing: that the vendor's
paper reports a number for the checkpoint we intend to evaluate, and that our job is to
land within a tolerance of it. Deliverable D2 was to write that tolerance down before the
hardware existed (Rule 5).

**It does not.** The plan of record is `1b_ttt_e2e_finetune_books_8k_1x_cc` — 1B, Books,
8K, 1× Chinchilla — and that configuration appears in no table and on no curve in
*End-to-End Test-Time Training for Long Context*. The full inventory is
`experiments/000-repro-baseline/TOLERANCE.md` §2. In summary:

- Every **printed** absolute loss is for the **760M** model (Table 1, §3.2, Appendix C).
- Every **plotted** absolute loss is for the **3B at 3×** model (Figures 6 and 9).
- The only place a **1B** model appears at all is Figure 5, and there it is at **32K**, not
  8K, and reported as a **difference** from full attention — which cannot be converted to an
  absolute without a same-size full-attention number the paper does not print and for which
  no checkpoint is released.

Two further facts shaped the decision. First, the vendor **does** release
`3b_ttt_e2e_finetune_books_8k_3x_cc` and `3b_ttt_e2e_finetune_books_128k_3x_cc`, and both
*are* points on Figure 9 — so a directly comparable run is available, at a price. Second,
Figure 9 is vector art, so its values are recoverable exactly rather than by eye; doing so
yields **TTT-E2E Books @8K = 2.314** and **@128K = 2.249** nats/token, cross-validated
against Figure 1 across all 35 plotted values (TOLERANCE.md §3).

## Decision

1. **Keep the 1B Books @8K checkpoint as the plan of record**, and **restate what its
   evaluation establishes.** Phase 0.5 is a check that *our environment is sound* — right
   weights, right tokenizer, right corpus, sane metric — which is what makes a later
   corruption measurement attributable. That was always the gate's actual purpose
   (`experiments/000-repro-baseline/README.md`: "any corruption measured in 001 is
   unattributable"). It is **not** a reproduction of a published number, because for this
   checkpoint there is none.

2. **The bar is a two-sided bracket, not a tolerance:** PASS iff
   `2.314 < train_holdout/loss < 2.805` nats/token. Both bounds come from values printed or
   plotted in the paper, adjusted only by differences whose **sign** is certain (more
   compute cannot raise loss; Books is easier than DCLM and is the corpus this checkpoint
   was fine-tuned on). No magnitude is guessed. Derivation and limits: TOLERANCE.md §4.

3. **Structural checks carry the discriminating power**, not the band. A falling per-token
   NLL curve, run-to-run determinism, a `dummy_dataset` negative control, and
   `is_installed() == False` are all required for PASS. TOLERANCE.md §5. They are needed
   because the band is wide enough that a *DCLM-instead-of-Books* mix-up would pass it — so
   the resolved `dataset_name` is also verified from the run's own config echo.

4. **Pre-register the 3B bars now, and do not spend on them yet.** `2.314 ± 0.010` at 8K and
   `2.249 ± 0.010` at 128K are fixed as of today (TOLERANCE.md §6). Recording them costs
   nothing and removes the possibility of a bar being invented after a number is seen.
   Neither run is authorised. If a genuinely quantitative anchor is later wanted, **3B @ 8K
   is the one to buy**: ≈ $12–25 on a single card, against ≈ $96–320 for 3B @ 128K, which
   needs a multi-GPU node (`COST_MODEL.md` §5).

5. **No claim of the form "we reproduced the published TTT-E2E baseline" may cite the 1B
   run.** Under decision 2 the honest claim is narrower: *our setup produces a loss
   consistent with the published compute-scaling behaviour.* A reproduction claim requires
   a 3B run under decision 4. This binds `docs/patent/invention-disclosure.md` §8.

## Alternatives rejected

**Substitute the 3B @8K number (2.314) as the 1B's target.** Wrong by roughly the whole
compute-scaling gap — the paper's own measurement of that gap on this corpus is ~0.39 nats
(TOLERANCE.md §4.3). It would manufacture a FAIL from a healthy run. This is precisely the
substitution the D2 spec forbids.

**Switch the plan of record to 3B @8K, for a real published number.** Tempting, and it is
the *scientifically* stronger choice. Rejected for now on cost and sequencing: it is ~2×
the money and, more importantly, the first booking's job is to find out whether our command
works at all. Buy the cheap failure first. Revisit if Phase 1 needs a quantitative anchor —
decision 4 keeps the option priced and pre-registered.

**Use the 125M checkpoint as the baseline.** It has no published number either (Figure 5's
125M point is at 32K and is a difference). It stays what it is: a plumbing rehearsal.

**Declare Phase 0.5 unblocked and skip the gate.** Rejected. The environment check is the
part that was always load-bearing; only the "match the paper" framing was mistaken.

## Consequences

- **`experiments/000-repro-baseline/README.md` and ROADMAP Phase 0.5 need rewording** —
  "match the paper's reported long-context numbers" is not achievable for this checkpoint.
  Done as part of D2/D7.
- **A PASS is weaker evidence than the roadmap implied**, and deliberately so. At 8K the
  sliding window spans the whole sequence, so TTT is worth only 0.014 nats at 3B and the
  fast weights take just **8** inner updates over 8192 tokens. Reproducing an 8K number
  does not validate the long-context TTT machinery. Experiment 001 attacks those updates;
  8 steps is enough for a spike, but a 128K run exercises the mechanism 16× harder.
  TOLERANCE.md §7.
- **The FAIL direction is unaffected.** A value outside the bracket still means stop and
  debug the environment, and that is the outcome the gate exists to force.
- **Two numbers now depend on a figure-extraction method rather than printed digits.** The
  method and its four validation checks are recorded in TOLERANCE.md §3 so a reader can
  audit them; the residual risk is bounded by the 0.001 grid the values land on, an order of
  magnitude below the ±0.010 bar.
- **Reversing decision 1 later is cheap; reversing decision 5 is not.** A misfiled claim of
  reproduction in the invention disclosure would have to be corrected in front of counsel.
