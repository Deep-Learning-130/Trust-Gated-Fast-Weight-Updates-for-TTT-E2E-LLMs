# Invention Disclosure — Trust-Gated Fast-Weight Updates for TTT-E2E LLMs

**Status: DRAFT SKELETON — not for filing.** Phase 2+. Sections are stubbed to
attorney-ready structure so the empirical work fills them in as it lands, rather
than being reconstructed at the end.

> **Already publicly disclosed.** This repository — including the mechanism described
> below — has been public since **2026-08-01**, a deliberate posture as of 2026-08-02.
> Confidentiality is no longer available and must not be claimed here. Rights in
> absolute-novelty jurisdictions (EPO/CN/JP/KR) are forfeit; the **US grace period runs to
> approximately 2027-08-01**, which is a filing deadline rather than a backstop. Full
> posture and the open questions for counsel: [`/DISCLOSURE.md`](../../DISCLOSURE.md).
>
> Two things this does *not* change: material covering a **future** invention not yet in
> the repo still has its full options and should reach counsel before it reaches `main`;
> and working attack artifacts still wait on coordinated disclosure to the TTT-E2E authors.

**Inventor(s):** <Enter Full Legal Name(s) of Inventors>
**Date of conception:** on/around 2026-07-28 (this repo's initial commit; the git
history is the supporting record — do not rewrite it).
**Date of first public disclosure:** **2026-08-01** (repository made public). This is the
date the US grace period runs from — record it with the same care as conception.
**Reduction to practice:** <Enter Exact Date of first working Phase 1/2 prototype> (Phase 1 attack spike + Phase 2 gate).

---

## 1. Field

Security of machine-learning systems that perform continual learning at inference
time; specifically, defending the test-time "fast-weight" updates of end-to-end
test-time-trained (TTT-E2E) language models against input-stream poisoning.

## 2. Problem

A model that updates its weights while it serves can be corrupted while it serves.
A slow, benign-looking input stream can steer fast-weight updates to degrade later
benign performance or implant a latent trigger, without ever touching the base
(slow) weights. (Establish the attack is real: `experiments/001-attack-spike/`.)

## 3. Prior art and its shortcomings *(cross-ref `prior-art/medbn-diff.md`)*

- Test-time poisoning attacks exist (2308.08505, 2410.04682, 2412.01154) — the
  *attack* side is mature and is not claimed.
- Poisoning *defenses* exist for vision TTA — notably MedBN (2403.19326), which
  robustifies batch-norm statistics. TTT-E2E has no batch norm, so MedBN does not
  apply, and no defense targets fast-weight LLM updates. **This is the gap.**

## 4. Summary of the invention

A trust gate that intercepts each proposed fast-weight update before commit and
admits it only if it passes a consistency check against a frozen anchor, subject
to a bounded cumulative-drift budget, backed by checkpoint/rollback.

## 5. Detailed description *(enablement — reference the implementation)*

- **5.1 Update Interceptor** — enforcement point between updater and committed
  weights. `src/trustgate/interceptor.py`; hook rationale in `ADR-003`.
- **5.2 Frozen anchor** — meta-learned init θ₀; behavioural reference on rotating
  probes. `src/trustgate/gate/anchor.py`, `src/trustgate/probes/`.
- **5.3 Trust gate** — anchor-consistency (primary) + update-uncertainty
  (secondary) + decision policy. `src/trustgate/gate/`; `ADR-F1`.
- **5.4 Drift accumulator** — cumulative distance from anchor; per-window budget ε;
  **breach latches**. This is the bounded-drift guarantee.
  `src/trustgate/drift/accumulator.py`.
- **5.5 Versioned fast-weight store** — ring-buffered checkpoints, O(1) rollback.
  `src/trustgate/store/versioned.py`.
- **5.6 Audit log** — accept/reject/rollback trail. `src/trustgate/audit/log.py`.

## 6. Claim seeds *(counsel drafts actual claims)*

1. **Independent (system):** interceptor + anchor-consistency gate + bounded-drift
   accumulator + versioned rollback store, operating on fast-weight updates of a
   test-time-trained LM.
2. **Independent (method):** the intercept → score → accept/reject → accumulate →
   rollback-on-breach loop.
3. Dependent: uncertainty secondary signal; rotating probes; behavioural
   divergence measure; **offline influence auditor** (`gate/influence.py`);
   ring-buffer depth sized to detection latency; reject-reverts-weights-keeps-state.

## 7. Bounded-drift guarantee *(the headline claim property)*

Over any window, committed fast weights cannot travel more than ε from the anchor
before a rollback is forced. Stated and testable — see the latch tests in
`tests/test_accumulator.py`. Formalise the statement and proof here.

## 8. Enablement support / experimental results *(fill from experiments)*

- Attack demonstration: `experiments/001-attack-spike/` (+ pre-registration).
- Gate operating-point curves, clean-accuracy regression, overhead: Phase 2.
- Beats the MedBN-analogue baseline (`baselines/medbn_analogue.py`): Phase 2.

## 9. Enforceability note *(business context, not a claim)*

Internal gating is not externally observable in a competitor's product (dossier
verdict, §7). Deterrence value for a defensive portfolio is limited regardless of
technical success. Recorded so the filing decision is made with eyes open.
