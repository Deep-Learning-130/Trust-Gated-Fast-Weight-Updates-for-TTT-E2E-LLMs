# F1 — Trust-Gated Fast-Weight Updates for TTT-E2E LLMs

**Verdict:** Likelihood **High** · Impact **Medium** · Patent-shape: clean method, **weak enforceability** (internal gating is invisible in a competitor's product). Best fit for the efficiency/SLM home turf; best treated as a *fallback* for a defensive portfolio.

> **Framing (applies to all fresh-frontier ideas).** This is a defense/monitoring method. Those are hard to *enforce* — you usually can't prove a competitor runs your gate inside their stack — so for a *defensive* portfolio the deterrence value is limited. Route the decision through the goal fork: if the objective is enforceable patents, this ranks behind F3 and behind DKT-01 (the mutable-KV serving manager); if the objective is a solid, publishable systems-security contribution on a hot architecture, it's strong.

---

## 1. The idea

**TTT-E2E** (End-to-End Test-Time Training, arXiv 2512.23675) reframes long-context language modeling as *continual learning at inference*: the model carries "fast weights" that it updates via next-token prediction as it reads the context, compressing the context into weights and matching full-attention scaling at near-RNN cost.

**A model that learns while it serves can be poisoned while it serves.** A crafted input *stream* — not a single adversarial token, but a slow, benign-looking sequence — could steer the fast-weight updates to degrade behavior on later benign inputs, or implant a latent trigger, all without touching the base (slow) weights.

**The invention is the defense:** a **trust gate** that admits a fast-weight update only if it passes a consistency/uncertainty check against a frozen anchor, subject to a **bounded cumulative-drift budget** and backed by **checkpoint/rollback**. The patentable core is the gate + drift-budget + rollback controller as a system, not the attack.

---

## 2. Verification & prior-art diff (as of Jul 2026)

| Item | arXiv / venue | Status | Meaning for F1 |
|---|---|---|---|
| TTT-E2E (paradigm) | 2512.23675 (Yu Sun, Dalal, X. Wang, Leskovec, Hashimoto, Guestrin, Choi et al.) | ✅ Real, high-profile | Target architecture confirmed; public code exists; solo-buildable |
| Test-time poisoning attacks | 2308.08505 (IEEE S&P), 2410.04682 (ICLR'25), R.I.P. 2412.01154 | ✅ Real | Attack side is **mature** — not your novelty |
| **MedBN** — robust TTA defense | 2403.19326 | ✅ Real | **Corrects the shortlist**: a poisoning *defense already exists* (batch-norm-statistics based). Your gate must be **distinct from BN-stat defenses** |

**Open (your lane):** poisoning *of TTT-E2E-LLM fast weights specifically*, and a *gate mechanism tailored to fast-weight continual-inference updates* (not BN statistics). Both appear untouched.
**Not open:** the generic "TTA can be poisoned / here's a defense" narrative. Lead with the fast-weight-LLM setting + a non-BN gate, or you read onto MedBN.

---

## 3. Brainstorm read (Fischbach–Walsh lenses)

- **Fix ONE parameter:** the setting — TTT-E2E-LLM fast weights. **Let float:** the gate signal (anchor-consistency vs. update-uncertainty vs. influence estimate).
- **Risk matrix (1 = safe, 5 = miracle):**

  | Assumption | Risk | Note |
  |---|---|---|
  | Fast-weight poisoning is feasible on TTT-E2E | 3 | Plausible by analogy to TTA; **unproven on this arch** — the demo *is* the de-risk |
  | A gate separates poison from benign drift without killing useful adaptation | 3 | Central tension; needs an operating-point curve |
  | Gate is novel & effective vs. MedBN / robust-TTA | 3 | Line-level diff mandatory before filing |
- **Multiple miracles?** No — single mechanism. ✅ (the "befriend risk, but only one miracle" sweet spot)
- **Traps:** Trap #2 (applying an existing tool — a TTA defense — to a new system). Defensible only because fast weights ≠ BN stats and the LLM-agent threat model is new. Watch for Trap #3 (over-loving a clean story) — the attack must be shown *real* first.

---

## 4. Execution plan

| Phase | Timebox | Work | Go / No-Go |
|---|---|---|---|
| **Attack spike** | 2 wks | Reproduce TTT-E2E at 1–3B on public code; implement a crafted-stream poisoning attack on the fast-weight update | **If no benign-looking stream measurably corrupts fast weights → STOP.** No attack ⇒ no defense |
| **Gate prototype** | 4–6 wks | Implement trust gate (frozen anchor + consistency/uncertainty score + bounded-drift accumulator + rollback); operating-point sweep | Gate must suppress attack while retaining clean accuracy |
| **Eval harness** | (parallel) | Benign long-context tasks + injected poison streams; ROC of gate, clean-task regression, compute overhead | Beat a MedBN-style baseline |
| **Filing gate** | — | File provisional **only if** gate beats baseline **and** novelty diff vs MedBN is clean | Legal + empirical both green |

---

## 5. System design

### 5.1 Requirements

**Functional**
- Intercept every proposed fast-weight update Δθ before it commits.
- Cheaply score the trustworthiness of Δθ against a frozen anchor.
- Maintain a cumulative-drift measure from the anchor; enforce a per-window budget ε.
- Checkpoint fast-weight state; roll back to the last trusted checkpoint on budget breach.
- Log accept/reject/rollback decisions for audit.

**Non-functional**
- Gate compute overhead ≤ ~10% of the per-window update cost (target; measure and report).
- Clean long-context accuracy regression ≤ small ε_acc vs. ungated TTT-E2E.
- Provably bounded drift per window (a stated, testable guarantee — key for the claim).
- Streaming/online; strict per-request isolation; **no access to other users' data** (also closes the strongest attack channel).

**Constraints**
- Public TTT code; single-GPU prototyping at 1–3B; no foundation-model training.

### 5.2 Component architecture

```
                       ┌───────────────────────────── frozen ─────────────┐
                       │              ANCHOR MODEL (meta-learned init θ0)   │
                       └───────────────┬───────────────────────────────────┘
                                       │ reference activations / logits
context   ┌──────────────┐  Δθ  ┌──────▼───────────────────────────────┐ commit ┌───────────────────┐
stream ──▶│ Fast-weight  │─────▶│              TRUST GATE               │───────▶│ Versioned fast-   │
          │ updater      │      │  • Anchor-consistency scorer          │        │ weight store      │
          └──────────────┘      │  • Update-uncertainty scorer          │◀──────▶│ (checkpoint +     │
                    ▲           │  • Decision policy (accept/reject)    │ roll   │ ring-buffer)      │
                    │           └──────────────┬───────────────────────┘ back   └───────────────────┘
                    │                          │ drift delta
                    │                 ┌────────▼─────────┐  budget exceeded → rollback signal
        rolled-back state            │ Drift accumulator│───────────────────────────────┐
                    └────────────────│ (budget ε/window)│                                │
                                     └──────────────────┘                                ▼
                                                                              last-trusted checkpoint
```

**Component notes**
- **Update Interceptor** — sits between the fast-weight updater and the committed weights; the enforcement point.
- **Anchor model** — frozen meta-learned init; provides a stable reference so "has this update pulled behavior away from the trusted baseline on held-out probes?" is answerable cheaply.
- **Trust Gate** — the novelty. Anchor-consistency scorer compares behavior on a small rotating probe set before/after Δθ; uncertainty scorer flags high-variance/low-confidence updates; decision policy combines them.
- **Drift accumulator** — maintains cumulative distance from anchor; the ε budget makes "bounded drift" a provable property rather than a hope.
- **Versioned fast-weight store** — ring-buffered checkpoints enabling O(1) rollback to the last trusted state (shared primitive with idea N2).

### 5.3 Data flow
1. Updater proposes Δθ for the current context window.
2. Interceptor pauses commit; Trust Gate scores Δθ (consistency + uncertainty).
3. If accept → commit, update drift accumulator, periodically checkpoint.
4. If reject → discard Δθ, log, continue.
5. If drift budget exceeded → roll back to last trusted checkpoint, reset budget, raise alert.

### 5.4 Interfaces
- `gate(delta_theta, anchor_state, probe_set) → {decision: accept|reject, drift_delta, confidence}`
- `accumulate(drift_delta) → {budget_remaining, breach: bool}`
- `checkpoint(fast_weights) → version_id` · `rollback(version_id) → fast_weights`

### 5.5 Error handling & reliability
- Gate low-confidence ⇒ conservative reject + log (fail-safe toward rejecting).
- Budget breach ⇒ rollback + alert; never silently continue on a drifted state.
- Checkpoint ring buffer sized to cover the attack-detection latency window.

### 5.6 Trade-off analysis
- **False-reject** (loses useful adaptation, hurts clean accuracy) vs. **false-accept** (admits poison). The deliverable is the operating-point curve, not a single threshold.
- **Overhead vs. sensitivity:** more probes / an extra anchor pass improves detection but costs compute. Anchor-consistency (Option A below) trades a forward pass for detection quality.

---

## 6. ADR-F1 — Gate signal

**Status:** Proposed **Date:** Jul 2026 **Deciders:** you (+ eventual patent counsel on the MedBN diff)

**Context.** We must admit useful test-time fast-weight updates while rejecting poison, at low overhead, in a way that is provably bounded and novel over BN-statistics defenses.

**Decision.** Use **frozen-anchor consistency** as the primary gate, **update-uncertainty** as a cheap secondary, and an **offline influence estimator** only as an auditor.

**Options**

### Option A — Frozen-anchor consistency
| Dimension | Assessment |
|---|---|
| Complexity | Medium |
| Cost | Medium (extra probe forward pass) |
| Novelty vs MedBN | High (behavioral, not BN-stat) |
| Poison detectability | Good |

**Pros:** directly measures behavioral drift; clean novelty story. **Cons:** extra compute per admitted window.

### Option B — Update-uncertainty threshold
| Dimension | Assessment |
|---|---|
| Complexity | Low |
| Cost | Low |
| Novelty vs MedBN | Medium |
| Poison detectability | Medium |

**Pros:** near-free. **Cons:** weaker against low-variance stealth poison.

### Option C — Influence / gradient estimate
| Dimension | Assessment |
|---|---|
| Complexity | High |
| Cost | High |
| Novelty | High |
| Poison detectability | Best |

**Pros:** strongest signal. **Cons:** too slow for the online path.

**Trade-off analysis.** A gives the best novelty-per-cost and a clean behavioral story; B backstops A cheaply on every window; C is impractical online but valuable as an offline auditor and as a dependent claim.

**Consequences.** Adds one anchor forward pass on admitted windows (revisit if overhead > budget). Establishes the bounded-drift guarantee as the headline property. Requires a documented line-level diff vs. MedBN before filing.

**Action items**
1. [ ] Reproduce TTT-E2E + attack (kill-gate).
2. [ ] Implement A + B; sweep operating points.
3. [ ] Diff claim vs. MedBN / robust-TTA; decide filing.

---

## 7. Risks & kill criteria
- **Kill** if no crafted stream corrupts fast weights (no attack surface).
- **Kill / repane** if the gate can't beat a MedBN-style baseline.
- **Do not file** if the mechanism reads onto MedBN or other robust-TTA defenses after a line-level diff.
- **Discount for portfolio** regardless of technical success: internal gating is not externally observable ⇒ low deterrence.

---

## 8. Sources
- TTT-E2E — End-to-End Test-Time Training for Long Context: arXiv 2512.23675
- Test-Time Poisoning Attacks Against TTA: arXiv 2308.08505 (IEEE S&P)
- Realistic Test-Time Data Poisoning: arXiv 2410.04682 (ICLR 2025)
- R.I.P. black-box attack on continual TTA: arXiv 2412.01154
- MedBN — robust TTA against malicious samples (defense prior art): arXiv 2403.19326

*Research-novelty + design dossier, not a freedom-to-operate opinion. A formal FTO search (esp. vs. MedBN) is required before filing.*
