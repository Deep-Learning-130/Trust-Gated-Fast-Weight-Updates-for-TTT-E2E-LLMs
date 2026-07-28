# ADR-F1 — Gate signal

**Status:** Accepted **Date:** 2026-07-28 **Deciders:** project owner (+ patent counsel on the MedBN diff)

*Extracted and adopted from dossier §6. The dossier proposed it; this ADR records
it as the accepted decision for the codebase. Implementation is Phase 2, gated on
the Phase 1 attack spike.*

## Context

We must admit useful test-time fast-weight updates while rejecting poison, at low
overhead, in a way that is provably bounded and novel over batch-norm-statistics
defenses (MedBN, arXiv 2403.19326).

## Decision

Use **frozen-anchor consistency** (Option A) as the primary gate,
**update-uncertainty** (Option B) as a cheap secondary on every window, and an
**offline influence estimator** (Option C) only as an auditor and dependent claim.

## Options

| | Complexity | Cost | Novelty vs MedBN | Poison detectability |
|---|---|---|---|---|
| **A — frozen-anchor consistency** | Medium | Medium (extra probe forward pass) | High (behavioural, not BN-stat) | Good |
| **B — update-uncertainty threshold** | Low | Low | Medium | Medium |
| **C — influence / gradient estimate** | High | High | High | Best |

- **A** directly measures behavioural drift from the trusted baseline on a small
  rotating probe set. Clean novelty story; costs one anchor forward pass per
  admitted window. → `src/trustgate/gate/anchor.py`
- **B** near-free, reuses inner-loop signals (grad norm, loss spike). Weak against
  low-variance stealth poison — exactly the slow-attack threat model — so it
  backstops A, it is not a defense alone. → `src/trustgate/gate/uncertainty.py`
- **C** strongest signal, impractical online. Valuable offline as ground truth for
  the operating-point curves and as an independently claimable auditor.
  → `src/trustgate/gate/influence.py`

## Trade-off

A gives the best novelty-per-cost and a behavioural story that reads clearly away
from MedBN (TTT-E2E has no batch norm for a MedBN-style test to read). B backstops
A cheaply. C is the offline auditor and a dependent claim. The false-reject /
false-accept tension (dossier §5.6) is settled by the **operating-point curve**,
not a single threshold — so `gate/policy.py` takes thresholds as parameters.

## Consequences

- Adds one anchor forward pass on admitted windows; revisit if overhead exceeds the
  ~10% budget (dossier §5.1). Measure, don't assume.
- Establishes **bounded drift per window** as the headline, claimable property,
  enforced by `drift/accumulator.py` (breach latches; see its tests).
- Requires a documented line-level diff vs. MedBN before filing — a hard gate
  (dossier §7). See `docs/patent/prior-art/medbn-diff.md`.

## Action items

1. [ ] Reproduce TTT-E2E + attack (kill-gate) — `experiments/001-attack-spike/`.
2. [ ] Implement A + B; sweep operating points.
3. [ ] Diff claim vs. MedBN / robust-TTA; decide filing.
