# Prior-art diff — trust gate vs. MedBN (arXiv 2403.19326)

**Status: UNFILLED — HARD FILING GATE.** Per dossier §7, we do **not** file if the
mechanism reads onto MedBN or other robust-TTA defenses after a line-level diff.
This document must be completed and reviewed by counsel *before* a provisional is
filed. It exists now, empty, so the obligation is visible from day one rather than
discovered at the end.

## Why MedBN is the reference to clear

The shortlist flagged MedBN as the closest existing poisoning *defense*. It is the
one an examiner is most likely to cite. Clearing it explicitly is the difference
between a filing and a rejection.

## What MedBN actually is (to be confirmed against the paper, not memory)

- **Setting:** test-time adaptation for image classification (CIFAR10-C,
  CIFAR100-C, ImageNet-C).
- **Mechanism:** replace the *mean* in batch-norm statistics with the *median*,
  which a single malicious sample cannot swing. Algorithm-agnostic drop-in for
  existing TTA methods.
- **Signal:** distributional — robust statistics over a test batch.

## What our gate is

- **Setting:** long-context language modeling with TTT-E2E fast weights. **No
  batch-norm layers exist** — there is no BN statistic for a MedBN-style test to
  operate on.
- **Mechanism:** intercept each proposed fast-weight update; score behavioural
  consistency against a frozen meta-learned anchor on a rotating probe set;
  enforce a bounded cumulative-drift budget with checkpoint/rollback.
- **Signal:** behavioural — divergence of model *outputs* from the anchor's,
  plus a system-level drift bound.

## Diff table (TO COMPLETE)

| Axis | MedBN | Trust gate (F1) | Distinct? |
|---|---|---|---|
| Domain | vision TTA | fast-weight LLM continual inference | TBD |
| Unit defended | BN statistics | fast-weight update Δθ | TBD |
| Signal type | distributional (median of stats) | behavioural (anchor consistency) | TBD |
| Guarantee | robust statistic | bounded cumulative drift per window | TBD |
| Rollback / versioning | none | ring-buffer checkpoints, O(1) rollback | TBD |
| Online vs batch | per-batch statistic | per-window streaming interception | TBD |

## The trap to avoid (dossier §3, Trap #2)

"Apply an existing TTA defense to a new system" is only defensible because
(a) fast weights ≠ BN statistics — there are no BN stats here at all, and (b) the
LLM-agent threat model is new. State both explicitly, with citations. If the diff
comes back thin on any row, that row is a **do-not-file** signal, not a wording
problem to paper over.

## Also diff against (stubs)

- Generic robust-TTA beyond MedBN — see `robust-tta-survey.md`.
- The MedBN-analogue baseline we build ourselves
  (`src/trustgate/baselines/medbn_analogue.py`) is our *empirical* comparison; it
  is not a substitute for this *legal* diff.
