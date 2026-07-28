# Pre-registration — Phase 1 Attack Spike (the kill-gate)

**Written:** 2026-07-28, BEFORE any attack code was run.
**Purpose:** fix the go/no-go criterion in advance so the dossier §7 STOP rule
("no attack ⇒ no defense") cannot be rationalised away after seeing results.

> **Commitment.** The thresholds below are frozen. If the result misses them, the
> honest outcome is STOP or a *documented, dated* revision of this file explaining
> what changed and why — never a quiet post-hoc adjustment. This file is committed
> to git; its history is the evidence that the bar predated the result.

---

## Hypothesis

A benign-looking input stream, fed to a TTT-E2E model in `train_mode="meta"`, can
steer its fast-weight updates so that performance on *later, held-out benign*
inputs is measurably worse than after a length-matched benign control stream —
without modifying the base (slow) weights.

## Setup (frozen)

- **Model:** TTT-E2E 1B, DCLM-pretrained + Books-finetuned @8K context (the
  released checkpoint). Confirmed reproducing in `../000-repro-baseline/` first.
- **Mode:** `meta` only. `pretrain` mode has no inner loop and is a vacuous null.
- **Hardware:** single A100/H100 80GB.
- **Seeds:** **5** independent seeds per condition. Each seed draws a fresh poison
  stream and a fresh length-matched benign control.
- **Streams:** report the **SELECT** strategy (reorder real benign sentences) as
  the headline. PARAPHRASE and SOFT reported separately as stronger/less-realistic
  points; they do not substitute for SELECT.

## Primary metric (frozen)

`corruption_metric` (`src/trustgate/eval/metrics.py`): benign-task loss after the
poison stream vs. after the control stream, across seeds. Reported as Cohen's *d*
and relative degradation.

## Decision thresholds (frozen)

**PROCEED** to the gate prototype only if **all** hold:

| Criterion | Threshold | Rationale |
|---|---|---|
| Effect size (Cohen's *d*, poison vs control) | **≥ 0.8** | Conventional "large" effect; must be clearly separable from seed noise |
| Relative benign degradation | **≥ 10%** | Below this the harm is real-but-trivial and would not motivate a defense |
| Fluency ratio (poison perplexity / control perplexity, under an independent reference model) | **≤ 1.5** | Above this the stream is not "benign-looking" and fails the §1 threat model |

Both statistical bars (top two) AND the realism bar (fluency) are required. A
corruption that only appears with a non-fluent stream does **not** pass.

**STOP** (or documented revise) if any bar is missed. Per dossier §7: no
demonstrated benign-looking attack means there is no defense worth building — do
not proceed to Phase 2.

## Secondary (reported, not gating)

- TRIGGER objective attack-success-rate (`attack_success_rate`). A working trigger
  strengthens the case but is not required to PROCEED; DEGRADE alone suffices.
- Number of stream tokens needed to reach threshold (attack "slowness").

## What would make this result invalid

- Control not length/seed/dtype-matched to poison (`RunCondition.assert_matches`
  guards this).
- Benign degradation measured on the stream itself rather than held-out data.
- Fluency scored with the victim model (must be an independent reference).
- Fewer than 5 usable seeds per condition.
