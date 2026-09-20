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

---

## Addendum 2026-09-20 -- sequence-position arms (secondary, non-gating)

**This addendum adds arms. It moves no bar.** The three PROCEED criteria above
are exactly as frozen on 2026-07-28: effect size >= 0.8, relative degradation
>= 10%, fluency ratio <= 1.5, all three required. Nothing below is consulted in
the PROCEED/STOP decision, and `report.render_sequence_markdown` renders no
verdict line at all so the two artefacts cannot be confused.

### Why

The frozen comparison measures benign loss once, after the whole stream, poison
against a length-matched benign control. Two things it cannot see:

- **The drift floor.** Fast weights move on any input at all, so the frozen
  comparison measures harm *above* a floor it never measures.
- **Whether accumulation is the mechanism.** The threat model in section 1
  accuses carry across windows specifically. Harm that survives resetting the
  carry before every window was never an accumulation effect.

### What is added

The stream is measured after every window (one window = one inner step) rather
than only at the end, giving a degradation-vs-position curve. This also supplies
the "attack slowness" secondary already listed above, which had no
implementation.

| Arm | Carry | Reads |
|---|---|---|
| `poison` | threaded | the poison stream |
| `control` | threaded | the length-matched benign control |
| `poison_no_carry` | reset each window | the poison stream |
| `control_no_carry` | reset each window | the benign control |
| `floor` | fresh, no stream | nothing |

Derived, all reported and none gating: `accumulation_excess`,
`drift_floor_excess`, `onset_window` (`src/trustgate/eval/metrics.py`).

The end-of-stream values of `poison` and `control` are the same quantity
`corruption_metric` consumes, and
`test_final_position_equals_the_single_shot_endpoint` holds the two
orchestrators to the same number. If they ever diverge, that is a bug in the
harness and not a second opinion about the result.

### What this does not change

- No threshold, in either direction.
- Not the primary metric: still benign-task loss on held-out data, not
  generated text.
- Not the invalidating conditions below, which apply to these arms too. Both
  no-carry arms exist precisely so the accumulation comparison is matched.

---

## Revision 2026-09-20 -- Phase 2 started before the verdict

**Decided by:** Manas Maahir, project owner.
**This revision moves no bar.** The three PROCEED criteria frozen on 2026-07-28
are unchanged: effect size >= 0.8, relative degradation >= 10%, fluency ratio
<= 1.5, all three required. The STOP rule still applies exactly as written, and
a STOP is still the honest outcome if any bar is missed.

### What changed

The Commitment clause above allows a documented, dated revision and forbids a
quiet one. This is that document.

Dossier section 7 and the Standing Rule the CI `hygiene` job enforces say: no
demonstrated attack means no defense worth building, and `src/trustgate/gate/`
stays stubbed until 001 returns PROCEED. **Phase 2 is being implemented before
001 has run.** The owner was advised of the rule twice and chose to build both,
so that a single GPU session can produce the attack verdict and the gate
measurement together rather than requiring a second rental.

### What it costs

Stated plainly, because a revision that only records the decision and not its
price is not worth writing.

- **If 001 returns STOP, the Phase 2 work is sunk cost.** It will have been
  built to defend an attack that was not demonstrated.
- **The project loses a claim it cannot get back.** `ORIENTATION.md` section 1
  calls it *the discipline*: that we refused to build the defense until the
  attack was demonstrated. After this revision the honest statement is weaker --
  that the defense was built in parallel, with the attack verdict still open.
- **The incentive to soften a STOP is now higher than it has ever been**, since
  a STOP also writes off the gate work. That pressure is the reason the bars
  are frozen in a file with a git history, and the reason this paragraph is
  here rather than only in a commit message.

### What is being built, and what is not

Implemented: `gate/anchor.py` (frozen-anchor behavioural consistency over a
rotating probe set), `gate/uncertainty.py`, `gate/policy.py`,
`probes/rotating.py`.

**Not implemented: the bounded-drift guarantee.** This is a structural blocker
recorded in `docs/adr/ADR-003-jax-interceptor-shape.md`'s 2026-08-08 correction,
not a scoping choice. The vendor's scan carry is fixed at
`(model, inner_opt_state, (state_all, state_suffix))` and a `scan` requires the
carry structure out to match the structure in, so the interceptor cannot thread
`DriftState` or `VersionedStore` without editing vendor code, which ADR-002
forbids. Both modules remain implemented, unit-tested and unreachable from the
hook. `docs/adr/ADR-P3-1-drift-carry.md` chooses a route on paper; no code
follows it until 001 returns PROCEED.

### Pre-verdict design defaults

`ADR-F1` and the `gate/anchor.py` docstring defer three questions "to settle
with Phase 1 data in hand": probe set size and rotation policy, the divergence
measure, and whether to score against theta_0 or the last trusted checkpoint.
**That data does not exist yet.** The values chosen are guesses made before the
verdict, labelled as such in code and docs, and they are not findings. Anything
downstream that reads them as tuned results is reading them wrong.

### What this does not change

- No threshold, in either direction.
- Not the primary metric, and not the invalidating conditions.
- Not the STOP rule. `clean_regression` was defined in advance for the same
  reason the bars were, and the Phase 2 numbers are reported whatever they say.
