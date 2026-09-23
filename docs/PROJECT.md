# TrustGate — the project

**What this is.** One of three documents that together describe the whole project:

| | |
|---|---|
| **`PROJECT.md`** (this file) | The idea, the threat, the defense, the discipline, and where it stands. |
| [`CODEBASE.md`](CODEBASE.md) | Every module, script, experiment and document — what it is and where. |
| [`OPERATIONS.md`](OPERATIONS.md) | How to run it, how to publish it, and what is currently broken or pending. |

[`ORIENTATION.md`](ORIENTATION.md) is the long-form companion to this file: the same
material at four times the length, with a full glossary and the per-finding detail. Read
this one first.

Status line, 2026-09-23: **the kill gate has not been run, there is no verdict, and total
spend is $0.13.**

---

## 1. The idea in one page

**The observation.** A new class of language model — *TTT-E2E* — keeps learning after it
is deployed. As it reads your input it modifies a subset of its own weights by training on
that input. That is what lets it handle very long contexts cheaply.

**The security consequence.** A model that learns while it serves can be poisoned while it
serves. An attacker who controls part of the input stream may be able to steer those live
updates so the model performs worse on **later, unrelated, perfectly innocent inputs** —
without touching the stored weights, without anything that looks like an attack in the
text, and without the operator noticing.

**The proposed defense.** A *trust gate*: a component between the model's update machinery
and the weights, which refuses to commit an update unless it passes a cheap trustworthiness
check, tracks how far the weights have cumulatively drifted from a frozen reference, and
forces a rollback when that drift exceeds a budget. The bounded-drift property is the
defensible part — a stated, testable guarantee rather than a claim about detection
accuracy.

**The discipline.** The project refuses to build the defense until the attack is
demonstrated. That decision point is the **kill gate**, and its criteria were frozen in
writing on 2026-07-28, before any attack code was run. If the attack does not materialise,
the correct outcome is **STOP**: publish the negative result and build nothing.

**Where it stands.** The measurement apparatus is built and tested. The kill gate has not
been run, so no verdict exists. Two serious flaws were found in the apparatus before it
could produce a wrong verdict — one in the upstream research code, one in our own pilot.
Both are fixed and documented.

---

## 2. Background: the two weight sets

A conventional transformer has fixed weights; training sets them, inference reads them.
Long context means attending over every previous token, at quadratic cost.

**Test-Time Training** treats the context not as something to attend over but as something
to *learn from*: as the model reads, it runs real gradient-descent steps on a small subset
of its parameters, using next-token prediction as the objective. The context is compressed
*into weights* rather than held in an attention cache. **TTT-E2E**
([arXiv 2512.23675](https://arxiv.org/abs/2512.23675)) is the end-to-end version for
long-context language modeling. It is real, recent, high-profile and public, which is why
it is the target architecture here.

The single most important distinction in the project:

| | **Slow weights** | **Fast weights** |
|---|---|---|
| Also called | base weights, outer parameters | inner parameters, `feed_forward_prime` |
| Set by | pre-training and fine-tuning | the inner loop, *during inference* |
| Changed at serving time | No | **Yes, constantly** |

An attack on fast weights leaves no trace in the model file. The checkpoint on disk is
byte-identical before and after. That is what makes this a distinct threat rather than
data poisoning under a new name.

---

## 3. The defense

The enforcement point is the **Update Interceptor** (`src/trustgate/interceptor.py`,
dossier §5.2). A proposed fast-weight update arrives; something decides accept, reject or
rollback.

| Signal | Module | Role |
|---|---|---|
| **A — frozen-anchor consistency** | `gate/anchor.py` | **The primary gate.** Score the proposed update against a frozen anchor on held-out probes. |
| B — update uncertainty | `gate/uncertainty.py` | Cheap secondary. |
| C — influence / gradient estimate | `gate/influence.py` | **Offline auditor only.** ADR-F1 keeps it off the serving path by design. |
| Policy | `gate/policy.py` | Combines the signals into accept / reject / rollback. |
| Drift budget | `drift/accumulator.py` | Cumulative drift against a per-window budget. |
| Rollback | `store/versioned.py` | Ring-buffered checkpoints, O(1) restore on breach. |

The gate costs something, and the budget is ~10% overhead (`ADR-F1` §5.1), measured by
`eval/overhead.py`.

**What is not enforced.** The drift/rollback half is implemented but not wired end to end.
The paper states this explicitly rather than implying a complete overlay. See `ADR-003`'s
correction and `ADR-P3-1`.

---

## 4. The kill gate

Four criteria, frozen 2026-07-28 in
[`experiments/001-attack-spike/PREREGISTERED.md`](../experiments/001-attack-spike/PREREGISTERED.md),
**all required jointly**:

| Criterion | Bar |
|---|---|
| Effect size (Cohen's d), poison vs control | ≥ 0.80 |
| Relative benign degradation | ≥ 10% |
| Fluency ratio vs control (max across seeds) | ≤ 1.50 |
| Usable seeds per condition | ≥ 5 |

The bars live in code as `src/trustgate/eval/prereg.py`, marked FROZEN, with a CI test that
parses the markdown and asserts equality. **Thresholds do not move once a result has been
seen.**

If the bars are not met jointly: **STOP**. Publish the negative result, build nothing
further.

---

## 5. The experiments

Numbered in dependency order, not chronology.

### 000 — Reproduce the baseline · *not run*

Run the unmodified upstream evaluation on the released 1B checkpoint and confirm the
environment is sound. A gate, not a formality: if upstream behaviour cannot be reproduced
on our hardware, any "corruption" measured later is unattributable.

`ADR-005` records the subtlety: **the paper reports no number for the checkpoint we intend
to evaluate.** So the bar is not "match the published number" but a two-sided bracket from
sign-determined monotonicity — **2.314 < loss < 2.805** nats/token — plus structural checks
carrying most of the discriminating power. Fixed in `TOLERANCE.md`.

Blocked on GPU access and a requester-pays download.

### 001 — Attack spike · *the kill gate* · **not run**

The PROCEED/STOP decision. The directory holds `PREREGISTERED.md`, `README.md` and
`FLUENCY_REFERENCE.md`. There is **no `results/` directory**, and that absence is the
honest state of the project.

Since 2026-09-16 the path is executable: `--random-init` builds a victim with no checkpoint
and exercises every seam a real run would — stream crafting, the five-seed loop,
run-condition matching, carry threading, fluency scoring, report generation. A model with
no weights has learned nothing, so it has nothing worth corrupting, and the report carries
a `NOT A RESULT` banner *above* the verdict line. The only missing input is the weights.

### 002 — Scaled-down pilot · *complete*

The smallest honestly TTT-shaped stand-in: a ~0.1M-parameter byte-level model with fast
weights updated by next-token prediction at inference, carried across chunks, base weights
frozen. ~35 seconds on CPU. No GPU, no download, no network, no spend.

What it deliberately does not claim: it is not TTT-E2E, the corpus is this repository's own
prose rather than Books3, there is no attention mechanism, and the fluency bar is
**unscored by construction**. It renders no verdict and touches no pre-registered bar.

### 003 — 125M smoke test · *partially complete*

First execution of the real upstream model and our carry overlay against it, at 125M
parameters, on free hardware. Instrument validation, not measurement.

The model executes; seven distinct first-launch failures were found and fixed, which is
exactly the value this tier exists to produce. The final check — "the carry is
non-trivial" — is blocked: an 8 GB consumer card is roughly 2 GB short.

---

## 6. Status

| | Status |
|---|---|
| Phase 0–1 scaffold | Complete |
| Phase 1 CPU implementation | Complete, 473 tests passing |
| Fluency reference model | Complete 2026-08-27, verified against HuggingFace GPT-2 |
| Phase 1 spike runnable end to end | Complete 2026-09-16, against a weightless victim |
| Cost model probed against the live bucket | Complete 2026-09-14 |
| Continuous integration | Complete |
| 002 pilot | Complete and reproducible |
| 003 smoke — upstream model executes | Complete |
| 003 smoke — carry verified non-trivial | **Blocked on GPU memory** |
| Checkpoint loading | Complete 2026-09-20. Refuses a partial restore rather than warning. |
| Real attacker wired to the CLI | Complete 2026-09-20 |
| 000 baseline reproduction | **Not run** |
| **001 kill gate** | **Not run** |
| Trust gate (Phase 2) | Anchor, uncertainty, policy, probe rotation implemented 2026-09-20 **before the verdict**. Bounded drift **not enforced**. |
| **Verdict** | **None** |
| **Total spend** | **$0.13** |

**Deliberately empty:** `gate/influence.py`, which ADR-F1 keeps offline by design, and the
MedBN comparison baseline, which needs per-example gradients the vendor's
`inner_loop_step` does not produce. Both are documented stubs naming what they must do and
why they are not done.

**Built before the evidence that would justify it:** the gate. The owner decided on
2026-09-20 to build Phase 2 without waiting for the 001 verdict, and `PREREGISTERED.md`
carries a dated revision recording who decided, why, and what it costs if the verdict is
STOP. The design values inside `gate/` are pre-verdict guesses labelled as such in code.
**They are not findings.**

**The single remaining technical risk** is binding the carry overlay to the real upstream
model. Everything downstream is written and tested.

**The largest untested assumption is cost, not correctness.** One ordering-search proposal
is one full adapt-and-eval — roughly 16 inner steps plus 2 prefix passes at 8192/1024 — and
`--max-iters` multiplies that by five seeds. Measure a single evaluation on the box before
choosing the budget; `run_deep.py`'s value of 40 was picked on CPU against a tiny model and
does not transfer.

---

## 7. The five findings that changed the plan

All obtained at a desk, for zero spend. Each would otherwise have cost a paid GPU session
or, worse, a confidently wrong answer. This is the most valuable output the project has
produced so far.

### 7.1 The upstream code throws the adapted weights away

`vendor/ttt-e2e/ttt/model/transformer.py:712` discards its scan carry. Fast weights are
adapted within a chunk and then **reset**.

Run the kill gate against stock upstream code and the consequence is mechanical: nothing
accumulates, so the measurement finds no effect, so the verdict is **STOP** — and it would
have looked exactly like a genuine negative result about the architecture's security.
`ADR-006` calls this a *structurally guaranteed false STOP*. Found by reading the source
before any GPU existed. Fixed by the carry overlay in `eval/carry.py`.

### 7.2 Our own pilot was unfalsifiable

Until 2026-09-15, experiment 002 built its corpus by **globbing this repository's own
markdown files at runtime**. Every commit silently changed the experiment's input; the
committed results predated the code that produced them. Nobody, including us, could have
reproduced those numbers. Found while building a live demo, which is a reasonable argument
for building live demos.

Fixed: the corpus is frozen to a committed 615,141-byte file, its SHA-256 pinned in source,
the loader verifies the digest and **refuses to fall back to a glob**, and the file is
marked binary in `.gitattributes` so line-ending conversion cannot break the digest on a
fresh clone.

### 7.3 The pre-registered threshold reports noise at n = 5

The d ≥ 0.80 bar is cleared by a negative control with no attacker present, often. See
`OPERATIONS.md` §5 — this is the live open issue, and the measured false-positive rate is
now a contribution of the paper in its own right.

### 7.4 The validation split is twice the largest figure the plan allowed for

`COST_MODEL.md` §2.2 budgeted a `/val` split of 50M–1B tokens. A free probe on 2026-09-14
measured **2,000,168,321 tokens, 8.4 GB uncompressed**, and flagged itself `OUT_OF_BAND`.
That turns one evaluation pass into ~7.5 GPU-hours, and `TOLERANCE.md`'s bar S2 requires
the identical command twice — a fifteen-hour job, colliding with the project's own
12-GPU-hour stop rule.

Recoverable because the number of evaluation batches is not configuration:
`ttt/dataloader/lm_dataset.py:27` derives it from the zarr array's declared **shape**, so
copying K chunks and rewriting `shape` in our own local `val/zarr.json` bounds the pass
exactly without touching the vendor tree. Tool: `scripts/make_val_subset.py`.

The trap worth naming: absent zarr chunks read as the **fill value** rather than raising, so
a declared shape larger than the chunks on disk evaluates the model on padding and returns
a confident, meaningless loss with no error anywhere.

### 7.5 The evaluation batch is 8 whatever the configuration says

`train.py:211` takes `max(eval_batch_size, global_batch_size // accum_steps * 4)`, flooring
the batch at **8** however low `global_batch_size` is set. Eight chunks of
`[1024, 128256]` logits, cast to fp32 *before* the log-softmax at `loss.py:18`, is where
the memory goes. Corrected estimate **36–49 GB**, which straddles 40 GB where the old
figure sat comfortably below it — so the recommendation moves to **80 GB**. The difference
between a session that runs and a session that pays for an out-of-memory error.

---

## 8. Working rules

- **Pre-registration is binding.** Thresholds do not move after a result is seen. A CI test
  keeps `prereg.py` equal to `PREREGISTERED.md`.
- **A negative control is run and reported whichever way it lands.** This is the only
  reason §7.3 was caught instead of published.
- **A seed where the poison *helped* the victim must be visible, not averaged away**
  (Standing Rule 6 — enforced in `make_figures.py`'s per-seed plot).
- **No number without its inputs.** Digests are checked and printed; a result that cannot
  name what produced it is not a result.
- **Rehearse for free before paying.** `preflight_wandb.py`, `make_dummy_checkpoint.py` and
  the `--random-init` path exist so a paid GPU hour never discovers a bad credential or a
  broken loader.
- **The repository is public** (`DISCLOSURE.md`, since 2026-08-01, deliberately so since
  2026-08-02). Anything committed is published.
- **Overlay, never fork.** The vendor tree is pinned and unmodified (`ADR-002`).

---

## 9. The paper

*Trust-Gated Fast-Weight Updates for TTT-E2E Large Language Models* — Kandaswamy, Maahir,
Kamlekar, Yadav (School of Computer Science and Engineering, VIT Vellore). IEEEtran,
5 sections, 35 references, 6 figures, 5 tables, 1 algorithm.

It formalises the threat, specifies the gate, proves a per-window drift bound for the gated
commit, and records which part of that bound our own overlay does not yet enforce. It then
reports the pilot honestly: the measured effect clears the pre-registered criterion, **and**
a negative control with no attacker clears it too, often enough that the observed value is
approximately the median of what no attack at all produces at this sample size.

**It therefore reports no attack, and contributes a measured false-positive rate for the
criterion itself.**

Build chain and the open numbers issue: `OPERATIONS.md` §4 and §5.
