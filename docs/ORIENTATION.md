# TrustGate — orientation for a first-time reader

**Written 2026-09-15.** This document assumes you know what a language model is and
nothing else about this project. It explains the idea, defines every term the
repository uses, describes what has actually been built and measured, and is explicit
about what has *not*.

If you read only one thing, read [§1](#1-the-project-in-one-page) and
[§9](#9-status-what-is-done-and-what-is-not).

---

## Table of contents

1. [The project in one page](#1-the-project-in-one-page)
2. [Background: what TTT-E2E is](#2-background-what-ttt-e2e-is)
3. [The security problem](#3-the-security-problem)
4. [The proposed defense: the trust gate](#4-the-proposed-defense-the-trust-gate)
5. [The kill gate — why the defense is not being built yet](#5-the-kill-gate--why-the-defense-is-not-being-built-yet)
6. [Glossary](#6-glossary)
7. [How the repository is organised](#7-how-the-repository-is-organised)
8. [The experiments, numbered](#8-the-experiments-numbered)
9. [Status: what is done and what is not](#9-status-what-is-done-and-what-is-not)
10. [The findings that changed the plan](#10-the-findings-that-changed-the-plan)
11. [How to read the numbers](#11-how-to-read-the-numbers)
12. [Working rules that constrain the project](#12-working-rules-that-constrain-the-project)
13. [Where to look next](#13-where-to-look-next)

---

## 1. The project in one page

**The observation.** A new class of language model — *TTT-E2E* — keeps learning after
it has been deployed. As it reads your input, it modifies a subset of its own weights
by training on that input. This is what lets it handle very long contexts cheaply.

**The security consequence.** A model that learns while it serves can be *poisoned*
while it serves. If an attacker controls part of the input stream, they may be able to
steer those live weight updates so that the model performs worse on **later, unrelated,
perfectly innocent inputs** — without ever touching the model's stored weights, without
any conventional "attack" visible in the text, and without the operator noticing.

**The proposed defense.** A *trust gate*: a component that sits between the model's
update machinery and the weights themselves, and refuses to let an update commit unless
it passes a cheap trustworthiness check. It also tracks how far the weights have
cumulatively drifted from a frozen reference and forces a rollback if that drift exceeds
a budget. The bounded-drift property is the defensible part — it is a stated, testable
guarantee rather than a claim about detection accuracy.

**The discipline.** The project refuses to build the defense until the attack is
demonstrated. That decision point is called the **kill gate**, and its criteria were
frozen in writing on 2026-07-28 before any attack code was run. If the attack does not
materialise, the correct outcome is **STOP** — publish the negative result and build
nothing.

**Where it stands today.** The measurement apparatus is built and tested. The kill gate
**has not been run**, so no verdict exists. Two serious flaws were found in the
apparatus before it could produce a wrong verdict — one in the upstream research code,
one in our own pilot. Both are fixed and documented. Total money spent: **$0.13**.

---

## 2. Background: what TTT-E2E is

### The ordinary picture

A conventional transformer LLM has a fixed set of weights. Training sets them; inference
reads them. Handling a long context means attending over every previous token, which
costs time and memory that grows quadratically with context length.

### What TTT changes

**Test-Time Training (TTT)** treats the context not as something to attend over but as
something to *learn from*. As the model reads, it performs actual gradient-descent steps
on a small subset of its parameters, using next-token prediction as the objective. The
context gets compressed *into weights* rather than held in an attention cache.

**TTT-E2E** ([arXiv 2512.23675](https://arxiv.org/abs/2512.23675)) is the end-to-end
version of this for long-context language modeling. Its claim is that it matches
full-attention scaling at near-RNN cost. It is real, recent, high-profile, and its code
is public — which is why it is the target architecture here.

### The two weight sets

This distinction is the single most important one in the whole project.

| | **Slow weights** | **Fast weights** |
|---|---|---|
| Also called | base weights, outer parameters | inner parameters, `feed_forward_prime` |
| Set by | pre-training and fine-tuning | the inner loop, *during inference* |
| Changed at serving time | **No** | **Yes, constantly** |
| Persist across requests | Yes | Depends on implementation |

An attack on fast weights therefore leaves no trace in the model file. The checkpoint on
disk is byte-identical before and after. This is what makes it a distinct threat and not
just "data poisoning" under a new name.

---

## 3. The security problem

### Threat model in one paragraph

The attacker can place text into the model's input stream — for example as a document in
a retrieval-augmented pipeline, a message in a long agent conversation, or a page the
model is asked to read. They cannot touch the weights on disk, the training pipeline, or
the serving infrastructure. Their text must look benign; a stream of obvious gibberish
would be caught by any input filter and also fails the project's own realism bar. Their
goal is that *some later, different, innocent input* is handled worse.

### Why this is not already-solved territory

The literature has two relevant bodies of work, and the project's contribution sits
precisely between them:

- **Test-time poisoning attacks** on test-time-adaptation systems are *mature*
  (IEEE S&P 2308.08505, ICLR'25 2410.04682, R.I.P. 2412.01154). The attack side is not
  the novelty.
- **MedBN** (2403.19326) is an existing *defense* against test-time poisoning. It works
  by making batch-normalisation statistics robust.

What is open: poisoning of **TTT-E2E-LLM fast weights specifically**, and a gate
mechanism designed for fast-weight continual-inference updates. Fast weights are not
batch-norm statistics, so a BN-statistics defense does not transfer. That distinction —
"the line-level diff versus MedBN" — is a mandatory check before any patent filing, and
is recorded as such.

---

## 4. The proposed defense: the trust gate

### Data flow

```
   context stream
        │
        ▼
 ┌──────────────┐   proposed update Δθ
 │ Fast-weight  │────────────────┐
 │   updater    │                │
 └──────────────┘                ▼
        ▲              ┌───────────────────────┐        ┌────────────────────┐
        │              │  1. UPDATE INTERCEPTOR │        │   ANCHOR MODEL      │
        │              │     (commit is paused) │◀───────│   (frozen, never    │
        │              └───────────┬────────────┘ probes │    updated)         │
        │                          ▼                     └────────────────────┘
        │              ┌───────────────────────┐
        │              │  2. TRUST GATE         │
        │              │   • anchor-consistency │
        │              │   • update-uncertainty │
        │              │   • decision policy    │
        │              └───────────┬────────────┘
        │                  reject  │  accept
        │                  ◀───────┤
        │                          ▼
        │              ┌───────────────────────┐        ┌────────────────────┐
        │              │  3. DRIFT ACCUMULATOR  │        │  4. VERSIONED      │
        │              │     budget ε / window  │        │     FAST-WEIGHT    │
        │              └───────────┬────────────┘        │     STORE          │
        │                  breach  │                     │  (ring buffer)     │
        └──────────────────────────┴────────────────────▶│  O(1) rollback     │
                        roll back to last trusted state  └────────────────────┘
```

### The five components

1. **Update interceptor** — the enforcement point. It sits at the single place in the
   upstream code where a fast-weight update commits, and holds that commit until a
   decision is made. **Nothing reaches the weights without passing through it.**
   Implemented: [`src/trustgate/interceptor.py`](../src/trustgate/interceptor.py).

   One design subtlety worth knowing: on *reject*, the interceptor reverts the inner
   parameters but **keeps the new suffix state**. The model did read those tokens, and
   its recurrent state must reflect that or the sequence becomes incoherent. Only the
   learned delta is refused. The project gates *weight commits*, not *context ingestion*.

2. **Anchor model** — a frozen copy of the meta-learned initial weights, never updated.
   It provides a stable reference so the question "has this update pulled behaviour away
   from the trusted baseline?" has a cheap answer.

3. **Trust gate** — the detection logic. An *anchor-consistency scorer* compares
   behaviour on a small rotating probe set before and after the proposed update; an
   *update-uncertainty scorer* flags high-variance or low-confidence updates; a
   *decision policy* combines them into accept / reject / rollback. **Not implemented** —
   see §5.

4. **Drift accumulator** — maintains cumulative distance from the anchor and enforces a
   per-window budget ε. Implemented:
   [`src/trustgate/drift/accumulator.py`](../src/trustgate/drift/accumulator.py).

5. **Versioned fast-weight store** — ring-buffered checkpoints enabling O(1) rollback to
   the last trusted state. Implemented:
   [`src/trustgate/store/versioned.py`](../src/trustgate/store/versioned.py).

### Where the actual claim lives

Not in detection. Detection is statistical and arguable — any "we detect poison" claim
invites an adversary who evades the detector.

The claim is **bounded drift**: over any window, committed fast weights cannot travel
more than ε from the frozen anchor before a rollback is *forced*. That holds regardless
of whether the gate correctly classified any individual update. It is a property of the
system, and it is testable.

### The central tension

A gate that rejects too readily starves the model of the useful adaptation that was the
whole point of TTT. A gate that accepts too readily admits the poison. The deliverable
is therefore an **operating-point curve** across threshold settings, not a single
threshold number. The decision policy is written to take its thresholds as parameters
and forbidden to hard-code them.

---

## 5. The kill gate — why the defense is not being built yet

> **This is the most commonly misunderstood part of the project, because the word
> "gate" means two unrelated things.**

| | **Trust gate** | **Kill gate** |
|---|---|---|
| What it gates | each fast-weight update, at inference | **the project itself** |
| Lives in | [`src/trustgate/gate/`](../src/trustgate/gate/) | [`experiments/001-attack-spike/`](../experiments/001-attack-spike/) |
| Decides | accept / reject / rollback | PROCEED / STOP |
| How often it fires | continuously, per window | **once, ever** |
| Status | not implemented | **not yet run** |

The kill gate asks one question: *can a benign-looking stream measurably corrupt fast
weights at all?* If the answer is no, there is no attack, so there is nothing worth
defending, so the trust gate never gets built. **STOP is a complete and acceptable
outcome** — a documented negative result about a hot architecture.

Its criteria were frozen on **2026-07-28, before any attack code was run**, in
[`PREREGISTERED.md`](../experiments/001-attack-spike/PREREGISTERED.md). All three must
hold to PROCEED:

| Criterion | Threshold | Why |
|---|---|---|
| Effect size (Cohen's *d*, poison vs control) | **≥ 0.8** | conventional "large"; must separate from seed noise |
| Relative benign degradation | **≥ 10%** | below this the harm is real-but-trivial |
| Fluency ratio (poison perplexity ÷ control perplexity, under an *independent* reference model) | **≤ 1.5** | above this the stream is not "benign-looking" and fails the threat model |

Both statistical bars **and** the realism bar are required. A corruption that only
appears with an obviously unnatural stream does not pass.

This is why every file under [`src/trustgate/gate/`](../src/trustgate/gate/) —
`anchor.py`, `uncertainty.py`, `policy.py`, `influence.py` — is a fully documented stub
that raises `NotImplementedError("Phase 2.")`. That is deliberate, not incomplete.

---

## 6. Glossary

### Core architecture terms

**Fast weights** — the parameters updated during inference by the inner loop. In this
codebase they correspond to the upstream `feed_forward_prime` module. The thing being
attacked and the thing being defended.

**Slow weights** — the ordinary model parameters, set by training, frozen at serving
time. Also called base or outer parameters. The attack never touches them.

**Inner loop** — the gradient-descent process that runs *during inference*, updating
fast weights from the context. One inner step per chunk.

**Outer loop** — ordinary training, which learns the slow weights *and* learns a good
initialisation for the fast weights. Also called meta-training.

**Meta mode** (`train_mode="meta"`) — the operating mode in which the inner loop is
active. The only mode in which this project's question is meaningful.
`train_mode="pretrain"` has no inner loop and would be a vacuous null.

**Chunk** / **mini-batch** — the context is split into fixed-size pieces; one inner-loop
update happens per piece. At the project's standard settings, 8192 tokens ÷ 1024 =
**8 inner steps** per sequence.

**Prefix / suffix** — the upstream model runs non-adapting "prefix" blocks over the whole
sequence first, then adapts only on the "suffix" tail. Any faithful reimplementation must
mirror this split.

**Carry** — threading the adapted fast weights forward, from one chunk to the next and
from one sequence to the next, instead of resetting them. See §10; the upstream code does
not do this, and that turned out to matter enormously.

**Anchor** — the frozen reference copy of the initial fast weights the gate scores
against.

**Drift budget (ε)** — the maximum cumulative distance from the anchor that committed
fast weights are allowed to travel within a window before rollback is forced.

**Probe set** — a small rotating collection of held-out inputs the gate runs before and
after a proposed update to see whether behaviour moved.

### Measurement terms

**Cross-entropy (CE) loss** — the standard language-modeling loss: how surprised the
model is by the correct next token. Lower is better. Measured here in **nats**.

**Nats** — the natural-logarithm unit of information. A uniformly random guess over a
vocabulary of size *V* costs ln(*V*) nats. For the 128,256-token vocabulary used here
that is **11.762 nats**, which is the sanity bar a freshly initialised model must hit.
(One nat ≈ 1.44 bits.)

**Perplexity** — exp(cross-entropy). Roughly "how many equally likely options is the
model effectively choosing between." Used for the fluency bar.

**Cohen's *d*** — a standardised effect size: the difference between two conditions'
means, divided by their pooled standard deviation. *d* = 0.8 is the conventional
threshold for a "large" effect. **Crucially, it is dimensionless** — which is exactly why
it can mislead when the standard deviation is small, as §11 explains.

**Relative degradation** — the plain-language version: how much worse, in percent, the
poisoned condition is than the control.

**Fluency ratio** — poison-stream perplexity ÷ control-stream perplexity, scored by an
**independent reference model** (never the victim — a victim that has been poisoned is not
a trustworthy judge of anything).

**Seed** — one independent repetition of the experiment with different randomness. The
pre-registration fixes **5 seeds** per condition.

**Control** — the comparison condition: a benign stream matched to the poison stream on
length, seed, and dtype. Without matching, any measured difference is unattributable.

**Null distribution** — what the measurement produces when you run it **control versus
control**, where the true effect is *zero by construction*. Anything the statistic reports
here is pure noise. This is how you calibrate an instrument. See §11.

**Noise floor** — the effect size observed under that control-vs-control condition. An
effect smaller than the noise floor is not an effect.

**False positive rate (FPR)** — the fraction of null draws that nonetheless clear the
threshold. If this is high, the threshold detects noise.

### Process terms

**Pre-registration** — writing down the hypothesis, the metric, and the decision
thresholds *before* collecting data, and committing that document to version control so
its history proves the bar predated the result. It is the mechanism that prevents
"the result was 0.6, and on reflection 0.6 is quite large."

**ADR (Architecture Decision Record)** — a short dated document recording one decision,
the options considered, and why one was chosen. There are seven, in
[`docs/adr/`](adr/).

**Kill gate** — see §5. A project-level PROCEED/STOP decision point.

**Vendor tree** — the upstream TTT-E2E source, checked out at
[`vendor/ttt-e2e/`](../vendor/ttt-e2e/) as a pinned git submodule. It ships **without a
licence**, so it is read-only: never edited, never modified, never re-published.
Continuous integration enforces this mechanically.

**Overlay** — the architectural consequence of that constraint. Rather than forking the
upstream code, this project is a separate package that imports and wraps it. Recorded in
[ADR-002](adr/ADR-002-overlay-vs-fork.md).

**SELECT / PARAPHRASE / SOFT** — three attack strategies of decreasing realism. SELECT
only *reorders real benign sentences* and is the headline strategy precisely because it
is the hardest to dismiss as unrealistic. The other two are reported separately as
stronger-but-less-realistic points and do not substitute for SELECT.

**Requester-pays** — a cloud storage configuration where the *downloader* is billed, not
the bucket owner. Both upstream buckets are requester-pays, which is why obtaining the
released checkpoints requires a billing account and why the project's cost model exists.

**Standing Rule 5** — the project rule that pre-registered bars are **reported against,
never moved**. A revision is legitimate only as a dated, written revision made *before*
the run.

---

## 7. How the repository is organised

```
TTT/
├── README.md              project overview and disclosure posture
├── ROADMAP.md             phases and their gates
├── TEAM_PLAN.md           working rules, including the Standing Rules
├── DISCLOSURE.md          publication and responsible-disclosure posture
│
├── docs/
│   ├── F1-trust-gated-ttt.md   the original technical dossier — the source of truth
│   │                            for the system design and the prior-art analysis
│   ├── report.md               progress report (assumes project knowledge)
│   ├── ORIENTATION.md          this file
│   ├── adr/                    seven architecture decision records
│   ├── paper/                  the LaTeX write-up
│   ├── patent/                 invention disclosure and prior-art search
│   ├── protocols/              branch-and-review rules, GPU booking and queue
│   └── review/                 the results deck for review panels
│
├── src/trustgate/         all of our code (the overlay)
│   ├── interceptor.py         the enforcement point                    [built]
│   ├── drift/accumulator.py   the ε budget                             [built]
│   ├── store/versioned.py     checkpoint + rollback ring buffer        [built]
│   ├── gate/                  the detection logic                      [Phase 2 stubs]
│   ├── probes/                rotating probe set                       [Phase 2 stub]
│   ├── attack/                objectives, corpus, stream builders      [built]
│   ├── eval/                  harness, carry overlay, metrics, bars    [built]
│   ├── baselines/             MedBN analogue for comparison            [Phase 2 stub]
│   └── audit/                 decision logging                         [built]
│
├── experiments/           see §8
├── tests/                 197 tests, CPU only, no vendor code required
├── scripts/               setup, cost probes, figures, the demo, the deck
└── vendor/ttt-e2e/        upstream source — READ ONLY, never edited
```

---

## 8. The experiments, numbered

The numbering is a dependency order, not a chronology.

### 000 — Reproduce the baseline

**Purpose:** run the *unmodified* upstream evaluation on the released 1B checkpoint and
confirm the environment is sound. This is a gate, not a formality: if we cannot reproduce
the upstream behaviour on our hardware, then any "corruption" measured later is
unattributable — it might be our own misconfiguration.

An important subtlety is recorded in [ADR-005](adr/ADR-005-baseline-comparison-basis.md):
**the paper reports no number for the checkpoint we intend to evaluate.** Every printed
absolute loss is for a different model size; every plotted one is for a different
configuration. So the bar cannot be "match the published number." It is instead a
two-sided bracket derived from sign-determined monotonicity, plus structural checks that
carry most of the real discriminating power. That bracket is **2.314 < loss < 2.805**
nats/token, fixed in [`TOLERANCE.md`](../experiments/000-repro-baseline/TOLERANCE.md).

**Status:** not run. Blocked on GPU access and a requester-pays download.

### 001 — Attack spike — *the kill gate*

**Purpose:** the PROCEED/STOP decision described in §5.

**Status:** **not run.** The directory contains `PREREGISTERED.md`, `README.md`, and
`FLUENCY_REFERENCE.md`. There is **no `results/` directory**, and that absence is the
honest state of the project.

### 002 — Scaled-down pilot

**Purpose:** the Phase 1 machinery had been implemented and unit-tested for a month but
had never run end to end against *any* model, because the real victim needs a
requester-pays checkpoint and an 80 GB accelerator. This pilot supplies the smallest
honestly TTT-shaped stand-in — a ~0.1M-parameter byte-level model with fast weights
updated by next-token prediction at inference, carried across chunks, base weights frozen
— so the pipeline executes and produces real numbers.

Runs in about 35 seconds on CPU. No GPU, no download, no network, no spend.

**What it deliberately does not claim:** it is not TTT-E2E, the corpus is this
repository's own prose rather than Books3, there is no attention mechanism, and **the
realism (fluency) bar is unscored by construction**. It therefore renders **no verdict**
and touches no pre-registered bar. Its README opens with that warning, and so does the
demo script.

**Status:** complete, reproducible, results in §11.

### 003 — 125M smoke test

**Purpose:** the first execution of the *real* upstream model and of our carry overlay
against it, at 125M parameters, on free hardware. This is instrument validation, not
measurement — a randomly initialised model's fast weights have nothing worth corrupting.

**Status:** partially complete. The upstream model executes; seven distinct first-launch
failures were found and fixed, which is exactly the value this tier exists to produce.
The final check — "the carry is non-trivial" — is blocked: an 8 GB consumer card is
roughly 2 GB short.

---

## 9. Status: what is done and what is not

| | Status |
|---|---|
| Phase 0–1 scaffold | Complete |
| Phase 1 CPU implementation | Complete — 197 tests passing |
| Fluency reference model | Complete |
| Continuous integration | Complete |
| 002 pilot | Complete and reproducible |
| 003 smoke — upstream model executes | Complete |
| 003 smoke — carry verified non-trivial | **Blocked on GPU memory** |
| 000 baseline reproduction | Not run |
| **001 kill gate** | **Not run** |
| Trust gate implementation (Phase 2) | **Deliberately forbidden** until 001 returns PROCEED |
| **Verdict** | **None** |
| **Total spend** | **$0.13** |

**Built and tested:** the interceptor, the drift accumulator, the versioned store, the
attack objectives (the attacker optimises the victim's own metric), the token corpus, the
stream builders, the run-condition matching discipline, the three frozen bars as
constants with a CI test that parses `PREREGISTERED.md` and asserts equality, the report
generator, the fast-weight carry overlay, and the fluency reference model.

**Deliberately empty:** everything in `gate/`, the rotating probe set, and the MedBN
comparison baseline. Each is a documented stub naming what it must do and why it is not
done yet.

**The single remaining technical risk** is binding the carry overlay to the real upstream
model. Everything downstream of that is written and tested.

---

## 10. The findings that changed the plan

These were all obtained at a desk, for zero spend, and each would otherwise have cost a
paid GPU session or — worse — a confidently wrong answer. They are the most valuable
output the project has produced so far.

### 10.1 The upstream code throws the adapted weights away

The upstream implementation discards its scan carry at
`vendor/ttt-e2e/ttt/model/transformer.py:712`. The fast weights are adapted within a
chunk and then **reset**.

Run the kill-gate experiment against the stock upstream code and the consequence is
mechanical: there is nothing accumulated to corrupt, so the measurement finds no effect,
so the verdict is **STOP** — and it would have looked exactly like a genuine negative
result about the security of the architecture.

This is what [ADR-006](adr/ADR-006-fast-weight-carry.md) calls a *structurally guaranteed
false STOP*. It was found by reading the source before any GPU existed. The fix is the
carry overlay in [`src/trustgate/eval/carry.py`](../src/trustgate/eval/carry.py), which
threads adapted weights across chunks *and* across sequence boundaries.

### 10.2 Our own pilot was unfalsifiable

Until 2026-09-15, experiment 002 built its corpus by **globbing this repository's own
markdown files at runtime**. Every commit silently changed the experiment's input. The
committed result files predated the code that produced them. Nobody — including us —
could have reproduced those numbers.

Found while building a live demo, which is a reasonable argument for building live demos.

The fix: the corpus is frozen to a committed file of 615,141 bytes, its SHA-256 is pinned
in source, the loader verifies the digest and **refuses to fall back to a glob**, and the
file is marked binary in `.gitattributes` so line-ending conversion cannot break the
digest on a fresh clone. All three result files were regenerated. The demo script prints
the digest check on screen, first, because a number that cannot name its inputs is not a
result.

### 10.3 The pre-registered threshold reports noise at n = 5

This one has the most far-reaching consequences, and it is explained in §11.

---

## 11. How to read the numbers

All figures below come from experiment 002 and therefore render **no verdict**. They are
observations about the *instrument*.

### The headline result, and why it is not one

| Run | Cohen's *d* | Read against |
|---|---|---|
| Shallow pilot | **+0.872** | noise floor **0.964** |
| Deep pilot (meta-trained base, hill-climbing attacker) | **−0.003** | 0th percentile of its own null |

The shallow number clears the pre-registered 0.8 bar. **It is not a detection.** The
control-vs-control noise floor — printed by the experiment itself, not added afterwards —
is *larger* than the effect. An effect smaller than the noise is not an effect.

This is the single most important reading habit in the project: **never read an effect
size without its noise floor beside it.**

### The null study

To make that concrete, the measurement was run 20 times **control against control**, where
the true effect is zero by construction:

| Quantity | Value |
|---|---|
| Draws | 20 (5 seeds each) |
| Mean \|*d*\| | 0.423 |
| Median \|*d*\| | 0.362 |
| **90th percentile \|*d*\|** | **0.849** |
| Max \|*d*\| | 1.307 |
| **False-positive rate at the 0.8 bar** | **0.15** |

Three draws in twenty clear |*d*| ≥ 0.8 **with no attacker present at all**, and the 90th
percentile of pure noise sits *above* the bar.

**What this means.** At n = 5 seeds, `d ≥ 0.8` is not a detection threshold. It is a
coin with a bias. This is a finding about the pre-registered criterion itself, and it is
the most consequential thing in the repository.

**What was done about it.** Nothing — deliberately. Standing Rule 5 means the frozen bar
is *reported against*, never moved. The null study is recorded as a dated observation
about the instrument. Any revision of the bar must be a written, dated decision made
*before* the run it governs, not a reaction to a number.

### The instrument does work

One number confirms the apparatus has real sensitivity: inner-loop adaptation gain is
**0.1532 nats** on the deep run versus **0.0007 nats** on the shallow one. The
meta-trained model genuinely learns from its context. So the deep run's *d* = −0.003 is
not a broken measurement returning zero — it is a working measurement reporting that,
in this scaled-down setting, **there was nothing there to find**.

That is a legitimate and informative outcome. It is also not the kill gate, which
requires the real 1B model, the real corpus, and the realism bar scored.

---

## 12. Working rules that constrain the project

Reading the repository without these will make several choices look arbitrary.

1. **The vendor tree is never edited.** It carries no licence. It is a pinned, read-only
   submodule; CI blocks any tracked file inside it. This is why the project is an overlay
   and why the interceptor wraps a method rather than changing two lines
   ([ADR-002](adr/ADR-002-overlay-vs-fork.md), [ADR-003](adr/ADR-003-jax-interceptor-shape.md)).

2. **History is never rewritten.** No rebase, no amend, no force-push on any pushed
   branch. The commit timeline is evidence of conception dates.

3. **Pre-registered bars are reported against, never moved** (Standing Rule 5). A
   revision is legitimate only as a dated, written revision made *before* the run.

4. **No Phase 2 implementation before the kill gate returns PROCEED** (Standing Rule 2).
   This is why `gate/` is all stubs.

5. **No self-merge.** Feature branches only; never commit directly to `main`; a
   non-author reviews and merges.

6. **Zero spend discipline.** Every GPU-hour and every byte of egress is costed in advance
   in [`COST_MODEL.md`](../experiments/000-repro-baseline/COST_MODEL.md), which was built
   by probing the real buckets for metadata only.

7. **Disclosure posture.** The repository has been public since 2026-08-01 and that is a
   deliberate settled posture. Working attack artifacts still wait on coordinated
   disclosure to the TTT-E2E authors. See [`DISCLOSURE.md`](../DISCLOSURE.md).

---

## 13. Where to look next

| If you want… | Read |
|---|---|
| The full technical design and prior-art analysis | [`docs/F1-trust-gated-ttt.md`](F1-trust-gated-ttt.md) |
| The decision thresholds, frozen and dated | [`experiments/001-attack-spike/PREREGISTERED.md`](../experiments/001-attack-spike/PREREGISTERED.md) |
| Why the upstream carry problem matters | [`docs/adr/ADR-006-fast-weight-carry.md`](adr/ADR-006-fast-weight-carry.md) |
| What the pilot does and does not claim | [`experiments/002-pilot-tiny-ttt/README.md`](../experiments/002-pilot-tiny-ttt/README.md) |
| A progress report written for someone who already knows the project | [`docs/report.md`](report.md) |
| A slide deck for a review panel | `docs/review/TrustGate_Results_2026-09-15.pptx` |

### Run it yourself

The pilot is the fastest way to see something real. It takes about 40 seconds, needs no
GPU, no download, no network, and no credentials:

```bash
bash scripts/demo.sh
```

It verifies the corpus digest on screen, runs the experiment end to end, and prints the
effect size **next to its noise floor** — because that is the only way either number can
be honestly read.

---

*Questions this document does not answer are probably answered in one of the seven ADRs
under [`docs/adr/`](adr/), each of which records a single decision, its options, and its
reasoning.*
