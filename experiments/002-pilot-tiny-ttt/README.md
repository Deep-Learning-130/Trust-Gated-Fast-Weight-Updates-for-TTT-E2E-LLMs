# Experiment 002 — Scaled-down pilot (NOT the kill-gate)

> **Read this before quoting any number from this directory.**
>
> This is **not** `001-attack-spike`. It renders no PROCEED/STOP verdict, it does
> not touch `PREREGISTERED.md`, and nothing here can discharge the kill-gate.
> Standing Rule 5 is untouched: the frozen bars are reported *against*, never
> moved, and the pre-registered verdict still requires the released 1B checkpoint.

## What this is

The Phase 1 machinery — crafted-stream builders, the matching discipline, the
fast-weight carry overlay, the corruption metric — has been implemented and CPU
unit-tested for a month, but had **never run end to end against a model**, because
`run_stream`/`eval_benign` are contracted against a victim that needs a
requester-pays checkpoint, a W&B key and an 80GB accelerator.

This pilot supplies the smallest honestly TTT-shaped stand-in — fast weights
updated by next-token prediction at inference, carried across chunks, base weights
frozen — so the pipeline executes and produces real numbers. It runs in **19
seconds on CPU** with no download, no GCS, and no spend.

```
PYTHONPATH=src python experiments/002-pilot-tiny-ttt/run_pilot.py
python scripts/make_figures.py
```

## What it deliberately does NOT claim

| | Pre-registered (001) | This pilot |
|---|---|---|
| Victim | TTT-E2E 1B, DCLM+Books @8K | ~0.1M-param byte-level model |
| Corpus | Books3 | this repository's own prose, 412 KB |
| Attention | full vendor architecture | causal window-mean, no attention |
| Crafting | `craft_stream`, fluency term inside the objective | greedy hardest-last ordering via `order_fn` |
| Realism bar | GPT-2 small, ratio ≤ 1.5 | **not scored** |
| Verdict | PROCEED / STOP | **none** |

Because the realism bar is unscored, the pilot **cannot render PROCEED under any
circumstances** — an unscored stream fails the realism bar safe. That is the
correct behaviour, not a gap.

`craft_stream` (T1.9) is left raising `NotImplementedError`. Its contract is
against the vendor victim, and committing a pilot-grade heuristic under that name
would misrepresent it.

> **Superseded 2026-09-14.** `craft_stream` now exists. The objection above was
> about *hardcoding* a toy scorer where a victim belongs, and it stands; the
> implementation resolves it by making the victim an injected seam, the way
> `carry.run_chunks` injects `step_fn`. The pilot's own hill-climber
> (`run_deep.py:163-192`) was promoted to `attack/craft.py:search_order`
> unchanged in method. Nothing in this pilot's numbers changes -- they were
> produced by the `order_fn` route described below, and are not re-run. The pilot instead drives `build_select_stream`'s existing
`order_fn` seam, which `attack/stream.py` documents as the hook a crafting
optimiser uses "without reimplementing the constraints".

## Two runs

`run_pilot.py` (shallow) trains the base weights with the fast weights frozen.
`run_deep.py` (headline) trains them **through** the inner loop — the end-to-end
objective — and gives the attacker a hill-climbing search instead of a
single-pass sort. The deeper run is the only one with an attack surface worth
interpreting, and it is what the figures plot.

| | Shallow | Deep |
|---|---|---|
| Base training | fast weights frozen | meta-trained through 6 unrolled inner steps |
| Inner-loop gain | 0.0007 nats | **0.0682 nats** (97×) |
| Attacker | single-pass sort | hill-climbing, 40 proposals/seed |
| Protocol | adapt on stream → score unrelated held-out | stream → benign prefix → **score that document's continuation** |

The protocol change was forced by S1 and matters more than the model size. The
shallow run asked the inner loop to generalise *across* documents, which is not
what test-time training buys; its benefit is within a sequence. The deep run
serves the victim in deployment order — attacker stream, then a benign document,
then score that document's continuation — so the measured quantity is the one the
threat model names.

## Result (deep run)

| Quantity | Observed | Pre-registered bar |
|---|---|---|
| Cohen's *d*, poison vs control | **0.831** | ≥ 0.80 — **crossed** |
| Relative benign degradation | **+0.120%** | ≥ 10% — missed by ~80× |
| **Control-vs-control noise floor** | **\|*d*\| = 0.557** | — |
| Fluency ratio | not scored | ≤ 1.50 |

**Read the third row before the first.** Cohen's *d* clears the 0.80 bar — and
the negative control, in which nothing is attacking anything, reaches
\|*d*\| = 0.557 on the same five seeds. The "signal" is within a factor of 1.5 of
pure noise between two benign streams.

So the honest reading is **not** that a weak attack was detected. It is that
**at five seeds the effect-size bar can be approached by noise alone**, and this
run happened to land above it. Three things corroborate that:

- Two of five seeds still show the poison stream *helping* the victim.
- The relative-degradation bar is missed by roughly 80×, and the two statistical
  bars are pre-registered as a conjunction precisely so neither can carry a
  verdict alone.
- The dose-response is non-monotone and mostly **negative** (−0.039%, −0.010%,
  −0.347%, −0.143%, +0.125%). A real attack that grows with budget does not look
  like this.

### The null distribution — this is the finding

One control-vs-control draw shows the bar *can* be cleared by noise. It does not
say how often. `run_null.py` answers that: **20 draws, each comparing two benign
streams over five seeds, so the true effect is exactly zero by construction.**

| | |
|---|---|
| Median \|*d*\| under the null | **0.816** |
| Mean / p90 / max | 0.695 / 1.227 / 1.575 |
| **Draws clearing the pre-registered bar** | **11 of 20 = 55%** |
| Percentile of the observed *d* = 0.831 in this null | **55th** |

With no attacker present anywhere, the gating criterion fires on **55% of draws**,
and the deep run's headline effect size sits almost exactly at the **median of the
null**. It is not a weak attack at the edge of significance. It is the middle of
what "nothing happening" looks like at this sample size.

That is a measured false-positive rate for a pre-registered criterion, and it is
the most useful thing this pilot produced.

### What this says about the pre-registration

The control-vs-control arm (T3.7) is listed in `001-attack-spike/README.md` as
"reported-but-not-gating", justified as establishing that the corruption metric
measures corruption rather than drift. **It just caught a false positive that the
gating statistic alone would have waved through**, and the null distribution shows
that would happen more often than not.

Two observations, recorded and not acted on — Rule 5 means bars do not move after
a result is seen, and a pilot is certainly not the place to move one:

- `n = 5` is under-powered for a `d ≥ 0.8` threshold in this setup.
- An arm that can overturn the gating statistic more than half the time is doing
  more work than a "not gating" label implies.

Whether either transfers to the 1B configuration is unknown — the noise floor
there has never been measured, and this run is the argument for measuring it
before the verdict rather than after.

## Result (shallow run)

| Quantity | Observed | Pre-registered bar |
|---|---|---|
| Cohen's *d*, poison vs control | **0.054** | ≥ 0.80 |
| Relative benign degradation | **+0.0017%** | ≥ 10% |
| Control-vs-control noise floor | ***d* = 0.077**, rel. **+0.0040%** | — |
| Fluency ratio | not scored | ≤ 1.50 |

**The effect is smaller than the noise floor.** Two of five seeds show the poison
stream *helping* the victim (both printed, Standing Rule 6). Under this attack, at
this scale, adversarial span ordering is indistinguishable from ordinary benign
adaptation drift.

The dose-response is the one place something moves: degradation rises
monotonically with stream length past 2048 tokens — 0.008% → 0.041% → 0.080% at
2k/4k/8k — climbing clear of the ±0.004% noise floor while remaining ~125× below
the 10% bar. The direction is the attacker's; the magnitude is not.

## Validity checks that ran

- **S1 — the inner loop does work.** Held-out loss 2.5352 → 2.5345 after 16 inner
  steps on benign text. The published S1 (a falling per-token NLL curve) does not
  transfer, because measurement freezes the fast weights; the transferable form of
  the same question is asked instead. Had this failed, the run aborts — an inert
  inner loop produces a null for reasons unrelated to the attack, which is the
  false STOP the carry overlay exists to prevent.
- **Disjoint splits.** Base-training, attacker-source and held-out slices do not
  overlap. Measuring on the stream itself is a pre-registered invalidating
  condition.
- **Matched arms.** Length, chunk count, valid-token count and dtype asserted
  equal per seed. Span multiplicity is 1 in both arms (spans drawn without
  replacement), so the repetition loophole in the fluency bar is not in play.
- **Control-vs-control (T3.7).** Run, and reported whichever way it landed.

## The limitation that matters most

The selected inner learning rate (0.003, chosen on benign text only, before any
poison stream existed) improves held-out loss by just **0.0007 nats**. The pilot's
inner loop barely does anything — so the pilot's null is **weak evidence**, not a
negative result about TTT poisoning. There is very little adaptation here to
corrupt.

That is the honest reading, and it is also the reason the pilot cannot substitute
for the real run: the pre-registered experiment attacks a mechanism worth 0.014
nats at 8K, roughly 20× more adaptation than this pilot exhibits.

## Outputs

`results/` is git-ignored. `results/pilot.json` feeds `scripts/make_figures.py`,
which writes Figures 4–5 and Table II to `docs/paper/figures/`.
