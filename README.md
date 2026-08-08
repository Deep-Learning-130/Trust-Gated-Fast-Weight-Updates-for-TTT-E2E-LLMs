<div align="center">

# Trust-Gated Fast-Weight Updates for TTT-E2E LLMs

**A runtime defense for language models that learn while they serve.**

`Phase 0–1 scaffold` · `JAX / Equinox` · `31 CPU tests passing` · `public since 2026-08-01`

</div>

> [!NOTE]
> **Public repository. No patent filed.** Public since 2026-08-01, and since 2026-08-02 that
> is a deliberate posture rather than an incident — so **nobody should change the
> visibility**. Foreign rights in absolute-novelty jurisdictions (EPO/CN/JP/KR) are forfeit;
> the US grace period runs to roughly **2027-08-01**, which is now a real deadline rather
> than a backstop. Pushing to a **new** remote is still a separate decision, and working
> attack artifacts still wait on coordinated disclosure to the TTT-E2E authors.
> Full posture and the open questions for counsel: [`DISCLOSURE.md`](DISCLOSURE.md).

---

## Overview

[TTT-E2E](https://arxiv.org/abs/2512.23675) reframes long-context language modeling as
*continual learning at inference*: the model carries **fast weights** that it updates by
next-token prediction as it reads the context, compressing that context into weights at
near-RNN cost while matching full-attention scaling.

The security consequence is direct: **a model that learns while it serves can be poisoned
while it serves.** A slow, benign-looking input stream — not a single adversarial token —
can steer the fast-weight updates so that *later, unrelated* benign inputs are handled
worse, or so that a latent trigger is implanted, all without ever touching the base
(slow) weights.

This project builds the **defense**: a **trust gate** that intercepts every proposed
fast-weight update and admits it only if it passes a consistency check against a frozen
anchor, subject to a **bounded cumulative-drift budget**, backed by **checkpoint and
rollback**. The defensible core is the *system* — interceptor + drift budget + rollback —
and its headline property is a **provable per-window drift bound**.

Full technical rationale and prior-art analysis: [`docs/F1-trust-gated-ttt.md`](docs/F1-trust-gated-ttt.md).

## Architecture

An **overlay**, not a fork. The upstream TTT-E2E code ships without a license, so it is
vendored as a **read-only pinned submodule** and never edited; all of our work lives in a
separate package that imports and wraps it. The interceptor hooks the single point where a
fast-weight update commits (`MetaModel.inner_loop_step`) via a confined, reversible
monkeypatch. Rationale: [`ADR-002`](docs/adr/ADR-002-overlay-vs-fork.md),
[`ADR-003`](docs/adr/ADR-003-jax-interceptor-shape.md).

```
                          proposed Δθ
   context ──▶ fast-weight ──────────▶ ┌───────────────────────────┐ ──commit──▶ versioned
   stream      updater                 │        TRUST GATE         │             store
   (vendor)    (vendor)                │  anchor-consistency (A)   │           (O(1) rollback)
                    ▲                   │  update-uncertainty (B)   │                 │
                    │                   └─────────────┬─────────────┘                 │
                    │                                 │ drift delta                   │
                    │                     ┌───────────▼───────────┐                   │
       rolled-back state                 │   drift accumulator    │  breach ──────────┘
                    └─────────────────────│  (budget ε, latches)   │  → rollback to
                                          └────────────────────────┘    last trusted checkpoint
```

Because JAX fast-weight updates are pure functions over a pytree, **reject** is a
`jnp.where` select between weight trees and **rollback** is a reference swap — genuinely
O(1), with no mutable state to unwind.

## Repository layout

```
src/trustgate/
├─ interceptor.py      ★ enforcement point — wraps the vendor update fn
├─ vendor_patch.py       install/uninstall the gate into the read-only submodule
├─ tree_ops.py           jit-safe pytree arithmetic over fast weights
├─ types.py              GateDecision, DriftState, Verdict (all trace-safe)
├─ gate/                 anchor (A) · uncertainty (B) · influence (C) · policy   [Phase 2]
├─ drift/accumulator.py  cumulative-drift budget ε — the bounded-drift guarantee
├─ store/versioned.py    ring-buffered checkpoints, O(1) rollback
├─ probes/               rotating held-out probe sets                            [Phase 2]
├─ baselines/            MedBN-analogue robust-aggregation baseline              [Phase 2]
├─ attack/               crafted-stream poisoning (Phase 1 kill-gate adversary)
├─ eval/                 corruption metrics · harness · go/no-go report
└─ audit/                accept/reject/rollback trail

vendor/ttt-e2e/          pinned submodule — READ-ONLY, unlicensed, never edited
docs/                    dossier · ADRs · patent disclosure + prior-art diffs
experiments/             000 baseline repro (gate) · 001 attack spike (pre-registered)
tests/                   CPU-only, no GPU, no checkpoints
scripts/                 vendor setup · checkpoint fetch · pre-push guard
graphify-out/            knowledge graph of this repo (graph.html, GRAPH_REPORT.md)
```

## Status

**Phase 0–1 scaffold.** The Phase 1 kill-gate — *does a benign-looking stream actually
corrupt fast weights?* — has not yet been run.

| Component | State |
|---|---|
| Update interceptor + pytree ops | ✅ implemented · CPU-tested |
| Drift accumulator (latching bounded-drift budget) | ✅ implemented · CPU-tested |
| Versioned store (O(1) rollback) | ✅ implemented · CPU-tested |
| Corruption metrics · audit log | ✅ implemented · CPU-tested |
| Attack (`attack/`) + evaluation harness | 🚧 specified — model steps blocked on checkpoints |
| Gate signals A/B/C · MedBN-analogue baseline | ⛔ Phase 2 — gated on the Phase 1 result |

## Quick start

CPU tests need no GPU and no checkpoints:

```bash
uv venv .venv
uv pip install --python .venv jax equinox optax numpy pytest
PYTHONPATH=src JAX_PLATFORMS=cpu .venv/bin/python -m pytest
```

Expected: **31 passing**. The suite verifies the interceptor's accept/reject/no-op
semantics, the store's round-trip and rollback, the drift latch, and the corruption
metrics — the parts of the invention that do not require a model.

## GPU path (Phase 0.5+)

```bash
export GCP_BILLING_PROJECT=<proj> WANDB_ENTITY=<e> WANDB_PROJECT=<p> WANDB_KEY=<k>

PROBE_ONLY=1 bash scripts/fetch_checkpoints.sh   # metadata-only: byte count before any egress
bash scripts/bootstrap_gpu_box.sh                # submodule → env → checkpoint → dataset → command
# then: experiments/000-repro-baseline  (must reproduce vendor numbers — a gate)
#       experiments/001-attack-spike    (the pre-registered kill-gate)
```

Both GCS buckets are **requester-pays**. The eval dataset must be copied to local disk — the
vendor's loader has no `gs://` backend. W&B credentials are mandatory and cannot be worked
around without editing vendor code. Costs, byte counts and the spending cap:
[`COST_MODEL.md`](experiments/000-repro-baseline/COST_MODEL.md). Exact eval command with
source citations: [`EVAL_ENTRYPOINT.md`](experiments/000-repro-baseline/EVAL_ENTRYPOINT.md).

## The one rule that governs everything

Phase 1 is a **hard kill-gate**, and its criterion is
[pre-registered](experiments/001-attack-spike/PREREGISTERED.md) — thresholds fixed *before*
any result is seen. If no benign-looking stream measurably corrupts fast weights against
that threshold, **the project stops.** No demonstrated attack means there is no defense to
build. This discipline is the point, not an obstacle.

## Development roadmap

1. **Phase 0.5 — Baseline.** Confirm the vendor eval runs soundly on our hardware, against a
   [pre-registered bar](experiments/000-repro-baseline/TOLERANCE.md). *(gate)* Note this is
   an environment check, not a reproduction: the paper publishes no number for the released
   1B checkpoint ([ADR-005](docs/adr/ADR-005-baseline-comparison-basis.md)).
2. **Phase 1 — Attack spike.** Demonstrate (or refute) benign-stream fast-weight poisoning
   against the pre-registered threshold. *(kill-gate)*
3. **Phase 2 — Gate prototype.** Implement anchor-consistency + uncertainty signals; sweep
   the operating-point curve; beat the MedBN-analogue baseline.
4. **Filing gate.** Complete the [line-level MedBN diff](docs/patent/prior-art/medbn-diff.md);
   file only if the gate beats the baseline *and* the novelty diff is clean.

## References

- **TTT-E2E** — End-to-End Test-Time Training for Long Context — [arXiv:2512.23675](https://arxiv.org/abs/2512.23675)
- **MedBN** — Robust Test-Time Adaptation against Malicious Samples (defense prior art) — [arXiv:2403.19326](https://arxiv.org/abs/2403.19326)
- Test-Time Poisoning Attacks Against TTA — [arXiv:2308.08505](https://arxiv.org/abs/2308.08505)
- Realistic Test-Time Data Poisoning — [arXiv:2410.04682](https://arxiv.org/abs/2410.04682)
- R.I.P. — black-box attack on continual TTA — [arXiv:2412.01154](https://arxiv.org/abs/2412.01154)

---

<div align="center">
<sub>Research-novelty + design dossier — not a freedom-to-operate opinion.
A formal FTO search (especially vs. MedBN) is required before filing.</sub>
</div>
