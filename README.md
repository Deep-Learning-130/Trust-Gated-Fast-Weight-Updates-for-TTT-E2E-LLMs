# F1 — Trust-Gated Fast-Weight Updates for TTT-E2E LLMs

> ⚠ **CONFIDENTIAL.** Provisional patent not yet filed. **Read [`DISCLOSURE.md`](DISCLOSURE.md)
> before pushing, publishing, or presenting.** No public disclosure until filed —
> it forfeits foreign patent rights.

A defense for language models that learn while they serve. [TTT-E2E](https://arxiv.org/abs/2512.23675)
updates "fast weights" at inference to compress long context; a model that learns
while it serves can be **poisoned** while it serves. This project builds a **trust
gate** that admits a fast-weight update only if it passes a frozen-anchor
consistency check, under a **bounded cumulative-drift budget** with
**checkpoint/rollback**.

Full rationale: [`docs/F1-trust-gated-ttt.md`](docs/F1-trust-gated-ttt.md).

## Status

**Phase 0–1 scaffold.** The kill-gate (does the attack even work?) is not yet run.

| Component | State |
|---|---|
| Update interceptor + tree ops | ✅ implemented, CPU-tested |
| Drift accumulator (bounded-drift latch) | ✅ implemented, CPU-tested |
| Versioned store (O(1) rollback) | ✅ implemented, CPU-tested |
| Corruption metrics / audit log | ✅ implemented, CPU-tested |
| Attack (`attack/`) + harness | ⏳ specified; model steps blocked on checkpoints |
| Gate signals A/B/C, MedBN-analogue baseline | ⛔ Phase 2 — blocked on the Phase 1 result |

31 CPU tests pass with no GPU and no checkpoints.

## Architecture

An **overlay** on the vendored TTT-E2E repo — we never edit it (no licence; see
[`ADR-002`](docs/adr/ADR-002-overlay-vs-fork.md)). `src/trustgate/` imports and
wraps `vendor/ttt-e2e/`. The interceptor monkeypatches the one vendor commit point
(`MetaModel.inner_loop_step`); see [`ADR-003`](docs/adr/ADR-003-jax-interceptor-shape.md).

```
context ──▶ fast-weight updater ──Δθ──▶ [ TRUST GATE ] ──commit──▶ versioned store
  (vendor)          (vendor)          anchor-consistency + uncertainty   (ours, O(1) rollback)
                                              │ drift delta
                                              ▼
                                      drift accumulator (budget ε, breach latches)
                                              │ breach → rollback
                                              ▼
                                      last trusted checkpoint
```

## Layout

- `src/trustgate/` — the overlay package (interceptor, gate, drift, store, attack, eval, audit)
- `vendor/ttt-e2e/` — pinned submodule, **read-only**
- `docs/` — dossier, ADRs, patent disclosure + prior-art diffs
- `experiments/` — `000` baseline repro (gate), `001` attack spike (kill-gate, **pre-registered**)
- `tests/` — CPU-only, no GPU

## Quick start (CPU tests)

```bash
uv venv .venv && uv pip install --python .venv jax equinox optax numpy pytest
PYTHONPATH=src JAX_PLATFORMS=cpu .venv/bin/python -m pytest
```

## GPU path (Phase 0.5+)

```bash
bash scripts/setup_vendor.sh                         # submodule at pinned SHA
GCP_BILLING_PROJECT=<proj> TTT_BUCKET=<gs://...> \
    bash scripts/fetch_checkpoints.sh                # 1B, requester-pays
# then experiments/000-repro-baseline (gate), then 001-attack-spike
```

## The one rule that governs everything

Phase 1 is a **hard kill-gate** ([`experiments/001-attack-spike/PREREGISTERED.md`](experiments/001-attack-spike/PREREGISTERED.md)).
If no benign-looking stream measurably corrupts fast weights against the
pre-registered threshold, **the project stops.** No attack ⇒ no defense.
