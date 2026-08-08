# ADR-003 — Interceptor shape: wrap the class method, gate inside the scan

**Status:** Accepted **Date:** 2026-07-28 **Deciders:** project owner

## Context

The design dossier (§5.2) draws the Update Interceptor as a block sitting between
the fast-weight updater and the committed weights. Turning that picture into JAX
requires deciding *where* and *how* it hooks the vendor code. Reading the vendor
at SHA `a4fc478`:

- The only place a fast-weight update commits is `MetaModel.inner_loop_step`
  (`ttt/model/transformer.py:593`), ending in `filter_apply_updates(self, updates)`.
- It is invoked through the **class**, not the instance, from inside
  `jax.lax.scan` in `loss_for_sequence` (`:703`): `MetaModel.inner_loop_step(model_inner, ...)`.
- The whole thing runs under `eqx.filter_jit` + `eqx.filter_checkpoint` +
  `scan_remat_chunk`.

## Decision

1. **Wrap the class method** `MetaModel.inner_loop_step` via monkeypatch, not by
   subclassing. Because the call site dispatches on the class, a
   `GatedMetaModel(MetaModel)` override would never run. The only leak-free
   alternative — copying `loss_for_sequence` into our tree — would mean copying
   unlicensed code (ADR-002). So: patch the class attribute.

2. **Gate decisions are traced arrays, never Python bools.** The gate runs
   inside `scan` under `jit`, so `verdict`, `drift_delta`, and `confidence` are
   `jnp` scalars, and accept/reject is a `jnp.where` select over both candidate
   weight trees — not an `if`. Commit and revert cost the same FLOPs; the saving
   from a reject is in **drift**, not compute (see `tree_ops.tree_select`).

3. **Reject reverts weights, keeps state.** On reject we restore the pre-update
   *inner parameters* but keep the new suffix/recurrent state — the model did
   read those tokens. We gate weight commits, not context ingestion. This
   distinction is load-bearing for the claim.

4. **Do not nest a `jit`.** The wrapper stays traceable, not compiled; the
   vendor's rematerialisation policy owns compilation. A nested `jit` fights
   `filter_checkpoint` and silently inflates memory.

## Consequences

- The interceptor is a pure function wrapper: `make_gated_inner_loop_step(original, gate)`.
- Fully CPU-testable with lightweight `eqx.Module` stand-ins — no GPU, no
  checkpoint — which is why `tests/test_interceptor.py` can prove the pass-through
  no-op property immediately (31 tests green as of scaffold).
- Tightly coupled to the vendor call-site shape; the test suite is the tripwire.
- The bounded-drift guarantee is enforceable inside the scan because the drift
  accumulator (`drift/accumulator.py`) is carried through the same `scan` carry.

## Correction — 2026-08-08

**The last consequence above is aspirational, not implemented.** `drift/accumulator.py`
and `store/versioned.py` exist and are unit-tested, but `interceptor.py` never touches
either: it calls `gate(delta, current)` and returns metrics. Today there is no enforced
budget and no rollback path.

Carrying them is blocked by a structural fact, not by effort. The vendor's scan carry is
fixed at `(model, inner_opt_state, (state_all, state_suffix))` inside `loss_for_sequence`,
and a `scan` requires the carry structure out to match the structure in — so the wrapper
cannot append a fourth slot without editing vendor code, which ADR-002 forbids. Three ways
out are on the table (optimizer-transformation wrapper, `eqx.tree_at` pytree smuggling, or
per-sequence granularity), and they do not buy the same guarantee: the third weakens the
claim from per-window to per-sequence.

That decision gets its own ADR before any code is written — TEAM_PLAN task **P3-1**,
gated on a Phase 2 PROCEED. Until it lands, treat the bounded-drift property as designed
but unproven, and do not describe it as enforced in the invention disclosure.
