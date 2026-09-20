# ADR-P3-1 — Carrying drift state through a scan we cannot change

**Status:** Accepted, unimplemented **Date:** 2026-09-20 **Deciders:** Manas Maahir, project owner

> **This ADR chooses a route and writes no code.** It exists because ADR-003's
> 2026-08-08 correction said the decision "gets its own ADR before any code is
> written", and because `PREREGISTERED.md`'s 2026-09-20 revision starts Phase 2
> before the 001 verdict. Implementation stays gated on 001 returning PROCEED.

## Context

`drift/accumulator.py` and `store/versioned.py` are implemented and unit-tested.
Neither is reachable from `interceptor.py`, which calls `gate(delta, current)`
and returns metrics. There is today no enforced budget and no rollback path, and
ADR-003 is explicit that the bounded-drift property must not be described as
enforced until this lands.

The blocker is structural. In `loss_for_sequence` the vendor scans with a carry
fixed at `(model, inner_opt_state, (state_all, state_suffix))`
(`transformer.py:709-715`), and `jax.lax.scan` requires the carry structure out
to match the structure in. The wrapper cannot append a fourth slot, and editing
the vendor tree is forbidden by ADR-002.

So the question is not *whether* to carry drift state. It is **which existing
slot can hold it**.

## Options

### A. Optimizer-transformation wrapper — CHOSEN

`inner_opt_state` is already a carry slot, and it exists to hold per-step
optimizer state. It is produced by `model.inner_optimizer(state_all).init(...)`
(`transformer.py:687`) and consumed by `.update(...)` inside `inner_loop_step`
(`transformer.py:624`). `inner_optimizer` (`transformer.py:575-579`) returns a
plain optax optimizer.

Wrap it in an `optax.GradientTransformation` whose state is
`(inner_state, DriftState, VersionedStore)`. The vendor's own `init` and
`update` calls then carry our state for us, in the slot already designed to
carry state, with no new slot and no vendor edit. `interceptor.py:116` already
passes `new_optimizer_state` through unchanged.

The patch technique is the one ADR-003 decision 1 already established: patch the
class attribute. `inner_optimizer` dispatches on the class exactly as
`inner_loop_step` does, so `vendor_patch` extends rather than changes shape.

- **Granularity:** per-window. Keeps the claim intact.
- **Cost:** the rollback ring buffer lives in the carry, `depth x |inner
  parameters|` resident for the whole sequence — the cost `store/versioned.py:6-18`
  already documents and requires be reported.
- **Risk:** the opt_state pytree structure changes. In eval mode nothing restores
  opt_state (`train.py:158-159` demotes `all` to `params`), so the checkpoint
  path is unaffected. Training would need the same wrapper installed to resume,
  which is acceptable because we do not train.
- **Cuts against it:** it is a slightly dishonest use of "optimizer state" —
  the rollback buffer is not optimizer state in any ordinary sense. That is a
  naming cost, paid once in a docstring, against a structural guarantee.

### B. `eqx.tree_at` pytree smuggling

Attach drift state as a field on the model, which is also in the carry.

Rejected. The model pytree is load-bearing in more places than the optimizer
state is: `filter_apply_updates`, `inner_parameters()`, `shard_params`, and the
checkpoint keypaths that `unify_dict_with_eqx_module` matches on. A structural
change there risks a silent partial restore — the exact failure
`build_from_checkpoint` was just written to refuse. Higher blast radius for the
same guarantee.

### C. Per-sequence granularity

Accumulate and roll back between sequences, outside the scan.

Rejected as the primary mechanism. ADR-003 already names the cost: it "weakens
the claim from per-window to per-sequence". The threat model in dossier section
1 accuses carry *across windows* specifically, so a per-sequence bound does not
bound the thing we claim to bound.

Worth keeping as a **fallback and as a test oracle**: it is simple enough to
implement outside the trace, and a per-sequence bound computed independently is
a good cross-check on whatever A reports.

## Decision

Route **A**, when Phase 2 is unblocked. Route C stays available as a cross-check.

## Consequences

- Until this is implemented, `gate/policy.py` composes a `DriftState` that
  nothing advances. Its docstring must say so, and the invention disclosure must
  continue to describe bounded drift as designed rather than enforced.
- `vendor_patch` grows a second patched class attribute, and the test suite's
  tripwire role (ADR-003 consequence 3) extends to `inner_optimizer`'s signature.
- The memory cost becomes a reported number, not a footnote: `depth x |inner
  parameters|` resident for the whole sequence, competing with the activation
  memory the vendor's rematerialisation already fights for.
- **`Verdict.ROLLBACK` gets its first producer and consumer.** It is defined in
  `types.py:24-30` and nothing in the repo currently emits or handles it.
