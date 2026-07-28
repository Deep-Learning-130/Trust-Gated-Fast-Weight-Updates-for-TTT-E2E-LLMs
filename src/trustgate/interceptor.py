"""The Update Interceptor -- the enforcement point of dossier section 5.2.

This is the claim boundary. Everything upstream of it is TTT-E2E as published;
everything downstream is ours. The interceptor sits between the fast-weight
updater and the committed weights, and no fast-weight update reaches the model
without passing through it.

Where it hooks
--------------
`MetaModel.inner_loop_step` (vendor `ttt/model/transformer.py:593`) is the sole
place a fast-weight update commits. Its last two lines are:

    updates, new_optimizer_state = self.inner_optimizer(...).update(...)
    new_model = filter_apply_updates(self, updates)     # <-- commit

We wrap the whole method rather than editing those lines, because the vendor
tree carries no licence and must not be modified (ADR-002).

What "reject" means here
------------------------
On reject we revert the *inner parameters* to their pre-update values but keep
the new suffix state. Rationale: the model still read those tokens, and its
recurrent/attention state must reflect that or the sequence becomes incoherent.
Only the learned delta is refused. This distinction matters for the claim --
we gate weight commits, not context ingestion.
"""

from __future__ import annotations

from typing import Callable

import jax.numpy as jnp

from trustgate.tree_ops import tree_global_norm, tree_select, tree_sub
from trustgate.types import FastWeights, Gate, GateDecision, Updates, Verdict


def passthrough_gate(
    updates: Updates,
    fast_weights: FastWeights,
    anchor: FastWeights | None = None,
) -> GateDecision:
    """Accepts everything. The identity element.

    Exists so the interceptor can be installed and verified as a provable no-op
    before any policy is written -- `tests/test_interceptor.py` asserts that a
    passthrough-gated step is bit-identical to the ungated vendor step.
    """
    del updates, fast_weights, anchor
    return GateDecision.accept_all()


def norm_threshold_gate(max_norm: float) -> Gate:
    """Rejects updates whose L2 norm exceeds `max_norm`.

    Deliberately crude. This is *not* the invention -- it is a sanity control
    for the Phase 1 attack spike, to check that the interception path can
    actually change an outcome. The real gate (frozen-anchor consistency) is
    `trustgate.gate.anchor`, unimplemented until the attack is demonstrated.
    """

    def gate(
        updates: Updates,
        fast_weights: FastWeights,
        anchor: FastWeights | None = None,
    ) -> GateDecision:
        del fast_weights, anchor
        norm = tree_global_norm(updates)
        rejected = norm > jnp.float32(max_norm)
        return GateDecision(
            verdict=jnp.where(rejected, jnp.int32(Verdict.REJECT), jnp.int32(Verdict.ACCEPT)),
            drift_delta=jnp.where(rejected, jnp.float32(0.0), norm),
            confidence=jnp.float32(1.0),
        )

    return gate


def make_gated_inner_loop_step(
    original_step: Callable,
    gate: Gate = passthrough_gate,
) -> Callable:
    """Wrap the vendor's `inner_loop_step` with the trust gate.

    Args:
        original_step: the unbound `MetaModel.inner_loop_step`.
        gate: scoring policy. Defaults to accept-everything.

    Returns:
        A drop-in replacement with the identical signature.
    """

    def gated_inner_loop_step(self, opt_state, state_tuple, seq, prefix_outputs):
        result = original_step(self, opt_state, state_tuple, seq, prefix_outputs)

        current = self.inner_parameters()
        proposed = result.new_model.inner_parameters()
        delta = tree_sub(proposed, current)

        decision = gate(delta, current)
        accepted = decision.verdict == jnp.int32(Verdict.ACCEPT)

        # Select between the updated and the pre-update model. Both branches are
        # already computed, so this is a select, not a saving of work.
        committed_model = tree_select(accepted, result.new_model, self)

        metrics = dict(result.metrics)
        metrics["gate/accepted"] = accepted.astype(jnp.float32)
        metrics["gate/drift_delta"] = decision.drift_delta
        metrics["gate/confidence"] = decision.confidence
        metrics["gate/update_norm"] = tree_global_norm(delta)

        return type(result)(
            new_model=committed_model,
            new_optimizer_state=result.new_optimizer_state,
            new_state=result.new_state,
            metrics=metrics,
        )

    # Deliberately NOT jitted. The caller (`loss_for_sequence`) is already inside
    # `eqx.filter_jit` + `eqx.filter_checkpoint` + `scan_remat_chunk`; a nested jit
    # would fight the vendor's rematerialisation policy and silently inflate memory.
    # Keep this traceable, not compiled.
    return gated_inner_loop_step
