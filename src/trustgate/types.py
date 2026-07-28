"""Core types for the trust gate.

Every type here must survive `jax.jit` tracing: the gate runs *inside*
`jax.lax.scan` in `MetaModel.loss_for_sequence`, so decisions are traced arrays,
never Python bools. See `docs/adr/ADR-003-jax-interceptor-shape.md`.
"""

from __future__ import annotations

from typing import Any, Protocol

import equinox as eqx
import jax.numpy as jnp

# A pytree with the same structure as `MetaModel.inner_parameters()` -- the
# inner-spec-filtered subtree that TTT-E2E updates at test time. These are the
# "fast weights". Outer/slow parameters are not part of this subtree.
FastWeights = Any

# A pytree matching FastWeights, holding the proposed update (delta-theta).
Updates = Any


class Verdict:
    """Traced-safe verdict constants. Not an IntEnum -- these are compared
    against int32 arrays under jit, where Python enum semantics do not apply."""

    ACCEPT = 0
    REJECT = 1
    ROLLBACK = 2


class GateDecision(eqx.Module):
    """Result of scoring one proposed fast-weight update.

    Implements the `gate(...) -> {decision, drift_delta, confidence}` interface
    from dossier section 5.4.
    """

    verdict: jnp.ndarray
    """int32 scalar, one of `Verdict`."""

    drift_delta: jnp.ndarray
    """float32 scalar. Distance this update would add to cumulative drift from
    the anchor. Only meaningful when verdict == ACCEPT; a rejected update
    contributes zero drift because it never commits."""

    confidence: jnp.ndarray
    """float32 scalar in [0, 1]. Low confidence triggers conservative reject
    (dossier section 5.5, fail-safe toward rejecting)."""

    @staticmethod
    def accept_all() -> GateDecision:
        """The identity decision. Used by the pass-through gate to prove the
        interceptor is a no-op when no policy is installed."""
        return GateDecision(
            verdict=jnp.int32(Verdict.ACCEPT),
            drift_delta=jnp.float32(0.0),
            confidence=jnp.float32(1.0),
        )


class DriftState(eqx.Module):
    """Cumulative drift from the frozen anchor, with a per-window budget.

    Implements `accumulate(drift_delta) -> {budget_remaining, breach}` from
    dossier section 5.4. Carried through the scan, never mutated in place.
    """

    cumulative: jnp.ndarray
    """float32 scalar. Distance from anchor accumulated in the current window."""

    budget: jnp.ndarray
    """float32 scalar. The epsilon of "bounded cumulative-drift budget". This is
    the quantity the patent's bounded-drift guarantee is stated over."""

    breached: jnp.ndarray
    """bool scalar. True once cumulative > budget; latches until window reset."""

    @property
    def remaining(self) -> jnp.ndarray:
        return self.budget - self.cumulative


class Gate(Protocol):
    """Scores a proposed update. Must be jit-safe and side-effect free."""

    def __call__(
        self,
        updates: Updates,
        fast_weights: FastWeights,
        anchor: FastWeights | None = None,
    ) -> GateDecision: ...
