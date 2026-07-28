"""Cumulative-drift accumulator with a per-window budget.

This is where the headline property of the patent lives. The claim is not "we
detect poison" -- detection is statistical and arguable. The claim is
**bounded drift**: over any window, committed fast weights cannot travel more
than epsilon from the frozen anchor before a rollback is forced. That is a
stated, testable guarantee, which is what makes it worth claiming.

Implements `accumulate(drift_delta) -> {budget_remaining, breach}` (dossier 5.4).
Pure and jit-safe -- carried through the scan, never mutated in place.
"""

from __future__ import annotations

import jax.numpy as jnp

from trustgate.types import DriftState


def init_drift(budget: float) -> DriftState:
    """Fresh accumulator with an empty window."""
    return DriftState(
        cumulative=jnp.float32(0.0),
        budget=jnp.float32(budget),
        breached=jnp.bool_(False),
    )


def accumulate(state: DriftState, drift_delta: jnp.ndarray) -> DriftState:
    """Add one accepted update's drift contribution.

    `breached` **latches**: once tripped it stays true until `reset_window`.
    Without latching, a subsequent small negative-drift update could silently
    clear the flag and the rollback would never fire -- which would make the
    bounded-drift guarantee false in exactly the case an adversary would aim for.
    """
    cumulative = state.cumulative + drift_delta.astype(jnp.float32)
    breached = jnp.logical_or(state.breached, cumulative > state.budget)
    return DriftState(cumulative=cumulative, budget=state.budget, breached=breached)


def reset_window(state: DriftState) -> DriftState:
    """Start a new window. Called after a rollback, or on window boundary.

    Budget is preserved; only the accumulated distance and the latch reset.
    """
    return DriftState(
        cumulative=jnp.float32(0.0),
        budget=state.budget,
        breached=jnp.bool_(False),
    )


def would_breach(state: DriftState, drift_delta: jnp.ndarray) -> jnp.ndarray:
    """Whether accepting `drift_delta` would exceed the budget.

    Lets the policy decline an update *before* committing it, so the guarantee
    holds as a strict invariant rather than being restored after the fact.
    """
    return (state.cumulative + drift_delta.astype(jnp.float32)) > state.budget
