"""Decision policy -- combines gate signals into accept / reject / rollback.

This is the component the operating-point curve is swept over. Dossier 5.6 is
emphatic that the deliverable is the *curve*, not a threshold, so this takes its
thresholds as parameters and hard-codes none.

Required behaviour (dossier 5.3 and 5.5):
  1. accept    -> commit, add drift_delta to the accumulator, maybe checkpoint
  2. reject    -> discard the update, log, continue
  3. breach    -> roll back to last trusted checkpoint, reset window, alert
  4. low confidence -> conservative REJECT. Fail-safe leans toward rejecting.

The interaction between 3 and 4 is the subtle part: a gate that is uncertain
often will reject often, which starves adaptation and looks like a
clean-accuracy regression rather than a security event. `combine` therefore
returns the verdict *and* the reason, so the two causes are logged separately.

WHAT DOES NOT WORK YET
----------------------
`combine` accepts a `DriftState` and returns ROLLBACK on a breach, and it is
unit-tested on that. **Nothing advances the accumulator it reads.**
`interceptor.py` cannot reach `drift/accumulator.py` or `store/versioned.py`:
the vendor's scan carry is fixed at
`(model, inner_opt_state, (state_all, state_suffix))` and a `scan` requires the
carry structure out to match in, so there is no slot for drift state without
editing vendor code, which ADR-002 forbids. See ADR-003's 2026-08-08 correction
and `docs/adr/ADR-P3-1-drift-carry.md`, which chooses the route and writes no
code.

Until that lands, a live gate passes a `DriftState` whose `cumulative` is
always zero, so branch 3 never fires in a real run. Do not describe the
bounded-drift property as enforced.
"""

from __future__ import annotations

import jax.numpy as jnp

from trustgate.types import DriftState, GateDecision, Verdict

#: Why a verdict came out the way it did. Traced-safe constants, same reasoning
#: as `Verdict`: these are compared against int32 arrays under jit.
class Reason:
    SIGNAL = 0
    """The primary or secondary signal rejected on its own merits."""

    LOW_CONFIDENCE = 1
    """Rejected because the gate was not sure, not because it saw an attack.
    Kept distinct so an over-cautious gate is not read as a security event."""

    BUDGET = 2
    """Cumulative drift breached the window budget."""


def combine(
    primary: GateDecision,
    secondary: GateDecision | None,
    drift: DriftState,
    *,
    min_confidence: float,
) -> tuple[GateDecision, jnp.ndarray]:
    """Fold Option A + Option B + budget state into one verdict.

    Traced-safe: no Python branching on array values. The `secondary is None`
    test is a Python-level branch on a *static* value -- whether a second signal
    was installed at all -- which is resolved at trace time and is not a branch
    on traced data.

    Returns `(decision, reason)`, where `reason` is a `Reason` int32 scalar.
    """
    if not 0.0 <= min_confidence <= 1.0:
        raise ValueError(
            f"min_confidence must be in [0, 1], got {min_confidence}; it is "
            f"compared against GateDecision.confidence, which types.py defines "
            f"over that range"
        )

    accept = jnp.int32(Verdict.ACCEPT)
    reject = jnp.int32(Verdict.REJECT)

    # Either signal rejecting is enough. Verdict constants ascend in severity
    # (ACCEPT=0, REJECT=1, ROLLBACK=2), so a max is the "most severe wins" fold
    # without a branch.
    verdict = primary.verdict
    confidence = primary.confidence
    drift_delta = primary.drift_delta
    if secondary is not None:
        verdict = jnp.maximum(verdict, secondary.verdict)
        # The pair is only as confident as its least confident member.
        confidence = jnp.minimum(confidence, secondary.confidence)
        # Drift is a property of the update, not of who judged it. The primary
        # signal is the behavioural one, so its measure is the one that counts;
        # taking a max here would let a cheap secondary inflate the budget.

    reason = jnp.int32(Reason.SIGNAL)

    # 4. Conservative reject on low confidence -- fail-safe toward rejecting.
    unsure = confidence < jnp.float32(min_confidence)
    reason = jnp.where(
        jnp.logical_and(unsure, verdict == accept),
        jnp.int32(Reason.LOW_CONFIDENCE),
        reason,
    )
    verdict = jnp.where(unsure, jnp.maximum(verdict, reject), verdict)

    # A verdict that is not ACCEPT commits nothing and therefore drifts nothing.
    drift_delta = jnp.where(verdict == accept, drift_delta, jnp.float32(0.0))

    # 3. Breach dominates everything above it: an already-latched breach, or one
    # this update would cause. `would_breach` is checked *before* committing so
    # the guarantee is a strict invariant rather than something restored after
    # the fact -- the same reason `drift.accumulator.would_breach` exists.
    would_breach = (drift.cumulative + drift_delta) > drift.budget
    breached = jnp.logical_or(drift.breached, would_breach)
    verdict = jnp.where(breached, jnp.int32(Verdict.ROLLBACK), verdict)
    reason = jnp.where(breached, jnp.int32(Reason.BUDGET), reason)
    drift_delta = jnp.where(breached, jnp.float32(0.0), drift_delta)

    return (
        GateDecision(
            verdict=verdict,
            drift_delta=jnp.float32(drift_delta),
            confidence=jnp.float32(confidence),
        ),
        jnp.int32(reason),
    )
