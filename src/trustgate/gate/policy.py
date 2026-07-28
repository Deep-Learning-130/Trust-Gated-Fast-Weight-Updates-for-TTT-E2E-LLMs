"""Decision policy -- combines gate signals into accept / reject / rollback.

Status: NOT IMPLEMENTED. Phase 2.

This is the component the operating-point curve is swept over. Dossier 5.6 is
emphatic that the deliverable is the *curve*, not a threshold, so this must take
its thresholds as parameters and never hard-code them.

Required behaviour (dossier 5.3 and 5.5):
  1. accept    -> commit, add drift_delta to the accumulator, maybe checkpoint
  2. reject    -> discard the update, log, continue
  3. breach    -> roll back to last trusted checkpoint, reset window, alert
  4. low confidence -> conservative REJECT. Fail-safe leans toward rejecting.

The interaction between 3 and 4 is the subtle part: a gate that is uncertain
often will reject often, which starves adaptation and looks like a clean-accuracy
regression rather than a security event. Log the two causes separately.
"""

from __future__ import annotations

import jax.numpy as jnp

from trustgate.types import DriftState, GateDecision, Verdict


def combine(
    primary: GateDecision,
    secondary: GateDecision | None,
    drift: DriftState,
    *,
    min_confidence: float,
) -> GateDecision:
    """Fold Option A + Option B + budget state into one verdict.

    Must be traced-safe: no Python branching on array values.
    """
    raise NotImplementedError("Phase 2.")
