"""Option B -- update-uncertainty threshold. CHEAP SECONDARY.

Status: NOT IMPLEMENTED. Phase 2.

Per ADR-F1: backstops Option A on every window at near-zero cost. Flags
high-variance or low-confidence updates using signals already computed by the
inner loop -- no extra forward pass. Candidate signals, all free at the hook
point in `trustgate.interceptor`:

  - inner-loop loss spike relative to a running mean over the sequence
  - gradient norm outliers (vendor already computes `global_norm_safe`)
  - disagreement across the mini-batch within a chunk

Known weakness, recorded honestly in ADR-F1: weak against low-variance stealth
poison. A patient adversary making many small in-distribution updates is exactly
the threat model of dossier section 1 ("a slow, benign-looking sequence"), and it
is the case Option B is worst at. It is a cost-effective backstop, not a defense
on its own -- do not let a good ROC on loud attacks disguise that.
"""

from __future__ import annotations

import jax.numpy as jnp

from trustgate.types import FastWeights, GateDecision, Updates


def uncertainty_gate(
    threshold: float,
    *,
    signal: str = "grad_norm_zscore",
    ema_decay: float = 0.99,
):
    """Build a gate thresholding on update uncertainty.

    Args:
        threshold: reject above this score.
        signal: "grad_norm_zscore" | "loss_spike" | "batch_disagreement".
        ema_decay: decay for the running baseline the z-score is taken against.

    Returns:
        A `Gate`.
    """
    raise NotImplementedError("Option B is Phase 2.")
