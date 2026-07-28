"""Option A -- frozen-anchor consistency. THE PRIMARY GATE.

Status: NOT IMPLEMENTED. Gated on the Phase 1 attack spike (dossier section 7:
no attack means no defense). Do not implement before `experiments/001-attack-spike`
returns a positive result against its pre-registered threshold.

Decided in ADR-F1 as the primary signal. The idea: hold a frozen meta-learned
anchor theta_0, keep a small rotating set of held-out probes, and ask whether
committing this update pulls probe behaviour away from the anchor's. Behavioural,
not distributional -- which is precisely the diff against MedBN, whose signal is
batch-norm *statistics*. TTT-E2E has no batch norm, so there is nothing for a
MedBN-style test to read; see `docs/patent/prior-art/medbn-diff.md`.

Cost: one extra anchor forward pass over the probe set per scored window. That
is the single largest overhead in the design and the reason the section 5.1
budget is stated as "<= ~10% of the per-window update cost". Measure it; do not
assume it.

Open design questions, to settle with Phase 1 data in hand:
  - Probe set size, and rotation policy. Fixed probes are cheap to overfit; an
    adversary who learns them can shape updates that look consistent on probes
    and drift everywhere else. Rotation is the mitigation and it is not free.
  - Divergence measure on probe logits: KL, symmetric KL, or top-k agreement.
  - Whether to score against theta_0 or against the last *trusted* checkpoint.
    Anchor is the stronger guarantee; last-trusted allows legitimate long-range
    adaptation. This is the central false-reject/false-accept tension of 5.6.
"""

from __future__ import annotations

import jax.numpy as jnp

from trustgate.types import FastWeights, GateDecision, Updates


def anchor_consistency_gate(
    anchor_params: FastWeights,
    probes,
    threshold: float,
    *,
    divergence: str = "symmetric_kl",
):
    """Build a gate scoring behavioural divergence from the frozen anchor.

    Args:
        anchor_params: frozen meta-learned init, theta_0. Never updated.
        probes: rotating held-out probe batch (see `trustgate.probes`).
        threshold: reject above this divergence.
        divergence: one of "symmetric_kl", "kl", "topk_agreement".

    Returns:
        A `Gate`.
    """
    raise NotImplementedError(
        "Option A is Phase 2. Blocked on the Phase 1 attack spike; see "
        "experiments/001-attack-spike/PREREGISTERED.md"
    )


def probe_divergence(
    updated_params: FastWeights,
    anchor_params: FastWeights,
    probes,
    divergence: str = "symmetric_kl",
) -> jnp.ndarray:
    """Behavioural distance between updated and anchor models on the probe set."""
    raise NotImplementedError("Phase 2.")
