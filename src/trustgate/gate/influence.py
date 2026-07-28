"""Option C -- influence / gradient estimate. OFFLINE AUDITOR ONLY.

Status: NOT IMPLEMENTED. Phase 2+, and never on the online path.

ADR-F1 is explicit: strongest detection signal, far too slow to sit inside the
serving loop. It has two jobs and neither is real-time:

  1. Ground truth for the operating-point curves. To claim Option A separates
     poison from benign drift, we need a reference for what "poison" was, and an
     offline influence estimate is the best available one.
  2. A dependent claim. An expensive-but-accurate auditor that runs out-of-band
     over the audit log is independently claimable and strengthens the family
     around the online gate.

Do not be tempted to optimise this into the online path. If it becomes fast
enough to run per-window, that is a different invention and needs its own
novelty diff.
"""

from __future__ import annotations

from trustgate.types import FastWeights, Updates


def offline_influence_audit(
    audit_log_path,
    anchor_params: FastWeights,
    eval_set,
):
    """Post-hoc: attribute downstream degradation to specific admitted updates.

    Consumes the accept/reject/rollback trail from `trustgate.audit.log`.

    Returns:
        Per-update influence scores on the evaluation objective.
    """
    raise NotImplementedError("Option C is Phase 2+, offline only.")
