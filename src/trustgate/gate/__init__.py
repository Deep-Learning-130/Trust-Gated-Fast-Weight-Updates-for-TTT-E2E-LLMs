"""Gate signal implementations. See ADR-F1 for the Option A/B/C decision.

Implemented before the Phase 1 verdict, per the dated revision in
`experiments/001-attack-spike/PREREGISTERED.md` (2026-09-20). The design values
inside are pre-verdict guesses, labelled as such in each module, and are not
findings.

Option C (`influence.py`) stays stubbed: it is an offline auditor by design and
ADR-F1 keeps it out of the online path.
"""

from trustgate.gate.anchor import (
    DIVERGENCES,
    anchor_consistency_gate,
    probe_divergence,
)
from trustgate.gate.policy import Reason, combine
from trustgate.gate.uncertainty import (
    IMPLEMENTED_SIGNALS,
    STATEFUL_SIGNALS,
    uncertainty_gate,
)

__all__ = [
    "DIVERGENCES",
    "IMPLEMENTED_SIGNALS",
    "Reason",
    "STATEFUL_SIGNALS",
    "anchor_consistency_gate",
    "combine",
    "probe_divergence",
    "uncertainty_gate",
]
