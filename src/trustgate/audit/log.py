"""Audit trail for accept / reject / rollback decisions (dossier 5.1).

Two consumers, two different reasons this exists:
  - the offline influence auditor (`trustgate.gate.influence`), which needs the
    decision history to attribute downstream degradation to specific updates;
  - the eval harness, which turns accept/reject counts into operating-point curves.

Design constraint: decisions are produced *inside* `jax.lax.scan` under jit, so
nothing here may be called from the traced path. Python-side logging inside a
trace runs once at trace time and then never again -- a classic way to end up
with a log containing exactly one entry. Instead the interceptor emits per-step
values into its returned `metrics` dict, they ride out of the scan as stacked
arrays, and *this* module records them once, outside the trace.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np


@dataclass
class DecisionRecord:
    """One scored fast-weight update."""

    step: int
    chunk: int
    accepted: bool
    drift_delta: float
    cumulative_drift: float
    confidence: float
    update_norm: float
    rolled_back: bool = False


class AuditLog:
    """Append-only JSONL log of gate decisions.

    JSONL rather than a single JSON document so a crashed or killed run still
    leaves a readable, parseable trail up to the failure point.
    """

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def record_batch(self, metrics: dict, step: int) -> int:
        """Write one entry per chunk from a scan's stacked metric arrays.

        Args:
            metrics: the `gate/*` arrays returned by the interceptor, each of
                shape [num_chunks].
            step: outer training/eval step these chunks belong to.

        Returns:
            Number of records written.
        """
        accepted = np.asarray(metrics["gate/accepted"]).reshape(-1)
        drift = np.asarray(metrics["gate/drift_delta"]).reshape(-1)
        confidence = np.asarray(metrics["gate/confidence"]).reshape(-1)
        norms = np.asarray(metrics["gate/update_norm"]).reshape(-1)

        cumulative = float(np.cumsum(drift)[-1]) if drift.size else 0.0
        running = np.cumsum(drift)

        with self.path.open("a", encoding="utf-8") as fh:
            for i in range(accepted.size):
                rec = DecisionRecord(
                    step=step,
                    chunk=i,
                    accepted=bool(accepted[i] > 0.5),
                    drift_delta=float(drift[i]),
                    cumulative_drift=float(running[i]),
                    confidence=float(confidence[i]),
                    update_norm=float(norms[i]),
                )
                fh.write(json.dumps(asdict(rec)) + "\n")

        del cumulative
        return int(accepted.size)

    def read(self) -> list[DecisionRecord]:
        if not self.path.exists():
            return []
        with self.path.open(encoding="utf-8") as fh:
            return [DecisionRecord(**json.loads(line)) for line in fh if line.strip()]
