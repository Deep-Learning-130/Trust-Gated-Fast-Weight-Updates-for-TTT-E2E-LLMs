"""Poisoning attack against TTT-E2E fast weights (Phase 1 kill-gate).

DEFENSIVE RESEARCH -- run only against models we control. See DISCLOSURE.md.
Exists to establish the threat model that motivates the trust gate and to serve
as the evaluation adversary. Do not publish working artifacts ahead of a
coordinated disclosure to the TTT-E2E authors.
"""

from trustgate.attack.objectives import AttackSpec, Objective

__all__ = ["AttackSpec", "Objective"]
