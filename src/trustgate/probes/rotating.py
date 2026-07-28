"""Rotating held-out probe sets for the anchor-consistency gate.

Status: NOT IMPLEMENTED. Phase 2.

The probe set is the gate's attack surface. A fixed set is learnable: an
adversary with query access can shape updates that stay consistent on the known
probes while drifting freely elsewhere, which defeats Option A entirely without
ever tripping it. Rotation is the mitigation, and it costs cache locality and
recompilation (changing probe shapes retriggers XLA compilation, so rotate
*contents* at fixed shape).

Requirements:
  - drawn from held-out data the served context cannot influence
  - rotated on a schedule unpredictable to the adversary
  - small enough that one anchor forward pass stays inside the ~10% overhead budget
  - fixed shape across rotations, to avoid recompilation
"""

from __future__ import annotations

import jax


def make_probe_rotation(probe_pool, probes_per_window: int, key: jax.Array):
    """Build a rotation schedule over a held-out probe pool.

    Returns:
        Callable mapping window index -> fixed-shape probe batch.
    """
    raise NotImplementedError("Phase 2.")
