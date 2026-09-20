"""Rotating held-out probe sets for the anchor-consistency gate.

The probe set is the gate's attack surface. A fixed set is learnable: an
adversary with query access can shape updates that stay consistent on the known
probes while drifting freely elsewhere, which defeats Option A entirely without
ever tripping it. Rotation is the mitigation, and it costs cache locality and
recompilation -- changing probe *shapes* retriggers XLA compilation, so this
rotates contents at fixed shape.

Requirements, and how each is met:
  - **drawn from held-out data the served context cannot influence** -- the
    caller's responsibility. `probe_pool` is whatever it hands in; this module
    cannot check provenance, and a pool drawn from the served stream would
    defeat the gate silently.
  - **rotated on a schedule unpredictable to the adversary** -- the schedule is
    `jax.random.fold_in(key, window_index)`. Without `key` the sequence is not
    predictable from observed probes; with it, it is fully reproducible, which
    is what makes a run auditable after the fact.
  - **small enough that one anchor forward pass stays inside the ~10% budget**
    -- `probes_per_window` is the knob, and it trades directly against that
    budget. Measure with `trustgate.eval.overhead`.
  - **fixed shape across rotations** -- `jnp.take` along the leading axis with a
    fixed-size index array, so every window produces the same shapes and the
    scan body compiles once.

PRE-VERDICT DEFAULT: sampling is without replacement within a window and
independent across windows. Phase 1 data was meant to inform whether a
lower-discrepancy schedule (e.g. a rolling partition of the pool) detects
better per probe. It does not exist yet; this is the simple choice, not the
tuned one.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp


def pool_size(probe_pool) -> int:
    """Leading-axis length of the pool, checked for agreement across leaves.

    A pool whose leaves disagree on their leading axis would silently sample
    mismatched rows -- probe inputs from one example and targets from another,
    which scores as behavioural divergence and looks exactly like an attack.
    """
    leaves = jax.tree_util.tree_leaves(probe_pool)
    if not leaves:
        raise ValueError("probe_pool has no array leaves")
    sizes = {int(leaf.shape[0]) for leaf in leaves if leaf.ndim >= 1}
    if len(sizes) != 1:
        raise ValueError(
            f"probe_pool leaves disagree on their leading axis: {sorted(sizes)}. "
            f"Mismatched rows would pair one example's inputs with another's "
            f"targets and score as divergence."
        )
    return sizes.pop()


def make_probe_rotation(probe_pool, probes_per_window: int, key: jax.Array):
    """Build a rotation schedule over a held-out probe pool.

    Returns:
        Callable mapping window index -> fixed-shape probe batch. The index may
        be a traced array: `fold_in` accepts one, which is what lets the
        rotation live inside the scan.
    """
    if probes_per_window < 1:
        raise ValueError(
            f"probes_per_window must be at least 1, got {probes_per_window}"
        )

    size = pool_size(probe_pool)
    if probes_per_window > size:
        raise ValueError(
            f"probes_per_window {probes_per_window} exceeds the pool size "
            f"{size}. Sampling without replacement cannot fill the window, and "
            f"padding it would put duplicate probes in every batch -- which "
            f"reduces the effective probe count without reducing the cost."
        )

    def rotation(window_index):
        # Per-window subkey. Deterministic given `key`, so a run is auditable;
        # unpredictable without it, so the schedule is not learnable from
        # observed probes.
        subkey = jax.random.fold_in(key, window_index)
        indices = jax.random.choice(
            subkey, size, shape=(probes_per_window,), replace=False
        )
        return jax.tree.map(
            lambda leaf: jnp.take(leaf, indices, axis=0), probe_pool
        )

    return rotation
