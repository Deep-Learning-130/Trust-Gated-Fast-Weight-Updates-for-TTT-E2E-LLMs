"""Versioned fast-weight store: ring-buffered checkpoints with O(1) rollback.

Implements `checkpoint(fast_weights) -> version_id` and
`rollback(version_id) -> fast_weights` (dossier 5.4).

A correction to the design dossier
----------------------------------
Section 5.2 describes rollback as cheap, and in wall-clock terms it is -- a
single indexed read, no state unwinding, because JAX updates are pure. But it is
**not free in memory**. Because the inner loop runs inside `jax.lax.scan`, the
buffer cannot be a Python list of references that grows on demand; it must be a
fixed-shape pytree living in the scan carry, allocated up front.

Cost is `depth x |inner_parameters|`, resident on device for the whole sequence.
At 1B with a 4-deep buffer in bf16 that is real HBM, and it competes with the
activation memory the vendor's rematerialisation is already fighting for. Size
`depth` to the attack-detection latency window (dossier 5.5) and no larger --
this is the main non-compute overhead of the defense and must be reported as such.
"""

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp

from trustgate.types import FastWeights


class VersionedStore(eqx.Module):
    """Fixed-depth ring buffer of fast-weight snapshots.

    Each inexact leaf carries a leading axis of size `depth`. `head` is the next
    slot to write; the most recent valid snapshot is at `head - 1 (mod depth)`.
    """

    slots: FastWeights
    head: jnp.ndarray
    count: jnp.ndarray
    depth: int = eqx.field(static=True)


def init_store(fast_weights: FastWeights, depth: int) -> VersionedStore:
    """Allocate a `depth`-slot buffer shaped after `fast_weights`.

    Allocates eagerly and deliberately: doing it here makes the memory cost
    visible at setup rather than as a mid-sequence OOM.
    """
    if depth < 1:
        raise ValueError(f"depth must be >= 1, got {depth}")

    slots = jax.tree.map(
        lambda x: jnp.zeros((depth, *x.shape), dtype=x.dtype) if eqx.is_inexact_array(x) else x,
        fast_weights,
    )
    return VersionedStore(
        slots=slots,
        head=jnp.int32(0),
        count=jnp.int32(0),
        depth=depth,
    )


def checkpoint(store: VersionedStore, fast_weights: FastWeights) -> VersionedStore:
    """Write a snapshot at `head` and advance. Overwrites the oldest when full."""
    idx = store.head

    slots = jax.tree.map(
        lambda buf, val: (
            jax.lax.dynamic_update_index_in_dim(buf, val.astype(buf.dtype), idx, axis=0)
            if eqx.is_inexact_array(val)
            else buf
        ),
        store.slots,
        fast_weights,
    )

    return VersionedStore(
        slots=slots,
        head=(idx + 1) % store.depth,
        count=jnp.minimum(store.count + 1, store.depth),
        depth=store.depth,
    )


def rollback(store: VersionedStore, steps_back: int | jnp.ndarray = 0) -> FastWeights:
    """Read the snapshot `steps_back` behind the most recent one.

    `steps_back=0` is the last trusted checkpoint -- the dossier 5.3 step-5
    behaviour. Index arithmetic is traced-safe so this works inside a scan.

    Reading further back than `count` returns a zero slot rather than raising:
    under jit we cannot branch on emptiness. Callers must gate on `has_checkpoint`.
    """
    idx = (store.head - 1 - jnp.asarray(steps_back, dtype=jnp.int32)) % store.depth

    return jax.tree.map(
        lambda buf: (
            jax.lax.dynamic_index_in_dim(buf, idx, axis=0, keepdims=False)
            if eqx.is_inexact_array(buf)
            else buf
        ),
        store.slots,
    )


def has_checkpoint(store: VersionedStore, steps_back: int | jnp.ndarray = 0) -> jnp.ndarray:
    """Whether a valid snapshot exists at that depth. Traced-safe bool."""
    return jnp.asarray(steps_back, dtype=jnp.int32) < store.count


def memory_bytes(fast_weights: FastWeights, depth: int) -> int:
    """Resident bytes a `depth`-slot buffer will occupy. Report this."""
    per_snapshot = sum(
        leaf.size * leaf.dtype.itemsize
        for leaf in jax.tree.leaves(fast_weights)
        if eqx.is_inexact_array(leaf)
    )
    return per_snapshot * depth
