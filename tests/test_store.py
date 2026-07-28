import jax.numpy as jnp

from trustgate.store.versioned import (
    checkpoint,
    has_checkpoint,
    init_store,
    memory_bytes,
    rollback,
)


def _fw(val):
    return {"w": jnp.full((2, 2), float(val), dtype=jnp.float32), "tag": "static"}


def test_checkpoint_and_rollback_roundtrip():
    store = init_store(_fw(0.0), depth=4)
    store = checkpoint(store, _fw(1.0))
    store = checkpoint(store, _fw(2.0))
    assert jnp.allclose(rollback(store, steps_back=0)["w"], 2.0)
    assert jnp.allclose(rollback(store, steps_back=1)["w"], 1.0)


def test_static_leaf_survives_store_roundtrip():
    store = init_store(_fw(0.0), depth=2)
    store = checkpoint(store, _fw(5.0))
    assert rollback(store, 0)["tag"] == "static"


def test_ring_buffer_overwrites_oldest():
    store = init_store(_fw(0.0), depth=2)
    for v in (1.0, 2.0, 3.0):
        store = checkpoint(store, _fw(v))
    # depth 2: latest=3, one back=2, the 1.0 snapshot is gone
    assert jnp.allclose(rollback(store, 0)["w"], 3.0)
    assert jnp.allclose(rollback(store, 1)["w"], 2.0)


def test_has_checkpoint_tracks_count():
    store = init_store(_fw(0.0), depth=3)
    assert not bool(has_checkpoint(store, 0))
    store = checkpoint(store, _fw(1.0))
    assert bool(has_checkpoint(store, 0))
    assert not bool(has_checkpoint(store, 1))


def test_count_saturates_at_depth():
    store = init_store(_fw(0.0), depth=2)
    for v in range(5):
        store = checkpoint(store, _fw(v))
    assert int(store.count) == 2


def test_memory_bytes_matches_hand_count():
    # 2x2 float32 = 16 bytes/snapshot; depth 4 = 64 bytes. The "tag" leaf is
    # static and must not be counted.
    assert memory_bytes(_fw(0.0), depth=4) == 64


def test_init_rejects_zero_depth():
    import pytest

    with pytest.raises(ValueError):
        init_store(_fw(0.0), depth=0)
