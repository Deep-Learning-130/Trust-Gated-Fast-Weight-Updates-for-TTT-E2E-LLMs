import jax.numpy as jnp
import numpy as np
import pytest

from trustgate.tree_ops import (
    tree_add,
    tree_global_norm,
    tree_select,
    tree_sub,
    tree_zeros_like,
)


def test_sub_then_add_roundtrips(tiny_fast_weights):
    a = tiny_fast_weights
    b = tree_zeros_like(a)
    delta = tree_sub(a, b)
    recovered = tree_add(b, delta)
    assert jnp.allclose(recovered["layer0"]["w"], a["layer0"]["w"])
    assert jnp.allclose(recovered["layer1"]["w"], a["layer1"]["w"])


def test_static_leaf_untouched(tiny_fast_weights):
    delta = tree_sub(tiny_fast_weights, tree_zeros_like(tiny_fast_weights))
    assert delta["meta"] == "not-an-array"


def test_global_norm_matches_numpy(tiny_fast_weights):
    got = float(tree_global_norm(tiny_fast_weights))
    leaves = [tiny_fast_weights["layer0"]["w"], tiny_fast_weights["layer0"]["b"], tiny_fast_weights["layer1"]["w"]]
    expected = float(np.sqrt(sum(float((np.asarray(l) ** 2).sum()) for l in leaves)))
    # float32 accumulation vs numpy float64 -- close, not bit-equal
    assert got == pytest.approx(expected, rel=1e-5)


def test_global_norm_empty_is_zero():
    assert float(tree_global_norm({"meta": "x"})) == 0.0


def test_select_picks_branch(tiny_fast_weights):
    a = tiny_fast_weights
    b = tree_zeros_like(a)
    on_true = tree_select(jnp.bool_(True), a, b)
    on_false = tree_select(jnp.bool_(False), a, b)
    assert jnp.allclose(on_true["layer0"]["w"], a["layer0"]["w"])
    assert jnp.allclose(on_false["layer0"]["w"], b["layer0"]["w"])
