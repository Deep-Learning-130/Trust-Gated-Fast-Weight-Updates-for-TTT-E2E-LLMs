"""`blocked_attention` against JAX's own XLA attention, on every call shape the vendor uses.

The reference is `jax.nn.dot_product_attention(..., implementation="xla")` --
the computation `attention_patch` claims to reproduce a block of queries at a
time. On real hardware the comparison that matters is against the vendor's
cuDNN kernel; that one is measured on the laptop, which can run both.
"""

from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from trustgate import attention_patch
from trustgate.attention_patch import blocked_attention

ATTENTION_PY = (
    Path(__file__).resolve().parents[1] / "vendor" / "ttt-e2e" / "ttt" / "model" / "attention.py"
)

#: The three call shapes in vendor attention.py (:183, :214, :244-251/:312).
CALLS = {
    "mask": lambda T: dict(mask=jnp.tril(jnp.ones((T, T), dtype=jnp.bool_))),
    "causal": lambda T: dict(is_causal=True),
    "causal+window": lambda T: dict(is_causal=True, local_window_size=(T // 3 - 1, 0)),
}


def qkv(seed=0, T=48, N=4, H=8, dtype=jnp.float32, batch=None):
    shape = (T, N, H) if batch is None else (batch, T, N, H)
    ks = jax.random.split(jax.random.PRNGKey(seed), 3)
    return tuple(jax.random.normal(k, shape, dtype) for k in ks)


def reference(q, k, v, **kw):
    return jax.nn.dot_product_attention(q, k, v, implementation="xla", **kw)


@pytest.mark.parametrize("call", sorted(CALLS))
@pytest.mark.parametrize("batch", [None, 2])
def test_matches_xla_attention_fp32(call, batch):
    q, k, v = qkv(batch=batch)
    kw = CALLS[call](q.shape[-3])
    got = blocked_attention(q, k, v, query_block=16, **kw)
    want = reference(q, k, v, **kw)
    assert got.shape == want.shape and got.dtype == want.dtype
    np.testing.assert_allclose(np.asarray(got), np.asarray(want), rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("call", sorted(CALLS))
def test_matches_xla_attention_bf16(call):
    q, k, v = qkv(seed=1, dtype=jnp.bfloat16)
    kw = CALLS[call](q.shape[0])
    got = jax.jit(lambda q, k, v: blocked_attention(q, k, v, query_block=16, **kw))(q, k, v)
    want = jax.jit(lambda q, k, v: reference(q, k, v, **kw))(q, k, v)
    assert got.dtype == want.dtype == jnp.bfloat16
    np.testing.assert_allclose(
        np.asarray(got, np.float32), np.asarray(want, np.float32), rtol=2e-2, atol=2e-2
    )


def test_gradients_match_xla_attention():
    """The inner loop differentiates through the suffix attention."""
    q, k, v = qkv(seed=2)
    kw = CALLS["causal+window"](q.shape[0])

    def loss(fn):
        return lambda q, k, v: jnp.sum(fn(q, k, v) ** 2)

    g_blocked = jax.grad(loss(lambda q, k, v: blocked_attention(q, k, v, query_block=16, **kw)),
                         argnums=(0, 1, 2))(q, k, v)
    g_ref = jax.grad(loss(lambda q, k, v: reference(q, k, v, **kw)), argnums=(0, 1, 2))(q, k, v)
    for g, w in zip(g_blocked, g_ref):
        np.testing.assert_allclose(np.asarray(g), np.asarray(w), rtol=1e-4, atol=1e-5)


def test_the_window_actually_masks():
    """Guard against a window that silently degenerates to plain causal."""
    q, k, v = qkv(seed=3)
    windowed = blocked_attention(q, k, v, query_block=16, **CALLS["causal+window"](q.shape[0]))
    causal = blocked_attention(q, k, v, query_block=16, is_causal=True)
    assert not np.allclose(np.asarray(windowed), np.asarray(causal))


def test_install_serves_cudnn_requests_and_uninstall_restores():
    q, k, v = qkv(seed=4)
    original = jax.nn.dot_product_attention
    attention_patch.install_blocked_attention()
    try:
        attention_patch.install_blocked_attention()  # no double wrap
        # implementation="cudnn" has no CPU kernel; served by the patch instead.
        got = jax.nn.dot_product_attention(q, k, v, is_causal=True, implementation="cudnn")
        np.testing.assert_allclose(
            np.asarray(got), np.asarray(reference(q, k, v, is_causal=True)), rtol=1e-5, atol=1e-5
        )
        assert attention_patch.is_installed()
    finally:
        attention_patch.uninstall_blocked_attention()
    assert jax.nn.dot_product_attention is original
    assert not attention_patch.is_installed()


def test_vendor_attention_call_sites_are_the_ones_covered():
    """Tripwire for a submodule bump: a new call shape would not be covered by
    the tests above. Re-read attention.py before trusting the patch if this fails."""
    if not ATTENTION_PY.is_file():
        pytest.skip("vendor submodule not checked out")
    src = ATTENTION_PY.read_text(encoding="utf-8")
    assert src.count("jax.nn.dot_product_attention(") == 4
    for fragment in (
        "jax.nn.dot_product_attention(xq, xk, xv, mask=attention_mask,",
        "jax.nn.dot_product_attention(xq, xk, xv, is_causal=True, implementation=",
        "local_window_size=(self.config.sliding_window_size - 1, 0),",
        "jax.nn.dot_product_attention(xq, xk, xv, is_causal=True, local_window_size=(self.window_size - 1, 0), implementation=\"cudnn\")",
    ):
        assert fragment in src, f"vendor attention.py no longer contains: {fragment}"
