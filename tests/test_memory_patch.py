"""The sliced LM loss against the vendor's own `loss.py`, not against a transcription.

`vendor/ttt-e2e/ttt/model/loss.py` imports only `jax` and `jaxtyping`, so it can
be loaded straight from the file without importing the rest of the vendor
package (which this environment deliberately cannot import). The reference in
these tests is therefore the vendor's code itself, run on full logits exactly
as `MetaModel.lm_loss` runs it. `memory_patch` must agree with it on the loss,
on the per-token NLL, and -- because the inner loop trains on it -- on the
gradients.
"""

import importlib.util
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from trustgate.memory_patch import (
    SLICE_TOKENS,
    reduce_like_vendor,
    slice_size,
    sliced_token_log_probs,
)

VENDOR = Path(__file__).resolve().parents[1] / "vendor" / "ttt-e2e" / "ttt"
LOSS_PY = VENDOR / "model" / "loss.py"
TRANSFORMER_PY = VENDOR / "model" / "transformer.py"

pytestmark = pytest.mark.skipif(
    not LOSS_PY.is_file(), reason="vendor submodule not checked out"
)


def vendor_loss():
    spec = importlib.util.spec_from_file_location("_vendor_loss", LOSS_PY)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def vendor_lm_loss(hidden, kernel, targets, valid):
    """`MetaModel.lm_loss` minus the transformer: full logits, vendor functions."""
    v = vendor_loss()
    logits = hidden @ kernel
    loss, pure = v.cross_entropy_loss_and_accuracy(logits, targets, valid)
    nll = -v.token_log_probs(logits, targets)
    return loss, pure, nll


def sliced_lm_loss(hidden, kernel, targets, valid, slice_tokens=4):
    lp32, lp_native = sliced_token_log_probs(
        hidden, targets, lambda hs: hs @ kernel, slice_tokens=slice_tokens
    )
    loss, pure = reduce_like_vendor(lp32, targets, valid)
    return loss, pure, -lp_native


def problem(seed=0, tokens=32, hidden=16, vocab=50, dtype=jnp.float32):
    rng = np.random.default_rng(seed)
    h = jnp.asarray(rng.normal(0, 1, (tokens, hidden)), dtype=dtype)
    k = jnp.asarray(rng.normal(0, 0.5, (hidden, vocab)), dtype=dtype)
    t = jnp.asarray(rng.integers(0, vocab, tokens), dtype=jnp.int32)
    return h, k, t


@pytest.mark.parametrize("masked", [False, True])
def test_loss_and_token_nll_match_the_vendor(masked):
    h, k, t = problem()
    valid = None
    if masked:
        valid = jnp.asarray(np.r_[0, np.ones(24), np.zeros(7)], dtype=jnp.float32)

    want = vendor_lm_loss(h, k, t, valid)
    got = sliced_lm_loss(h, k, t, valid)

    for w, g in zip(want, got):
        np.testing.assert_allclose(np.asarray(g), np.asarray(w), rtol=1e-6, atol=1e-6)


def test_gradients_match_the_vendor():
    """The inner loop trains on this loss, so equal values are not enough."""
    h, k, t = problem(seed=1)
    valid = jnp.asarray(np.r_[0, np.ones(31)], dtype=jnp.float32)

    def f_vendor(h, k):
        return vendor_lm_loss(h, k, t, valid)[0]

    def f_sliced(h, k):
        return sliced_lm_loss(h, k, t, valid)[0]

    gv = jax.grad(f_vendor, argnums=(0, 1))(h, k)
    gs = jax.grad(f_sliced, argnums=(0, 1))(h, k)
    for w, g in zip(gv, gs):
        np.testing.assert_allclose(np.asarray(g), np.asarray(w), rtol=1e-5, atol=1e-7)


def test_bf16_token_nll_keeps_the_vendors_precision():
    """`token_log_probs` does not cast to fp32, so neither may the patch: the
    per-token NLL the baseline reports must be the vendor's bf16 number."""
    h, k, t = problem(seed=2, dtype=jnp.bfloat16)

    # Both sides jitted, because in bf16 the VENDOR disagrees with itself between
    # eager and jit: 5.881573 eager vs 5.881661 jitted on this problem (XLA fuses
    # the bf16 dot with the fp32 convert and skips one rounding). The vendor only
    # ever runs jitted, so that is the number to match. Jitted vs jitted the two
    # agree to ~8e-8 relative; eager vs jitted would have compared the vendor's
    # two execution modes rather than the patch.
    loss_v, _, nll_v = jax.jit(lambda h, k: vendor_lm_loss(h, k, t, None))(h, k)
    loss_s, _, nll_s = jax.jit(lambda h, k: sliced_lm_loss(h, k, t, None))(h, k)

    assert nll_s.dtype == nll_v.dtype == jnp.bfloat16
    np.testing.assert_array_equal(
        np.asarray(nll_s, dtype=np.float32), np.asarray(nll_v, dtype=np.float32)
    )
    assert loss_s.dtype == loss_v.dtype == jnp.float32
    np.testing.assert_allclose(float(loss_s), float(loss_v), rtol=1e-6)


def test_leading_batch_dimension_is_preserved():
    """Under train.py's eval the loss is vmapped; a batched call must also agree."""
    h, k, t = problem(seed=3, tokens=24)
    hb, tb = h.reshape(2, 12, -1), t.reshape(2, 12)
    lp32, lp_native = sliced_token_log_probs(hb, tb, lambda hs: hs @ k, slice_tokens=4)
    assert lp32.shape == lp_native.shape == (2, 12)

    flat32, _ = sliced_token_log_probs(h, t, lambda hs: hs @ k, slice_tokens=4)
    np.testing.assert_allclose(np.asarray(lp32).ravel(), np.asarray(flat32), rtol=1e-6)


def test_slice_size_divides_the_chunk():
    assert slice_size(1024) == SLICE_TOKENS == 128
    assert slice_size(96, 128) == 96
    assert slice_size(1000, 128) == 125
    assert slice_size(7, 4) == 1


def test_vendor_lm_loss_still_has_the_shape_the_patch_replaces():
    """Tripwire for a submodule bump. If any of these lines moved, the patch
    may now compute something the vendor no longer does -- re-read
    `transformer.py:617-645` and `:814-828` before trusting it."""
    if not TRANSFORMER_PY.is_file():
        pytest.skip("vendor submodule not checked out")
    src = TRANSFORMER_PY.read_text(encoding="utf-8")
    for line in (
        "value_and_grad_fn = eqx.filter_value_and_grad(MetaModel.lm_loss, has_aux=True)",
        "lm_outputs = self.language_model.suffix_call(prefix_outputs=prefix_outputs, state=state, seq=seq)",
        "loss, loss_pure_ce = cross_entropy_loss_and_accuracy(lm_outputs.logits, seq.target_tokens, seq.loss_masks)",
        "token_nll_loss = -token_log_probs(lm_outputs.logits, seq.target_tokens)",
        "return loss, (loss_pure_ce, token_nll_loss, lm_outputs.new_state)",
        "hidden_states, shared_kernel = promote_dtype(hidden_states, shared_kernel, dtype=self.compute_dtype)",
        "lm_logits = hidden_states @ shared_kernel",
    ):
        assert line in src, f"vendor transformer.py no longer contains: {line}"
    assert "logits = logits.astype(jnp.float32)" in LOSS_PY.read_text(encoding="utf-8")
