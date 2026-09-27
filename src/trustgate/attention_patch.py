"""Exact attention without cuDNN, for GPUs the vendor's fused kernel does not support.

Why this exists
---------------
Every prefix block and the suffix sliding-window attention call
`jax.nn.dot_product_attention(..., implementation="cudnn")`
(vendor `ttt/model/attention.py:183, 214, 244-251, 312`). That kernel has no
engine for this graph on a Turing T4, in bf16 or fp16, with cuDNN 9.8 or 9.26
("No valid engine configs", Kaggle 2026-09-27). The owner chose to run C1 on
Kaggle anyway (2026-09-27), which needs the call served by something else.

What this computes
------------------
The same thing JAX's own XLA attention computes (`_dot_product_attention_core`,
jax 0.5.3 `jax/_src/nn/functions.py`), step for step: q.k in the input dtype
accumulated in fp32, times `scale`, masked with -0.7 * finfo.max, softmax in
fp32, probabilities cast to the key dtype, then times v. The only difference is
that queries are processed `QUERY_BLOCK` at a time. A softmax row is one query
against all its keys, so blocking over queries is exact -- no online-softmax
rescaling, nothing approximated -- and the [T, S] score matrix never exists
whole. That matters: the plain XLA path at T = 8192 holds 12 x 8192 x 8192 fp32
scores (3.2 GB) per layer per sequence, and C1 runs 4 sequences at once.

What it is NOT: the vendor's kernel. cuDNN fuses the whole thing and rounds
differently, so results are close, not bit-identical. The gap is measured on
real hardware in `experiments/003-smoke-125m` (the laptop can run both), not
assumed.

Installed by replacing the `jax.nn.dot_product_attention` attribute, which the
vendor looks up at call time. No vendor file is edited (ADR-002).
"""

from __future__ import annotations

import math
from functools import partial
from typing import Callable

import jax
import jax.numpy as jnp

#: Queries per block. [4 seqs, 12 heads, 512, 8192] fp32 scores = 805 MB at C1's
#: batch 4, against 6.4 GB for the whole matrix.
QUERY_BLOCK = 512

_original: Callable | None = None


def _block_size(t: int, target: int = QUERY_BLOCK) -> int:
    for s in range(min(target, t), 0, -1):
        if t % s == 0:
            return s
    return 1


def dot_precision(dtype, device=None):
    """The dot algorithm JAX's XLA attention asks for, where the device has it.

    JAX requests BF16_BF16_F32 (F16_F16_F32) and wraps the request in a bare
    `except` to fall back. On a Turing T4 the refusal surfaces at execution
    ("UNIMPLEMENTED: Unsupported algorithm ... ALG_DOT_BF16_BF16_F32", Kaggle
    2026-09-27), long after that `except` has passed, so the fallback never runs.
    Below Ampere there is no bf16 hardware to ask for, so ask for nothing: XLA
    then upcasts the bf16 operands to f32, where a bf16 x bf16 product is exact
    and the sum accumulates in f32 -- the arithmetic the preset names anyway.
    """
    if dtype == jnp.bfloat16:
        preset = jax.lax.DotAlgorithmPreset.BF16_BF16_F32
    elif dtype == jnp.float16:
        preset = jax.lax.DotAlgorithmPreset.F16_F16_F32
    else:
        return None
    device = device if device is not None else jax.devices()[0]
    if device.platform == "gpu":
        major = int(str(getattr(device, "compute_capability", "0")).split(".")[0] or 0)
        if major < 8:
            return None
    return preset


def _to_4d(x):
    return x if x is None or x.ndim == 4 else x.reshape((1,) * (4 - x.ndim) + x.shape)


def blocked_attention(
    query,
    key,
    value,
    *,
    mask=None,
    is_causal: bool = False,
    local_window_size=None,
    scale: float | None = None,
    query_block: int = QUERY_BLOCK,
):
    """`jax.nn.dot_product_attention(..., implementation="xla")`, a block of
    queries at a time. Inputs are `[T, N, H]` or `[B, T, N, H]`, as for JAX's."""
    squeeze = query.ndim == 3
    q, k, v = (x[None] if squeeze else x for x in (query, key, value))
    B, T, N, H = q.shape
    S, K = k.shape[1], k.shape[2]
    if K != N:  # grouped-query attention, as JAX broadcasts it
        k = jnp.repeat(k, N // K, axis=2)
        v = jnp.repeat(v, N // K, axis=2)
    if scale is None:
        scale = 1.0 / math.sqrt(H)
    if isinstance(local_window_size, int):
        local_window_size = (local_window_size, local_window_size)
    m4 = _to_4d(mask)

    logits_dtype = jnp.promote_types(q.dtype, jnp.float32)
    precision = dot_precision(q.dtype)
    large_negative = jnp.asarray(-0.7 * jnp.finfo(logits_dtype).max, dtype=logits_dtype)

    tb = _block_size(T, query_block)
    key_pos = jnp.arange(S)

    @partial(jax.checkpoint, prevent_cse=False)
    def one_block(i):
        start = i * tb
        qb = jax.lax.dynamic_slice_in_dim(q, start, tb, axis=1)
        try:
            logits = jnp.einsum("BTNH,BSNH->BNTS", qb, k, precision=precision,
                                preferred_element_type=logits_dtype)
        except Exception:  # noqa: BLE001 - JAX's own fallback, same reason
            logits = jnp.einsum("BTNH,BSNH->BNTS", qb, k, preferred_element_type=logits_dtype)
        logits = logits * jnp.array(scale, dtype=logits.dtype)

        keep = jnp.ones((1, 1, tb, S), dtype=jnp.bool_)
        query_pos = start + jnp.arange(tb)
        if is_causal:
            keep = keep & (query_pos[:, None] >= key_pos[None, :])[None, None]
        if local_window_size is not None:
            left, right = local_window_size
            keep = keep & (
                (query_pos[:, None] <= key_pos[None, :] + left)
                & (query_pos[:, None] >= key_pos[None, :] - right)
            )[None, None]
        if m4 is not None:
            mb = m4 if m4.shape[2] == 1 else jax.lax.dynamic_slice_in_dim(m4, start, tb, axis=2)
            keep = keep & mb
        if mask is not None or is_causal or local_window_size is not None:
            logits = jnp.where(keep, logits, large_negative)

        probs = jax.nn.softmax(logits.astype(jnp.float32), axis=-1).astype(k.dtype)
        return jnp.einsum("BNTS,BSNH->BTNH", probs, v)

    blocks = jax.lax.map(one_block, jnp.arange(T // tb))  # [T//tb, B, tb, N, H]
    out = jnp.moveaxis(blocks, 0, 1).reshape(B, T, N, v.shape[-1])
    return out[0] if squeeze else out


def _patched(query, key, value, bias=None, mask=None, *, scale=None, is_causal=False,
             query_seq_lengths=None, key_value_seq_lengths=None,
             local_window_size=None, implementation=None):
    # Anything the vendor never passes goes to the original, untouched.
    if bias is not None or query_seq_lengths is not None or key_value_seq_lengths is not None:
        return _original(query, key, value, bias, mask, scale=scale, is_causal=is_causal,
                         query_seq_lengths=query_seq_lengths,
                         key_value_seq_lengths=key_value_seq_lengths,
                         local_window_size=local_window_size, implementation=implementation)
    return blocked_attention(query, key, value, mask=mask, is_causal=is_causal,
                             local_window_size=local_window_size, scale=scale)


def install_blocked_attention() -> None:
    """Serve every `jax.nn.dot_product_attention` call with `blocked_attention`,
    whatever `implementation` it asked for. Re-installing is a no-op."""
    global _original
    if _original is None:
        _original = jax.nn.dot_product_attention
    jax.nn.dot_product_attention = _patched


def uninstall_blocked_attention() -> None:
    global _original
    if _original is not None:
        jax.nn.dot_product_attention = _original
        _original = None


def is_installed() -> bool:
    return _original is not None
