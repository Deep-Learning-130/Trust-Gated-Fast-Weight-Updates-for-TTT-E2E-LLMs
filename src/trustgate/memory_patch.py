"""Compute the vendor's LM loss without holding a `[tokens, vocab]` logit array.

Why this exists
---------------
`MetaModel.lm_loss` (vendor `ttt/model/transformer.py:631-645`) builds the full
`[1024, 128256]` logits for a chunk, casts them to fp32, and runs `log_softmax`
over them twice -- once in `cross_entropy_loss_and_accuracy` (`loss.py:7-29`)
and once in `token_log_probs` (`loss.py:32-41`). Each copy is ~525 MB, the
inner step's `value_and_grad` holds more, and that term scales with the 128K
vocabulary rather than with the model. It is the ~1.5 GiB allocation that
`carry-is-non-trivial` could not get on the 8 GB laptop card (003 smoke,
2026-09-15), and no amount of shrinking the model would remove it.

What this changes, and what it does not
---------------------------------------
Every quantity the vendor computes is still computed, from the same hidden
states, the same tied embedding and the same dtypes. `log_softmax` normalises
over the vocabulary axis of each token independently, so computing it one
slice of tokens at a time is the same arithmetic, not an approximation. The
only thing that changes is how much of it is alive at once: `SLICE_TOKENS`
rows instead of all of them, recomputed in the backward pass
(`jax.checkpoint`) instead of stored.

Faithfully kept, because a result is only as good as its equivalence:

- `loss` is taken from **fp32** log-probs (`loss.py:16`), with the vendor's
  mask handling and its mean-of-(sum / valid_length) reduction.
- `token_nll_loss` is taken from log-probs in the **logits' own dtype** (bf16
  at the default config), because `token_log_probs` does not cast
  (`loss.py:32-41`). It is the per-token number the baseline reports, so it
  must not quietly become more precise than the vendor's.
- The logits are `hidden @ wte.weight.T` after the vendor's own
  `promote_dtype` to `compute_dtype` (`transformer.py:822-828`), or `lm_head`
  when embeddings are untied.

The only residual difference is GEMM rounding: a 128-row matmul may be tiled
differently from a 1024-row one. In fp32 on CPU the two agree to ~1e-6
(`tests/test_memory_patch.py`); on the GPU at bf16 the smoke run measures the
gap on the real model (`chunked-ce-equivalence`) rather than asserting it.

Installed the same way as `vendor_patch.install_gate` -- by replacing the class
attribute, because the inner step reaches the loss as `MetaModel.lm_loss`
(`transformer.py:617`), not through the instance. ADR-002 holds: no vendor file
is edited. Pinned to the vendor commit it was written against.
"""

from __future__ import annotations

from functools import partial
from typing import Callable

import jax
import jax.numpy as jnp

#: Vendor commit this patch was written against (same pin as vendor_patch).
PINNED_VENDOR_SHA = "a4fc4788ace38e29b5067916d4f4be33da894085"

#: Tokens per slice. 128 x 128256 x 4 B = 66 MB of fp32 log-probs alive at once,
#: against ~525 MB per copy for a whole 1024-token chunk.
SLICE_TOKENS = 128

_original_lm_loss: Callable | None = None


def slice_size(n_tokens: int, target: int = SLICE_TOKENS) -> int:
    """Largest divisor of `n_tokens` that is <= `target`.

    A divisor, so every slice is the same shape and `lax.map` traces once.
    1024-token chunks give 128.
    """
    for s in range(min(target, n_tokens), 0, -1):
        if n_tokens % s == 0:
            return s
    return 1


def sliced_token_log_probs(hidden, targets, logits_fn, slice_tokens: int = SLICE_TOKENS):
    """Target-token log-probs, computed a slice of tokens at a time.

    Args:
        hidden: `[..., H]` final hidden states.
        targets: `[...]` integer target ids, same leading shape as `hidden`.
        logits_fn: maps `[s, H]` hidden to `[s, V]` logits in the vendor's dtype.

    Returns:
        `(log_prob_fp32, log_prob_native)`, each shaped like `targets`: the
        target's log-prob from an fp32 `log_softmax` (what `loss.py:16-22`
        uses for the loss) and from one in the logits' own dtype (what
        `token_log_probs` uses).
    """
    lead = targets.shape
    n = targets.size
    s = slice_size(n, slice_tokens)
    h = hidden.reshape(n // s, s, hidden.shape[-1])
    t = targets.reshape(n // s, s)

    @partial(jax.checkpoint, prevent_cse=False)
    def one_slice(args):
        hs, ts = args
        logits = logits_fn(hs)
        index = jnp.expand_dims(ts, -1)
        lp32 = jnp.squeeze(
            jnp.take_along_axis(
                jax.nn.log_softmax(logits.astype(jnp.float32), axis=-1), index, axis=-1
            ),
            -1,
        )
        lp_native = jnp.squeeze(
            jnp.take_along_axis(jax.nn.log_softmax(logits, axis=-1), index, axis=-1),
            -1,
        )
        return lp32, lp_native

    lp32, lp_native = jax.lax.map(one_slice, (h, t))
    return lp32.reshape(lead), lp_native.reshape(lead)


def reduce_like_vendor(token_log_prob, targets, valid=None):
    """`cross_entropy_loss_and_accuracy`'s reduction (`loss.py:13-29`), given
    fp32 target log-probs instead of logits. Returns `(loss, loss_pure_ce)`."""
    if valid is None:
        valid = jnp.ones(targets.shape[:2])
    valid = valid.astype(jnp.float32)
    valid_text_length = jnp.maximum(jnp.sum(valid, axis=-1), 1e-10)
    token_log_prob = jnp.where(valid > 0.0, token_log_prob, jnp.array(0.0))
    token_wise_loss = -token_log_prob
    loss_pure_ce = jnp.mean(jnp.sum(token_wise_loss, axis=-1) / valid_text_length)
    loss = jnp.mean(jnp.sum(token_wise_loss, axis=-1) / valid_text_length)
    return loss, loss_pure_ce


def _logits_fn(language_model, hidden):
    """The vendor's logit head (`transformer.py:822-828`), returned as a
    per-slice function. The dtype promotion of the (large) tied kernel happens
    once here, outside the slice loop, exactly as the vendor does it once."""
    from ttt.utils.jax_utils import promote_dtype

    if language_model.config.tie_word_embeddings:
        kernel = language_model.model.wte.weight.T
        hidden, kernel = promote_dtype(hidden, kernel, dtype=language_model.compute_dtype)
        return hidden, (lambda hs: hs @ kernel)
    return hidden, language_model.lm_head


def chunked_lm_loss(self, seq, state, *, prefix_outputs=None):
    """Drop-in for `MetaModel.lm_loss` with the same signature and return value:
    `loss, (loss_pure_ce, token_nll_loss, new_state)`."""
    lm = self.language_model
    if prefix_outputs is None:
        outputs = lm.model(state, seq)  # CausalLM.__call__, transformer.py:845
    else:
        outputs = lm.model.suffix_call(prefix_outputs, state, seq)  # :814-818
    hidden, logits_fn = _logits_fn(lm, outputs.last_hidden_state)

    lp32, lp_native = sliced_token_log_probs(hidden, seq.target_tokens, logits_fn)
    loss, loss_pure_ce = reduce_like_vendor(lp32, seq.target_tokens, seq.loss_masks)
    token_nll_loss = -lp_native
    return loss, (loss_pure_ce, token_nll_loss, outputs.state)


def install_chunked_ce() -> None:
    """Replace `MetaModel.lm_loss`. Re-installing is a no-op, never a double wrap."""
    from ttt.model.transformer import MetaModel

    global _original_lm_loss
    if _original_lm_loss is None:
        _original_lm_loss = MetaModel.lm_loss
    MetaModel.lm_loss = chunked_lm_loss


def uninstall_chunked_ce() -> None:
    """Restore the vendor's method. Safe to call when not installed."""
    from ttt.model.transformer import MetaModel

    global _original_lm_loss
    if _original_lm_loss is not None:
        MetaModel.lm_loss = _original_lm_loss
        _original_lm_loss = None


def is_installed() -> bool:
    return _original_lm_loss is not None
