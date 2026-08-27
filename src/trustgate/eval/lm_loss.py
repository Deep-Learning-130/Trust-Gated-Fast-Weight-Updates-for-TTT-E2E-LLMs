"""The victim's language-model loss, transcribed from the vendor.

Why transcribed rather than imported
------------------------------------
The vendor is not importable in this environment: `einops`, `zarr`, `grain`,
`orbax`, `wandb` and `hydra` are all absent from `.venv`, which is deliberate --
the CPU test suite is the fast feedback loop and must run anywhere, including CI
without the unlicensed submodule checked out. So the formula is transcribed
here, the source line is cited, and `tests/test_lm_loss.py` checks it against a
reference computed from first principles rather than against the vendor call.

The quantity
------------
`cross_entropy_loss_and_accuracy` (vendor `ttt/model/loss.py:6-29`):

    valid_text_length = jnp.maximum(jnp.sum(valid, axis=-1), 1e-10)
    logits = logits.astype(jnp.float32)
    log_prob = jax.nn.log_softmax(logits, axis=-1)
    token_log_prob = take_along_axis(log_prob, targets[..., None], -1).squeeze(-1)
    token_log_prob = jnp.where(valid > 0.0, token_log_prob, 0.0)
    loss = jnp.mean(jnp.sum(-token_log_prob, axis=-1) / valid_text_length)

Mean cross-entropy in **nats**, natural log, no label smoothing, no division by
`log(2)`. The denominator is the count of *valid* tokens, not the sequence
length. Getting the attacker to optimise this exact quantity matters: it is the
same number the victim reports as `train_holdout/loss` and the same one the
paper plots as "Loss (log perplexity)", so no `exp()` conversion enters anywhere
(ADR-004, closed 2026-08-08).

The mask
--------
There is no shifting here. The vendor's dataloader does it
(`lm_dataset.py:51-53`):

    input_ids=tokens[:-1], target_tokens=tokens[1:],
    loss_masks=(tokens[1:] != bos_token_id)

so logits at position *i* score `targets[i]`, and **BOS positions are masked out
of the loss**. That mask is not cosmetic. Because the denominator is the valid
count of *that chunk*, a chunk holding fewer valid tokens up-weights every
surviving token's gradient -- so BOS is a per-inner-step gradient-scale
confound, not merely a sequence-length one. This is why crafted streams are
built from BOS-free spans (`attack/corpus.py`).
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from trustgate.tokens import BOS_TOKEN_ID


def valid_mask(targets: jnp.ndarray, bos_token_id: int = BOS_TOKEN_ID) -> jnp.ndarray:
    """The vendor's loss mask: every position whose target is not BOS.

    Mirrors `lm_dataset.py:53`. Float32 rather than bool because the vendor
    casts it (`loss.py:13`) and sums it as a length.
    """
    return (targets != bos_token_id).astype(jnp.float32)


def count_valid(targets: jnp.ndarray, bos_token_id: int = BOS_TOKEN_ID) -> int:
    """Number of loss-bearing positions. The matching quantity, not raw length.

    Two streams with equal token counts but different BOS counts have different
    effective lengths and different per-chunk gradient scales, so this is what
    `RunCondition` has to match on.
    """
    return int(jnp.sum(valid_mask(targets, bos_token_id)))


def masked_mean_ce(
    logits: jnp.ndarray,
    targets: jnp.ndarray,
    loss_masks: jnp.ndarray | None = None,
    *,
    bos_token_id: int = BOS_TOKEN_ID,
) -> jnp.ndarray:
    """Mean cross-entropy in nats over valid positions.

    Transcribes `ttt/model/loss.py:6-29` exactly. When `loss_masks` is omitted it
    is derived as `targets != bos_token_id`, which is what the vendor's
    dataloader would have supplied (`lm_dataset.py:53`).

    Args:
        logits: `[..., seq, vocab]`. Cast to float32 internally, as the vendor
            does, because bf16 log-softmax over a 128k vocab is not accurate
            enough to threshold against.
        targets: `[..., seq]` token ids -- already shifted by the caller.
        loss_masks: `[..., seq]`, nonzero where the position counts.

    Returns:
        float32 scalar.
    """
    valid = valid_mask(targets, bos_token_id) if loss_masks is None else loss_masks.astype(jnp.float32)

    valid_text_length = jnp.maximum(jnp.sum(valid, axis=-1), 1e-10)
    logits = logits.astype(jnp.float32)

    log_prob = jax.nn.log_softmax(logits, axis=-1)
    token_log_prob = jnp.squeeze(
        jnp.take_along_axis(log_prob, jnp.expand_dims(targets, -1), axis=-1),
        -1,
    )
    token_log_prob = jnp.where(valid > 0.0, token_log_prob, jnp.array(0.0))

    return jnp.mean(jnp.sum(-token_log_prob, axis=-1) / valid_text_length)
