"""The victim's LM loss, transcribed from vendor `ttt/model/loss.py:6-29`.

The vendor cannot be imported here (no `einops`, `zarr`, `grain`, `orbax`,
`wandb`, `hydra`), so the equivalence claim is checked against a reference
computed from first principles in this file rather than against the vendor call.
That reference is written in float64 numpy, deliberately independently of the
implementation, so a shared mistake is unlikely.
"""

import numpy as np
import pytest

from trustgate.eval.lm_loss import count_valid, masked_mean_ce, valid_mask
from trustgate.tokens import BOS_TOKEN_ID


def reference_masked_mean_ce(logits, targets, mask):
    """Masked mean CE in nats, from first principles. No jax, no vendor."""
    logits = np.asarray(logits, dtype=np.float64)
    targets = np.asarray(targets)
    mask = np.asarray(mask, dtype=np.float64)

    shifted = logits - logits.max(axis=-1, keepdims=True)
    log_prob = shifted - np.log(np.exp(shifted).sum(axis=-1, keepdims=True))
    token_log_prob = np.take_along_axis(
        log_prob, targets[..., None], axis=-1
    ).squeeze(-1)

    nll = -token_log_prob * mask
    return nll.sum(axis=-1) / max(mask.sum(axis=-1), 1e-10)


def test_masked_mean_ce_matches_first_principles_reference():
    rng = np.random.default_rng(0)
    logits = rng.normal(0, 2.0, (6, 11))
    targets = np.array([3, 0, 7, 1, 10, 4])
    mask = np.ones(6)

    assert float(masked_mean_ce(logits, targets, mask)) == pytest.approx(
        reference_masked_mean_ce(logits, targets, mask), rel=1e-5
    )


def test_masked_mean_ce_is_in_nats_not_bits():
    # Uniform logits over V: every log-prob is -ln(V), so the loss is ln(V).
    # In bits it would be log2(10) = 3.32, which is the confusion the vendor's
    # "Loss (log perplexity)" axis label invites.
    vocab = 10
    logits = np.zeros((4, vocab))
    targets = np.array([1, 2, 3, 4])
    assert float(masked_mean_ce(logits, targets)) == pytest.approx(np.log(vocab), rel=1e-6)


#: Stand-in BOS so the tests can use a small vocab. The real id is 128000 and
#: allocating a 128k-wide logit array per test would be pure waste; the
#: default-argument path is covered by the tests that use the real constant.
FAKE_BOS = 3


def test_masked_mean_ce_ignores_bos_target_positions():
    logits = np.zeros((3, 8))
    targets = np.array([5, FAKE_BOS, 7])

    before = float(masked_mean_ce(logits, targets, bos_token_id=FAKE_BOS))

    # Wreck the row whose target is BOS. It must not move the loss at all.
    wrecked = logits.copy()
    wrecked[1, :] = np.linspace(-50, 50, wrecked.shape[-1])
    after = float(masked_mean_ce(wrecked, targets, bos_token_id=FAKE_BOS))

    assert after == pytest.approx(before, rel=1e-9)


def test_masked_mean_ce_denominator_is_valid_count_not_length():
    # Same per-token NLL in both cases; the second is padded with BOS targets.
    # If the denominator were the sequence length the padded case would report a
    # lower loss -- which is how a stream carrying more BOS would look
    # artificially better than its length-matched control.
    logits = np.zeros((2, 8))
    unpadded = float(masked_mean_ce(logits, np.array([4, 6]), bos_token_id=FAKE_BOS))

    padded = float(
        masked_mean_ce(
            np.zeros((4, 8)),
            np.array([4, 6, FAKE_BOS, FAKE_BOS]),
            bos_token_id=FAKE_BOS,
        )
    )

    assert padded == pytest.approx(unpadded, rel=1e-9)


def test_explicit_loss_masks_override_the_bos_derivation():
    vocab = 12
    logits = np.zeros((3, vocab))
    targets = np.array([1, 2, 3])

    # Masking two of three positions leaves the mean unchanged (uniform logits)
    # but the mask must actually be honoured rather than re-derived.
    assert float(masked_mean_ce(logits, targets, np.array([1.0, 0.0, 0.0]))) == pytest.approx(
        np.log(vocab), rel=1e-6
    )


def test_valid_mask_mirrors_the_dataloader_rule():
    targets = np.array([7, BOS_TOKEN_ID, 9, BOS_TOKEN_ID])
    assert list(np.asarray(valid_mask(targets))) == [1.0, 0.0, 1.0, 0.0]


def test_count_valid_excludes_bos():
    targets = np.array([7, BOS_TOKEN_ID, 9, BOS_TOKEN_ID, 11])
    assert count_valid(targets) == 3


def test_count_valid_of_an_all_bos_sequence_is_zero():
    assert count_valid(np.array([BOS_TOKEN_ID] * 4)) == 0
