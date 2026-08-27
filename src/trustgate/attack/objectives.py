"""Attack objectives for the Phase 1 spike.

DEFENSIVE RESEARCH. Run only against models we control. See DISCLOSURE.md.

Dossier section 1 names two distinct harms from a poisoned fast-weight stream.
They are kept separately switchable because they may not both be achievable, and
conflating them would let a success on one hide a failure on the other:

  DEGRADE  -- steer fast weights so later *benign* inputs are handled worse.
              Untargeted. Success = clean-task loss rises after the poison
              stream, relative to a length-matched benign control.

  TRIGGER  -- implant a latent association so a specific trigger produces
              attacker-chosen behaviour, while benign performance is untouched.
              Targeted, and the more alarming of the two: it is silent under
              any monitor that only watches aggregate quality.

Neither objective may touch the base (slow) weights. The whole premise of the
threat model is that this happens at inference, through the context stream only.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass

import jax.numpy as jnp

from trustgate.eval.lm_loss import masked_mean_ce
from trustgate.tokens import BOS_TOKEN_ID


class Objective(enum.Enum):
    DEGRADE = "degrade"
    TRIGGER = "trigger"


@dataclass(frozen=True)
class AttackSpec:
    """Configuration for one attack run."""

    objective: Objective

    stream_tokens: int
    """Length of the crafted stream. The threat model is a *slow* attack, so
    this is expected to be large relative to a single window."""

    target_tokens: jnp.ndarray | None = None
    """TRIGGER only: the attacker-desired continuation."""

    trigger_tokens: jnp.ndarray | None = None
    """TRIGGER only: the phrase that should activate the implanted behaviour."""

    fluency_weight: float = 1.0
    """Weight on the fluency constraint. Zero produces token soup that would
    never survive a human or perplexity filter, and proves nothing about the
    threat model of dossier section 1 ("benign-looking"). Do not set to zero
    except as a diagnostic upper bound on attack strength."""

    def __post_init__(self):
        if self.objective is Objective.TRIGGER:
            if self.trigger_tokens is None or self.target_tokens is None:
                raise ValueError("TRIGGER objective requires trigger_tokens and target_tokens")


def degrade_loss(
    clean_logits: jnp.ndarray,
    clean_targets: jnp.ndarray,
    loss_masks: jnp.ndarray | None = None,
    *,
    bos_token_id: int = BOS_TOKEN_ID,
) -> jnp.ndarray:
    """Attacker maximises benign loss. We return its negation to minimise.

    Measured on held-out benign data the victim will see *after* the poison
    stream -- not on the stream itself. Poisoning the stream's own loss is
    trivial and meaningless.

    The quantity maximised is the victim's own reported loss, transcribed in
    `trustgate.eval.lm_loss.masked_mean_ce` from vendor `loss.py:6-29`. Using
    anything else -- an unmasked CE, a per-token sum, a base-2 log -- would
    optimise against a number nobody reports, and the attack would be tuned for
    a target the pre-registered metric does not measure.

    Returns a value that gets **more negative** as the victim gets worse, so
    every caller minimises. `trigger_loss` follows the same convention.
    """
    return -masked_mean_ce(
        clean_logits, clean_targets, loss_masks, bos_token_id=bos_token_id
    )


def trigger_loss(
    triggered_logits: jnp.ndarray,
    target_tokens: jnp.ndarray,
    clean_logits: jnp.ndarray,
    clean_targets: jnp.ndarray,
    stealth_weight: float = 1.0,
    *,
    bos_token_id: int = BOS_TOKEN_ID,
) -> jnp.ndarray:
    """Attacker wants target behaviour on the trigger, normal behaviour elsewhere.

    The `stealth_weight` term is what makes this hard to catch: it explicitly
    penalises benign degradation, so aggregate-quality monitoring stays flat.

    Both terms are ordinary masked CE, so minimising this drives the target
    continuation's likelihood up *and* holds benign loss down. At
    `stealth_weight=0` the stealth constraint is off and the objective collapses
    toward DEGRADE-with-a-target -- a diagnostic, not a result to report.
    """
    trigger_term = masked_mean_ce(
        triggered_logits, target_tokens, bos_token_id=bos_token_id
    )
    benign_term = masked_mean_ce(
        clean_logits, clean_targets, bos_token_id=bos_token_id
    )
    return trigger_term + stealth_weight * benign_term
