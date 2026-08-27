"""Attack objectives.

DEFENSIVE RESEARCH. Pure functions on toy logits -- no model, no GPU, no
checkpoint. Fixing their behaviour before any attack result exists is the point:
the pre-registration commits to thresholds expressed over these quantities, so
they must not become tunable after the fact.
"""

import numpy as np
import pytest

from trustgate.attack.objectives import (
    AttackSpec,
    Objective,
    degrade_loss,
    trigger_loss,
)
from trustgate.eval.lm_loss import masked_mean_ce

VOCAB = 8


def _confident_logits(targets, strength=6.0):
    """Logits that put most mass on `targets` -- a model doing well."""
    logits = np.zeros((len(targets), VOCAB))
    logits[np.arange(len(targets)), targets] = strength
    return logits


def test_degrade_loss_is_the_negated_benign_ce():
    targets = np.array([1, 2, 3, 4])
    logits = _confident_logits(targets)
    assert float(degrade_loss(logits, targets)) == pytest.approx(
        -float(masked_mean_ce(logits, targets)), rel=1e-6
    )


def test_degrade_loss_falls_as_the_victim_gets_worse():
    # The attacker minimises, so a more-corrupted victim must give a *lower*
    # value. Getting this sign backwards would optimise the attack into a
    # defense and the spike would return a confident, wrong STOP.
    targets = np.array([1, 2, 3, 4])
    healthy = float(degrade_loss(_confident_logits(targets, strength=6.0), targets))
    damaged = float(degrade_loss(_confident_logits(targets, strength=0.5), targets))
    assert damaged < healthy


def test_degrade_loss_honours_an_explicit_mask():
    targets = np.array([1, 2, 3, 4])
    logits = _confident_logits(targets)
    masked = float(degrade_loss(logits, targets, np.array([1.0, 1.0, 0.0, 0.0])))
    assert masked == pytest.approx(
        -float(masked_mean_ce(logits, targets, np.array([1.0, 1.0, 0.0, 0.0]))), rel=1e-6
    )


def test_trigger_loss_rewards_target_likelihood():
    target = np.array([2, 5])
    clean_targets = np.array([1, 3])
    clean_logits = _confident_logits(clean_targets)

    hits = float(trigger_loss(_confident_logits(target), target, clean_logits, clean_targets))
    misses = float(trigger_loss(np.zeros((2, VOCAB)), target, clean_logits, clean_targets))

    assert hits < misses


def test_trigger_loss_stealth_weight_penalises_benign_drift():
    # This is what makes TRIGGER the more alarming objective: the attacker is
    # explicitly paying to keep aggregate quality flat, so a monitor watching
    # only benign loss sees nothing.
    target = np.array([2, 5])
    clean_targets = np.array([1, 3])
    triggered = _confident_logits(target)

    healthy = _confident_logits(clean_targets, strength=6.0)
    degraded = _confident_logits(clean_targets, strength=0.5)

    with_stealth_ok = float(trigger_loss(triggered, target, healthy, clean_targets, 1.0))
    with_stealth_bad = float(trigger_loss(triggered, target, degraded, clean_targets, 1.0))

    assert with_stealth_bad > with_stealth_ok


def test_trigger_loss_without_stealth_ignores_benign_degradation():
    target = np.array([2, 5])
    clean_targets = np.array([1, 3])
    triggered = _confident_logits(target)

    healthy = float(
        trigger_loss(triggered, target, _confident_logits(clean_targets, 6.0), clean_targets, 0.0)
    )
    degraded = float(
        trigger_loss(triggered, target, _confident_logits(clean_targets, 0.5), clean_targets, 0.0)
    )

    assert healthy == pytest.approx(degraded, rel=1e-6)


def test_trigger_spec_requires_trigger_and_target_tokens():
    with pytest.raises(ValueError, match="TRIGGER objective requires"):
        AttackSpec(objective=Objective.TRIGGER, stream_tokens=1024)


def test_degrade_spec_needs_no_trigger_tokens():
    spec = AttackSpec(objective=Objective.DEGRADE, stream_tokens=1024)
    assert spec.fluency_weight == 1.0  # never default to the token-soup setting
