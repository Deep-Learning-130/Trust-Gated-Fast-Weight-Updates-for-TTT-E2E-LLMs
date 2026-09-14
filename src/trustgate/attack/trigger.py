"""TRIGGER objective: implant a latent association.

DEFENSIVE RESEARCH. See DISCLOSURE.md.

The TRIGGER objective differs from DEGRADE: instead of broadly corrupting fast
weights, it implants a specific trigger-target association. When the trigger
appears in input, the model produces attacker-chosen behavior; otherwise, it
behaves normally. This is stealthier than DEGRADE (aggregate quality stays flat)
but harder to demonstrate.

Blocked on GPU + model access (T1.10), but the orchestration is specified here
so the shape is reviewable and testable on CPU with toy data.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from trustgate.attack.objectives import AttackSpec, Objective
from trustgate.attack.stream import CraftedStream, StreamStrategy
from trustgate.eval.metrics import attack_success_rate


@dataclass
class TriggerSpec:
    """Configuration for one TRIGGER attack run.

    The trigger is a specific token sequence that, when presented to the model
    after poisoning, should elicit the target output. The trigger must be
    present in the poison stream (so the model learns the association) and
    absent from the control stream (so we can measure the causal effect).
    """

    trigger_tokens: np.ndarray
    """The phrase that should activate the implanted behavior. Must be present
    in the poison stream at least once."""

    target_tokens: np.ndarray
    """The attacker-desired continuation when the trigger appears."""

    trigger_count: int = 1
    """How many times the trigger appears in the poison stream. More repetitions
    strengthen the association but increase the risk of detection."""


def build_trigger_stream(
    corpus,
    length_tokens: int,
    seed: int,
    trigger_spec: TriggerSpec,
    *,
    span_tokens: int = 64,
    mini_batch_size: int = 1024,
    offset_range: tuple[int, int] | None = None,
) -> CraftedStream:
    """Build a poison stream with the trigger implanted.

    T2.8 implementation. Takes a base SELECT stream and injects the trigger
    at a deterministic position (derived from seed). The trigger is concatenated
    with the target tokens, so the model sees (trigger, target) pairs during
    adaptation.

    Args:
        corpus: token-ID stream to draw spans from
        length_tokens: target stream length
        seed: for reproducible span sampling and trigger placement
        trigger_spec: trigger tokens, target tokens, repetition count
        span_tokens: span size (default 64)
        mini_batch_size: vendor's inner-loop chunk size (default 1024)
        offset_range: restrict sampling to this corpus region
    """
    raise NotImplementedError(
        "TRIGGER stream building blocked on T1.10 (GPU session). "
        "Interface is ready; implementation requires model access to verify "
        "trigger placement and target concatenation."
    )


def evaluate_trigger(
    victim_model,
    poison_stream: CraftedStream,
    trigger_spec: TriggerSpec,
    eval_tokens: np.ndarray,
) -> float:
    """Evaluate trigger success rate after adapting on the poison stream.

    Presents the trigger in the eval context and checks if the model produces
    the target tokens. Returns the fraction of trigger presentations that yield
    the exact target (strict match).

    Args:
        victim_model: the model with adapted fast weights
        poison_stream: the stream the model adapted on
        trigger_spec: trigger and target tokens
        eval_tokens: held-out eval tokens (must not contain the trigger)

    Returns:
        attack_success_rate: fraction of trigger presentations yielding target
    """
    raise NotImplementedError(
        "TRIGGER evaluation blocked on T1.10 (GPU session). "
        "Interface is ready; implementation requires forward passes with the "
        "victim model."
    )


def run_trigger_objective(
    corpus,
    length_tokens: int,
    seeds: list[int],
    trigger_spec: TriggerSpec,
    *,
    span_tokens: int = 64,
    mini_batch_size: int = 1024,
    offset_range: tuple[int, int] | None = None,
) -> dict:
    """Execute the full TRIGGER attack across seeds.

    For each seed: build a poison stream with the trigger implanted, adapt the
    model, evaluate trigger success rate. Report the mean and per-seed rates.

    This is the secondary objective (reported, not gating). DEGRADE alone
    suffices for PROCEED; TRIGGER strengthens the case but is not required.

    Args:
        corpus: token-ID stream
        length_tokens: stream length
        seeds: list of seeds (typically [0, 1, 2, 3, 4])
        trigger_spec: trigger and target configuration
        span_tokens: span size
        mini_batch_size: vendor's inner-loop chunk size
        offset_range: restrict sampling to this corpus region

    Returns:
        dict with keys:
            - 'mean_success_rate': float
            - 'per_seed_rates': list[float]
            - 'trigger_spec': TriggerSpec
            - 'seeds': list[int]
    """
    if not seeds:
        raise ValueError("no seeds: the pre-registration requires 5 per condition")

    raise NotImplementedError(
        "TRIGGER objective run blocked on T1.10 (GPU session). "
        "Orchestration is specified; implementation requires model access."
    )
