"""Optimisation loop that crafts a poison stream against a fixed objective.

DEFENSIVE RESEARCH. See DISCLOSURE.md.

Blocked on checkpoints + baseline reproduction, like the rest of the model-touching
Phase 1 code. The control flow is specified here so the shape is reviewable now.

Threat-model realism is a constraint, not an afterthought
--------------------------------------------------------
The optimiser searches only within whatever `StreamStrategy` allows -- SELECT
reorders real sentences, PARAPHRASE stays in a similarity ball, SOFT relaxes into
embedding space. The fluency term from `AttackSpec.fluency_weight` is inside the
objective, not a post-hoc filter, because filtering after optimisation just throws
away compute and biases toward whatever slipped through. An attack that needs the
fluency constraint switched off to work is reported as such -- it establishes a
weaker threat model, and hiding that would make the whole defense look better than
the evidence supports.
"""

from __future__ import annotations

from dataclasses import dataclass

from trustgate.attack.objectives import AttackSpec
from trustgate.attack.stream import CraftedStream, StreamStrategy


@dataclass
class CraftConfig:
    strategy: StreamStrategy
    max_iters: int
    reference_model_path: str
    """Independent model for the fluency term. NOT the victim -- scoring a
    stream's fluency with the model under attack is circular."""
    early_stop_patience: int = 50


def craft_stream(
    victim_model,
    spec: AttackSpec,
    config: CraftConfig,
    corpus,
    seed: int,
) -> CraftedStream:
    """Optimise a poison stream against `spec` within the strategy's constraints.

    Loop: propose stream (constrained) -> run through victim inner loop -> score
    objective + fluency -> keep/perturb. SELECT and PARAPHRASE are discrete
    (search / hill-climbing); SOFT is gradient-based in embedding space with a
    projection back to tokens each step.
    """
    raise NotImplementedError(
        "Blocked on checkpoints + experiments/000-repro-baseline passing."
    )
