"""Crafted-stream construction for the Phase 1 attack spike.

DEFENSIVE RESEARCH. See DISCLOSURE.md.

The hard constraint, and the reason this is not just GCG
--------------------------------------------------------
Dossier section 1 specifies "not a single adversarial token, but a slow,
benign-looking sequence". Unconstrained token optimisation produces fluent-adjacent
garbage that any perplexity filter, and any human skimming the context, would
catch. An attack that only works with visibly adversarial input does not
establish the threat model and would not justify the defense.

So the stream is constrained to stay in-distribution. Three strategies, in
increasing order of realism and difficulty:

  SELECT    -- choose and order sentences from a benign corpus. Never synthesises
               text, so fluency is exact by construction. Weakest attack, but any
               positive result here is immediately credible.
  PARAPHRASE -- perturb benign text within a semantic-similarity ball.
  SOFT      -- optimise in embedding space, project back to tokens. Strongest,
               least realistic; use as an upper bound on what is achievable, not
               as the headline result.

Report SELECT separately. If only SOFT works, say so plainly -- that is a much
weaker threat model and it changes whether the defense is worth building.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass


class StreamStrategy(enum.Enum):
    SELECT = "select"
    PARAPHRASE = "paraphrase"
    SOFT = "soft"


@dataclass
class CraftedStream:
    """A poison stream plus the evidence that it is benign-looking."""

    tokens: "jnp.ndarray"  # noqa: F821
    strategy: StreamStrategy

    perplexity: float
    """Under an independent reference model -- never the victim. Scoring
    fluency with the model being attacked is circular."""

    control_perplexity: float
    """Same measure on the length-matched benign control. The stream is only
    'benign-looking' if these are close; report the ratio, not just the value."""

    @property
    def fluency_ratio(self) -> float:
        return self.perplexity / max(self.control_perplexity, 1e-6)


def build_select_stream(corpus, length_tokens: int, seed: int) -> CraftedStream:
    """SELECT strategy: order existing benign sentences adversarially.

    Fluency is exact -- every sentence is real text. The only attacker freedom
    is choice and ordering, which is a genuinely weak lever and therefore a
    genuinely strong result if it works.
    """
    raise NotImplementedError("Phase 1.")


def build_benign_control(corpus, length_tokens: int, seed: int) -> CraftedStream:
    """Length-matched benign control stream.

    Required by the pre-registration. Fast weights drift on *any* stream; without
    this control, benign adaptation is indistinguishable from poisoning and the
    attack result is uninterpretable.
    """
    raise NotImplementedError("Phase 1.")
