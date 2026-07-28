"""MedBN-analogue baseline -- robust aggregation of fast-weight updates.

Status: NOT IMPLEMENTED. Phase 2.

Read this before assuming you can download a baseline
-----------------------------------------------------
MedBN (arXiv 2403.19326, CVPR 2024) **cannot be run here.** It is a vision
method. Its mechanism is replacing the *mean* in batch-norm statistics with the
*median*, evaluated on CIFAR10-C / CIFAR100-C / ImageNet-C. TTT-E2E is a
transformer with sliding-window attention and has no batch-norm layers, so there
is no statistic for MedBN to robustify. The official code
(github.com/ml-postech/MedBN-robust-test-time-adaptation) is not portable to
this setting.

So the baseline must be **built**, by porting MedBN's *principle* rather than its
code: the insight that a mean is corruptible by a single malicious sample while
a median is not. Transposed onto fast weights, the natural analogue is robust
aggregation of the per-example gradients that form the inner-loop update --
coordinate-wise median or trimmed mean in place of the mean.

This cuts both ways and both matter for filing:

  - It weakens the section 7 "reads onto MedBN" risk. Our gate is behavioural
    (anchor consistency over probes); this baseline is distributional (robust
    statistics over gradients). Genuinely different mechanisms.
  - It obliges us to build a *strong* version of it. A baseline we constructed
    ourselves and then beat is worth very little, to reviewers or to an examiner
    assessing non-obviousness. Tune it as hard as we tune our own gate, and say
    in writing that we did.

See `docs/patent/prior-art/medbn-diff.md` -- the line-level diff is a hard
filing gate per dossier section 7.
"""

from __future__ import annotations

import jax.numpy as jnp

from trustgate.types import Updates


def robust_aggregate(
    per_example_grads,
    *,
    method: str = "coordinate_median",
    trim_fraction: float = 0.1,
) -> Updates:
    """Aggregate per-example gradients robustly instead of by mean.

    Args:
        per_example_grads: pytree with a leading per-example axis.
        method: "coordinate_median" | "trimmed_mean" | "geometric_median".
        trim_fraction: fraction trimmed from each tail for "trimmed_mean".

    Returns:
        Aggregated update, same structure minus the example axis.
    """
    raise NotImplementedError("Phase 2 baseline.")
