#!/usr/bin/env python
"""Primary acceptance check for the reference model: match a reference implementation.

Why this and not the published perplexity
-----------------------------------------
T3.2 asks for "a published perplexity on a fixed text as the acceptance check".
That was tried first (`acceptance_fluency_reference.py`) and it turns out **not to
be decisive**, for a reason worth recording rather than working around.

GPT-2's reported WikiText-103 perplexity (37.50, Radford et al. 2019 Table 3) is
a *per-word* figure, and the word-count denominator is not specified precisely
enough to reproduce. On the WikiText-103 test split our implementation gives a
per-BPE-token perplexity of 22.66, which converts to:

    62.29 per word  using 204,464 whitespace words after detokenization
    31.18 per word  using WikiText-103's canonical 245,569-token count

The published 37.50 sits **between** those two conventions. So the check cannot
separate "the implementation is wrong" from "the denominator convention differs",
which makes it useless as a pass/fail gate on the thing we actually care about.

This script answers that question directly instead: run the same input through
our jax implementation and through HuggingFace's `GPT2LMHeadModel`, and compare
logits elementwise. Equivalence to float32 noise means the forward pass is
correct -- no interpretation required.

`torch` is needed **only here**. It is not a runtime dependency, nothing under
`src/` imports it, and the fluency path does not need it. Install with:

    uv pip install --python .venv "torch>=2.2" --index-url https://download.pytorch.org/whl/cpu

Usage:
    python scripts/verify_gpt2_equivalence.py
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from trustgate.eval.fluency import reference_cache_root  # noqa: E402
from trustgate.eval.gpt2_reference import GPT2Reference  # noqa: E402

#: float32 matmul accumulation order differs between backends, so exact equality
#: is not achievable. This band is ~4 orders of magnitude tighter than any real
#: implementation bug: a transposed Conv1D weight, exact GELU instead of
#: gelu_new, or a wrong LayerNorm epsilon all shift logits by O(0.1) or more.
MAX_ABS_LOGIT_DIFF = 1e-3
MAX_ABS_NLL_DIFF = 1e-3


def log_softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - logits.max(axis=-1, keepdims=True)
    return shifted - np.log(np.exp(shifted).sum(axis=-1, keepdims=True))


def main() -> int:
    warnings.filterwarnings("ignore")
    model_dir = reference_cache_root() / "gpt2"

    print("=" * 68)
    print("GPT-2 forward-pass equivalence: ours (jax) vs HuggingFace (torch)")
    print("=" * 68)
    print(f"weights: {model_dir}")

    ours = GPT2Reference(model_dir)

    try:
        import torch
        from transformers import GPT2LMHeadModel
    except ImportError as exc:
        print(f"\ntorch/transformers unavailable ({exc}).")
        print("This is a verification-only dependency; see the module docstring.")
        return 1

    theirs = GPT2LMHeadModel.from_pretrained(str(model_dir), local_files_only=True)
    theirs.eval()
    print(f"config: {ours.config}\n")

    rng = np.random.default_rng(0)
    worst_logit_diff = 0.0
    worst_nll_diff = 0.0
    failures = []

    for length in (1, 7, 64, 256, 1024):
        ids = rng.integers(0, ours.config.vocab_size, length).astype(np.int32)

        mine = np.asarray(ours.logits(ids), dtype=np.float64)
        with torch.no_grad():
            reference = (
                theirs(torch.tensor(ids[None, :], dtype=torch.long))
                .logits[0]
                .numpy()
                .astype(np.float64)
            )

        diff = np.abs(mine - reference)
        argmax_agreement = float((mine.argmax(-1) == reference.argmax(-1)).mean())
        worst_logit_diff = max(worst_logit_diff, float(diff.max()))

        nll_diff = 0.0
        if length > 1:
            ours_nll = -log_softmax(mine[:-1])[np.arange(length - 1), ids[1:]]
            ref_nll = -log_softmax(reference[:-1])[np.arange(length - 1), ids[1:]]
            nll_diff = float(np.abs(ours_nll - ref_nll).max())
            worst_nll_diff = max(worst_nll_diff, nll_diff)

        ok = diff.max() <= MAX_ABS_LOGIT_DIFF and argmax_agreement == 1.0
        if not ok:
            failures.append(length)

        print(
            f"  seq={length:5d}  max|logit diff|={diff.max():.3e}  "
            f"max|NLL diff|={nll_diff:.3e}  argmax agree={argmax_agreement:.4f}  "
            f"{'ok' if ok else 'MISMATCH'}"
        )

    print()
    print("-" * 68)
    print(f"  worst |logit diff| : {worst_logit_diff:.3e}  (band {MAX_ABS_LOGIT_DIFF:.0e})")
    print(f"  worst |NLL diff|   : {worst_nll_diff:.3e}  (band {MAX_ABS_NLL_DIFF:.0e})")
    print("-" * 68)

    passed = (
        not failures
        and worst_logit_diff <= MAX_ABS_LOGIT_DIFF
        and worst_nll_diff <= MAX_ABS_NLL_DIFF
    )
    print(f"\n  VERDICT: {'PASS' if passed else 'FAIL'}")
    if passed:
        print("  The implemented forward pass is equivalent to the reference")
        print("  implementation to float32 noise. Differences at this magnitude are")
        print("  matmul accumulation order, not semantics.")
    else:
        print(f"  Mismatch at sequence lengths {failures}. Do NOT use this model to")
        print("  score fluency: a wrong forward pass produces plausible perplexities")
        print("  that are silently wrong, and the fluency ratio would be meaningless.")

    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
