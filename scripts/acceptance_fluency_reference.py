#!/usr/bin/env python
"""Acceptance check for the fluency reference model (T3.2).

The reference model's forward pass is **implemented** in this repo rather than
imported (`transformers` 5.x dropped Flax models; see
`trustgate/eval/gpt2_reference.py`). That is a deliberate trade -- control over
tokenization, windowing and masking, which a *ratio* depends on -- and it moves
the risk to "did we implement GPT-2 correctly?".

This script is the answer to that question: reproduce a **published** perplexity.

Target
------
Radford et al. 2019, *Language Models are Unsupervised Multitask Learners*,
Table 3: the 117M model scores **37.50** on WikiText-103.

Two things about that number are easy to get wrong:

1. **It is per *word*, not per BPE token.** GPT-2's reported WikiText perplexities
   divide the total negative log-likelihood by the count of whitespace words, not
   by the number of BPE tokens. The per-token number is necessarily lower, and
   comparing against it would look like a failed reproduction when it is a
   different quantity.
2. **It uses an invertible detokenizer.** WikiText ships pre-tokenized -- ` @-@ `
   for hyphens, spaces before punctuation, ` = = = ` for headings. The paper
   reverses that before scoring. Skipping the step inflates perplexity
   substantially, because the model is charged for artifacts no natural text has.

The detokenizer below is the standard WikiText one from the Transformer-XL /
lm-evaluation-harness lineage.

Note on the data: WikiText-2 and WikiText-103 share identical validation and test
splits, so the same 732KB file serves both.

Usage:
    python scripts/acceptance_fluency_reference.py
    STRIDE=512 MAX_DOCS=0 python scripts/acceptance_fluency_reference.py
"""

from __future__ import annotations

import io
import os
import re
import sys
import time
import urllib.request
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from trustgate.eval.fluency import reference_cache_root, sliding_window_nll  # noqa: E402
from trustgate.eval.gpt2_reference import GPT2Reference  # noqa: E402

WIKITEXT_TEST_URL = (
    "https://huggingface.co/datasets/Salesforce/wikitext/resolve/main/"
    "wikitext-103-raw-v1/test-00000-of-00001.parquet"
)

PUBLISHED_WIKITEXT103_PPL = 37.50
PUBLISHED_SOURCE = "Radford et al. 2019, Table 3 (117M model, WikiText-103)"


def wikitext_detokenize(text: str) -> str:
    """The standard WikiText detokenizer (Transformer-XL / lm-eval-harness)."""
    text = text.replace("s '", "s'")
    text = re.sub(r"/' [0-9]/", r"/'[0-9]/", text)
    text = text.replace(" @-@ ", "-")
    text = text.replace(" @,@ ", ",")
    text = text.replace(" @.@ ", ".")
    text = text.replace(" : ", ": ")
    text = text.replace(" ; ", "; ")
    text = text.replace(" . ", ". ")
    text = text.replace(" ! ", "! ")
    text = text.replace(" ? ", "? ")
    text = text.replace(" , ", ", ")
    text = re.sub(r"\(\s*([^\)]*?)\s*\)", r"(\1)", text)
    text = re.sub(r"\[\s*([^\]]*?)\s*\]", r"[\1]", text)
    text = re.sub(r"{\s*([^}]*?)\s*}", r"{\1}", text)
    text = re.sub(r"\"\s*([^\"]*?)\s*\"", r'"\1"', text)
    text = re.sub(r"'\s*([^']*?)\s*'", r"'\1'", text)
    text = text.replace("= = = =", "====")
    text = text.replace("= = =", "===")
    text = text.replace("= =", "==")
    text = text.replace(" " + chr(176) + " ", chr(176))
    text = text.replace(" \n", "\n")
    text = text.replace("\n ", "\n")
    text = text.replace(" N ", " 1 ")
    text = text.replace(" 's", "'s")
    return text


def load_wikitext_test(cache: Path) -> str:
    import pyarrow.parquet as pq

    cache.parent.mkdir(parents=True, exist_ok=True)
    if not cache.exists():
        print(f"fetching {WIKITEXT_TEST_URL}")
        request = urllib.request.Request(
            WIKITEXT_TEST_URL, headers={"User-Agent": "trustgate-acceptance"}
        )
        with urllib.request.urlopen(request, timeout=120) as response:
            cache.write_bytes(response.read())
        print(f"  saved {cache} ({cache.stat().st_size / 1e6:.2f} MB)")

    table = pq.read_table(io.BytesIO(cache.read_bytes()))
    lines = table.column("text").to_pylist()
    print(f"  {len(lines)} rows in the test split")
    return "".join(lines)


def main() -> int:
    stride = int(os.environ.get("STRIDE", "512"))
    root = reference_cache_root()

    print("=" * 66)
    print("T3.2 acceptance check -- independent fluency reference model")
    print("=" * 66)
    print(f"reference cache : {root}")
    print(f"target          : {PUBLISHED_WIKITEXT103_PPL} per word")
    print(f"source          : {PUBLISHED_SOURCE}")
    print(f"stride          : {stride}")
    print()

    reference = GPT2Reference(root / "gpt2")
    tokenizer = reference.tokenizer()
    print(f"model loaded: {reference.config}")
    print()

    raw = load_wikitext_test(root / "wikitext" / "wikitext-103-raw-test.parquet")
    detokenized = wikitext_detokenize(raw)

    # Word count convention: whitespace split on the detokenized text, which is
    # what the published per-word perplexities divide by.
    n_words = len([w for w in re.split(r"\s+", detokenized) if w])
    ids = tokenizer.encode(detokenized).ids
    print(f"  {len(raw)} raw chars -> {len(detokenized)} detokenized chars")
    print(f"  {n_words} words, {len(ids)} BPE tokens "
          f"({len(ids) / n_words:.3f} tokens/word)")
    print()

    print("scoring (this takes a few minutes on CPU)...")
    started = time.time()
    windows = 0

    def logits_fn(window_ids):
        nonlocal windows
        windows += 1
        if windows % 50 == 0:
            print(f"    window {windows}, {time.time() - started:.0f}s elapsed", flush=True)
        return np.asarray(reference.logits(window_ids))

    score = sliding_window_nll(
        logits_fn, ids, max_length=reference.max_length, stride=stride
    )
    elapsed = time.time() - started
    print(f"  done in {elapsed:.0f}s over {windows} windows")
    print()

    # Per-token is what sliding_window_nll returns. Convert to per-word by
    # re-dividing the same total NLL by the word count.
    total_nll = score.mean_nll * score.n_scored_tokens
    word_ppl = float(np.exp(total_nll / n_words))

    print("-" * 66)
    print(f"  per-BPE-token perplexity : {score.perplexity:.3f}")
    print(f"  per-BPE-token mean NLL   : {score.mean_nll:.5f} nats")
    print(f"  scored tokens            : {score.n_scored_tokens}")
    print()
    print(f"  PER-WORD perplexity      : {word_ppl:.3f}")
    print(f"  published target         : {PUBLISHED_WIKITEXT103_PPL}")
    delta = word_ppl - PUBLISHED_WIKITEXT103_PPL
    print(f"  delta                    : {delta:+.3f} "
          f"({100 * delta / PUBLISHED_WIKITEXT103_PPL:+.1f}%)")
    print("-" * 66)

    # A 5% band. Tight enough that a real implementation bug (transposed Conv1D
    # weights, exact-GELU instead of gelu_new, wrong LayerNorm eps) lands far
    # outside; loose enough to absorb residual preprocessing differences the
    # paper does not fully specify.
    within = abs(delta) / PUBLISHED_WIKITEXT103_PPL <= 0.05
    print(f"\n  VERDICT: {'PASS' if within else 'OUTSIDE THE 5% BAND'}")
    if not within:
        print("  Record the number as measured. Do not tune the implementation")
        print("  toward the target -- the ratio the fluency bar uses does not")
        print("  depend on the absolute, and a fudged absolute hides real bugs.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
