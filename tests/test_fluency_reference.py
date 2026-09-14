"""Opt-in tests against the real GPT-2 reference model.

**Skipped unless the weights have been fetched.** CI has no network and must not
download 548MB, so these are gated on the cache existing:

    python scripts/fetch_reference_model.py

What is checked here is the thing the unit tests cannot check with a fake: that
the implemented forward pass actually behaves like a language model. The
published-perplexity reproduction lives in
`scripts/acceptance_fluency_reference.py` and its result is recorded in
`experiments/001-attack-spike/FLUENCY_REFERENCE.md`; these are the fast
invariants worth having in the suite.
"""

import numpy as np
import pytest

from trustgate.eval.fluency import (
    Llama3Decoder,
    reference_cache_root,
    sliding_window_nll,
)
from trustgate.tokens import BOS_TOKEN_ID, VOCAB_SIZE

ROOT = reference_cache_root()
GPT2_DIR = ROOT / "gpt2"
LLAMA3_TOKENIZER = ROOT / "llama3" / "tokenizer.json"

requires_reference = pytest.mark.skipif(
    not (GPT2_DIR / "model.safetensors").exists(),
    reason="reference model not fetched; run scripts/fetch_reference_model.py",
)
requires_decoder = pytest.mark.skipif(
    not LLAMA3_TOKENIZER.exists(),
    reason="llama3 tokenizer not fetched; run scripts/fetch_reference_model.py",
)

FLUENT = (
    "It is a truth universally acknowledged, that a single man in possession "
    "of a good fortune, must be in want of a wife."
)


@pytest.fixture(scope="module")
def reference():
    from trustgate.eval.gpt2_reference import GPT2Reference

    return GPT2Reference(GPT2_DIR)


def perplexity(reference, text):
    ids = reference.tokenizer().encode(text).ids
    return sliding_window_nll(
        lambda window: np.asarray(reference.logits(window)),
        ids,
        max_length=reference.max_length,
        stride=512,
    ).perplexity


@requires_reference
def test_config_matches_gpt2_small(reference):
    assert reference.config.n_layer == 12
    assert reference.config.n_head == 12
    assert reference.config.n_embd == 768
    assert reference.config.n_positions == 1024
    assert reference.config.vocab_size == 50257
    assert reference.config.layer_norm_epsilon == pytest.approx(1e-5)


@requires_reference
def test_reference_is_not_the_victim(reference):
    # Different vocabulary, hence a different tokenizer and a different corpus.
    # This is the structural half of the independence claim.
    assert reference.config.vocab_size != VOCAB_SIZE


@requires_reference
def test_fluent_english_scores_in_a_plausible_range(reference):
    # A broken forward pass (transposed Conv1D weights, exact GELU instead of
    # gelu_new, wrong LayerNorm eps) lands in the hundreds or thousands, not
    # here. This is the cheap version of the acceptance check.
    assert 10.0 < perplexity(reference, FLUENT) < 100.0


@requires_reference
def test_scrambling_words_makes_perplexity_much_worse(reference):
    # The discrimination the fluency bar actually relies on.
    words = FLUENT.split()
    np.random.default_rng(0).shuffle(words)
    assert perplexity(reference, " ".join(words)) > 5 * perplexity(reference, FLUENT)


@requires_reference
def test_repetition_scores_LOW_which_perplexity_alone_cannot_catch(reference):
    # Documented limitation, not a bug: a repetitive stream is trivially
    # predictable, so perplexity *rewards* it. The realism bar therefore cannot
    # detect a poison stream that reuses one span many times -- which is why
    # spans are drawn without replacement and span multiplicity is reported
    # separately (ADR-007).
    repetitive = " ".join(["the quick brown fox"] * 30)
    assert perplexity(reference, repetitive) < perplexity(reference, FLUENT)


@requires_reference
def test_logits_shape_and_finiteness(reference):
    logits = np.asarray(reference.logits(np.arange(16, dtype=np.int32)))
    assert logits.shape == (16, 50257)
    assert np.all(np.isfinite(logits))


@requires_reference
def test_a_sequence_past_the_context_window_is_refused(reference):
    with pytest.raises(ValueError, match="exceeds the model"):
        reference.logits(np.zeros(reference.max_length + 1, dtype=np.int32))


@requires_decoder
def test_decoder_vocabulary_matches_the_vendor_configs():
    # Asserted at construction: a tokenizer with the wrong vocab would decode to
    # plausible but wrong text, and nothing downstream could detect it.
    decoder = Llama3Decoder(LLAMA3_TOKENIZER)
    assert decoder is not None


@requires_decoder
def test_decoder_round_trips_real_text():
    from tokenizers import Tokenizer

    tokenizer = Tokenizer.from_file(str(LLAMA3_TOKENIZER))
    ids = tokenizer.encode(FLUENT, add_special_tokens=False).ids

    assert BOS_TOKEN_ID not in ids  # crafted streams are BOS-free
    assert Llama3Decoder(LLAMA3_TOKENIZER).decode(ids).strip() == FLUENT.strip()


@requires_reference
@requires_decoder
def test_end_to_end_stream_scoring_discriminates():
    """Ids -> decode -> score, the full path the harness uses per seed."""
    from tokenizers import Tokenizer

    from trustgate.attack.stream import CraftedStream, StreamStrategy
    from trustgate.eval.fluency import load_default_scorer

    tokenizer = Tokenizer.from_file(str(LLAMA3_TOKENIZER))
    fluent_ids = tokenizer.encode(FLUENT, add_special_tokens=False).ids

    words = FLUENT.split()
    np.random.default_rng(1).shuffle(words)
    scrambled_ids = tokenizer.encode(" ".join(words), add_special_tokens=False).ids

    def stream(ids):
        return CraftedStream(
            tokens=np.asarray(ids, dtype=np.int32), strategy=StreamStrategy.SELECT
        )

    scorer = load_default_scorer()
    poison, control = stream(scrambled_ids), stream(fluent_ids)

    ratio = scorer(poison, control)

    assert poison.is_scored
    # A scrambled poison arm must blow the pre-registered <= 1.5 bar.
    assert ratio > 1.5


requires_torch = pytest.mark.skipif(
    __import__("importlib.util", fromlist=["util"]).find_spec("torch") is None,
    reason="torch is a verification-only dependency; see FLUENCY_REFERENCE.md",
)


@requires_reference
@requires_torch
def test_forward_pass_matches_the_reference_implementation(reference):
    """The primary acceptance check, in miniature.

    The published-perplexity check turned out undecidable (the per-word
    denominator convention is unspecified, and 37.50 falls between two plausible
    conventions), so equivalence against HuggingFace's implementation is what
    actually establishes the forward pass. Full sweep:
    `scripts/verify_gpt2_equivalence.py`.
    """
    import warnings

    warnings.filterwarnings("ignore")
    import torch
    from transformers import GPT2LMHeadModel

    theirs = GPT2LMHeadModel.from_pretrained(
        str(GPT2_DIR), local_files_only=True
    ).eval()

    ids = np.random.default_rng(0).integers(0, 50257, 64).astype(np.int32)
    mine = np.asarray(reference.logits(ids), dtype=np.float64)
    with torch.no_grad():
        theirs_logits = (
            theirs(torch.tensor(ids[None, :], dtype=torch.long))
            .logits[0]
            .numpy()
            .astype(np.float64)
        )

    # float32 accumulation order differs between backends; a real bug is O(0.1).
    assert np.abs(mine - theirs_logits).max() < 1e-3
    assert (mine.argmax(-1) == theirs_logits.argmax(-1)).all()
