"""Tests for the independent fluency reference model."""

import numpy as np
import pytest

from trustgate.attack.stream import CraftedStream, StreamStrategy
from trustgate.eval.fluency import score_perplexity, fluency_scorer


class MockModel:
    """A tiny deterministic mock for testing perplexity math without network access."""
    
    def __init__(self, vocab_size: int = 100):
        self.device = "cpu"
        self.vocab_size = vocab_size
        
        # We need a stable output for testing. 
        # We'll just return uniform logits for simplicity, so NLL = ln(vocab_size)
        # PPL = exp(ln(vocab_size)) = vocab_size.
        
    def __call__(self, input_ids, past_key_values=None, use_cache=True):
        import torch
        
        batch, seq_len = input_ids.shape
        # Uniform logits = 0 for all tokens -> softmax is uniform 1/V.
        logits = torch.zeros((batch, seq_len, self.vocab_size))
        
        class Output:
            def __init__(self, logits):
                self.logits = logits
                self.past_key_values = past_key_values
        
        return Output(logits=logits)


def test_perplexity_math():
    """Verify that perplexity equals the vocab size when logits are uniform zeros."""
    vocab_size = 123
    mock_model = MockModel(vocab_size=vocab_size)
    
    # 10 tokens. The scorer splits into input/target (len 9).
    tokens = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], dtype=np.int32)
    
    ppl = score_perplexity(tokens, model=mock_model, stride=4)
    
    # ln(1/123) is -4.81. -NLL is 4.81. exp(4.81) = 123.
    # We should recover the vocab size exactly.
    assert np.isclose(ppl, vocab_size, rtol=1e-5)


def test_fluency_scorer_ratio():
    """Verify the ratio computation correctly accesses both streams."""
    vocab_size = 50
    mock_model = MockModel(vocab_size=vocab_size)
    
    poison = CraftedStream(
        tokens=np.array([1, 2, 3, 4]),
        strategy=StreamStrategy.SELECT,
    )
    control = CraftedStream(
        tokens=np.array([1, 2, 3, 4]),
        strategy=StreamStrategy.SELECT,
    )
    
    ratio = fluency_scorer(poison, control, model=mock_model)
    
    # Both streams have identical perplexity (50.0). Ratio = 1.0.
    assert np.isclose(ratio, 1.0)
    assert np.isclose(poison.perplexity, 50.0)
    assert np.isclose(control.perplexity, 50.0)
