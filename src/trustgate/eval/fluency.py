"""Independent fluency reference model for the Phase 1 attack spike.

T3.2 Implementation:
Uses `unsloth/Llama-3.2-1B` as the reference model.
Rationale:
1. Independent: Meta's Llama 3.2 is independent of the victim TTT-E2E model.
2. Exact Vocab Match: Both use the Llama-3 tokenizer (128256 vocab size). This is
   critical because it means we do NOT need to decode the `CraftedStream` back to
   text and re-encode it. We can feed the generated token IDs directly into the
   reference model, eliminating tokenization seams and ensuring perfect fidelity.
3. Ungated: `unsloth/Llama-3.2-1B` does not require Hugging Face authentication,
   making it reproducible without secrets.
4. CPU Compatible: At 1B parameters, it runs on CPU within a reasonable time
   budget for the evaluation spike (which runs on an 80GB GPU node anyway, but
   ensuring the scoring itself uses CPU avoids GPU memory fragmentation).
"""

from __future__ import annotations

import logging
import numpy as np

from trustgate.attack.stream import CraftedStream

logger = logging.getLogger(__name__)

# Loaded lazily to avoid downloading/allocating weights on module import
_REFERENCE_MODEL = None


def _get_reference_model():
    """Lazy load the reference model on CPU."""
    global _REFERENCE_MODEL
    if _REFERENCE_MODEL is None:
        import torch
        from transformers import AutoModelForCausalLM
        
        logger.info("Loading independent fluency reference model (unsloth/Llama-3.2-1B)...")
        # Default loads to CPU. We avoid device_map to prevent requiring accelerate.
        _REFERENCE_MODEL = AutoModelForCausalLM.from_pretrained("unsloth/Llama-3.2-1B")
        _REFERENCE_MODEL.eval()
    return _REFERENCE_MODEL


def score_perplexity(tokens: np.ndarray, model=None, stride: int = 1024) -> float:
    """Compute exact causal LM perplexity for a sequence of token IDs.
    
    Formula: exp(mean(negative_log_likelihood))
    
    Uses past_key_values and chunking (default 1024) to avoid allocating massive
    (L, V) logit tensors in memory, while maintaining exact mathematical equivalence
    to a single full-sequence forward pass.
    """
    import torch

    if model is None:
        model = _get_reference_model()

    # The tokens array includes the +1 lookahead token if it's from CraftedStream.
    # tokens shape: (L+1,)
    # inputs: (L,) targets: (L,) shifted by 1.
    input_ids = torch.tensor(tokens[:-1], dtype=torch.long).unsqueeze(0).to(model.device)
    target_ids = torch.tensor(tokens[1:], dtype=torch.long).unsqueeze(0).to(model.device)

    max_len = input_ids.shape[1]
    nlls = []
    past_key_values = None
    
    # We do NOT reduce immediately so we can concatenate and mean at the end,
    # ensuring no bias from unequal chunk sizes on the final boundary.
    loss_fn = torch.nn.CrossEntropyLoss(reduction="none")
    
    with torch.no_grad():
        for i in range(0, max_len, stride):
            chunk_input = input_ids[:, i:i+stride]
            chunk_target = target_ids[:, i:i+stride]
            
            outputs = model(
                input_ids=chunk_input,
                past_key_values=past_key_values,
                use_cache=True
            )
            past_key_values = outputs.past_key_values
            
            logits = outputs.logits[0]  # (chunk_len, V)
            
            nll = loss_fn(logits, chunk_target[0])  # (chunk_len,)
            nlls.append(nll.cpu().numpy())
            
    all_nlls = np.concatenate(nlls)
    return float(np.exp(np.mean(all_nlls)))


def fluency_scorer(poison: CraftedStream, control: CraftedStream, model=None) -> float:
    """The injected callback for `run_attack_spike`.
    
    Computes perplexity of both streams and returns the ratio:
    poison perplexity / control perplexity.
    
    A ratio <= 1.5 indicates the poison stream is reasonably benign-looking.
    """
    poison_ppl = score_perplexity(poison.tokens, model=model)
    control_ppl = score_perplexity(control.tokens, model=model)
    
    # Store these on the stream objects so they can be retrieved later if needed
    poison.perplexity = poison_ppl
    control.perplexity = control_ppl
    
    return poison_ppl / max(control_ppl, 1e-6)
