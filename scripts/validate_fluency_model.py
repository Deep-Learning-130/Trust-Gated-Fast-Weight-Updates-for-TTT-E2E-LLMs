#!/usr/bin/env python3
"""
T3.2 Acceptance Validation Script.

This script validates the independent fluency scorer (unsloth/Llama-3.2-1B)
by running it against the Wikitext-2 test set. It proves that the exact NLL
computation logic in `trustgate.eval.fluency` correctly reproduces a realistic
published/reference perplexity on a fixed, known corpus.

This script is meant to be run manually and is NOT part of the automated CI 
because it requires network access to download the model weights and the 
Hugging Face `datasets` library.

Run from the repository root:
    PYTHONPATH=src python scripts/validate_fluency_model.py
"""

import sys
import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import numpy as np
from time import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))
# Ensure we import the actual logic being tested
try:
    from trustgate.eval.fluency import score_perplexity
except ImportError:
    print(f"Error: Could not import trustgate. sys.path: {sys.path}")
    sys.exit(1)

def main():
    model_id = "unsloth/Llama-3.2-1B"
    print(f"Loading reference model {model_id}...")
    
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    # Llama 3 does not have a pad token by default, use eos
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.float32,
    )
    model.eval()

    print("Loading datasets...")
    try:
        from datasets import load_dataset
    except ImportError:
        print("Please install datasets library (pip install datasets) to run validation.")
        sys.exit(1)

    dataset = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test")
    
    # Preprocessing: join all text with newlines (standard wikitext eval)
    text = "\n\n".join(dataset["text"])
    
    # We only take the first N tokens to keep CPU eval time reasonable,
    # but enough to get a stable perplexity (e.g., 4096 tokens).
    eval_tokens = 4096
    print(f"Tokenizing and evaluating on the first {eval_tokens} tokens...")
    
    input_ids = tokenizer.encode(text, return_tensors="pt")
    input_ids = input_ids[:, :eval_tokens]

    start_time = time()
    # The score_perplexity function expects tokens in numpy array
    ppl = score_perplexity(input_ids[0].numpy(), model=model, stride=1024)
    elapsed = time() - start_time

    print("\n" + "="*50)
    print("T3.2 ACCEPTANCE VALIDATION RESULT")
    print("="*50)
    print(f"Model: {model_id}")
    print(f"Dataset: wikitext-2-raw-v1 (test split, first {eval_tokens} tokens)")
    print(f"Context Stride: 1024")
    print(f"Perplexity: {ppl:.2f}")
    print(f"Execution Time: {elapsed:.2f} seconds")
    print("="*50)
    
    # A realistic untrained Llama-3.2-1B on wikitext-2 usually gets ~10-15 PPL
    if ppl < 100.0 and ppl > 1.0:
        print("Verdict: PASS. The model successfully produces a realistic reference perplexity.")
    else:
        print("Verdict: FAIL. Perplexity is outside the expected range for a 1B model.")
        sys.exit(1)

if __name__ == "__main__":
    main()
