# Prior art — robust TTA defenses beyond MedBN

**Status: STUB.** Complete before filing. MedBN gets its own dedicated diff
(`medbn-diff.md`); this file catches the rest of the robust-TTA landscape so the
novelty argument is not blindsided by a non-MedBN reference.

## Survey of Robust TTA Methods

### 1. Non-Batch-Norm Robust TTA
Methods defending against distribution shift typically rely on pseudo-label filtering or confidence thresholding (e.g., ignoring updates for samples where model confidence is low).
- **Unit Defended**: Entire model or specific adapter layers.
- **Signal**: Confidence scores, prediction entropy.
- **Diff vs Our System**: These methods filter the *input* or *loss* before the update is computed. Our system intercepts the *update itself* just before it is applied to the weights, using anchor consistency rather than output confidence. None operate on LLM fast weights.

### 2. Entropy-Minimization (TENT) and Failure Modes
TENT adapts by minimizing the entropy of model predictions.
- **Failure Mode**: Susceptible to confirmation bias and gradient explosion on out-of-distribution or malicious streams, leading to model collapse.
- **Diff vs Our System**: TENT provides no structural guarantee against divergence. Our system uses a hard cumulative-drift budget ($\epsilon$) and a fallback anchor, guaranteeing that even if the objective (like entropy minimization or autoregressive loss) degrades, the weights can never drift beyond a safe radius.

### 3. Continual-Learning Sanitization
Methods in continual learning (e.g., rehearsal buffers, gradient episodic memory) attempt to prevent catastrophic forgetting.
- **Unit Defended**: Global base weights across tasks.
- **Signal**: Gradients compared against a memory buffer of past tasks.
- **Diff vs Our System**: Continual learning methods typically operate offline or at task boundaries, require storing historical data, and defend base weights. Our system operates purely online, stores only a scalar accumulator and a ring-buffer of states, and explicitly defends *fast weights* during the forward pass. Our rollback is O(1) using versioned stores rather than requiring retraining.
