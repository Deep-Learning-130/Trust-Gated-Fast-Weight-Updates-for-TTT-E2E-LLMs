# Robust TTA Survey

This document surveys defenses in the domain of Test-Time Adaptation (TTA), highlighting why existing techniques (often built for vision models) do not directly apply to the fast-weight LLM scenario.

## 2403.19326: MedBN: Robust Test-Time Adaptation against Malicious Test Samples
- **Citation:** arXiv:2403.19326
- **Mechanism:** Defends against TTA poisoning by replacing batch-normalization statistics from the current malicious test samples with robust statistics aggregated from benign samples.
- **Threat Model:** Adversary can submit batches of malicious inputs to poison TTA models.
- **Target:** Batch Normalization layers in vision models.
- **Distinction from our work:** TTT-E2E models (and LLMs in general) do not use batch normalization, rendering MedBN inapplicable. Furthermore, our trust gate operates directly on the fast-weight updates themselves, enforcing a bounded-drift and consistency check against an anchor, rather than manipulating feature statistics. 
- **Relevance:** Serves as the primary analogue/baseline for poisoning defenses in the TTA space, highlighting the gap for defending weight-updating models in NLP.
