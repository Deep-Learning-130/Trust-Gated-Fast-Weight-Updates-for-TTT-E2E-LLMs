# Prior art — test-time poisoning attacks (the *attack* side is not claimed)

The attack in `src/trustgate/attack/` is a *tool* to establish the threat model
and to serve as the evaluation adversary. It is deliberately **not** part of the
patent claim — the attack side is mature. Recorded here so the disclosure is
honest about what is old.

| Work | arXiv / venue | Relevance |
|---|---|---|
| Test-Time Poisoning Attacks Against TTA | 2308.08505 (IEEE S&P) | Foundational: TTA is poisonable |
| Realistic Test-Time Data Poisoning | 2410.04682 (ICLR 2025) | Realistic threat model; informs our benign-looking constraint |
| R.I.P. — black-box attack on continual TTA | 2412.01154 | Black-box continual-TTA attack |

**Our novelty is the defense on the fast-weight-LLM setting, not the attack.**
What is arguably new on the attack side — poisoning TTT-E2E fast weights
*specifically* — is a research contribution and a motivation for the defense, but
we lead the filing with the gate.

## Per-Paper Mechanism Notes

- **2308.08505**: Demonstrates foundational vulnerability of TTA by injecting malicious gradients via input modifications. Primarily targets classification models using white-box gradients to maximize loss on subsequent benign batches.
- **2410.04682**: Highlights the necessity of stealth. It argues that obvious adversarial perturbations are easily filtered. For text, it constrains adversarial generation to maintain fluency (often using simple post-hoc filtering or bounding perplexity degradation).
- **2412.01154**: Operates in a black-box setting for continual TTA. Relies on query feedback rather than full gradient access to construct the poison sequence over time.

## Diff: Our Crafted-Stream Constraint vs. 2410.04682

While 2410.04682 introduces realism constraints (such as fluency bounds for text), our crafted-stream constraint differs in the following ways:
1. **Objective Integration**: Rather than acting as a post-hoc filter, our fluency weight sits directly inside the search objective during stream construction.
2. **Independent Scoring**: We use an entirely independent reference model (GPT-2) to score fluency. 2410.04682 often relies on the victim model itself, which becomes circular when the victim's fast weights are actively adapting.
3. **Sub-chunk Granularity**: Our constraint is tailored to the exact sub-chunk span sizes processed by the TTT-E2E inner loop (as detailed in ADR-007), rather than generic sequence-level constraints.
