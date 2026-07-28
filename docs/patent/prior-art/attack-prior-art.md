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
we lead the filing with the gate. TO COMPLETE: per-paper mechanism notes and the
exact diff of our crafted-stream constraint vs. 2410.04682.
