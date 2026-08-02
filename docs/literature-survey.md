# Literature Survey

**Topic:** Trustworthy Inference-Time Self-Modification — Validated Rollback and Poisoning Defenses for Self-Adapting Deep Learning Models

15 original research articles (journals only) from IEEE Xplore and ScienceDirect, published June 2024 onwards. Every DOI below was resolved against Crossref and confirmed to redirect to the publisher page.

> Naming note: this is the coursework/report "Phase 1". It is unrelated to roadmap Phase 1 (the attack-spike kill-gate) elsewhere in this repo.

---

## Quick reference

| # | Paper | Journal | Date | DOI |
|---|-------|---------|------|-----|
| 1 | Compromising LLM-Driven Embodied Agents with Contextual Backdoor Attacks | IEEE Trans. Information Forensics and Security, v20, 3979–3994 | 2025 | [10.1109/TIFS.2025.3555410](https://doi.org/10.1109/TIFS.2025.3555410) |
| 2 | Mutual Information Guided Backdoor Mitigation for Pre-Trained Encoders | IEEE Trans. Information Forensics and Security, v20, 3414–3428 | 2025 | [10.1109/TIFS.2025.3550062](https://doi.org/10.1109/TIFS.2025.3550062) |
| 3 | Enhanced Model Poisoning Attack and Multi-Strategy Defense in Federated Learning | IEEE Trans. Information Forensics and Security, v20, 3877–3892 | 2025 | [10.1109/TIFS.2025.3555193](https://doi.org/10.1109/TIFS.2025.3555193) |
| 4 | LDPGuard: Defenses Against Data Poisoning Attacks to Local Differential Privacy Protocols | IEEE Trans. Knowledge and Data Engineering, v36(7), 3195–3209 | Jul 2024 | [10.1109/TKDE.2024.3358909](https://doi.org/10.1109/TKDE.2024.3358909) |
| 5 | Protecting Machine Learning from Poisoning Attacks: A Risk-Based Approach | Computers & Security, v155, 104468 | Aug 2025 | [10.1016/j.cose.2025.104468](https://doi.org/10.1016/j.cose.2025.104468) |
| 6 | AntidoteFL: Enhancing Defense Against Poisoning Attacks in Federated Learning | Computer Networks, v269, 111427 | Sep 2025 | [10.1016/j.comnet.2025.111427](https://doi.org/10.1016/j.comnet.2025.111427) |
| 7 | DPAD: Data Poisoning Attack Defense Mechanism for Federated Learning-Based System | Computers and Electrical Engineering, v121, 109893 | Jan 2025 | [10.1016/j.compeleceng.2024.109893](https://doi.org/10.1016/j.compeleceng.2024.109893) |
| 8 | Novel Defense Strategies for Concurrent Data and Model Poisoning Attacks in Federated Learning | Knowledge-Based Systems, v345, 116057 | Jun 2026 | [10.1016/j.knosys.2026.116057](https://doi.org/10.1016/j.knosys.2026.116057) |
| 9 | Defending Federated Learning-Based Intrusion Detection Systems Against Model Poisoning Attacks | Neurocomputing, v698, 134208 | Oct 2026 | [10.1016/j.neucom.2026.134208](https://doi.org/10.1016/j.neucom.2026.134208) |
| 10 | FedDRLPD: Deep Reinforcement Learning-Based Defense Mechanism Against Poisoning Attacks in Federated Learning | Knowledge-Based Systems, v339, 115558 | Apr 2026 | [10.1016/j.knosys.2026.115558](https://doi.org/10.1016/j.knosys.2026.115558) |
| 11 | Adversarial Self-Training for Robustness and Generalization | Pattern Recognition Letters, v185, 117–123 | Sep 2024 | [10.1016/j.patrec.2024.07.020](https://doi.org/10.1016/j.patrec.2024.07.020) |
| 12 | Multi-View Contrastive Learning for Graph Adversarial Defense | Neural Networks, v192, 107868 | Dec 2025 | [10.1016/j.neunet.2025.107868](https://doi.org/10.1016/j.neunet.2025.107868) |
| 13 | Adversarial Attack-Defense Framework for Enhancing the Robustness of Power Insulator Detection in Cloud-Edge Deployment | Engineering Applications of Artificial Intelligence, v166, 113625 | Feb 2026 | [10.1016/j.engappai.2025.113625](https://doi.org/10.1016/j.engappai.2025.113625) |
| 14 | Momentum Gradient-Based Untargeted Poisoning Attack on Hypergraph Neural Networks | Neurocomputing, v634, 129835 | Jun 2025 | [10.1016/j.neucom.2025.129835](https://doi.org/10.1016/j.neucom.2025.129835) |
| 15 | H3NI: Non-Target-Specific Node Injection Attacks on Hypergraph Neural Networks via Genetic Algorithm | Neurocomputing, v613, 128746 | Jan 2025 | [10.1016/j.neucom.2024.128746](https://doi.org/10.1016/j.neucom.2024.128746) |

---

## 1. Compromising LLM-Driven Embodied Agents with Contextual Backdoor Attacks

- **Journal:** IEEE Transactions on Information Forensics and Security, vol. 20, pp. 3979–3994 (2025)
- **What it addresses / relation:** Shows that a crafted sequence of benign-looking contextual inputs can implant a hidden backdoor into an LLM-driven agent at deployment, establishing the inference-time poisoning threat model that a rollback/validation controller must defend against.
- **DOI:** https://doi.org/10.1109/TIFS.2025.3555410
- **Publisher page:** https://ieeexplore.ieee.org/document/10943262/

## 2. Mutual Information Guided Backdoor Mitigation for Pre-Trained Encoders

- **Journal:** IEEE Transactions on Information Forensics and Security, vol. 20, pp. 3414–3428 (2025)
- **What it addresses / relation:** Removes an implanted backdoor from a pre-trained encoder using mutual-information-guided distillation with only a small clean set, informing how compromised model state can be restored to a trusted version without full retraining.
- **DOI:** https://doi.org/10.1109/TIFS.2025.3550062
- **Publisher page:** https://ieeexplore.ieee.org/document/10930652/

## 3. Enhanced Model Poisoning Attack and Multi-Strategy Defense in Federated Learning

- **Journal:** IEEE Transactions on Information Forensics and Security, vol. 20, pp. 3877–3892 (2025)
- **What it addresses / relation:** Proposes a stronger scaling/sign poisoning attack and a multi-strategy defense that filters malicious model updates before they are committed, directly relevant to a gate that accepts or rejects proposed state updates.
- **DOI:** https://doi.org/10.1109/TIFS.2025.3555193
- **Publisher page:** https://ieeexplore.ieee.org/document/10942405/

## 4. LDPGuard: Defenses Against Data Poisoning Attacks to Local Differential Privacy Protocols

- **Journal:** IEEE Transactions on Knowledge and Data Engineering, vol. 36, no. 7, pp. 3195–3209 (July 2024)
- **What it addresses / relation:** Builds a framework that detects and counteracts data poisoning that skews aggregated estimates, illustrating budgeted detection of adversarially injected data in an online collection setting.
- **DOI:** https://doi.org/10.1109/TKDE.2024.3358909
- **Publisher page:** https://ieeexplore.ieee.org/document/10415225/

## 5. Protecting Machine Learning from Poisoning Attacks: A Risk-Based Approach

- **Journal:** Computers & Security (Elsevier), vol. 155, art. 104468 (August 2025)
- **What it addresses / relation:** Scores the poisoning risk of each incoming data point and routes high-risk points away from training — a risk-gating idea directly analogous to the validation signal that admits, rejects, or rolls back an inference-time update.
- **DOI:** https://doi.org/10.1016/j.cose.2025.104468
- **Publisher page:** https://www.sciencedirect.com/science/article/pii/S0167404825001579

## 6. AntidoteFL: Enhancing Defense Against Poisoning Attacks in Federated Learning

- **Journal:** Computer Networks (Elsevier), vol. 269, art. 111427 (September 2025)
- **What it addresses / relation:** Distinguishes benign from malicious model updates using PCA and cosine-similarity mapping before aggregation — a consistency-based acceptance test comparable to an anchor-consistency validator.
- **DOI:** https://doi.org/10.1016/j.comnet.2025.111427
- **Publisher page:** https://www.sciencedirect.com/science/article/pii/S1389128625003949

## 7. DPAD: Data Poisoning Attack Defense Mechanism for Federated Learning-Based System

- **Journal:** Computers and Electrical Engineering (Elsevier), vol. 121, art. 109893 (January 2025)
- **What it addresses / relation:** Combines cosine-similarity probing, L2-norm filtering, anomaly detection and temporal suspicion scoring into a multi-metric pipeline catching both abrupt and stealthy poisoning of updates.
- **DOI:** https://doi.org/10.1016/j.compeleceng.2024.109893
- **Publisher page:** https://www.sciencedirect.com/science/article/pii/S004579062400819X

## 8. Novel Defense Strategies for Concurrent Data and Model Poisoning Attacks in Federated Learning

- **Journal:** Knowledge-Based Systems (Elsevier), vol. 345, art. 116057 (June 2026)
- **What it addresses / relation:** Defends against simultaneous data- and model-poisoning using parameter clamping and cross-space consistency validation, mirroring the bounded-drift and consistency-check components of the proposed controller.
- **DOI:** https://doi.org/10.1016/j.knosys.2026.116057
- **Publisher page:** https://www.sciencedirect.com/science/article/pii/S0950705126007835

## 9. Defending Federated Learning-Based Intrusion Detection Systems Against Model Poisoning Attacks

- **Journal:** Neurocomputing (Elsevier), vol. 698, art. 134208 (October 2026)
- **What it addresses / relation:** Stabilizes update dynamics with intra-layer directional gradient projection and magnitude normalization to bound the effect of adversarial updates without needing clean validation data.
- **DOI:** https://doi.org/10.1016/j.neucom.2026.134208
- **Publisher page:** https://www.sciencedirect.com/science/article/pii/S0925231226016061

## 10. FedDRLPD: Deep Reinforcement Learning-Based Defense Mechanism Against Poisoning Attacks in Federated Learning

- **Journal:** Knowledge-Based Systems (Elsevier), vol. 339, art. 115558 (April 2026)
- **What it addresses / relation:** Uses a deep-Q-network policy to adaptively decide which incoming updates to trust and which contributors to exclude — an adaptive accept/reject decision policy relevant to the controller's validator.
- **DOI:** https://doi.org/10.1016/j.knosys.2026.115558
- **Publisher page:** https://www.sciencedirect.com/science/article/pii/S095070512600300X

## 11. Adversarial Self-Training for Robustness and Generalization

- **Journal:** Pattern Recognition Letters (Elsevier), vol. 185, pp. 117–123 (September 2024)
- **What it addresses / relation:** Shows how self-training on adversarial examples improves robustness while preserving clean accuracy, addressing the same robustness-versus-utility trade-off the validation gate must balance.
- **DOI:** https://doi.org/10.1016/j.patrec.2024.07.020
- **Publisher page:** https://www.sciencedirect.com/science/article/pii/S016786552400223X

## 12. Multi-View Contrastive Learning for Graph Adversarial Defense

- **Journal:** Neural Networks (Elsevier), vol. 192, art. 107868 (December 2025)
- **What it addresses / relation:** Improves model robustness against structural adversarial manipulation via multi-view contrastive learning — an approach to keeping learned representations reliable under adversarial input.
- **DOI:** https://doi.org/10.1016/j.neunet.2025.107868
- **Publisher page:** https://www.sciencedirect.com/science/article/pii/S0893608025007488

## 13. Adversarial Attack-Defense Framework for Enhancing the Robustness of Power Insulator Detection in Cloud-Edge Deployment

- **Journal:** Engineering Applications of Artificial Intelligence (Elsevier), vol. 166, art. 113625 (February 2026)
- **What it addresses / relation:** Pairs a strong perturbation attack with a two-stage denoising defense at the inference/deployment stage under edge constraints — an example of low-overhead inference-time defense and recovery.
- **DOI:** https://doi.org/10.1016/j.engappai.2025.113625
- **Publisher page:** https://www.sciencedirect.com/science/article/pii/S0952197625036577

## 14. Momentum Gradient-Based Untargeted Poisoning Attack on Hypergraph Neural Networks

- **Journal:** Neurocomputing (Elsevier), vol. 634, art. 129835 (June 2025)
- **What it addresses / relation:** Crafts untargeted feature poisoning via a momentum-gradient mechanism to degrade model behavior, characterizing how gradient-driven poisoning corrupts a learner and what a validator must detect.
- **DOI:** https://doi.org/10.1016/j.neucom.2025.129835
- **Publisher page:** https://www.sciencedirect.com/science/article/pii/S0925231225005077

## 15. H3NI: Non-Target-Specific Node Injection Attacks on Hypergraph Neural Networks via Genetic Algorithm

- **Journal:** Neurocomputing (Elsevier), vol. 613, art. 128746 (January 2025)
- **What it addresses / relation:** Introduces a black-box, budget-constrained injection attack that degrades performance without target access — the stealthy, budget-limited poisoning regime that a bounded-drift budget is designed to contain.
- **DOI:** https://doi.org/10.1016/j.neucom.2024.128746
- **Publisher page:** https://www.sciencedirect.com/science/article/pii/S0925231224015170

---

*DOIs resolved via the Crossref REST API and each confirmed to redirect to a live publisher page. If a publisher link ever breaks, the `https://doi.org/...` form is permanent — use that.*
