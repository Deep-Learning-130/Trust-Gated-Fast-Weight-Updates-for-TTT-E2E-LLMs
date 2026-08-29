# Attack Prior-Art Notes

This document surveys test-time poisoning mechanisms. 
It establishes that while the *attack* side of test-time poisoning is well-documented (and therefore not claimed as novel here), our defense mechanism targets a specific gap in the literature.

## 2308.08505: Test-Time Poisoning Attacks Against Test-Time Adaptation Models
- **Citation:** arXiv:2308.08505
- **Mechanism:** Injects maliciously crafted samples into the test stream to manipulate the test-time adaptation process.
- **Threat Model:** Adversary can submit inputs at test time.
- **Target:** General test-time adaptation (typically computer vision).
- **Distinction from our work:** Establishes the viability of the attack vector, but does not address TTT-E2E specifically, nor does it provide a defense targeting fast-weight LLM updates.

## 2410.04682: On the Adversarial Risk of Test Time Adaptation: An Investigation into Realistic Test-Time Data Poisoning
- **Citation:** arXiv:2410.04682
- **Mechanism:** Demonstrates realistic test-time data poisoning scenarios against TTA where models update weights during inference.
- **Threat Model:** Black-box or grey-box injection of adversarial inputs during inference.
- **Target:** Continual Test-time Adaptation models.
- **Distinction from our work:** Highlights the vulnerability we are attempting to solve. Confirms that unbounded fast-weight updates are a severe risk.

## 2412.01154: R.I.P.: A Simple Black-box Attack on Continual Test-time Adaptation
- **Citation:** arXiv:2412.01154
- **Mechanism:** Uses self-supervised learning signals natively used by the TTA model to craft adversarial inputs that disrupt the continual learning process.
- **Threat Model:** Black-box, streaming inputs.
- **Target:** Continual TTA parameters.
- **Distinction from our work:** Provides another attack technique proving that continual adaptation is fragile without a trust gate.
