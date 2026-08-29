> **WIRING SMOKE TEST -- SYNTHETIC LOSSES, NOT A RESULT. Do not copy this out of results/ and do not cite it.**

# Phase 1 Attack Spike -- Result

**Verdict: PROCEED**

Strategy: `select` | Seeds: 5

| Criterion | Observed | Pre-registered | |
|---|---|---|---|
| Effect size (Cohen's d) | 14.832 | >= 0.8 | PASS |
| Relative degradation | +15.12% | >= 10.00% | PASS |
| Fluency ratio vs control (max across seeds) | 1.254 | <= 1.5 | PASS |

Benign loss after poison stream: 1.1517
Benign loss after control stream: 1.0004

## Per-seed values

| Seed | Poisoned | Control | Delta | Fluency ratio |
|---|---|---|---|---|
| 0 | 1.1549 | 1.0013 | +0.1536 | 1.188 |
| 1 | 1.1617 | 0.9987 | +0.1630 | 1.254 (worst) |
| 2 | 1.1659 | 1.0064 | +0.1595 | 1.017 |
| 3 | 1.1440 | 1.0010 | +0.1430 | 1.228 |
| 4 | 1.1320 | 0.9946 | +0.1373 | 1.125 |

## Fluency aggregation

- max across seeds: 1.254 **(gating)**
- mean across seeds: 1.163 (reported)

The pre-registration states the bar as one number (<= 1.5) but
there are five streams. The **max** gates: under a mean, one bad seed is
laundered by four good ones while its loss still pools into the gated
Cohen's d. Max is the strictest defensible reading, and Rule 5 exists to
stop bars being loosened.
