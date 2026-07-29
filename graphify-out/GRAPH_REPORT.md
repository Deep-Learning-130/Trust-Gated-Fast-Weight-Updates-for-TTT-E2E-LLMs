# Graph Report - .  (2026-07-29)

## Corpus Check
- Corpus is ~26,586 words - fits in a single context window. You may not need a graph.

## Summary
- 302 nodes · 510 edges · 26 communities (14 shown, 12 thin omitted)
- Extraction: 77% EXTRACTED · 23% INFERRED · 0% AMBIGUOUS · INFERRED: 116 edges (avg confidence: 0.61)
- Token cost: 0 input · 0 output

## Community Hubs (Navigation)
- [[_COMMUNITY_Gate Signals & MedBN Baseline|Gate Signals & MedBN Baseline]]
- [[_COMMUNITY_Attack Stream Crafting|Attack Stream Crafting]]
- [[_COMMUNITY_Design Docs, ADRs & Experiments|Design Docs, ADRs & Experiments]]
- [[_COMMUNITY_Update Interceptor|Update Interceptor]]
- [[_COMMUNITY_Versioned Rollback Store|Versioned Rollback Store]]
- [[_COMMUNITY_Pytree Fast-Weight Ops|Pytree Fast-Weight Ops]]
- [[_COMMUNITY_Corruption Metrics|Corruption Metrics]]
- [[_COMMUNITY_Drift Accumulator|Drift Accumulator]]
- [[_COMMUNITY_Audit Log|Audit Log]]
- [[_COMMUNITY_Attack Objectives|Attack Objectives]]
- [[_COMMUNITY_GoNo-Go Report|Go/No-Go Report]]
- [[_COMMUNITY_Probe Rotation|Probe Rotation]]
- [[_COMMUNITY_Uncertainty Gate (Option B)|Uncertainty Gate (Option B)]]
- [[_COMMUNITY_Test Fixtures|Test Fixtures]]
- [[_COMMUNITY_Attack Package Init|Attack Package Init]]
- [[_COMMUNITY_Audit Package Init|Audit Package Init]]
- [[_COMMUNITY_Baselines Package Init|Baselines Package Init]]
- [[_COMMUNITY_Checkpoint Fetch Script|Checkpoint Fetch Script]]
- [[_COMMUNITY_Pre-Push Guard Script|Pre-Push Guard Script]]
- [[_COMMUNITY_Vendor Setup Script|Vendor Setup Script]]
- [[_COMMUNITY_Drift Package Init|Drift Package Init]]
- [[_COMMUNITY_Eval Package Init|Eval Package Init]]
- [[_COMMUNITY_Gate Package Init|Gate Package Init]]
- [[_COMMUNITY_Probes Package Init|Probes Package Init]]
- [[_COMMUNITY_Store Package Init|Store Package Init]]
- [[_COMMUNITY_Trustgate Package Init|Trustgate Package Init]]

## God Nodes (most connected - your core abstractions)
1. `FastWeights` - 14 edges
2. `Trust Gate Mechanism` - 14 edges
3. `make_gated_inner_loop_step()` - 13 edges
4. `GateDecision` - 12 edges
5. `Updates` - 12 edges
6. `Invention Disclosure — Trust-Gated Fast-Weight Updates` - 12 edges
7. `AttackSpec` - 11 edges
8. `init_store()` - 10 edges
9. `StreamStrategy` - 9 edges
10. `accumulate()` - 9 edges

## Surprising Connections (you probably didn't know these)
- `Invention Disclosure — Trust-Gated Fast-Weight Updates` --references--> `make_gated_inner_loop_step()`  [EXTRACTED]
  docs/patent/invention-disclosure.md → src/trustgate/interceptor.py
- `test_cohens_d_requires_two_per_group()` --calls--> `cohens_d()`  [INFERRED]
  tests/test_metrics.py → src/trustgate/eval/metrics.py
- `test_cohens_d_sign_follows_direction()` --calls--> `cohens_d()`  [INFERRED]
  tests/test_metrics.py → src/trustgate/eval/metrics.py
- `test_cohens_d_zero_when_identical()` --calls--> `cohens_d()`  [INFERRED]
  tests/test_metrics.py → src/trustgate/eval/metrics.py
- `test_corruption_metric_detects_degradation()` --calls--> `corruption_metric()`  [INFERRED]
  tests/test_metrics.py → src/trustgate/eval/metrics.py

## Import Cycles
- None detected.

## Hyperedges (group relationships)
- **Gate-Signal Options A/B/C** — adr_adr_f1_gate_signal_frozen_anchor_consistency, adr_adr_f1_gate_signal_update_uncertainty, adr_adr_f1_gate_signal_influence_estimator [EXTRACTED 1.00]
- **Test-Time Poisoning Prior-Art Papers** — prior_art_attack_prior_art_2308_08505, prior_art_attack_prior_art_2410_04682, prior_art_attack_prior_art_2412_01154 [EXTRACTED 1.00]
- **Trust-Gate System (interceptor + anchor + drift + rollback)** — docs_f1_trust_gated_ttt_update_interceptor, docs_f1_trust_gated_ttt_frozen_anchor, docs_f1_trust_gated_ttt_drift_accumulator, docs_f1_trust_gated_ttt_versioned_store [EXTRACTED 1.00]

## Communities (26 total, 12 thin omitted)

### Community 0 - "Gate Signals & MedBN Baseline"
Cohesion: 0.07
Nodes (41): MedBN-analogue baseline -- robust aggregation of fast-weight updates.  Status: N, Aggregate per-example gradients robustly instead of by mean.      Args:, robust_aggregate(), anchor_consistency_gate(), probe_divergence(), Option A -- frozen-anchor consistency. THE PRIMARY GATE.  Status: NOT IMPLEMENTE, Build a gate scoring behavioural divergence from the frozen anchor.      Args:, Behavioural distance between updated and anchor models on the probe set. (+33 more)

### Community 1 - "Attack Stream Crafting"
Cohesion: 0.08
Nodes (36): craft_stream(), CraftConfig, Optimisation loop that crafts a poison stream against a fixed objective.  DEFENS, Optimise a poison stream against `spec` within the strategy's constraints., AttackSpec, Configuration for one attack run., build_benign_control(), build_select_stream() (+28 more)

### Community 2 - "Design Docs, ADRs & Experiments"
Cohesion: 0.13
Nodes (33): Experiment 000 — Reproduce TTT-E2E Baseline, Pre-registration — Phase 1 Attack Spike, Crafted-Stream Strategies (SELECT / PARAPHRASE / SOFT), DEGRADE Objective, Phase 1 Kill-Gate / Pre-registration, Crafted-Stream Poisoning Attack, TRIGGER Objective, Experiment 001 — Attack Spike (kill-gate) (+25 more)

### Community 3 - "Update Interceptor"
Cohesion: 0.12
Nodes (25): JAX Interceptor Shape (wrap class method, gate inside scan), Update Interceptor, Gate, FakeModel, FakeResult, make_original(), _model(), Interceptor behaviour, tested against lightweight stand-ins for the vendor MetaM (+17 more)

### Community 4 - "Versioned Rollback Store"
Cohesion: 0.18
Nodes (23): FastWeights, ndarray, checkpoint(), has_checkpoint(), init_store(), memory_bytes(), Versioned fast-weight store: ring-buffered checkpoints with O(1) rollback.  Impl, Whether a valid snapshot exists at that depth. Traced-safe bool. (+15 more)

### Community 5 - "Pytree Fast-Weight Ops"
Cohesion: 0.18
Nodes (19): FastWeights, Updates, test_global_norm_empty_is_zero(), test_global_norm_matches_numpy(), test_select_picks_branch(), test_static_leaf_untouched(), test_sub_then_add_roundtrips(), _is_inexact() (+11 more)

### Community 6 - "Corruption Metrics"
Cohesion: 0.16
Nodes (18): attack_success_rate(), clean_regression(), cohens_d(), corruption_metric(), Metrics for the Phase 1 kill-gate decision.  These are pure functions over array, Phase 2 metric: clean-accuracy cost of running the gate.      Positive means the, Standardised mean difference with pooled SD.      Returns 0.0 when both groups a, THE kill-gate metric.      Benign-task loss after a poison stream vs. after a le (+10 more)

### Community 7 - "Drift Accumulator"
Cohesion: 0.21
Nodes (16): accumulate(), init_drift(), Cumulative-drift accumulator with a per-window budget.  This is where the headli, Fresh accumulator with an empty window., Add one accepted update's drift contribution.      `breached` **latches**: once, Start a new window. Called after a rollback, or on window boundary.      Budget, Whether accepting `drift_delta` would exceed the budget.      Lets the policy de, reset_window() (+8 more)

### Community 8 - "Audit Log"
Cohesion: 0.22
Nodes (7): AuditLog, DecisionRecord, Audit trail for accept / reject / rollback decisions (dossier 5.1).  Two consume, One scored fast-weight update., Append-only JSONL log of gate decisions.      JSONL rather than a single JSON do, Write one entry per chunk from a scan's stacked metric arrays.          Args:, Path

### Community 9 - "Attack Objectives"
Cohesion: 0.29
Nodes (7): degrade_loss(), Objective, Attack objectives for the Phase 1 spike.  DEFENSIVE RESEARCH. Run only against m, Attacker maximises benign loss. We return its negation to minimise.      Measure, Attacker wants target behaviour on the trigger, normal behaviour elsewhere., trigger_loss(), ndarray

### Community 10 - "Go/No-Go Report"
Cohesion: 0.38
Nodes (6): Render a Phase 1 spike result against its pre-registration.  The report always s, Produce the go/no-go report., render_markdown(), write_report(), SpikeResult, Path

### Community 11 - "Probe Rotation"
Cohesion: 0.40
Nodes (4): Array, make_probe_rotation(), Rotating held-out probe sets for the anchor-consistency gate.  Status: NOT IMPLE, Build a rotation schedule over a held-out probe pool.      Returns:         Call

### Community 12 - "Uncertainty Gate (Option B)"
Cohesion: 0.50
Nodes (3): Option B -- update-uncertainty threshold. CHEAP SECONDARY.  Status: NOT IMPLEMEN, Build a gate thresholding on update uncertainty.      Args:         threshold: r, uncertainty_gate()

### Community 13 - "Test Fixtures"
Cohesion: 0.50
Nodes (3): Shared fixtures. Force JAX onto CPU before it is imported anywhere.  These tests, A minimal pytree standing in for `MetaModel.inner_parameters()`.      A dict of, tiny_fast_weights()

## Knowledge Gaps
- **13 isolated node(s):** `fetch_checkpoints.sh script`, `pre-push-guard.sh script`, `setup_vendor.sh script`, `Objective`, `Path` (+8 more)
  These have ≤1 connection - possible missing edges or undocumented components.
- **12 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `make_gated_inner_loop_step()` connect `Update Interceptor` to `Gate Signals & MedBN Baseline`, `Design Docs, ADRs & Experiments`?**
  _High betweenness centrality (0.129) - this node is a cross-community bridge._
- **Why does `Invention Disclosure — Trust-Gated Fast-Weight Updates` connect `Design Docs, ADRs & Experiments` to `Update Interceptor`?**
  _High betweenness centrality (0.087) - this node is a cross-community bridge._
- **Why does `Gate` connect `Gate Signals & MedBN Baseline` to `Update Interceptor`?**
  _High betweenness centrality (0.051) - this node is a cross-community bridge._
- **Are the 13 inferred relationships involving `FastWeights` (e.g. with `FastWeights` and `ndarray`) actually correct?**
  _`FastWeights` has 13 INFERRED edges - model-reasoned connections that need verification._
- **Are the 3 inferred relationships involving `Trust Gate Mechanism` (e.g. with `Phase 1 Kill-Gate / Pre-registration` and `MedBN — Robust TTA Defense`) actually correct?**
  _`Trust Gate Mechanism` has 3 INFERRED edges - model-reasoned connections that need verification._
- **Are the 8 inferred relationships involving `GateDecision` (e.g. with `FastWeights` and `ndarray`) actually correct?**
  _`GateDecision` has 8 INFERRED edges - model-reasoned connections that need verification._
- **Are the 11 inferred relationships involving `Updates` (e.g. with `Updates` and `FastWeights`) actually correct?**
  _`Updates` has 11 INFERRED edges - model-reasoned connections that need verification._