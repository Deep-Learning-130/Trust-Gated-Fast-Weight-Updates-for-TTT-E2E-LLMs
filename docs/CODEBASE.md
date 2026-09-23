# TrustGate — the codebase

Every module, script, experiment and document: what it is and where. The second of three
documents — see [`PROJECT.md`](PROJECT.md) for the idea and
[`OPERATIONS.md`](OPERATIONS.md) for how to run and publish it.

Every Python file in `src/` and `scripts/` carries a header docstring giving its reason to
exist, not just its behaviour. This file is the index; the docstring is the detail.

---

## 1. Layout

```
src/trustgate/      the library — 35 modules, installed from src/
scripts/            operational scripts — 27 files + 2 notebooks
experiments/        000, 001, 002, 003 — one directory per experiment
tests/              32 test files, 473 tests
docs/               these three files, ORIENTATION, ADRs, paper, patent, protocols
docs/paper/         the IEEE paper and its build chain
vendor/ttt-e2e/     pinned upstream submodule — unmodified, never forked (ADR-002)
```

The dependency runs one way: `scripts/` uses `src/trustgate`, never the reverse.

---

## 2. `src/trustgate` — the library

### The enforcement path

A proposed fast-weight update arrives and something decides accept, reject or rollback.
These four files are that path.

| Module | Role |
|---|---|
| `interceptor.py` | **The Update Interceptor** — the enforcement point of dossier §5.2. Start here. |
| `gate/anchor.py` | **Option A, frozen-anchor consistency. THE PRIMARY GATE.** |
| `gate/policy.py` | Decision policy — combines gate signals into accept / reject / rollback. |
| `store/versioned.py` | Versioned fast-weight store: ring-buffered checkpoints with O(1) rollback. |

Supporting signals and state:

| Module | Role |
|---|---|
| `gate/uncertainty.py` | Option B — update-uncertainty threshold. Cheap secondary. |
| `gate/influence.py` | Option C — influence / gradient estimate. **Offline auditor only** (ADR-F1). |
| `drift/accumulator.py` | Cumulative-drift accumulator with a per-window budget. |
| `probes/rotating.py` | Rotating held-out probe sets the anchor gate scores against. |
| `audit/log.py` | Audit trail for accept / reject / rollback decisions (dossier §5.1). |

> **Scope.** Phase 2 wired anchor, uncertainty, policy and probe rotation. The
> drift/rollback half is implemented but **not enforced end to end** — `ADR-P3-1` and the
> Phase 2 scope row of the pre-registration.

### Plumbing

| Module | Role |
|---|---|
| `types.py` | Core types for the trust gate. |
| `tree_ops.py` | Pytree arithmetic over fast weights. |
| `tokens.py` | Facts about the token space, recovered from the vendor's configs. |
| `vendor_patch.py` | Installs the gate into the **unmodified** vendor tree (ADR-002: overlay, not fork). |

### `attack/` — the Phase 1 adversary

| Module | Role |
|---|---|
| `corpus.py` | Benign token corpus and span sampling for crafted streams. |
| `stream.py` | Crafted-stream construction for the Phase 1 attack spike. |
| `craft.py` | Optimisation loop that crafts a poison stream against a fixed objective. |
| `objectives.py` | Attack objectives — the attacker optimises the victim's own metric. |
| `trigger.py` | TRIGGER objective: implant a latent association. |

Span granularity is settled by `ADR-007`: SELECT spans are sub-chunk, not sentences and not
inner-loop chunks.

### `eval/` — measurement

Entry point is `cli.py`; everything else is called by it.

| Module | Role |
|---|---|
| `cli.py` | CLI behind `python -m trustgate.eval.harness` — the Phase 1 spike entry point. |
| `harness.py` | Run poison and control streams under matched conditions. |
| `ledger.py` | Crash-safe record of finished evaluations, so a rerun **resumes rather than restarts**. |
| `model_build.py` | Build a real vendor `MetaModel` from the vendor's own hydra config. |
| `vendor_bind.py` | Bind `eval.carry` to the real vendor `MetaModel`. |
| `carry.py` | Carry adapted fast weights across chunks *and* across sequences (ADR-006). |
| `attacker.py` | The seam between `attack.craft` and a live victim. |
| `lm_loss.py` | The victim's language-model loss, transcribed from the vendor. |
| `metrics.py` | Metrics for the Phase 1 kill-gate decision. |
| `prereg.py` | **The frozen decision thresholds**, plus a check that they still match the doc. |
| `report.py` | Render a Phase 1 spike result against its pre-registration. |
| `sequence.py` | Sequence-position evaluation: where in a stream the damage appears. |
| `fluency.py` | Fluency scoring under an independent reference model. |
| `gpt2_reference.py` | GPT-2's forward pass, on the jax stack the project already has. |
| `overhead.py` | Measure what the gate costs, against ADR-F1 §5.1's ~10% budget. |
| `gate_eval.py` | Phase 2 measurement: run the anchor gate against the real vendor step. |

`prereg.py` is **frozen**; a CI test keeps it equal to `PREREGISTERED.md`.

### `baselines/`

| Module | Role |
|---|---|
| `medbn_analogue.py` | MedBN-analogue baseline — robust aggregation of fast-weight updates. A documented stub: it needs per-example gradients the vendor's `inner_loop_step` does not produce. |

---

## 3. `scripts/` — operations

### GPU session lifecycle

Run in this order; driven by `docs/protocols/gpu-session-1-runbook.md`.

| Script | What it does |
|---|---|
| `setup_vendor.sh` | Initialise the vendored TTT-E2E submodule at its pinned SHA and install its deps. |
| `bootstrap_gpu_box.sh` | Cold-start a fresh GPU box: nothing → ready to run the baseline eval. |
| `prepare_phase1.sh` | Prepare the box for runbook C2–C5 (001 spike, sequence arms, Phase 2 gate). |
| `run_gpu_session.sh` | Run GPU session 1 end to end, **unattended**, after bootstrap succeeded. |
| `collect_results.py` | Copy a session's artefacts into the repo, **with every secret redacted**. |

`run_gpu_session.sh` is crash-safe: rerunning the same command resumes from the ledger
rather than restarting.

### Checkpoints, data and tokens

| Script | What it does |
|---|---|
| `probe_gcs_access.sh` | One-command GCS probe. Settles every open question in task-allocation T1.3. |
| `fetch_checkpoints.sh` | Fetch a released TTT-E2E checkpoint from GCS and fingerprint it. |
| `fingerprint_checkpoint.sh` | Fingerprint a checkpoint on local disk, or verify against a recorded one. |
| `make_dummy_checkpoint.py` | Write a small orbax checkpoint so the loading path can be rehearsed **for free**. |
| `make_val_subset.py` | Build a TRUNCATED local copy of the books3 `/val` array, recording what was cut. |
| `dump_tokens.py` | Dump the `.npy` token files the 001 spike and the Phase 2 gate measurement read. |

**Notebooks** — the routes opened after the funded ones closed:

| Notebook | What it does |
|---|---|
| `colab/fetch_checkpoint_colab.ipynb` | Fetch the checkpoint from a Colab runtime — needs no service-account key. |
| `kaggle/ttt_session_kaggle.ipynb` | Run the session on Kaggle's accelerators. |

### Acceptance checks — run before trusting a number

| Script | What it checks |
|---|---|
| `check_baseline_acceptance.py` | Scores a collected baseline session against `TOLERANCE.md`, bar by bar. |
| `fetch_reference_model.py` | Fetch the independent fluency reference model and the Llama-3 decoder. |
| `verify_gpt2_equivalence.py` | Primary acceptance check for the reference model. |
| `acceptance_fluency_reference.py` | Acceptance check for the fluency reference model (T3.2). |
| `preflight_wandb.py` | Prove the W&B credentials work the way the vendor uses them — on a laptop, free. |

### Paper and decks

| Script | Output |
|---|---|
| `make_figures.py` | The paper's tables and figures → `docs/paper/figures/`, both `.tex` and `.png`. |
| `make_results_deck.py` | The results deck: what has been done, stated plainly. |
| `make_review_deck.py` | The project review deck, in the existing deck's design language. |
| `update_review_deck.py` | Update the Gamma-authored review deck: text only, plus watermark removal. |
| `make_update_deck.py` | The 2026-09-22 project-update deck. Reads its numbers live from the result files. |

### Repo safety

| Script | What it does |
|---|---|
| `pre-push-guard.sh` | Pre-push guard: make pushing to a **new** remote a deliberate act. |
| `pre-push-guard.ps1` | The same guard, PowerShell, for Windows. |
| `pre-push-hook-wrapper.sh` | Windows shim: Git Bash hook → the PowerShell guard. |

### Demo

| Script | What it does |
|---|---|
| `demo.sh` | A live demo of experiment 002, for a room. About 40 seconds, CPU only. |

---

## 4. `experiments/`

| Directory | Contents |
|---|---|
| `000-repro-baseline/` | `TOLERANCE.md` (the two-sided bracket, source of the paper's Table I), `COST_MODEL.md`, `EVAL_ENTRYPOINT.md`, `README.md` |
| `001-attack-spike/` | **`PREREGISTERED.md`** (frozen 2026-07-28), `FLUENCY_REFERENCE.md`, `README.md`. **No `results/` — that absence is the honest state.** |
| `002-pilot-tiny-ttt/` | `pilot_model.py`, `run_pilot.py`, `run_deep.py`, `run_null.py`, `run_sequence.py`, `corpus.txt` (615,141 bytes, SHA-256 pinned, marked binary in `.gitattributes`), `README.md` |
| `003-smoke-125m/` | `README.md` and the smoke runner |

`experiments/*/results/` is **gitignored** — see `OPERATIONS.md` §5 for what that has cost.

---

## 5. `tests/`

32 files, 473 tests, CPU-only, run in CI.

`test_accumulator` · `test_attacker` · `test_baseline_acceptance` · `test_carry` ·
`test_cli` · `test_cli_end_to_end` · `test_collect_results` · `test_corpus` · `test_craft` ·
`test_dump_tokens` · `test_fluency` · `test_fluency_reference` · `test_gate` ·
`test_gate_eval` · `test_harness` · `test_interceptor` · `test_lm_loss` · `test_metrics` ·
and the rest.

The one worth knowing about: a test parses `PREREGISTERED.md` and asserts the thresholds in
`prereg.py` still equal it. The bars cannot drift from the document that froze them.

---

## 6. `docs/`

### These three

`PROJECT.md` · `CODEBASE.md` · `OPERATIONS.md`

### Long form

| Document | What it is |
|---|---|
| `ORIENTATION.md` | 747 lines, written for a first-time reader who knows what an LLM is and nothing else. Full glossary. |
| `F1-trust-gated-ttt.md` | The original dossier. §5.1 is the audit trail, §5.2 the interceptor. |
| `literature-survey.md` | The survey behind the paper's 35 references. |
| `report.md` | Project report (stale — 2026-08-27). |
| `task-allocation.md` *(repo root)* | All 35 tasks with status (stale — 2026-09-14). |

### Decisions — `docs/adr/`

| ADR | Decision |
|---|---|
| `ADR-002` | Overlay vs fork — the vendor tree is never modified. |
| `ADR-003` | JAX interceptor shape (carries a correction: bounded drift is not enforced). |
| `ADR-004` | Baseline eval invocation. |
| `ADR-005` | Baseline comparison basis — why the bar is a bracket, not a published number. |
| `ADR-006` | Fast-weight carry — the *structurally guaranteed false STOP*. |
| `ADR-007` | SELECT span granularity. |
| `ADR-F1` | Gate signal — keeps influence offline, sets the ~10% overhead budget. |
| `ADR-P3-1` | Drift carry. |

### Protocols — `docs/protocols/`

`gpu-session-1-runbook.md` (the operating procedure) · `gpu-bookings.md` (plan, owners,
budget) · `gpu-queue.md` · `branch-and-review.md`

### Patent — `docs/patent/`

`invention-disclosure.md` and prior art: `attack-prior-art.md`, `medbn-diff.md`,
`robust-tta-survey.md`.

### Paper — `docs/paper/`

| File | What it is |
|---|---|
| `main-overleaf.tex` | **Self-contained.** Paste into Overleaf, compile with pdfLaTeX. No `.bib`, no image files. |
| `main.tex` | Same document, `\input`ing the figures. |
| `p3-intro-related-methodology.tex` | Source: Sections I–III + the 35-entry bibliography. |
| `p4-experiments-results.tex` | Source: Sections IV–V. |
| `figures/` | 7 figures + 2 tables, each as `.tex` (pgfplots, inline coordinates) and `.png`. |
| `build_main.py` → `make_standalone.py` → `check_main.py` | The build chain. |

`figures/fig7_null_distribution.tex` is the **one hand-maintained figure** — its header
claims `make_figures.py` generates it and that is not true.

### Reviews — `docs/review/`

Decks from 2026-08-27 onward, plus `TrustGate_Review_Transcript.md`. Newest:
`TrustGate_Update_2026-09-22_v2.pptx`.

---

## 7. Local-only, by design

Gitignored. Absence from the remote is correct, not an oversight.

| Path | Rule |
|---|---|
| `ROADMAP.md`, `HANDOFF.md`, `TEAM_PLAN.md` | `.gitignore:42-44` — local working docs |
| `CONTINUATION.local.md` | `.gitignore:39` — `*.local.*` |
| `experiments/*/results/` | `.gitignore:16` — **all result JSON** |
| `checkpoints/`, `*.safetensors`, `reference-models/` | size |
| `.env`, `*sa-key*.json`, `*service-account*.json` | secrets — the repo is public |

The `*.local.*` rule has a story attached, recorded in `.gitignore` itself: `*.local` does
**not** match `CONTINUATION.local.md`. The file claimed to be ignored for weeks and was
merely untracked — one `git add -A` from a public repo.
