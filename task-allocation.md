# Project Task Allocation

> Working doc, same class as `ROADMAP.md` / `HANDOFF.md` / `TEAM_PLAN.md`. The repo is
> public — add `task-allocation.md` to `.gitignore` before the next push if this should
> not ship.
>
> **Labels.** Owners are `P1`, `P2`, `P3`. Priorities are written as `Priority P0`
> (blocks wrap-up), `Priority P1` (required for a complete deliverable), `Priority P2`
> (optional cleanup — only if time remains). Task IDs `T1.x` / `T2.x` / `T3.x` belong to
> the matching owner. References like _(TEAM_PLAN P1-4)_ point at the older plan's IDs.

---

## Current State

_Measured on `main` = `9556165`, 2026-09-14._

**Code.** Overlay package `src/trustgate/` — 29 modules across nine packages. **244 CPU
tests pass and 11 skip** with no reference model fetched (what CI sees), **255 pass** once
`scripts/fetch_reference_model.py` has run. Quote both numbers or the check is
unfalsifiable. Vendor submodule pinned at `a4fc478`, clean, never edited.

- **Implemented + tested:** `interceptor.py`, `vendor_patch.py`, `tree_ops.py`, `types.py`,
  `tokens.py`, `drift/accumulator.py`, `store/versioned.py`, `audit/log.py`, all of
  `eval/` (`carry`, `cli`, `fluency`, `gpt2_reference`, `harness`, `lm_loss`, `metrics`,
  `prereg`, `report`, `vendor_bind`), and `attack/corpus.py`, `attack/objectives.py`,
  `attack/stream.py` (SELECT), `attack/craft.py` (SELECT + DEGRADE).
- **Interface-only (`NotImplementedError`, 8 call sites outside `gate/`):**
  `build_paraphrase_stream` and `build_soft_stream` (T2.5), all of `attack/trigger.py`
  (T2.8), and `craft_stream`'s non-SELECT / non-DEGRADE branches. Each stub names its
  blocker in the message.
- **Deliberately still stubbed (Standing Rule 2):** all of `gate/`, `probes/rotating.py`,
  `baselines/medbn_analogue.py`. CI asserts they still raise, so gate work cannot start
  before the verdict by accident.
- **Implemented but unwired:** the drift accumulator and the versioned store are still
  never called by `interceptor.py`. Per-window bounded drift is designed, not enforced
  (ADR-003, 2026-08-08 correction). Unchanged since the last revision of this document.
- **Plumbing that was missing and now exists:** `python -m trustgate.eval.harness` is real
  (`eval/cli.py`, exactly the five documented flags); `.github/workflows/ci.yml` runs the
  CPU suite plus three mechanised hygiene checks on every push and PR.

**Desk prep.** Phase 0.4 (D1–D7) complete, unchanged. Since then:
`experiments/001-attack-spike/FLUENCY_REFERENCE.md` (T3.2), `docs/adr/ADR-006` (the
vendor discards its adapted fast weights) and `ADR-007` (SELECT spans are sub-chunk).

**Hardware and spend.** Still zero. No GPU box provisioned, no checkpoint fetched, no
booking row in `docs/protocols/gpu-bookings.md`, no GCS object ever egressed. The one
thing that did get settled for free: all three buckets were confirmed requester-pays by
anonymous probe (HTTP 400, "no user project provided"), recorded in `COST_MODEL.md`.
`experiments/003-smoke-125m/` prepares a $0 rehearsal of the whole vendor path on Colab
or Kaggle — written, not yet run, and not yet merged.

**Numbers that now exist.** `experiments/002-pilot-tiny-ttt/` ran a scaled-down pilot and
a null study on CPU. They are **not** the kill-gate and render no verdict; the pilot's own
README says so at the top. They do appear in the paper's Sections IV–V.

**Paper.** `docs/paper/` is tracked. `p3-intro-related-methodology.tex` carries Sections
I–III and the 35-entry bibliography; `p4-experiments-results.tex` carries Sections IV–V
around the 002 pilot; `build_main.py` splices them into `main.tex`; seven pgfplots figures
and two tables are generated into `docs/paper/figures/` by `scripts/make_figures.py`.
**Live contradiction:** Section III still states the paper reports no measurements, while
Sections IV–V report the pilot's. That is T1.12's job and is not yet done.

**Open admin.** `invention-disclosure.md` inventors field still reads `TBD` (T3.9). The
acknowledgement table in `docs/protocols/branch-and-review.md` is signed by Person 2 only;
Person 1 and Person 3 rows are blank. Pre-push guard installation is verifiable on this
clone only.

---

## Status — read off the repository, 2026-09-14

Every status below was checked against the tree at `main` = `9556165` rather than
reported by its owner. The single exception is **T1.2**, which leaves no trace in the
repository and is marked as reported, not verified.

**`DONE` does not mean merged.** Four branch tips are unmerged on `org`, each verified
conflict-free against `main` today:

| Branch | Ahead | Carries |
|---|---|---|
| `fix/seed-pairs-degenerate-arms` | +2 | the T2.6 fix — **a live bug on `main`** |
| `exp/gcs-probe-findings` | +1 | T1.9's stream scoring, and this document |
| `exp/003-smoke-125m` | +4 | the free 125M rehearsal of the vendor path |
| `docs/review-deck-refresh` | +3 | the exported review deck |

Rule 3 forbids self-merge, so all four need somebody other than their author to press the
button. PR #6 was merged at `45c23c5`, one commit before the T1.9 scoring fix landed, which
is why `exp/gcs-probe-findings` is still ahead.

**One live bug on `main`.** `generate_seed_pairs` returns byte-identical poison and control
arms — 5 of 5 pairs, re-verified today. `assert_matches` passes on them because it checks
structural match, not distinctness, so the suite is green and the defect is silent. Until
`fix/seed-pairs-degenerate-arms` merges, T2.6's output is unusable and T3.6 has nothing
meaningful to score: an attack measured against a byte-identical copy of itself cannot show
an effect, and would produce a **false STOP** for the same structural reason ADR-006
describes.

**Delivery does not match allocation.** Most of the P3 lane's shipped artefacts were
written by P1; P2 delivered one commit. The per-lane notes below record who actually
delivered what. This is stated as fact and not as a complaint: the commit timeline is the
project's conception evidence, and an allocation document that contradicts it is worse than
no document.

## P1

> ### Status note — 2026-09-14
>
> Of 14: 3 done (T1.1, T1.8, T1.9), 2 partial (T1.3, T1.12), 1 reported-not-verifiable
> (T1.2), 1 in progress (T1.13), 5 blocked (T1.5–T1.7, T1.10, T1.11), 2 not started
> (T1.4, T1.14).
>
> **Nothing in P1 is blocked on code any more.** T1.3 and T1.5 need a GCP billing project,
> not a budget — egress is priced under $2. T1.6 and T1.10 need an accelerator. Everything
> after that is blocked on those two. See `experiments/000-repro-baseline/COST_MODEL.md`.
>
> P1 also delivered most of the P3 lane and four of the nine P2 tasks; those are marked
> where they sit rather than renumbered here.

### Priority Tasks

**T1.1 — Commit the paper working tree** · `Priority P0` · depends: —

> **Status (2026-09-14):** `DONE` — merged (`a225c91`, on `main`).
> Two deliberate deviations. The branch was `exp/002-pilot-and-paper`, not the name below.
> And the second bullet — *"the tracked text claims no measurement anywhere"* — is no longer
> true: the same branch landed Section IV and the pilot figures, so `main.tex` and
> `p4-experiments-results.tex` now carry the 002 pilot's numbers while
> `p3-intro-related-methodology.tex` still declares the paper reports none. That
> contradiction is real and is **T1.12**'s explicit job to reconcile.

- Branch `docs/paper-sections-1-3`; add `docs/paper/p3-intro-related-methodology.tex` and
  `intro-related-work.tex`.
- Confirm the tracked text claims no measurement anywhere (it currently does not).
- Confirm no `results/`, `*.npy` or `.env` path slipped in with it.
- Open the PR; have P2 or P3 press merge (no self-merge).

**T1.2 — Procure the W&B entity, project and API key** _(TEAM_PLAN P0-10)_ · `Priority P0` ·
depends: —

> **Status (2026-09-14):** `REPORTED DONE` by P1 — the only task in this document that
> cannot be verified from the repository. A key in a git-ignored `.env` leaves no trace by
> design. Nothing downstream has exercised it yet.

- Create the entity and project; generate the key; store it in a git-ignored `.env`.
- Verify `wandb login` succeeds and an authenticated `api.runs()` query returns from a
  throwaway shell. `training.log_wandb=false` does not avoid this path (ADR-004 §3).
- Post the entity/project names to P2 and P3 so nobody blocks on asking.

**T1.3 — Verify the GCS paths and probe `/val`** · `Priority P0` · depends: T1.2

> **Status (2026-09-14):** `PARTIAL` — one of four sub-items complete.
> `/train/zarr.json` is in the copy list (`scripts/bootstrap_gpu_box.sh:190`) — the one
> whose absence kills a run *after* billing starts. Separately, requester-pays was
> confirmed on all three buckets for free (anonymous HTTP 400, "no user project provided"),
> which settles a standing `COST_MODEL.md` assumption but completes no sub-item. The
> remaining three — `gcloud auth login`, the `PROBE_ONLY=1` run, and the `/val` byte count —
> all need a GCP billing project. Egress is priced under $2, so this is a **billing account,
> not a budget**.

- `gcloud auth login`; export `GCP_BILLING_PROJECT`. Both buckets are requester-pays.
- `PROBE_ONLY=1 bash scripts/fetch_checkpoints.sh` — the first execution of the script and
  the first listing of any path. Metadata only, no egress.
- Record the `/val` byte count against `COST_MODEL.md` §4–5. This is the single number that
  can invalidate the cost model.
- Confirm `/train/zarr.json` is in the copy list — without it the job dies after billing
  starts (`COST_MODEL.md` §1.1).

**T1.4 — GPU session 1: provision, bootstrap, rehearse** _(TEAM_PLAN P0-1)_ · `Priority P0` ·
depends: T1.3

> **Status (2026-09-14):** `NOT STARTED` — no booking row claimed, no box provisioned.
> A $0 rehearsal of the same path is written in `experiments/003-smoke-125m/` (125M,
> random-init, `dummy_dataset`, Colab or Kaggle) and is unmerged and unrun. Running it first
> converts most of this task's first-launch failures into free ones.

- Claim booking row #1 in `docs/protocols/gpu-bookings.md` **before** the instance starts:
  owner, task ID, hour estimate, budget.
- Provision A100/H100 80GB on CUDA 12.8 / cuDNN 9.8. Run `bash scripts/bootstrap_gpu_box.sh`
  (never executed before — expect to debug it).
- Set `training.exp_dir` outside the repo tree; lower `training.global_batch_size` (eval
  batch is 128 inside a `max()` that `eval_batch_size` cannot lower).
- **Rehearse on 125M in the same session** before touching the 1B run.
- Fill the Outcome column on release.

**T1.5 — Fetch and hash the 1B DCLM+Books @8K checkpoint** _(TEAM_PLAN P0-5)_ ·
`Priority P0` · depends: T1.4

> **Status (2026-09-14):** `BLOCKED` on a GCP billing project (via T1.3), not on money.

- Full `fetch_checkpoints.sh` run with `MAX_BYTES` set from the T1.3 probe.
- sha256 manifest + byte size written to `experiments/000-repro-baseline/results/`.
- Confirm `checkpoints/` is still git-ignored and nothing was force-added.

**T1.6 — Run the unmodified vendor eval** _(TEAM_PLAN P1-1)_ · `Priority P0` · depends: T1.5

> **Status (2026-09-14):** `BLOCKED` on T1.5 and on an 80GB accelerator.

- Exactly the command in `EVAL_ENTRYPOINT.md`: `training.eval_mode=true` on `train`, base
  config `ext-1b-e2e-32K`, `seq_length=8192`.
- `trustgate.vendor_patch.is_installed()` returns `False` throughout; no `trustgate` import
  in the run.
- Save raw logs and copy `train_holdout_token_nll_loss.npy` out of the git-ignored tree.

**T1.7 — Check the baseline against the pre-registered bar** _(TEAM_PLAN P1-2, P1-3)_ ·
`Priority P0` · depends: T1.6

> **Status (2026-09-14):** `BLOCKED` on T1.6. `TOLERANCE.md` is written and the bar is
> frozen, so this task is a comparison and nothing else once the number exists.

- **Read `TOLERANCE.md` before reading the number** (Standing Rule 5).
- Check all five: band `2.314 < loss < 2.805` nats/token; monotonically falling per-token
  NLL; run-to-run determinism; `dummy_dataset` control lands far outside the band; resolved
  `dataset_name == books3` from the run's own config echo.
- Write ours, the bar, and PASS/FAIL into `experiments/000-repro-baseline/results/`, with
  checkpoint hash, vendor SHA and env versions in one rerunnable block.
- FAIL ⇒ stop and debug the environment. Nothing measured downstream is attributable until
  this passes.

**T1.8 — Implement `run_stream` and `eval_benign`** _(TEAM_PLAN P2-1)_ · `Priority P0` ·
depends: T1.7

> **Status (2026-09-14):** `DONE` — merged (`9a3356b`, on `main`).
> `eval/vendor_bind.py` transcribes the vendor's meta branch and *returns* the carry the
> vendor discards at `transformer.py:712`; `run_stream` and `eval_benign` no longer raise.
> Tested against an injected fake vendor (`tests/test_vendor_bind.py`), never against the
> real model — the binding has still never executed on hardware. That is T1.4/003's job.

- `train_mode="meta"` only; returns adapted fast weights.
- Benign loss measured on held-out data, never on the stream itself.
- Accepts the `RunCondition` produced by P3's harness half and the `CraftedStream` produced
  by P2's builders — settle both signatures with P2 and P3 before writing the bodies.

**T1.9 — Implement `craft_stream`** _(TEAM_PLAN P2-2)_ · `Priority P0` · depends: T1.8,
T2.2, T2.4, T3.2

> **Status (2026-09-14):** `DONE for SELECT + DEGRADE`; PARAPHRASE/SOFT blocked on T2.5,
> TRIGGER on T2.8. The search (`search_order`, pairwise-swap hill-climb with early stop) is
> merged on `main` (`f969348`). The fourth acceptance bullet — *returns a `CraftedStream`
> with both `perplexity` and `control_perplexity` filled* — was **not** met by that commit
> and is fixed on `exp/gcs-probe-findings`, **unmerged**: the scorer takes `(poison, control)`
> because the realism bar is a ratio, the control is built once and held fixed across the
> whole search, and the rebuilt winner is scored once at the end.

- Discrete search for SELECT/PARAPHRASE, embedding-space with projection for SOFT.
- The `fluency_weight` term sits **inside** the objective, never as a post-hoc filter.
- Runs to `max_iters` or early-stops on `early_stop_patience`; returns a `CraftedStream`
  with both `perplexity` and `control_perplexity` filled.

**T1.10 — GPU session 2: the five-seed poison and control runs** _(TEAM_PLAN P2-3)_ ·
`Priority P0` · depends: T1.9, T2.6

> **Status (2026-09-14):** `BLOCKED` on T1.7 (a validated baseline) and on T2.6, whose
> seed pairs are currently degenerate on `main`.

- Claim the booking row first. This is the longest serial block in the plan; estimate
  generously and announce overruns rather than absorbing them.
- Five usable seeds per condition, both arms. No seed dropped without a written reason in
  the booking Outcome.
- Hand raw per-seed losses to P3 as they land so scoring runs in parallel, not after.

**T1.11 — Make the PROCEED/STOP call** _(TEAM_PLAN P2-4)_ · `Priority P0` · depends: T1.10,
T3.6

> **Status (2026-09-14):** `BLOCKED` on T1.10 and T3.6. The machinery that renders the
> verdict exists and is tested (`eval/report.py`, `eval/cli.py`, thresholds pinned from
> `PREREGISTERED.md` via `eval/prereg.FROZEN`); only the numbers are missing.

- Verdict lands in `experiments/001-attack-spike/results/report.md`, generated by
  `trustgate.eval.report` — never hand-written.
- Copy `report.md` out of the git-ignored `results/` and commit it.
- Any bar missed ⇒ STOP, or a dated written revision inside `PREREGISTERED.md` explaining
  what changed. Never a silent edit. This is the moment Rule 5 exists for.

**T1.12 — Paper results and final assembly** · `Priority P0` · depends: T1.11, T2.7, T3.10

> **Status (2026-09-14):** `PARTIAL, and ahead of its dependency`. Sections IV–V exist
> (`p4-experiments-results.tex`), but they are written around the **002 pilot**, not around
> a verdict that does not yet exist — the section says so explicitly. Two of the four
> bullets are therefore still open: reconciling Section III against what actually ran, and
> writing the abstract from the result. `build_main.py` assembles `main.tex` so a stale
> splice is not possible.

- Write Section IV (Results) around the verdict, and Section V (Discussion + Conclusion).
- Reconcile Section III against what actually ran — including the sentence that currently
  promises no measurements, and any strategy that was not executed.
- Write the abstract last, from the result rather than the hypothesis.
- Merge P2's methodology reconciliation and P3's tables/figures; compile clean on stock
  IEEEtran with no external `.bib` and no image files.

**T1.13 — Integration merge to `main`** _(TEAM_PLAN P4-3)_ · `Priority P1` · depends: T1.12

> **Status (2026-09-14):** `IN PROGRESS` — 13 branches merged, 4 tips outstanding and all
> four verified conflict-free against `main` today. See the table in the status section
> above. The review checklist is partly mechanised now: `ci.yml` enforces the vendor-tree,
> secrets/checkpoints and `gate/`-stubs checks on every push.

- Review and merge every P2 and P3 branch. P1's own branches are reviewed and merged by P2
  or P3 — no self-merge in either direction.
- Per the review checklist: `git -C vendor/ttt-e2e status` clean; CPU tests green; no
  `gate/` code; no secrets, checkpoints or datasets; pre-registered bars untouched;
  unflattering numbers still present.
- Confirm the passthrough gate is still a provable no-op.

**T1.14 — Update the invention disclosure** · `Priority P1` · depends: T1.11

> **Status (2026-09-14):** `NOT STARTED` — needs T1.11. Note that §7's caveat still holds
> exactly as written: the accumulator and store remain unwired, so the enforced granularity
> is still not the designed one.

- §8: real numbers and file references, or the STOP result stated plainly.
- §7: state the drift granularity **actually enforced today** — the accumulator and store
  are still unwired. Do not restate the designed guarantee as an implemented one.
- Update _Reduction to practice_ from `TBD`.

### Definition of Done

> **Rolled up 2026-09-14:** 2 of 7 met. The three code items are done (`run_stream`, `eval_benign`, `craft_stream` for SELECT) and the paper compiles with Sections I–V, though around the 002 pilot rather than a verdict. Everything that needs a box, a bucket or a billing account is untouched: no booking rows, no baseline number, no seeds run, no `report.md`. `main` does not yet carry all three workstreams — four tips are unmerged.

- W&B credentials exist and are proven to authenticate.
- Two booking rows in `gpu-bookings.md`, each claimed before its instance started and each
  with an Outcome filled on release. Total spend inside the $325 cap.
- `experiments/000-repro-baseline/results/` holds the baseline number, the bar, a PASS, and
  a rerunnable environment block.
- `run_stream`, `eval_benign` and `craft_stream` no longer raise; five seeds ran on both
  arms.
- `experiments/001-attack-spike/results/report.md` is committed, machine-generated, and
  names PROCEED or STOP.
- The paper compiles end to end with Sections I–V and no unresolved references.
- `main` carries all three workstreams; vendor tree clean; CPU tests green; no `gate/` code.

---

## P2

> ### Status note — 2026-09-14
>
> Of 9: 4 done (T2.1–T2.4), 1 shipped-but-broken (T2.6), 1 partial (T2.9), 3 not done
> (T2.5, T2.7, T2.8).
>
> P2 delivered one commit, `344a973`, merged as PR #5: the PARAPHRASE/SOFT and TRIGGER
> interfaces, `generate_seed_pairs`, and the Person 2 acknowledgement signature. T2.1–T2.4
> were already on `main` before it, written by P1 (`bbb6dea`, `c2783ee`, `5a7bd17`).
>
> **Still open in this lane:** T2.5 (interfaces only), T2.6 (shipped but broken on `main`),
> T2.7 (not started), T2.8 (interfaces only, blocked on a victim).

All P2 work below runs on CPU with toy pytrees and needs no hardware. Start T2.1 immediately.

### Priority Tasks

**T2.1 — Corpus and tokenizer pipeline** _(TEAM_PLAN P1-4)_ · `Priority P0` · depends: —

> **Status (2026-09-14):** `DONE` — merged (`bbb6dea`, on `main`). Delivered by P1, not P2.
> `attack/corpus.py` + `tests/test_corpus.py`. Spans are sub-chunk and BOS-free per ADR-007;
> the corpus records its split name so a stream drawn from the eval split is rejected
> (`test_stream_from_the_eval_split_is_rejected`) — drawing control spans from the split the
> benign eval reads would manufacture a positive result.

- Use the vendor tokenizer at the pinned SHA. A mismatch here silently breaks length
  matching and invalidates every later comparison.
- Deterministic given a seed; exact token counts, asserted not estimated.
- CPU unit tests: same seed ⇒ byte-identical output; token count exact at several lengths.

**T2.2 — `build_select_stream`** _(TEAM_PLAN P1-5)_ · `Priority P0` · depends: T2.1

> **Status (2026-09-14):** `DONE` — merged (`c2783ee`, on `main`). Delivered by P1, not P2.
> All four bullets met: real spans only, exact `length_tokens`, byte-identical under a
> repeated seed at several lengths, and the scoring call left injectable. The last bullet
> closed on `exp/gcs-probe-findings`, where `craft_stream` now actually calls the T3.2
> scorer and fills both perplexities.

- Real sentences only, chosen and ordered — never synthesised text.
- `length_tokens` honoured exactly.
- Same seed reproduces the same stream; assert it in a test.
- Fill `perplexity` / `control_perplexity` from P3's reference model once T3.2 lands; until
  then leave the scoring call injectable.

**T2.3 — `build_benign_control`** _(TEAM_PLAN P1-6)_ · `Priority P0` · depends: T2.1

> **Status (2026-09-14):** `DONE` — merged (`c2783ee`, on `main`). Delivered by P1, not P2.
> The failure direction is tested in all three: token-count, chunk-count and valid-token
> mismatches each raise. **Caveat:** `assert_matches` checks structural match, not
> distinctness, which is exactly why T2.6's degenerate pairs pass it silently.

- Matches the poison stream in token count, chunking and dtype.
- `RunCondition.assert_matches` passes on every generated pair.
- Test the failure direction too: a deliberately mismatched pair must raise.

**T2.4 — `degrade_loss` and `trigger_loss`** _(TEAM_PLAN P1-7)_ · `Priority P0` · depends: —

> **Status (2026-09-14):** `DONE` — merged (`5a7bd17`, on `main`). Delivered by P1, not P2.
> Pure functions on toy logits, measured against held-out benign targets. The stealth term
> is tested in both directions: `test_trigger_loss_stealth_weight_penalises_benign_drift`
> and `test_trigger_loss_without_stealth_ignores_benign_degradation`.

- Pure functions, CPU-tested on toy logits.
- `degrade_loss` measured against held-out benign targets, not the stream's own tokens.
- Demonstrate in a test that the `stealth_weight` term in `trigger_loss` penalises benign
  degradation.

**T2.5 — PARAPHRASE and SOFT builders** _(TEAM_PLAN P2-5)_ · `Priority P1` · depends: T2.2

> **Status (2026-09-14):** `NOT DONE` — interfaces only (`344a973`, P2). Both builders
> raise `NotImplementedError` and each names its blocker: PARAPHRASE needs a Llama-3
> decode/encode round trip and cannot work in ID space (so it waits on T3.2's tokenizer);
> SOFT needs the victim's embedding matrix and gradients through the stream. The third
> bullet — *both return `CraftedStream` through the same interface as SELECT* — is met by
> the signatures. Neither is required for the headline result, which is SELECT.

- PARAPHRASE stays inside a semantic-similarity ball; SOFT optimises in embedding space and
  projects back to tokens.
- Both return `CraftedStream` through the same interface as SELECT.
- SOFT is labelled in code and in the report as an upper bound, never a headline.

**T2.6 — Per-seed stream and control pairs** _(TEAM_PLAN P2-6)_ · `Priority P0` · depends:
T1.9

> **Status (2026-09-14):** `SHIPPED BUT BROKEN ON MAIN` — **this is the one live defect in
> the tree.** `generate_seed_pairs` exists and is merged (`344a973`), but when no `order_fn`
> is supplied it returns a control that is byte-identical to the poison: 5 of 5 pairs,
> re-verified today. Every pair passes `assert_matches`, because matching is structural.
> The fix is on `fix/seed-pairs-degenerate-arms` (**unmerged**): a deterministic shuffle
> guaranteed non-identity, plus `assert_arms_distinguishable`. The same-seed design is
> correct and unchanged — only the `order_fn is None` path was broken.

- Five fresh poison streams and five fresh length-matched controls, seeds 0–4.
- Every pair passes `assert_matches`; the pairs are regenerable from seed alone.
- Hand off to P1 for the GPU runs and to P3 for fluency scoring in the same drop.

**T2.7 — Methodology reconciliation for the paper** · `Priority P1` · depends: T2.6

> **Status (2026-09-14):** `NOT STARTED`. Section III-C still describes all three
> strategies as instantiated and says SELECT and PARAPHRASE *"are searched by hill-climbing
> with restarts"* — PARAPHRASE is a stub, and the shipped search has no restarts. There is
> no crafted-stream pseudocode listing (the one `algorithm` environment in the file is the
> gate loop). Not blocked: the code it must describe is on `main`.

- Rewrite Section III-C (Crafted-Stream Construction) to describe what was implemented, not
  what was planned — including any strategy that was built but not run.
- Supply the crafted-stream pseudocode listing so it matches `attack/craft.py` line for
  line in structure.
- Confirm the threat-model subsections still hold against the code; flag anything that
  drifted to P1 rather than editing Section III-A silently.

**T2.8 — TRIGGER objective run** _(TEAM_PLAN P2-7)_ · `Priority P2` · depends: T1.10

> **Status (2026-09-14):** `NOT DONE` — interfaces only (`344a973`, P2). All three of
> `build_trigger_stream`, `evaluate_trigger` and `run_trigger_objective` raise, blocked on
> T1.10. `craft_stream` also refuses a TRIGGER objective, and says why: only DEGRADE has a
> scalar the ordering search can hill-climb. `Priority P2` — correctly last.

- Only if box time remains inside the cap after the five headline seeds.
- Report `attack_success_rate`; state plainly in the report that it is secondary and not
  required for PROCEED.

**T2.9 — Clone hygiene** _(TEAM_PLAN P0-7, P0-3)_ · `Priority P1` · depends: —

> **Status (2026-09-14):** `PARTIAL`. The Person 2 acknowledgement row is signed,
> ✓ 2026-09-12. Guard installation is a local `.git/hooks` change and is not verifiable from
> the repository by anyone but its owner. A `ci.yml` check that both guard scripts allow the
> same remotes — which would catch a drifted copy centrally — is written on
> `fix/seed-pairs-degenerate-arms` and is **not yet merged**.

- `cp scripts/pre-push-guard.sh .git/hooks/pre-push` on your clone; verify it passes
  `origin` silently and blocks the `Deep-Learning-130` `org` remote.
- Sign the Person 2 row of the acknowledgement table in
  `docs/protocols/branch-and-review.md`.

### Definition of Done

> **Rolled up 2026-09-14:** 2 of 6 met. `objectives.py` is clean; `stream.py` still raises in the PARAPHRASE and SOFT builders. The test suite is green and far larger than 31 (244 + 11 skipped in CI, 255 locally), and `gate/` was not touched. The seed pairs exist but are degenerate on `main`, Section III-C does not match the shipped code, and guard installation is not verifiable from here.

- `attack/stream.py` and `attack/objectives.py` raise `NotImplementedError` nowhere.
- New CPU tests cover determinism, exact token counts, pair matching (both directions), and
  the stealth term; the suite is green and larger than 31.
- Five seed-reproducible poison/control pairs exist and every one passes `assert_matches`.
- Section III-C of the paper matches the shipped code, with a pseudocode listing.
- Pre-push guard installed and verified on your clone; acknowledgement row signed.
- Nothing under `src/trustgate/gate/` was touched.

---

## P3

> ### Status note — 2026-09-14
>
> Of 12: 5 done (T3.1–T3.4, T3.10), 1 partial (T3.5), 3 blocked on hardware or on T2.6
> (T3.6–T3.8), 3 not done (T3.9, T3.11, T3.12).
>
> This lane is materially complete, but **it was not delivered by P3**. P3's commits were
> `de30eb7`, `6de6363` and `af23748` on `jay-v1`, merged as PR #3 and reverted in full by
> PR #4 (`80d5f85`). No line of them survives on `main` — including the one-line `ci.yml`
> change, which the revert undid. Everything marked `DONE` below was written by P1.
>
> Recorded because the git timeline is patent conception evidence (see
> `docs/protocols/branch-and-review.md`, rule 4) and this document must not contradict it.
>
> **Still open in this lane and owned by nobody else:** T3.5 (partial), T3.9, T3.11, T3.12.

All P3 work below runs on CPU and needs no hardware except T3.8. Start T3.1 immediately.

### Priority Tasks

**T3.1 — CI over the CPU test suite** _(TEAM_PLAN P0-6)_ · `Priority P0` · depends: —

> **Status (2026-09-14):** `DONE` — merged (`d5d5ea9`, on `main`). Delivered by P1, not P3.
> Two jobs: the CPU suite on Python 3.11 and 3.13, and three mechanised hygiene checks
> (no vendor-tree contents, no secrets/checkpoints/results, `gate/` still stubbed). The
> submodule is deliberately not checked out, per the second bullet. **Third bullet not
> evidenced:** there is no record of a deliberately broken commit being pushed to confirm
> CI fails. Cheap to do and worth doing — a green CI nobody has seen go red proves nothing.

- Add `.github/workflows/` running `PYTHONPATH=src JAX_PLATFORMS=cpu pytest` on every push
  and PR. There is no `.github/` directory today.
- Do **not** check out the vendor submodule in CI — the CPU tests do not need it and it is
  unlicensed.
- Prove it works: push a deliberately broken commit on a scratch branch, confirm CI fails,
  revert.

**T3.2 — Independent fluency reference model** _(TEAM_PLAN P1-8)_ · `Priority P0` · depends: —

> **Status (2026-09-14):** `DONE` — merged (`332c6f4`, on `main`). Delivered by P1, not P3.
> GPT-2 as the reference, justified in `experiments/001-attack-spike/FLUENCY_REFERENCE.md`,
> with `eval/gpt2_reference.py`, `scripts/fetch_reference_model.py` and a scoring function
> P2's builders can call. **Read the document before quoting it:** the acceptance check this
> task specifies — reproducing a published perplexity — does not work as written, and the
> file records why and what replaced it rather than quietly substituting a weaker check.
> The 11 tests CI skips are this model's; they pass locally once it is fetched.

- It is **not** the victim model. Scoring fluency with the model under attack is circular
  and voids the realism bar.
- Reproduce a published perplexity on a fixed text as the acceptance check.
- Write the choice, the reason, and the reproduced number into
  `experiments/001-attack-spike/`; P1 and P2 both depend on this being settled.
- Expose a scoring function P2's builders can call.

**T3.3 — Harness CLI and report generation** _(TEAM_PLAN P1-9, P1-10)_ · `Priority P0` ·
depends: —

> **Status (2026-09-14):** `DONE` — merged (`caab1fd`, on `main`). Delivered by P1, not P3.
> `python -m trustgate.eval.harness` exists with exactly the five documented flags.
> Thresholds come from `eval/prereg.FROZEN`, pinned to `PREREGISTERED.md` and never spelled
> out in the report code. The verdict is computed in the CLI layer deliberately, so
> `run_attack_spike` cannot transitively decide go/no-go. Proven end to end on a synthetic
> `SpikeResult` in `tests/test_spike_orchestration.py`.

- Add the `python -m trustgate.eval.harness` entry point with exactly the flags
  `experiments/001-attack-spike/README.md` already documents: `--objective`, `--strategy`,
  `--checkpoint`, `--seeds`, `--out`.
- Wire it to `trustgate.eval.report.write_report` with the thresholds passed in from the
  `PREREGISTERED.md` values — never hard-coded inside the report code.
- Prove it end to end on a synthetic `SpikeResult`: prints the comparison table and a
  PROCEED/STOP verdict.

**T3.4 — `RunCondition` test coverage** _(TEAM_PLAN P1-9)_ · `Priority P1` · depends: —

> **Status (2026-09-14):** `DONE` — merged (`3f1491a`, on `main`). Delivered by P1, not P3.
> Both bullets met: a parametrised per-field mismatch test that asserts the message names
> the offending field, and a matching pair that passes silently. It also closed a hole the
> pre-registration already claimed was guarded and was not.

- One test per field: any single-field mismatch raises with a readable message naming that
  field.
- A matching pair passes silently.

**T3.5 — Results archiving layout** _(TEAM_PLAN P1-11)_ · `Priority P1` · depends: T3.3

> **Status (2026-09-14):** `PARTIAL`. The `results/` convention is stated in
> `experiments/001-attack-spike/README.md` (git-ignored; copy `report.md` out; commit only
> that) and enforced by `.gitignore` plus a CI hygiene check. **Missing:** the written
> layout for per-seed losses, crafted streams, raw logs and the environment block, and the
> `git status` verification on a populated tree — which cannot be done until a run produces
> one. The third bullet matters most and is the least done: `*.npy` is git-ignored, so
> without a documented copy-out step a run can leave zero record.

- Write the directory convention down: per-seed losses, crafted streams, raw logs,
  environment block.
- `results/` stays git-ignored; only the copied-out `report.md` is committed. Verify with
  `git status` on a populated tree.
- `*.npy` is git-ignored — document the deliberate copy-out step so a run does not leave
  zero record.

**T3.6 — Fluency scoring across seeds** _(TEAM_PLAN P2-8)_ · `Priority P0` · depends: T2.6,
T3.2

> **Status (2026-09-14):** `BLOCKED` — on T2.6 (degenerate on `main`) and on a GPU run.
> The scorer itself is built and tested (T3.2), so this is data, not code.

- Per-seed `fluency_ratio` for every poison/control pair, under the reference model only.
- SELECT's ratio is the headline; PARAPHRASE and SOFT reported separately.
- Hand the ratios to P1 before the verdict — T1.11 blocks on this.

**T3.7 — Negative control** _(TEAM_PLAN P2-10)_ · `Priority P1` · depends: T1.10

> **Status (2026-09-14):** `BLOCKED` on T1.10. The closest thing that exists is the 002
> pilot's null study, which is P1's and is **not** this task: it found that at n=5 the
> `d ≥ 0.8` criterion fires on 55% of draws with no attacker present. That result makes this
> task more important, not less — it is the evidence that separates corruption from
> ordinary adaptation drift.

- Score two independent benign controls against each other.
- Cohen's _d_ well under 0.8 is the evidence that the metric measures corruption rather
  than ordinary adaptation drift. Report the number whichever way it lands.

**T3.8 — Mixed metrics-dict smoke test** _(TEAM_PLAN P0-9)_ · `Priority P1` · depends: T1.4

> **Status (2026-09-14):** `BLOCKED` on T1.4 — by design, it piggybacks on that session.
> The free 125M rehearsal in `experiments/003-smoke-125m/` can carry it at $0 instead, which
> would retire it before any paid box exists.

- Piggyback on P1's first box session — minutes of box time, not a booking of its own.
- Install the passthrough gate, run one tiny meta-mode step, confirm no vendor code breaks
  on string `gate/*` keys sitting alongside `MetricType` keys.
- Confirm the passthrough output is bit-identical to ungated **on the GPU**, not only CPU.

**T3.9 — Record inventorship** _(TEAM_PLAN P0-8)_ · `Priority P0` · depends: —

> **Status (2026-09-14):** `NOT DONE` — `docs/patent/invention-disclosure.md:18` still
> reads `TBD`. `Priority P0`, no dependencies, nothing blocking it, and the same file
> already records a public-disclosure date of 2026-08-01 that the US grace period runs
> from. This is the cheapest open P0 in the document.

- `docs/patent/invention-disclosure.md` §Inventor(s) currently reads `TBD`.
- Name all three, or explicitly exclude someone, with dates. Match the paper's author list
  or state why they differ.

**T3.10 — Paper results tables and figures** · `Priority P1` · depends: T3.6, T3.7

> **Status (2026-09-14):** `DONE` — merged (`a225c91`, on `main`). Delivered by P1, not P3.
> Seven pgfplots figures and two tables in `docs/paper/figures/`, generated by
> `scripts/make_figures.py` and `\input` by `p4-experiments-results.tex`. Image-free and
> self-contained as required. **Caveat:** they plot the 002 pilot, not the kill-gate, so
> they will be regenerated once T1.11 produces a verdict.

- Generate the results table from `results/`, in the same column shape as `tab:prereg` —
  observed next to pre-registered, per criterion.
- Per-seed scatter or bar of poison vs control benign loss, drawn in TikZ so the document
  stays image-free and self-contained.
- Hand LaTeX fragments to P1 for Section IV, not prose.

**T3.11 — Clone hygiene** _(TEAM_PLAN P0-7, P0-3)_ · `Priority P1` · depends: —

> **Status (2026-09-14):** `NOT DONE` — the Person 3 acknowledgement row is blank. So is
> Person 1's; the table is signed by Person 2 only. Rule 4 (never rewrite history) is the
> one that most needs to be acknowledged before a first push, because it is the rule that
> protects the conception evidence.

- Install and verify `pre-push-guard.sh` on your clone.
- Sign the Person 3 row of the acknowledgement table.

**T3.12 — Prior-art notes** · `Priority P2` · depends: —

> **Status (2026-09-14):** `NOT DONE` — `attack-prior-art.md` is 18 lines and
> `robust-tta-survey.md` is 16; both are still skeletons. `Priority P2`, correctly last.
> `medbn-diff.md` is untouched, as instructed.

- Only if time remains. Finish the per-paper mechanism notes in
  `docs/patent/prior-art/attack-prior-art.md` and the survey in `robust-tta-survey.md`.
- `medbn-diff.md` stays untouched — it is a filing gate needing counsel, not wrap-up work.

### Definition of Done

> **Rolled up 2026-09-14:** 4 of 9 met — CI, the fluency reference model, the harness entry point, and the Section IV tables and figures. All four were delivered by P1. Outstanding: the archiving layout (partial), per-seed fluency ratios and the negative control (blocked), inventorship (not started, unblocked), and the acknowledgement row. CI has never been observed to go red.

- CI runs the CPU suite on every push and has been proven to fail on a broken commit.
- The fluency reference model is chosen, justified in writing, and reproduces a published
  perplexity.
- `python -m trustgate.eval.harness` runs end to end on a synthetic result and emits a
  verdict report.
- Per-seed fluency ratios and the negative control are computed and delivered to P1.
- The archiving convention is written down and `results/` leaks nothing into git.
- Inventorship is recorded with dates.
- Section IV tables and figures exist as compilable LaTeX fragments.
- Pre-push guard installed and verified; acknowledgement row signed.
- Nothing under `src/trustgate/gate/` was touched.

---

## Dependencies & Parallel Execution

**Start immediately, in parallel, no cross-blocking:**

| Owner | Starts on day 1 with no dependency                                                  |
| ----- | ----------------------------------------------------------------------------------- |
| P1    | T1.1 commit the paper · T1.2 W&B key · T1.3 GCS probe                               |
| P2    | T2.1 corpus/tokenizer · T2.4 objectives · T2.9 clone hygiene                        |
| P3    | T3.1 CI · T3.2 fluency model · T3.3 harness CLI · T3.9 inventorship · T3.11 hygiene |

**The genuine dependencies — there are only six:**

| Blocked                    | Blocked by         | Why                                                                                     |
| -------------------------- | ------------------ | --------------------------------------------------------------------------------------- |
| T1.4 → T1.7 (all GPU work) | T1.2 W&B key       | The vendor login runs before the eval branch; no workaround without editing vendor code |
| T1.9 `craft_stream`        | T2.2, T2.4, T3.2   | It optimises P2's streams against P2's objectives under P3's fluency term               |
| T2.6 per-seed pairs        | T1.9               | Pairs are produced by the crafting loop                                                 |
| T3.6 fluency scoring       | T2.6, T3.2         | Needs both the streams and the reference model                                          |
| T1.11 verdict              | T1.10, T3.6        | All three bars must be in hand at once                                                  |
| T1.12 paper results        | T1.11, T2.7, T3.10 | Assembly of the other two workstreams                                                   |

**Interface contracts to settle in the first sitting**, so the three streams never block on
each other afterwards: the `CraftedStream` construction signature (P1↔P2), the
`RunCondition` field set (P1↔P3), and the fluency scoring call signature (P2↔P3). Fix them
once, in writing, before anyone writes a body.

**Hardware fallback.** If the box or the budget slips, T2._ and T3._ continue untouched and
P1 validates the whole pipeline on toy pytrees and the 125M rehearsal path, so that when
hardware arrives only the numbers are missing. Nothing in P2's or P3's lane waits on
procurement.

**Standing rules that constrain scheduling:** one box, one holder, booking claimed before
billing starts; feature branches only; the merge button is never pressed by the author.

---

## Final Integration

**Stage 1 — Freeze the verdict.** P1 runs both arms across five seeds (T1.10), P3 delivers
per-seed fluency ratios and the negative control (T3.6, T3.7), and `report.md` is generated
by `trustgate.eval.report` and copied out of `results/`. The bar is read before the number.
If a bar is missed the outcome is STOP or a dated revision inside `PREREGISTERED.md` — the
report is committed either way.

**Stage 2 — Paper.** P2 delivers the reconciled Section III-C plus pseudocode; P3 delivers
Section IV tables and TikZ figures; P1 writes Sections IV and V around them, reconciles
Section III against what actually ran, and writes the abstract last. Compiles clean on
stock IEEEtran, self-contained.

**Stage 3 — Repo consolidation.** P1 reviews and merges every branch into `main` (T1.13),
with P2 or P3 merging P1's own. Checklist per merge: vendor tree clean, CPU tests green, no
`gate/` code, no secrets or checkpoints, pre-registered bars untouched, unflattering numbers
still present. CI must be green on `main`.

**Stage 4 — Record and close.** P1 updates the invention disclosure §7/§8 to the granularity
actually enforced (T1.14). Booking rows carry filled Outcomes; total spend recorded against
the $325 cap. `README.md` status table updated to the real state.

**If the verdict is STOP**, Stages 2–4 run unchanged. The negative result is the deliverable:
the paper reports it, the disclosure records it, and the project closes there. That is a
complete outcome, not a failed one.

---

## Final Project Checklist

_Ticked against the repository, 2026-09-14. A tick means the acceptance criteria were
checked and met; it does **not** mean merged — see the status section for the four
outstanding branch tips._

- [x] `docs/paper/` committed and tracked (P1)
- [x] W&B entity, project and key exist and authenticate (P1) — **reported, not verifiable**
- [ ] GCS paths listed and `/val` size probed for free; cost model confirmed (P1) —
      requester-pays confirmed at $0; the listing and the `/val` count need a billing project
- [ ] Booking rows claimed before billing, Outcomes filled, spend inside the $325 cap (P1) —
      no rows; spend is $0
- [ ] 125M rehearsal completed before the 1B run (P1) — written in
      `experiments/003-smoke-125m/`, unmerged and unrun
- [ ] `experiments/000-repro-baseline/results/` holds the number, the bar, a PASS, and a
      rerunnable environment block (P1) — the bar exists, the number does not
- [x] `run_stream`, `eval_benign`, `craft_stream` implemented (P1) — SELECT + DEGRADE;
      PARAPHRASE, SOFT and TRIGGER refuse and name their blockers
- [ ] Five usable seeds per condition, both arms (P1)
- [ ] `experiments/001-attack-spike/results/report.md` committed, machine-generated, naming
      PROCEED or STOP (P1) — the generator is built and tested; only the numbers are missing
- [ ] Paper Sections I–V compile clean and self-contained (P1) — all five sections exist and
      are image-free, but Section III still claims the paper reports no measurements while
      Sections IV–V report the pilot's
- [ ] Everything merged to `main`; vendor tree clean; no `gate/` code (P1) — 13 branches
      merged, 4 tips outstanding; vendor tree clean; `gate/` still stubbed and CI-enforced
- [ ] Invention disclosure §7/§8 updated to what is actually enforced (P1)
- [ ] `attack/stream.py` and `attack/objectives.py` fully implemented and tested (P2) —
      `objectives.py` yes; `stream.py` still raises in the PARAPHRASE and SOFT builders
- [ ] Five seed-reproducible poison/control pairs, all passing `assert_matches` (P2) — they
      pass, and are byte-identical; `assert_matches` checks structure, not distinctness
- [ ] PARAPHRASE and SOFT built and reported separately, SOFT labelled an upper bound (P2) —
      interfaces only
- [ ] Section III-C matches the shipped code, with pseudocode (P2)
- [ ] CI running the CPU suite on every push, proven to fail on a broken commit (P3) —
      running on every push and PR; **never observed to fail**, and that is the half of this
      item that carries the evidence
- [x] Fluency reference model chosen, justified, and reproducing a published perplexity (P3)
      — GPT-2, justified in `FLUENCY_REFERENCE.md`, which also records why the specified
      acceptance check does not work and what replaced it
- [x] `python -m trustgate.eval.harness` entry point working end to end (P3)
- [ ] Per-seed fluency ratios and the control-vs-control negative control delivered (P3)
- [ ] Results archiving convention written; `results/` leaks nothing into git (P3) — the
      leak side is done and CI-enforced; the layout is not written down
- [ ] Inventorship recorded with dates (P3) — still `TBD`; unblocked and `Priority P0`
- [x] Section IV tables and TikZ figures delivered as LaTeX fragments (P3) — against the 002
      pilot, to be regenerated once a verdict exists
- [x] CPU test suite green and larger than 31 (P2, P3) — 244 passed / 11 skipped in CI,
      255 passed locally
- [ ] Pre-push guard installed on all three clones (P1, P2, P3) — verifiable on this clone
      only; a CI check that the two guard scripts allow the same remotes is written but
      unmerged (`fix/seed-pairs-degenerate-arms`)
- [ ] Acknowledgement table in `branch-and-review.md` signed by all three (P1, P2, P3) —
      Person 2 only
- [x] Repository visibility unchanged; no push to a new remote without a recorded decision
