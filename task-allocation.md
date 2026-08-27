# Project Task Allocation

> Working doc, same class as `ROADMAP.md` / `HANDOFF.md` / `TEAM_PLAN.md`. The repo is
> public — add `task-allocation.md` to `.gitignore` before the next push if this should
> not ship.
>
> **Labels.** Owners are `P1`, `P2`, `P3`. Priorities are written as `Priority P0`
> (blocks wrap-up), `Priority P1` (required for a complete deliverable), `Priority P2`
> (optional cleanup — only if time remains). Task IDs `T1.x` / `T2.x` / `T3.x` belong to
> the matching owner. References like *(TEAM_PLAN P1-4)* point at the older plan's IDs.

---

## Current State

**Code.** Overlay package `src/trustgate/` (1,517 lines). 31 CPU tests pass
(`PYTHONPATH=src JAX_PLATFORMS=cpu pytest`). Vendor submodule pinned at `a4fc478`, clean,
never edited.

- **Implemented + tested:** `interceptor.py`, `vendor_patch.py`, `tree_ops.py`, `types.py`,
  `drift/accumulator.py`, `store/versioned.py`, `eval/metrics.py`, `eval/report.py`,
  `audit/log.py`.
- **Stubbed (`NotImplementedError`, 15 call sites):** all of `attack/`, the model-touching
  half of `eval/harness.py`, all of `gate/`, `probes/rotating.py`,
  `baselines/medbn_analogue.py`.
- **Implemented but unwired:** the drift accumulator and the versioned store are never
  called by `interceptor.py`. Per-window bounded drift is designed, not enforced (ADR-003,
  2026-08-08 correction).
- **Missing plumbing:** `experiments/001-attack-spike/README.md` documents
  `python -m trustgate.eval.harness --objective ... --strategy ... --seeds ...`; no such
  entry point exists. There is no `.github/` and no CI.

**Desk prep.** Phase 0.4 (D1–D7) is complete: `EVAL_ENTRYPOINT.md` (the exact eval
command), `TOLERANCE.md` (the pre-registered bar), `COST_MODEL.md` ($325 cap, $10–35
expected), `scripts/bootstrap_gpu_box.sh`, corrected `scripts/fetch_checkpoints.sh`,
`docs/protocols/` (queue, bookings ledger, branch-and-review), reconciled disclosure
wording.

**Never done.** No GPU box provisioned. No W&B key. No checkpoint fetched. No GCS path ever
listed. No booking row in `docs/protocols/gpu-bookings.md`. Zero spend. No number produced.

**Paper.** `docs/paper/` is **untracked**. `p3-intro-related-methodology.tex` (1,633 lines)
carries Sections I–III plus a 35-entry bibliography and states explicitly that it *reports
no measurements*. `intro-related-work.tex` is the superseded earlier snapshot. Results,
discussion and conclusion do not exist.

**Open admin.** `invention-disclosure.md` inventors field reads `TBD`. The acknowledgement
table in `docs/protocols/branch-and-review.md` is unsigned by all three people. The
pre-push guard is installed on this clone only.

**Wrap-up scope (this document).** Reach the pre-registered kill-gate verdict and publish
the paper that carries it. Everything downstream of that verdict stays closed — see
*Not in scope* below.

### Not in scope for this wrap-up

- Any code in `src/trustgate/gate/`, `probes/rotating.py`, `baselines/medbn_analogue.py`.
  Standing Rule 2: no gate work before the spike returns PROCEED. Do not start it.
- Wiring the drift accumulator / versioned store into the interceptor (TEAM_PLAN P3-1).
- Operating-point sweeps, MedBN head-to-head, ROC, adaptive adversary (TEAM_PLAN Phase 4).
- Filing. `docs/patent/prior-art/medbn-diff.md` stays a filing gate, not wrap-up work.
- **Repository visibility. Nobody changes it.** Public since 2026-08-01, deliberate since
  2026-08-02. A push to a *new* remote is a separate decision that gets recorded first.

---

## P1

### Priority Tasks

**T1.1 — Commit the paper working tree** · `Priority P0` · depends: —
- Branch `docs/paper-sections-1-3`; add `docs/paper/p3-intro-related-methodology.tex` and
  `intro-related-work.tex`.
- Confirm the tracked text claims no measurement anywhere (it currently does not).
- Confirm no `results/`, `*.npy` or `.env` path slipped in with it.
- Open the PR; have P2 or P3 press merge (no self-merge).

**T1.2 — Procure the W&B entity, project and API key** *(TEAM_PLAN P0-10)* · `Priority P0` ·
depends: —
- Create the entity and project; generate the key; store it in a git-ignored `.env`.
- Verify `wandb login` succeeds and an authenticated `api.runs()` query returns from a
  throwaway shell. `training.log_wandb=false` does not avoid this path (ADR-004 §3).
- Post the entity/project names to P2 and P3 so nobody blocks on asking.

**T1.3 — Verify the GCS paths and probe `/val`** · `Priority P0` · depends: T1.2
- `gcloud auth login`; export `GCP_BILLING_PROJECT`. Both buckets are requester-pays.
- `PROBE_ONLY=1 bash scripts/fetch_checkpoints.sh` — the first execution of the script and
  the first listing of any path. Metadata only, no egress.
- Record the `/val` byte count against `COST_MODEL.md` §4–5. This is the single number that
  can invalidate the cost model.
- Confirm `/train/zarr.json` is in the copy list — without it the job dies after billing
  starts (`COST_MODEL.md` §1.1).

**T1.4 — GPU session 1: provision, bootstrap, rehearse** *(TEAM_PLAN P0-1)* · `Priority P0` ·
depends: T1.3
- Claim booking row #1 in `docs/protocols/gpu-bookings.md` **before** the instance starts:
  owner, task ID, hour estimate, budget.
- Provision A100/H100 80GB on CUDA 12.8 / cuDNN 9.8. Run `bash scripts/bootstrap_gpu_box.sh`
  (never executed before — expect to debug it).
- Set `training.exp_dir` outside the repo tree; lower `training.global_batch_size` (eval
  batch is 128 inside a `max()` that `eval_batch_size` cannot lower).
- **Rehearse on 125M in the same session** before touching the 1B run.
- Fill the Outcome column on release.

**T1.5 — Fetch and hash the 1B DCLM+Books @8K checkpoint** *(TEAM_PLAN P0-5)* ·
`Priority P0` · depends: T1.4
- Full `fetch_checkpoints.sh` run with `MAX_BYTES` set from the T1.3 probe.
- sha256 manifest + byte size written to `experiments/000-repro-baseline/results/`.
- Confirm `checkpoints/` is still git-ignored and nothing was force-added.

**T1.6 — Run the unmodified vendor eval** *(TEAM_PLAN P1-1)* · `Priority P0` · depends: T1.5
- Exactly the command in `EVAL_ENTRYPOINT.md`: `training.eval_mode=true` on `train`, base
  config `ext-1b-e2e-32K`, `seq_length=8192`.
- `trustgate.vendor_patch.is_installed()` returns `False` throughout; no `trustgate` import
  in the run.
- Save raw logs and copy `train_holdout_token_nll_loss.npy` out of the git-ignored tree.

**T1.7 — Check the baseline against the pre-registered bar** *(TEAM_PLAN P1-2, P1-3)* ·
`Priority P0` · depends: T1.6
- **Read `TOLERANCE.md` before reading the number** (Standing Rule 5).
- Check all five: band `2.314 < loss < 2.805` nats/token; monotonically falling per-token
  NLL; run-to-run determinism; `dummy_dataset` control lands far outside the band; resolved
  `dataset_name == books3` from the run's own config echo.
- Write ours, the bar, and PASS/FAIL into `experiments/000-repro-baseline/results/`, with
  checkpoint hash, vendor SHA and env versions in one rerunnable block.
- FAIL ⇒ stop and debug the environment. Nothing measured downstream is attributable until
  this passes.

**T1.8 — Implement `run_stream` and `eval_benign`** *(TEAM_PLAN P2-1)* · `Priority P0` ·
depends: T1.7
- `train_mode="meta"` only; returns adapted fast weights.
- Benign loss measured on held-out data, never on the stream itself.
- Accepts the `RunCondition` produced by P3's harness half and the `CraftedStream` produced
  by P2's builders — settle both signatures with P2 and P3 before writing the bodies.

**T1.9 — Implement `craft_stream`** *(TEAM_PLAN P2-2)* · `Priority P0` · depends: T1.8,
T2.2, T2.4, T3.2
- Discrete search for SELECT/PARAPHRASE, embedding-space with projection for SOFT.
- The `fluency_weight` term sits **inside** the objective, never as a post-hoc filter.
- Runs to `max_iters` or early-stops on `early_stop_patience`; returns a `CraftedStream`
  with both `perplexity` and `control_perplexity` filled.

**T1.10 — GPU session 2: the five-seed poison and control runs** *(TEAM_PLAN P2-3)* ·
`Priority P0` · depends: T1.9, T2.6
- Claim the booking row first. This is the longest serial block in the plan; estimate
  generously and announce overruns rather than absorbing them.
- Five usable seeds per condition, both arms. No seed dropped without a written reason in
  the booking Outcome.
- Hand raw per-seed losses to P3 as they land so scoring runs in parallel, not after.

**T1.11 — Make the PROCEED/STOP call** *(TEAM_PLAN P2-4)* · `Priority P0` · depends: T1.10,
T3.6
- Verdict lands in `experiments/001-attack-spike/results/report.md`, generated by
  `trustgate.eval.report` — never hand-written.
- Copy `report.md` out of the git-ignored `results/` and commit it.
- Any bar missed ⇒ STOP, or a dated written revision inside `PREREGISTERED.md` explaining
  what changed. Never a silent edit. This is the moment Rule 5 exists for.

**T1.12 — Paper results and final assembly** · `Priority P0` · depends: T1.11, T2.7, T3.10
- Write Section IV (Results) around the verdict, and Section V (Discussion + Conclusion).
- Reconcile Section III against what actually ran — including the sentence that currently
  promises no measurements, and any strategy that was not executed.
- Write the abstract last, from the result rather than the hypothesis.
- Merge P2's methodology reconciliation and P3's tables/figures; compile clean on stock
  IEEEtran with no external `.bib` and no image files.

**T1.13 — Integration merge to `main`** *(TEAM_PLAN P4-3)* · `Priority P1` · depends: T1.12
- Review and merge every P2 and P3 branch. P1's own branches are reviewed and merged by P2
  or P3 — no self-merge in either direction.
- Per the review checklist: `git -C vendor/ttt-e2e status` clean; CPU tests green; no
  `gate/` code; no secrets, checkpoints or datasets; pre-registered bars untouched;
  unflattering numbers still present.
- Confirm the passthrough gate is still a provable no-op.

**T1.14 — Update the invention disclosure** · `Priority P1` · depends: T1.11
- §8: real numbers and file references, or the STOP result stated plainly.
- §7: state the drift granularity **actually enforced today** — the accumulator and store
  are still unwired. Do not restate the designed guarantee as an implemented one.
- Update *Reduction to practice* from `TBD`.

### Definition of Done

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

All P2 work below runs on CPU with toy pytrees and needs no hardware. Start T2.1 immediately.

### Priority Tasks

**T2.1 — Corpus and tokenizer pipeline** *(TEAM_PLAN P1-4)* · `Priority P0` · depends: —
- Use the vendor tokenizer at the pinned SHA. A mismatch here silently breaks length
  matching and invalidates every later comparison.
- Deterministic given a seed; exact token counts, asserted not estimated.
- CPU unit tests: same seed ⇒ byte-identical output; token count exact at several lengths.

**T2.2 — `build_select_stream`** *(TEAM_PLAN P1-5)* · `Priority P0` · depends: T2.1
- Real sentences only, chosen and ordered — never synthesised text.
- `length_tokens` honoured exactly.
- Same seed reproduces the same stream; assert it in a test.
- Fill `perplexity` / `control_perplexity` from P3's reference model once T3.2 lands; until
  then leave the scoring call injectable.

**T2.3 — `build_benign_control`** *(TEAM_PLAN P1-6)* · `Priority P0` · depends: T2.1
- Matches the poison stream in token count, chunking and dtype.
- `RunCondition.assert_matches` passes on every generated pair.
- Test the failure direction too: a deliberately mismatched pair must raise.

**T2.4 — `degrade_loss` and `trigger_loss`** *(TEAM_PLAN P1-7)* · `Priority P0` · depends: —
- Pure functions, CPU-tested on toy logits.
- `degrade_loss` measured against held-out benign targets, not the stream's own tokens.
- Demonstrate in a test that the `stealth_weight` term in `trigger_loss` penalises benign
  degradation.

**T2.5 — PARAPHRASE and SOFT builders** *(TEAM_PLAN P2-5)* · `Priority P1` · depends: T2.2
- PARAPHRASE stays inside a semantic-similarity ball; SOFT optimises in embedding space and
  projects back to tokens.
- Both return `CraftedStream` through the same interface as SELECT.
- SOFT is labelled in code and in the report as an upper bound, never a headline.

**T2.6 — Per-seed stream and control pairs** *(TEAM_PLAN P2-6)* · `Priority P0` · depends:
T1.9
- Five fresh poison streams and five fresh length-matched controls, seeds 0–4.
- Every pair passes `assert_matches`; the pairs are regenerable from seed alone.
- Hand off to P1 for the GPU runs and to P3 for fluency scoring in the same drop.

**T2.7 — Methodology reconciliation for the paper** · `Priority P1` · depends: T2.6
- Rewrite Section III-C (Crafted-Stream Construction) to describe what was implemented, not
  what was planned — including any strategy that was built but not run.
- Supply the crafted-stream pseudocode listing so it matches `attack/craft.py` line for
  line in structure.
- Confirm the threat-model subsections still hold against the code; flag anything that
  drifted to P1 rather than editing Section III-A silently.

**T2.8 — TRIGGER objective run** *(TEAM_PLAN P2-7)* · `Priority P2` · depends: T1.10
- Only if box time remains inside the cap after the five headline seeds.
- Report `attack_success_rate`; state plainly in the report that it is secondary and not
  required for PROCEED.

**T2.9 — Clone hygiene** *(TEAM_PLAN P0-7, P0-3)* · `Priority P1` · depends: —
- `cp scripts/pre-push-guard.sh .git/hooks/pre-push` on your clone; verify it passes
  `origin` silently and blocks the `Deep-Learning-130` `org` remote.
- Sign the Person 2 row of the acknowledgement table in
  `docs/protocols/branch-and-review.md`.

### Definition of Done

- `attack/stream.py` and `attack/objectives.py` raise `NotImplementedError` nowhere.
- New CPU tests cover determinism, exact token counts, pair matching (both directions), and
  the stealth term; the suite is green and larger than 31.
- Five seed-reproducible poison/control pairs exist and every one passes `assert_matches`.
- Section III-C of the paper matches the shipped code, with a pseudocode listing.
- Pre-push guard installed and verified on your clone; acknowledgement row signed.
- Nothing under `src/trustgate/gate/` was touched.

---

## P3

All P3 work below runs on CPU and needs no hardware except T3.8. Start T3.1 immediately.

### Priority Tasks

**T3.1 — CI over the CPU test suite** *(TEAM_PLAN P0-6)* · `Priority P0` · depends: —
- Add `.github/workflows/` running `PYTHONPATH=src JAX_PLATFORMS=cpu pytest` on every push
  and PR. There is no `.github/` directory today.
- Do **not** check out the vendor submodule in CI — the CPU tests do not need it and it is
  unlicensed.
- Prove it works: push a deliberately broken commit on a scratch branch, confirm CI fails,
  revert.

**T3.2 — Independent fluency reference model** *(TEAM_PLAN P1-8)* · `Priority P0` · depends: —
- It is **not** the victim model. Scoring fluency with the model under attack is circular
  and voids the realism bar.
- Reproduce a published perplexity on a fixed text as the acceptance check.
- Write the choice, the reason, and the reproduced number into
  `experiments/001-attack-spike/`; P1 and P2 both depend on this being settled.
- Expose a scoring function P2's builders can call.

**T3.3 — Harness CLI and report generation** *(TEAM_PLAN P1-9, P1-10)* · `Priority P0` ·
depends: —
- Add the `python -m trustgate.eval.harness` entry point with exactly the flags
  `experiments/001-attack-spike/README.md` already documents: `--objective`, `--strategy`,
  `--checkpoint`, `--seeds`, `--out`.
- Wire it to `trustgate.eval.report.write_report` with the thresholds passed in from the
  `PREREGISTERED.md` values — never hard-coded inside the report code.
- Prove it end to end on a synthetic `SpikeResult`: prints the comparison table and a
  PROCEED/STOP verdict.

**T3.4 — `RunCondition` test coverage** *(TEAM_PLAN P1-9)* · `Priority P1` · depends: —
- One test per field: any single-field mismatch raises with a readable message naming that
  field.
- A matching pair passes silently.

**T3.5 — Results archiving layout** *(TEAM_PLAN P1-11)* · `Priority P1` · depends: T3.3
- Write the directory convention down: per-seed losses, crafted streams, raw logs,
  environment block.
- `results/` stays git-ignored; only the copied-out `report.md` is committed. Verify with
  `git status` on a populated tree.
- `*.npy` is git-ignored — document the deliberate copy-out step so a run does not leave
  zero record.

**T3.6 — Fluency scoring across seeds** *(TEAM_PLAN P2-8)* · `Priority P0` · depends: T2.6,
T3.2
- Per-seed `fluency_ratio` for every poison/control pair, under the reference model only.
- SELECT's ratio is the headline; PARAPHRASE and SOFT reported separately.
- Hand the ratios to P1 before the verdict — T1.11 blocks on this.

**T3.7 — Negative control** *(TEAM_PLAN P2-10)* · `Priority P1` · depends: T1.10
- Score two independent benign controls against each other.
- Cohen's *d* well under 0.8 is the evidence that the metric measures corruption rather
  than ordinary adaptation drift. Report the number whichever way it lands.

**T3.8 — Mixed metrics-dict smoke test** *(TEAM_PLAN P0-9)* · `Priority P1` · depends: T1.4
- Piggyback on P1's first box session — minutes of box time, not a booking of its own.
- Install the passthrough gate, run one tiny meta-mode step, confirm no vendor code breaks
  on string `gate/*` keys sitting alongside `MetricType` keys.
- Confirm the passthrough output is bit-identical to ungated **on the GPU**, not only CPU.

**T3.9 — Record inventorship** *(TEAM_PLAN P0-8)* · `Priority P0` · depends: —
- `docs/patent/invention-disclosure.md` §Inventor(s) currently reads `TBD`.
- Name all three, or explicitly exclude someone, with dates. Match the paper's author list
  or state why they differ.

**T3.10 — Paper results tables and figures** · `Priority P1` · depends: T3.6, T3.7
- Generate the results table from `results/`, in the same column shape as `tab:prereg` —
  observed next to pre-registered, per criterion.
- Per-seed scatter or bar of poison vs control benign loss, drawn in TikZ so the document
  stays image-free and self-contained.
- Hand LaTeX fragments to P1 for Section IV, not prose.

**T3.11 — Clone hygiene** *(TEAM_PLAN P0-7, P0-3)* · `Priority P1` · depends: —
- Install and verify `pre-push-guard.sh` on your clone.
- Sign the Person 3 row of the acknowledgement table.

**T3.12 — Prior-art notes** · `Priority P2` · depends: —
- Only if time remains. Finish the per-paper mechanism notes in
  `docs/patent/prior-art/attack-prior-art.md` and the survey in `robust-tta-survey.md`.
- `medbn-diff.md` stays untouched — it is a filing gate needing counsel, not wrap-up work.

### Definition of Done

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

| Owner | Starts on day 1 with no dependency |
|---|---|
| P1 | T1.1 commit the paper · T1.2 W&B key · T1.3 GCS probe |
| P2 | T2.1 corpus/tokenizer · T2.4 objectives · T2.9 clone hygiene |
| P3 | T3.1 CI · T3.2 fluency model · T3.3 harness CLI · T3.9 inventorship · T3.11 hygiene |

**The genuine dependencies — there are only six:**

| Blocked | Blocked by | Why |
|---|---|---|
| T1.4 → T1.7 (all GPU work) | T1.2 W&B key | The vendor login runs before the eval branch; no workaround without editing vendor code |
| T1.9 `craft_stream` | T2.2, T2.4, T3.2 | It optimises P2's streams against P2's objectives under P3's fluency term |
| T2.6 per-seed pairs | T1.9 | Pairs are produced by the crafting loop |
| T3.6 fluency scoring | T2.6, T3.2 | Needs both the streams and the reference model |
| T1.11 verdict | T1.10, T3.6 | All three bars must be in hand at once |
| T1.12 paper results | T1.11, T2.7, T3.10 | Assembly of the other two workstreams |

**Interface contracts to settle in the first sitting**, so the three streams never block on
each other afterwards: the `CraftedStream` construction signature (P1↔P2), the
`RunCondition` field set (P1↔P3), and the fluency scoring call signature (P2↔P3). Fix them
once, in writing, before anyone writes a body.

**Hardware fallback.** If the box or the budget slips, T2.* and T3.* continue untouched and
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

- [ ] `docs/paper/` committed and tracked (P1)
- [ ] W&B entity, project and key exist and authenticate (P1)
- [ ] GCS paths listed and `/val` size probed for free; cost model confirmed (P1)
- [ ] Booking rows claimed before billing, Outcomes filled, spend inside the $325 cap (P1)
- [ ] 125M rehearsal completed before the 1B run (P1)
- [ ] `experiments/000-repro-baseline/results/` holds the number, the bar, a PASS, and a
      rerunnable environment block (P1)
- [ ] `run_stream`, `eval_benign`, `craft_stream` implemented (P1)
- [ ] Five usable seeds per condition, both arms (P1)
- [ ] `experiments/001-attack-spike/results/report.md` committed, machine-generated, naming
      PROCEED or STOP (P1)
- [ ] Paper Sections I–V compile clean and self-contained (P1)
- [ ] Everything merged to `main`; vendor tree clean; no `gate/` code (P1)
- [ ] Invention disclosure §7/§8 updated to what is actually enforced (P1)
- [ ] `attack/stream.py` and `attack/objectives.py` fully implemented and tested (P2)
- [ ] Five seed-reproducible poison/control pairs, all passing `assert_matches` (P2)
- [ ] PARAPHRASE and SOFT built and reported separately, SOFT labelled an upper bound (P2)
- [ ] Section III-C matches the shipped code, with pseudocode (P2)
- [ ] CI running the CPU suite on every push, proven to fail on a broken commit (P3)
- [ ] Fluency reference model chosen, justified, and reproducing a published perplexity (P3)
- [ ] `python -m trustgate.eval.harness` entry point working end to end (P3)
- [ ] Per-seed fluency ratios and the control-vs-control negative control delivered (P3)
- [ ] Results archiving convention written; `results/` leaks nothing into git (P3)
- [ ] Inventorship recorded with dates (P3)
- [ ] Section IV tables and TikZ figures delivered as LaTeX fragments (P3)
- [ ] CPU test suite green and larger than 31 (P2, P3)
- [ ] Pre-push guard installed on all three clones (P1, P2, P3)
- [ ] Acknowledgement table in `branch-and-review.md` signed by all three (P1, P2, P3)
- [ ] Repository visibility unchanged; no push to a new remote without a recorded decision
