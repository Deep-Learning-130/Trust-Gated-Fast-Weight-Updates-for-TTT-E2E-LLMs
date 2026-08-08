# Design — Phase 0 Lead Prep Package

> Written 2026-08-03. Owner: Lead (Person 1).
> Scope: TEAM_PLAN.md Phase 0, Lead-owned tasks only. No GPU, no spend.

## Purpose

Reach the first GPU booking with no open question whose answer would change what we
rent, how long we hold it, or whether the number it produces counts.

Today there is no GPU box and none has been procured. Everything in TEAM_PLAN.md past
the scaffold is blocked on hardware — but a meaningful share of the risk in that first
booking is resolvable at a desk, for nothing. This package is that share.

## Context

The Lead owns P0-1 (provision the box), P0-2 (GPU queue protocol), P0-3 (branch and
review protocol), and then holds the GPU alone for the entire baseline reproduction
(P1-1 through P1-3). The baseline is a gate: until the vendor's published numbers
reproduce on our hardware, any corruption measured in experiment 001 is unattributable.

## Findings that motivated this package

Three unknowns were found by reading `vendor/ttt-e2e/README.md`, which is already
checked out locally. Each would otherwise have surfaced while paying for a GPU.

1. **`scripts/fetch_checkpoints.sh` is wrong in shape, not only in its placeholder.**
   It builds `gs://${TTT_BUCKET}/${SIZE}`. The real bucket is `gs://ttt-e2e-checkpoints/`
   with flat named directories. The 1B DCLM+Books @8K checkpoint is
   `gs://ttt-e2e-checkpoints/1b_ttt_e2e_finetune_books_8k_1x_cc`.
2. **No eval entry point is documented.** The vendor README shows only
   `uv run --exact train`. `experiments/000-repro-baseline/README.md` step 3 assumes an
   eval entry point exists. Nobody has established what that command is.
3. **Egress is larger than the plan assumes.** The dataloader needs `deploy_paths`
   pointing at `gs://llama3-dclm-filter-8k/` and `gs://llama3-books3/` — separate
   buckets, also requester-pays. Weights & Biases credentials
   (`training.wandb_entity`, `training.wandb_project`, `training.wandb_key`) are
   mandatory for any launch.

A 125M checkpoint also exists (`125m_ttt_e2e_finetune_books_8k_1x_cc`). It is not used
in this package, but it is the natural cheap-rehearsal target for the first dollar spent
and is recorded here so that option stays visible.

## Deliverables

### D1 — Resolve the eval entry point

Read `vendor/ttt-e2e/ttt/train.py`, `vendor/ttt-e2e/configs/experiment/`, and
`vendor/ttt-e2e/pyproject.toml` (`[project.scripts]`) to determine whether eval exists
as a separate entry point, a mode on `train`, a Hydra config, or not at all.

Output: `experiments/000-repro-baseline/EVAL_ENTRYPOINT.md` naming the exact command
and the config overrides it needs, with file-and-line citations into vendor source.

If no eval path exists, that document must say so plainly. P1-1 then means "write an
eval harness against the vendor's model", which is a scope change to the critical path
and must be escalated before any further work in this package.

**This is a hard checkpoint.** D4 and D5 depend on its answer. Report the result before
building either.

### D2 — Pre-write the P1-2 tolerance

Rule 5 of TEAM_PLAN.md: pre-registered thresholds do not move after a result is seen.
The baseline tolerance therefore has to exist before the hardware does.

Source: the TTT-E2E paper at `https://test-time-training.github.io/e2e.pdf`, linked from
the vendor README. It is not in `docs/literature-survey.md`. Extract the reported
long-context loss and/or perplexity for the 1B DCLM+Books @8K configuration.

Output: `experiments/000-repro-baseline/TOLERANCE.md`, dated, recording the paper's
number, the table or figure it came from, the tolerance, and the reasoning behind that
tolerance. The reasoning matters more than the number — it is what makes a later PASS
or FAIL defensible.

If the paper does not report a directly comparable number for this configuration, the
document states that and defines the closest defensible comparison instead. It does not
silently substitute a different configuration's number.

### D3 — Correct `scripts/fetch_checkpoints.sh`

- `BUCKET` defaults to `gs://ttt-e2e-checkpoints`.
- Replace `SIZE` with `CKPT`, defaulting to `1b_ttt_e2e_finetune_books_8k_1x_cc`.
- Add a metadata-only size probe (`gsutil -u "$GCP_BILLING_PROJECT" du -s`) that runs
  and prints before any transfer, so byte count is known before egress is paid.
- Record sha256 of the fetched checkpoint into
  `experiments/000-repro-baseline/results/`, satisfying P0-5.
- Keep `checkpoints/` git-ignored.

Writing the script needs nothing. *Verifying* the paths resolve needs `gcloud` authed
against a billing project; listing and `du` are metadata operations and effectively
free, but the auth setup is a prerequisite and is called out as such.

### D4 — Cost model and budget cap

Sum checkpoint bytes (from D3's probe), eval-dataset bytes for the buckets D1 shows the
eval path actually touches, and hourly GPU rate times estimated hold time. Price the 1B
run as the plan of record and the 3B/128k run alongside it, so a later scope change is
priced rather than discovered.

Output: `experiments/000-repro-baseline/COST_MODEL.md`, ending in a written cap and the
assumptions it rests on.

### D5 — Cold-start bootstrap script

One idempotent script taking a fresh box from nothing to: vendor environment installed,
`python -c "import ttt"` succeeding, submodule verified at `a4fc478`, checkpoint on disk,
hash recorded, all output logged to a file.

`set -euo pipefail`. Safe to re-run after a partial failure.

**Known limit:** this is the only deliverable that cannot be validated without hardware.
It can be shellchecked and dry-run, but its real test is the first booking. Its value is
that the first session is scripted rather than improvised.

### D6 — P0-2 and P0-3 protocols

Two short documents, both Phase 0 acceptance criteria:

- GPU queue: how the box is booked, who holds the calendar, the rule that two people
  never hold it at once, and that every GPU task carries a named owner and an hour
  estimate.
- Branch and review: feature branches, Lead merges to `main`, no self-merge, no history
  rewrite. TEAM_PLAN.md Rule 4 exists because the commit timeline is conception
  evidence.

### D7 — Reconcile the disclosure wording

TEAM_PLAN.md P0-7 instructs each person to confirm the repo is Private in the GitHub UI.
The repo has been public since 2026-08-01 and that is now a deliberate posture, not an
incident. Reword P0-7 so a teammate does not "fix" the visibility, and note that the
pre-push guard still matters for any *new* remote (the Deep-Learning-130 org repo is a
separate surface).

This deliverable changes wording in local working docs only. It does not reopen the
disclosure decision.

## Order and dependencies

D1 first — it is the highest-risk unknown and its answer can invalidate D4 and D5.
D2 runs in parallel; it depends only on the paper. D3 runs when gcloud auth is available.
D4 needs D3's byte counts and D1's answer about which data the eval path touches.
D5 composes D1 and D3. D6 and D7 are independent and can be done at any point.

## Definition of done

A first GPU booking can be made with a written command list, a known cost ceiling, and a
pre-registered pass/fail bar — all signed off before the instance starts billing.

## Out of scope

- Any `gate/` work. TEAM_PLAN.md Rule 2: no gate code before Phase 2 returns PROCEED.
- Attack machinery. That is Worker A's, in Phase 1.
- Provider selection and procurement. Deliberately left open.
- Any edit inside `vendor/ttt-e2e/`. ADR-002; the tree has no license.
  `git -C vendor/ttt-e2e status` must stay clean.
