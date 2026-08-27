# Branch and review protocol

> TEAM_PLAN.md **P0-3**. Owner: Lead. Written 2026-08-08.
> Acceptance: feature branches, Lead merges to `main`, no self-merge, no history rewrite —
> stated in writing and acknowledged by both workers.

## The four rules

1. **Feature branches. Never commit directly to `main`.**
2. **The Lead merges.** Workers open PRs; the Lead reviews and merges.
3. **No self-merge.** Including the Lead — the Lead's own branches are reviewed by a worker.
   Someone other than the author presses the button, always.
4. **Never rewrite history.** No `rebase`, no `--amend`, no force-push, on any branch that
   has been pushed. See below — this one is not a style preference.

## Why rule 4 is different from the others

TEAM_PLAN.md Rule 4 and `DISCLOSURE.md` both rest on the same fact: **the commit timeline is
conception evidence.** Dated commits support conception and reduction-to-practice for the
provisional. Rewriting history destroys that evidentiary value, and it cannot be undone by
apologising afterwards.

So: commit granularly, write honest messages, and if a commit is wrong, **fix it forward**
with another commit. A messy-but-true history is worth more here than a tidy fabricated one.

This holds even though the repo is public (see `DISCLOSURE.md`). Publication changed which
jurisdictions are available; it did not change the value of a dated, unrewritten timeline.

## Branch naming

```
<area>/<short-description>
```

`area` ∈ `attack` | `gate` | `eval` | `infra` | `docs` | `exp`. Examples:
`attack/select-stream`, `eval/report-cli`, `infra/fetch-checkpoints`, `exp/000-baseline`.

One branch, one reviewable idea. A branch that touches the attack, the harness and the docs
is three branches wearing a coat, and it will get a worse review than any of the three.

## What a review checks

In order. The first four are mechanical and non-negotiable; the rest need judgement.

1. **`git -C vendor/ttt-e2e status` is clean.** ADR-002: the vendor tree has no licence and
   must never be edited. Any diff inside `vendor/` blocks the merge outright.
2. **CPU tests pass:** `PYTHONPATH=src JAX_PLATFORMS=cpu pytest` — **165 tests** as of
   2026-08-27. The count is expected to grow; a branch that *reduces* it is the thing to
   question, and a branch that changes it without updating this line makes the check a lie.
3. **No `gate/` code before the Phase 1 spike returns PROCEED** (Rule 2). Gate work is sunk
   cost if the verdict is STOP. Attack machinery is fine early; it is needed either way.
   (This previously read "before Phase 2 returns PROCEED", which was muddled — Phase **1**
   returns the verdict that gates Phase 2.) Note that `interceptor.py` already ships
   `norm_threshold_gate` as a deliberate Phase 1 sanity control; that is not `gate/` code and
   is not the invention, but it must stay unwired from the harness and off the CLI.
4. **No secrets, no checkpoints, no datasets.** `checkpoints/`, `*.npy`, `results/`, `.env`
   are git-ignored — confirm nothing slipped past with `git add -f` or a new path.
5. **Pre-registered thresholds are untouched** (Rule 5). A diff that moves a bar in
   `PREREGISTERED.md` or `TOLERANCE.md` is rejected unless it is a dated, reasoned revision
   in that file's own revision log. Quiet edits are the specific thing these files exist to
   prevent.
5b. **A diff to the threshold constants or their parser is reviewed as a threshold change.**
   `src/trustgate/eval/prereg.py` holds the three frozen bars in code, and its parser maps
   the doc's table rows onto them. Both are new surfaces on which a bar can move while
   `PREREGISTERED.md` itself shows a clean diff — so rule 5's check does not cover them.
   `tests/test_thresholds.py` pins the constants to the document and CI goes red if they
   diverge, but a reviewer still has to read a change to either file as a change to the bar.
   Watch specifically for the percent/fraction boundary: the doc says `10%`, the code
   compares a fraction, and `10.0` instead of `0.10` sets the bar at 1000% and manufactures
   a STOP from a genuine positive.
6. **Unflattering numbers are still there** (Rule 6). A result that got worse gets reported,
   not dropped. Concretely: the per-seed table in `report.md` must be present and complete.
   Aggregates alone can hide a seed where the poison stream *helped* the victim — which is
   exactly what happened until 2026-08-27, when `render_markdown` was found never to print
   the per-seed lists `SpikeResult` had been carrying since the scaffold.
7. **Decisions that are expensive to reverse have an ADR** in `docs/adr/`.

## Merging

- Squash or merge commit, either is fine. **Never rebase a pushed branch.**
- The branch is deleted after merge; the commits stay in `main`'s history.
- If a merge needs a fix, it is a new commit on a new branch. Not an amend.

## Working with agents

Agent-authored commits follow the same rules — branch, PR, human review, no self-merge. The
reviewer is accountable for the diff regardless of who typed it. Attribute honestly in the
commit trailer; the timeline is evidence, and evidence that misstates authorship is worse
than no evidence.

## Acknowledgement

Both workers acknowledge this document before their first push. Record it here.

| Person | Role | Acknowledged | Date |
|---|---|---|---|
| Person 1 | Lead | — | — |
| Person 2 | Worker A | — | — |
| Person 3 | Worker B | — | — |
