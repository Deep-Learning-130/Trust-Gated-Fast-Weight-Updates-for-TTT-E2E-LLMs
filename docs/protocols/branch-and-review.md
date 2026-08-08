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
2. **CPU tests pass:** `PYTHONPATH=src JAX_PLATFORMS=cpu pytest` — currently 31 tests.
3. **No `gate/` code before Phase 2 returns PROCEED** (Rule 2). Gate work is sunk cost if the
   verdict is STOP. Attack machinery is fine early; it is needed either way.
4. **No secrets, no checkpoints, no datasets.** `checkpoints/`, `*.npy`, `results/`, `.env`
   are git-ignored — confirm nothing slipped past with `git add -f` or a new path.
5. **Pre-registered thresholds are untouched** (Rule 5). A diff that moves a bar in
   `PREREGISTERED.md` or `TOLERANCE.md` is rejected unless it is a dated, reasoned revision
   in that file's own revision log. Quiet edits are the specific thing these files exist to
   prevent.
6. **Unflattering numbers are still there** (Rule 6). A result that got worse gets reported,
   not dropped.
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
