# TrustGate — operations

How to run it, how to publish it, and what is currently broken or pending. The third of
three documents — see [`PROJECT.md`](PROJECT.md) for the idea and
[`CODEBASE.md`](CODEBASE.md) for the map.

Audit state below is from **2026-09-23**, branch `infra/gpu-session-1` at `c9cf1fb`.

---

## 1. Setup

```bash
git clone <remote> && cd TTT
bash scripts/setup_vendor.sh          # pinned TTT-E2E submodule + its deps
uv sync                                # or: pip install -e .
pytest                                 # 473 tests, CPU only
```

`setup_vendor.sh` checks out the vendor at its pinned SHA. The tree is never modified —
the gate is installed over it by `vendor_patch.py` (ADR-002).

## 2. Run the pilot — free, 35 seconds, no GPU

```bash
bash scripts/demo.sh                   # the version for a room, ~40s
python experiments/002-pilot-tiny-ttt/run_pilot.py
python experiments/002-pilot-tiny-ttt/run_deep.py       # the headline run
python experiments/002-pilot-tiny-ttt/run_null.py       # 20 control-vs-control draws
python experiments/002-pilot-tiny-ttt/run_sequence.py
```

Results land in `experiments/002-pilot-tiny-ttt/results/`, which is **gitignored**. The
demo prints the corpus digest check on screen first, because a number that cannot name its
inputs is not a result.

## 3. Run a GPU session

Driven by `docs/protocols/gpu-session-1-runbook.md`; owners and budget in
`gpu-bookings.md`. Operated by Jaykay with Manas remote.

```bash
# on the box, in order
bash scripts/bootstrap_gpu_box.sh      # nothing -> ready. Step 3 catches a bad GPU image.
bash scripts/run_gpu_session.sh        # T1.5-T1.7 baseline, DEADLINE_HOURS=2.5
bash scripts/prepare_phase1.sh         # ready for C2-C5
# C2 timing -> sets --max-iters for C3. Agree it before spending.
python scripts/collect_results.py      # copies artefacts back, secrets redacted
```

**Rehearse for free first.** `preflight_wandb.py` proves the W&B credentials work the way
the vendor uses them, `make_dummy_checkpoint.py` rehearses the loading path, and the CLI's
`--random-init` exercises every seam without weights. A paid GPU hour should never be the
thing that discovers a bad credential.

**Stop rules that already exist:** delete the box within 30 minutes if the fingerprint step
fails; a baseline FAIL stops the session; the project's overall budget is 12 GPU-hours.

`run_gpu_session.sh` is crash-safe — rerun the same command with the same `--out` and the
ledger replays finished evaluations instead of recomputing them.

### Compute situation

The funded routes kept closing. GCP A100 quota was refused; E2E Networks will not launch an
80 GB card without roughly 24 h of credit in the wallet up front, on **UPI-only** payment,
which rules out the card-only providers. Two free routes are now committed:
`scripts/colab/fetch_checkpoint_colab.ipynb` (data by Drive, no service-account key) and
`scripts/kaggle/ttt_session_kaggle.ipynb`.

Memory sizing is settled and not negotiable: the evaluation batch floors at 8 regardless of
configuration, putting the real requirement at 36–49 GB, so **80 GB** is the
recommendation. See `PROJECT.md` §7.5.

## 4. Build the paper

```bash
python scripts/make_figures.py         # regenerate figures + tables from the result files
python docs/paper/build_main.py        # p3 + p4        -> main.tex
python docs/paper/make_standalone.py   # main.tex       -> main-overleaf.tex
python docs/paper/check_main.py        # static checks; no compiler needed
```

Then paste `docs/paper/main-overleaf.tex` into Overleaf as `main.tex` and compile with
**pdfLaTeX**. It is self-contained: no `.bib`, no image uploads, every figure is
TikZ/pgfplots with inline coordinates, bibliography is a `thebibliography` block. Stock
IEEEtran is all it needs.

Never hand-edit `main.tex` or `main-overleaf.tex`. Edit
`p3-intro-related-methodology.tex` (Sections I–III + bibliography) or
`p4-experiments-results.tex` (Sections IV–V) and rebuild.

`check_main.py` currently passes: 5 sections, 35 bibitems, 6 figures, 5 tables, 1
algorithm, no dangling refs, no undefined citation keys, balanced environments.

> **No compiler has ever run on this paper.** There is no LaTeX toolchain on the
> development machine — `check_main.py` says so in its own docstring. Static checks passing
> is not the same as compiling.

---

## 5. Open issue: the paper's numbers contradict the results

**This blocks submission and is not recoverable from git.**

`p4-experiments-results.tex` §IV-D and the Fig. 7 caption report a null distribution with
median |d| 0.816 and 11 of 20 draws clearing the pre-registered bar (55% false positives),
against an observed d of 0.831. No file in `experiments/002-pilot-tiny-ttt/results/`
produces those numbers:

| Quantity | Paper prose | `null.json` on disk |
|---|---|---|
| Median \|d\| | 0.816 | 0.362 |
| Mean \|d\| | 0.695 | 0.423 |
| 90th percentile | 1.227 | 0.849 |
| Maximum | 1.575 | 1.307 |
| Draws clearing d ≥ 0.80 | 11/20 → **55%** | 3/20 → **15%** |
| Observed d | 0.831, 55th percentile | −0.003, 0th percentile |

The headline disagrees the same way: prose and abstract say **d = 0.831**, `deep.json`
gives **0.122**, and its per-seed control losses sit at ≈2.71 where the figures committed
alongside the prose came from a run at ≈2.46. Running `make_figures.py` rewrites Table II
from 0.831 to 0.122, so the assembled paper contradicts itself two pages apart.

**Which side is current is answerable.** `scripts/make_update_deck.py`, written 2026-09-22,
reads its numbers live from those same result files and puts 0.122, 0.225 and "15% (3 of
20)" on the slides the team is presenting. The deck and the result files agree; **the paper
prose is the stale side**, left over from a run that predates them.

Because `.gitignore:16` excludes `experiments/*/results/`, that earlier run was never
committed and cannot be recovered. Either the prose is rewritten against the current files,
or the earlier run is reproduced.

**The verdict is unaffected.** At 15% or at 55%, a negative control with no attacker clears
the pre-registered bar often enough that the observed effect cannot be read as an attack,
and Section IV already reports no attack. What changes is which false-positive rate the
paper publishes as its contribution.

---

## 6. Git, remotes and publishing

### The remotes

| Remote | Repository | Role |
|---|---|---|
| `org` | `Deep-Learning-130/Trust-Gated-Fast-Weight-Updates-for-TTT-E2E-LLMs` | **The live line.** |
| `origin` | `Manas-Maahir/Trust-Gated-Fast-Weight-Updates-for-TTT-E2E-LLMs` | Personal fork. |

**The repository is public.** `DISCLOSURE.md` records it: public since 2026-08-01,
deliberately so since 2026-08-02. Anything committed is published. Check before every push.

### The pre-push guard

`.git/hooks/pre-push` is installed (the bash guard). Install it on a fresh clone:

```bash
cp scripts/pre-push-guard.sh .git/hooks/pre-push && chmod +x .git/hooks/pre-push
# Windows: copy pre-push-hook-wrapper.sh to .git/hooks/pre-push
#          and pre-push-guard.ps1 to .git/hooks/pre-push.ps1
```

It permits pushes to `origin` and `org` by design and fires on a remote the project has not
decided about. It used to refuse any push to a public-looking host, which fired on every
single push to a remote already decided about — and a guard everyone reflexively dismisses
protects nothing. A *different* remote is a *different* decision.

### Branch state, 2026-09-23

| Ref | State |
|---|---|
| `infra/gpu-session-1` = `org/…` = `origin/…` | **In sync.** Every commit is on both remotes. |
| …vs `org/main` | 37 ahead — a pending merge, not an unpushed one. |
| `main` (local) | 22 behind `org/main`, 0 ahead. |
| `origin/main` (fork) | **52 behind `org/main`.** Stale since PR #1. |

Every one of the 25 local branches exists on at least one remote; there is no local-only
branch.

`git status` displays "ahead 37" because the branch tracks `org/main` rather than its own
upstream. That reads alarmingly and is not.

**Open decision:** either sync `origin/main` to `org/main` or stop treating the fork as a
mirror. It has been drifting rather than being decided.

---

## 7. Documentation coverage

Audited 2026-09-23 across all 35 library modules and 27 scripts.

- Every module and script carries a header docstring explaining why it exists.
- Three modules (`eval/gate_eval.py`, `eval/ledger.py`, `eval/model_build.py`) and five
  scripts (`dump_tokens.py`, `make_review_deck.py`, `make_update_deck.py`,
  `pre-push-guard.ps1`, `pre-push-hook-wrapper.sh`) were reachable from **no** markdown
  file. `CODEBASE.md` now indexes all of them.
- Eight ADRs plus a frozen pre-registration with a CI test binding it to code. This is the
  strongest part of the documentation.

### Stale

| Document | Last commit | Note |
|---|---|---|
| `README.md` | 2026-08-08 | **The front door, seven weeks behind.** Predates Phase 2, the pilot, the paper and three compute pivots. |
| `docs/report.md` | 2026-08-27 | Predates the pilot results it would want to report. |
| `task-allocation.md` | 2026-09-14 | Predates the 003 smoke work and the GCP refusal. |

`README.md` is the largest remaining documentation gap and these three files do not fix it.

---

## 8. What to do next

1. **Rewrite paper §IV-D and the Fig. 7 caption** against the current result files, so the
   paper, Table II and the 2026-09-22 deck tell the same story. Blocks submission.
2. **Compile the paper.** Nobody has seen it rendered.
3. **Refresh `README.md`.** It is what a new reader opens first.
4. **Decide about `origin/main`** — sync it, or stop calling it a mirror.
5. **Run the 003 carry check** on a card with enough memory, then the 000 baseline, then
   the 001 kill gate. That is the path to a verdict.
