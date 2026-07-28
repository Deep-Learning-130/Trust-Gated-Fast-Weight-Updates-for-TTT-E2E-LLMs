# ⚠ DISCLOSURE CONTROL — READ BEFORE PUSHING, PUBLISHING, OR PRESENTING

**Status:** Provisional patent application **NOT YET FILED**.
**Therefore: this repository and its contents are confidential.**

---

## The rule

**No public disclosure of the trust-gate mechanism before the provisional is filed.**

Most jurisdictions (EPO, China, Japan, Korea) apply **absolute novelty**: any public
disclosure before the priority date destroys patentability there, permanently, with no
grace period. The US 12-month inventor grace period under the AIA is a **backstop against
accident, not a plan** — relying on it forfeits all foreign rights.

Filing a single provisional before disclosure secures the priority date and preserves
foreign filing options for 12 months.

## What counts as public disclosure

Not just publication. All of the following can start the clock or destroy foreign rights:

- Pushing this repo, or any fork of it, to a **public** remote
- arXiv, preprints, workshop papers, blog posts, newsletters
- Conference or meetup talks, posters, demos — including internal talks with unrestricted attendance
- Public GitHub issues, PRs, or comments on `test-time-training/e2e` describing the gate
- Social media threads describing the mechanism
- Sharing the invention disclosure with anyone outside an NDA or attorney-client privilege
- Offering the mechanism for sale or license

Discussing the **published prior art** (TTT-E2E, MedBN, the TTA poisoning papers) is fine.
Discussing **our gate, drift budget, or rollback controller** is not.

## Checklist before any push

- [ ] `git remote -v` shows a **private** remote only
- [ ] The remote's visibility is confirmed Private in the host UI (not just assumed)
- [ ] No submodule or fork of this work has been pushed publicly
- [ ] Nothing in this commit is scheduled for external sharing

## Attack code

`src/trustgate/attack/` implements a poisoning attack against TTT-E2E fast weights.
It exists to establish the threat model that motivates the defense (dossier §4, Phase 1)
and to serve as the evaluation adversary.

It is for **defensive research on models we control**. Do not run it against third-party
or production systems. Do not publish working attack artifacts ahead of a coordinated
disclosure to the TTT-E2E authors.

## Conception evidence

Commits in this repo are dated and serve as supporting evidence of conception and
reduction to practice. Commit granularly and write honest messages. Do not rewrite
history (`rebase`, `commit --amend`, force-push) on this repo — it destroys the
evidentiary value of the timeline.

## Lifting this notice

When the provisional is filed, replace the status line at the top with the application
number and filing date, and relax this file to reflect the 12-month non-provisional window.
Do not delete the file — the disclosure history matters.

---

*Not legal advice. Confirm the disclosure strategy with patent counsel before relying on it.*
