# DISCLOSURE POSTURE — READ BEFORE PUSHING TO A NEW REMOTE

**This repository has been PUBLIC since 2026-08-01.**
**Provisional patent application: NOT FILED.**
**As of 2026-08-02 the public posture is a deliberate decision, not an incident.**

> **Do not "fix" the visibility.** Setting `origin` back to Private would achieve nothing —
> the disclosure has already occurred and cannot be undone — while destroying the clean
> dated record of when it occurred, which is the one thing still worth protecting.
> The decision is settled. This file records its consequences; it does not reopen it.

Superseded: every earlier version of this file said "this repository is confidential" and
gated all publication on filing. That was written before 2026-08-01 and no longer describes
reality. The history is deliberately preserved in git rather than rewritten.

---

## What happened, and what follows

The repository was made public on 2026-08-01. It was noticed on 2026-08-02 and the trade-off
was put to the owner, who chose to leave it public and keep pushing — treating publication as
the route the dossier already named as the alternative to patents. Engagement at discovery
was zero, which bounds the practical exposure but does not change the legal position.

**Forfeit.** Absolute-novelty jurisdictions — **EPO, China, Japan, Korea** — apply no general
grace period. Patent rights there are gone for everything disclosed in this repository as of
2026-08-01. Some of those jurisdictions have narrow exceptions with short deadlines and
procedural requirements; treat them as forfeit unless counsel says otherwise.

**Still live.** The **US** grace period under AIA §102(b)(1)(A) runs **12 months from first
public disclosure**, so approximately **2027-08-01**. A US provisional filed before that date
can still claim priority over the intervening disclosure.

**The single most important consequence:** the US grace period used to be a backstop against
accident. **It is now the actual clock**, and it has a date on it. Filing after ~2027-08-01
forfeits the US too.

---

## The operative rules now

1. **Nobody changes any remote's visibility.** Not to Private, not to Public. If a change is
   ever genuinely needed, it is a written decision by the owner, recorded here first.
2. **New remotes are still a live surface.** `origin` is public and settled; a *different*
   remote is a *different* decision. `scripts/pre-push-guard.sh` still has a job — it just
   changed from "stop any public push" to "make sure a new remote is a deliberate choice".
   Install it: `cp scripts/pre-push-guard.sh .git/hooks/pre-push`. Remotes with a recorded
   decision are listed below and pass the guard silently.
3. **Never rewrite history.** No `rebase`, no `--amend`, no force-push. Dated commits are
   conception evidence and now also fix the disclosure date. This rule got *more* important
   when the repo went public, not less. See [`docs/protocols/branch-and-review.md`](docs/protocols/branch-and-review.md).
4. **The filing question is now a deadline, not a gate.** `ROADMAP.md` Phase 3 still requires
   the empirical result and a clean prior-art diff before filing is *worthwhile*; the
   ~2027-08-01 date decides whether it is still *possible*. Raise both with counsel together.

## Remotes — recorded decisions

Verified 2026-08-08 by unauthenticated `GET https://api.github.com/repos/OWNER/REPO`
(200 = publicly readable, 404 = private or nonexistent) plus `git ls-remote`, which resolves
with our stored credentials and so proves existence and access independently of the 404.

| Remote | URL | Publicly readable | Decision |
|---|---|---|---|
| `origin` | `Manas-Maahir/Trust-Gated-Fast-Weight-Updates-for-TTT-E2E-LLMs` | **Yes** (200) | **Public, deliberately, since 2026-08-01.** Push freely. |
| `org` | `Deep-Learning-130/Trust-Gated-Fast-Weight-Updates-for-TTT-E2E-LLMs` | **No** (404) | **Private mirror. Pushing is fine** — see below. |

**On the org remote.** It exists, we have access, and at the time of checking its `main` was
at exactly the same commit as `origin`'s — `64aec91`. So it **already holds the identical
history**, and pushing to it discloses nothing that is not already public via `origin`. There
was never a meaningful decision to make here; what was missing was the record. This is it.

Two things follow that are easy to get backwards:

- **Its privacy protects nothing today.** Identical content is public on `origin`. Nobody
  should treat "it's in the private org repo" as meaning "this is confidential" — that
  inference is false for every commit up to `64aec91`, and it is exactly the kind of mistake
  that leads to putting genuinely new material somewhere it does not belong.
- **Its privacy is nonetheless an asset going forward.** If material covering a *future*
  invention is ever staged anywhere, a private repo whose members are under NDA/employment
  is the right home for it and the public `origin` is not. That would be a new decision, and
  it would be recorded here before the push, not after.

## What secrecy still buys — and what it does not

Secrecy no longer protects the mechanism. The interceptor, drift budget and rollback design
are public. Do not spend effort pretending otherwise, and do not let stale "confidential"
labelling stop the team from discussing published work.

Two things are still worth withholding, for reasons that have nothing to do with patents:

- **Working attack artifacts.** `src/trustgate/attack/` implements poisoning against TTT-E2E
  fast weights. It exists to establish the threat model that motivates the defense and to
  serve as the evaluation adversary. It is for **defensive research on models we control**.
  Do not run it against third-party or production systems, and **do not publish working
  attack artifacts ahead of a coordinated disclosure to the TTT-E2E authors.** This is
  research ethics and it survived the visibility change untouched.
- **Anything covering a *future* invention.** The grace period runs from *this* disclosure.
  A new mechanism, not yet in the repo, still has its full foreign-filing options — and
  loses them the moment it is pushed here. Anything genuinely new goes to counsel before it
  goes to `main`.

## Checklist before pushing to a remote you have not pushed to before

- [ ] `git remote -v` — is this remote one you have pushed to before?
- [ ] If new: is publishing here a decision someone actually made? (`origin` is; others are not.)
- [ ] Nothing in the commit covers a *future* invention outside the current disclosure.
- [ ] No working attack artifacts beyond what is already public.
- [ ] No credentials — W&B keys, `GCP_BILLING_PROJECT`, `.env`.

For `origin`, this is already settled: push freely.

## Counsel questions — open

Consolidated so they are asked once, together, rather than piecemeal:

1. Is a US provisional still worth filing given the 2026-08-01 disclosure, and does the
   ~2027-08-01 date hold on these facts?
2. The repository carries an OSS **LICENSE** committed at init. What does that grant, and how
   does it interact with a later patent claim? (`ROADMAP.md` lists this as open.)
3. Do any of EPO/CN/JP/KR have an exception that is realistically still available here?
4. Does the public commit history help or hurt as conception evidence now that it is also
   the disclosure record?

---

*Not legal advice. Nothing in this file substitutes for patent counsel, and the dates in it
are approximate. Confirm the strategy before relying on it.*
