# GPU queue protocol

> TEAM_PLAN.md **P0-2**. Owner: Lead. Written 2026-08-08.
> Standing rule 1: *one GPU; every GPU task names an owner and an expected duration;
> two people never hold the box; the Lead keeps the queue.*

**Status: no box has been provisioned.** This protocol is written before the first booking
so the first session is scheduled rather than improvised. It costs nothing to have it ready
and it is a Phase 0 exit criterion in its own right.

## The rule

**One box. One holder. Always a named owner and an hour estimate.**

There is a single shared A100/H100 80 GB. It is not time-sliced and it is not shared. If two
people run at once, both runs are slow, memory-fragile, and — worse — mutually
uninterpretable, because neither can attribute a timing or OOM to their own work.

## The booking record

`docs/protocols/gpu-bookings.md`, appended in chronological order, one row per booking. It
is a plain file in the repo on purpose: it is version-controlled, it needs no external
service, and it is dated evidence like the rest of the tree.

| Field | Meaning |
|---|---|
| **Start / End (UTC)** | Planned window. Actual end recorded on release. |
| **Owner** | One person. Not a team, not "whoever is around". |
| **Task ID** | The TEAM_PLAN.md task, e.g. `P1-1`. If there isn't one, it isn't a booking. |
| **Est. hours** | Written *before* starting. Compared against actual afterwards. |
| **Budget** | Expected spend, against the cap in `experiments/000-repro-baseline/COST_MODEL.md`. |
| **Outcome** | Filled on release: what ran, what it produced, what broke. |

A booking with no task ID, no estimate, or no owner is not a booking. Ad-hoc "just checking
something" use is the failure mode this exists to prevent — it is how a box ends up held for
a day with nothing written down.

## Booking

1. **Claim it in the record before the instance starts.** Not after, and not "I'll add it
   later" — the record is the lock.
2. **The Lead resolves collisions.** Ties break toward whatever unblocks the most other
   people; when that's equal, toward whoever has held the box least.
3. **Overruns are announced, not absorbed.** Passing the estimate means telling the queue and
   posting a revised end time. Silent overruns are what make the calendar useless.
4. **Release explicitly**, by filling in the Outcome. The box is not free until that's done.

## Before an instance starts billing

Every one of these is a **procurement item, not a configuration item** — none is fixable on a
billing clock, which is exactly why they are checked beforehand:

- [ ] W&B entity, project and key in hand (P0-10; mandatory — ADR-004 §3).
- [ ] `gcloud` authenticated against `GCP_BILLING_PROJECT`; both buckets are requester-pays.
- [ ] Image verified for **CUDA 12.8 / cuDNN 9.8**. A mismatch eats days.
- [ ] The run command has been read: `experiments/000-repro-baseline/EVAL_ENTRYPOINT.md`.
- [ ] Byte counts probed: `PROBE_ONLY=1 bash scripts/fetch_checkpoints.sh`.
- [ ] The relevant pre-registered bar has been read **before** any number is produced
      (`experiments/000-repro-baseline/TOLERANCE.md`, Rule 5).
- [ ] The booking row exists.

## During

- **Run `scripts/bootstrap_gpu_box.sh` first.** It is idempotent and safe to re-run after a
  partial failure. Do not hand-roll the setup.
- **Rehearse on the 125M checkpoint before the real run.** It costs minutes and exercises
  every failure mode except memory. See `COST_MODEL.md` §4.
- **Copy results out before releasing.** `results/`, `*.npy` and `wandb/` are all git-ignored.
  A run whose numbers are not deliberately transcribed into a tracked file leaves no record,
  and re-running costs another booking.

## Stop rules

Any one of these ends the session. They are not judgement calls — the point of writing them
down in advance is that the decision is already made when the moment is worst.

- **12 GPU-hours without a completed eval.** Release the box, write down what broke, re-book.
  Session 2 is always cheaper than hour 13.
- **The environment is wrong** (CUDA mismatch, driver failure). Kill it inside 30 minutes.
  Do not debug a broken image while it bills. Pick a different image.
- **Cumulative Phase 0.5 spend passes the cap** in `COST_MODEL.md` §7. Raising a cap is a
  written decision by the Lead, not a reflex at 2am.

## Handover

The next holder starts from the booking record, not from a conversation. On release, the
Outcome field records: what ran, the exact command if it differed from the written one, what
was produced and where, what failed and how far it got, and anything the next holder must
know. Two honest sentences beat a clean-looking blank.
