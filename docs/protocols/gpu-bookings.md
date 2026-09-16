# GPU booking record

> The queue. Rules: [`gpu-queue.md`](gpu-queue.md). Append in chronological order.
> Claim the row **before** the instance starts; fill in Outcome on release.
> A row with no task ID, no owner or no estimate is not a booking.

**No box has been provisioned. No booking has been made. Nothing has been spent.**

| # | Start (UTC) | End (UTC) | Owner | Task | Est. h | Actual h | Budget | Outcome |
|---|---|---|---|---|---|---|---|---|
| — | — | — | — | — | — | — | — | *(no bookings yet)* |

## Expected first entries

Not bookings — a plan, recorded so the first claim is quick to make. **Revised 2026-09-16**
against the 2026-09-14 GCS probe and `gpu-session-1-runbook.md`; the superseded plan is below.

| Task | Owner | Est. h | Budget | Notes |
|---|---|---|---|---|
| `P0-1`/`T1.4` provision + `bootstrap_gpu_box.sh` + smoke pass | Lead | 1.5 | ₹380 | Bootstrap has never executed. The smoke pass is the 1B itself at two eval batches — it covers every first-launch failure mode *including memory*, which a 125M rehearsal cannot. |
| `T1.5`/`T1.6` baseline eval, 1B Books @8K, ×2 for bar S2 | Lead | 1.5 | ₹380 | Over a truncated `/val` — see `COST_MODEL.md` §8 and `TOLERANCE.md` §8. |
| `T1.7` negative control, acceptance checks, copy-out | Lead | 0.5 | ₹130 | Read the bar before the number. |

Nominal total ≈ **3.5 h / ≈ ₹1,020 (~$12)** on an H100 80 GB at `COST_MODEL.md` §9.3 rates,
against a Phase 0.5 cap of **$325**. One session, not two.

**Superseded 2026-09-16:** the earlier plan was two sessions totalling ≈6.5 h / ≈$25, with a
125M rehearsal first. Two things changed it. The probe put `/val` at 2.0B tokens, making a
full pass ~7.5 GPU-hours — so the eval is now truncated and sized from measured throughput.
And the rehearsal moved onto the 1B checkpoint itself: `COST_MODEL.md` §4 concedes the 125M
exercises "every failure mode except memory", and memory is the one that ends a 1B session.

## Claiming row #1

Copy this into the table above and fill the blanks **before** the instance starts. It is not
a claim until Start, Owner and Est. h are real.

```
| 1 | YYYY-MM-DDTHH:MMZ | — | <name> | P0-1/T1.4-T1.7 | 4 | — | ₹1,020 | *(in progress)* |
```

On release, replace the Outcome with what ran, what it produced, what broke and how far it
got, and anything the next holder must know. The box is not free until that is written.
