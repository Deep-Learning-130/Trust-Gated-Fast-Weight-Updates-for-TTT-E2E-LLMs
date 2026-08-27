# GPU booking record

> The queue. Rules: [`gpu-queue.md`](gpu-queue.md). Append in chronological order.
> Claim the row **before** the instance starts; fill in Outcome on release.
> A row with no task ID, no owner or no estimate is not a booking.

**No box has been provisioned. No booking has been made. Nothing has been spent.**

| # | Start (UTC) | End (UTC) | Owner | Task | Est. h | Actual h | Budget | Outcome |
|---|---|---|---|---|---|---|---|---|
| — | — | — | — | — | — | — | — | *(no bookings yet)* |

## Expected first entries

Not bookings — a plan, recorded so the first claim is quick to make. Estimates and budgets
from `experiments/000-repro-baseline/COST_MODEL.md` §4–5.

| Task | Owner | Est. h | Budget | Notes |
|---|---|---|---|---|
| P0-1 provision + `bootstrap_gpu_box.sh` + **125M rehearsal** | Lead | 3 | $12 | Same session as P1-1. The rehearsal is the cheap way to hit every failure mode except memory. |
| P1-1 baseline eval, 1B Books @8K | Lead | 3 | $12 | Expect the first launch to fail; four known ways in COST_MODEL.md §4. |
| P1-2 compare against `TOLERANCE.md`, P1-3 record | Lead | 0.5 | — | Read the bar before the number. |

Nominal total ≈ **6.5 h / ≈ $25**, against a Phase 0.5 cap of **$325**. The gap funds roughly
two failed attempts, deliberately: the command has never been executed.
