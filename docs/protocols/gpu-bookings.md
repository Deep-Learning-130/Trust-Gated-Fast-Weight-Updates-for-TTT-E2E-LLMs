# GPU booking record

> The queue. Rules: [`gpu-queue.md`](gpu-queue.md). Append in chronological order.
> Claim the row **before** the instance starts; fill in Outcome on release.
> A row with no task ID, no owner or no estimate is not a booking.

**No box has been provisioned. No booking has been made. Nothing has been spent.**

| # | Start (UTC) | End (UTC) | Owner | Task | Est. h | Actual h | Budget | Outcome |
|---|---|---|---|---|---|---|---|---|
| — | — | — | — | — | — | — | — | *(no bookings yet)* |

## Expected first entries

Not bookings. This is a plan, recorded so the first claim is quick to make. **Revised 2026-09-16 (second pass):**
the provider is **E2E Networks** (UPI; Jarvislabs takes cards only, see `COST_MODEL.md` §9.3), and
the session is driven by `scripts/run_gpu_session.sh` per `gpu-session-1-runbook.md`.

| Task | Owner | Est. h | Budget | Notes |
|---|---|---|---|---|
| **Prerequisite, no GPU:** checkpoint + val subset side-loaded, W&B preflight passes | Lead | 0 | ₹0 | **Blocking.** The weights exist only in requester-pays GCS and this plan uses no Google Cloud account. Do not book until both directories are in hand (runbook §0.1). |
| `P0-1`/`T1.4` provision + `bootstrap_gpu_box.sh` + smoke pass | Lead | 1.5 | at booking rate | Rehearsed end to end on stub binaries; never run on a GPU. The smoke pass is the 1B itself at two eval batches, so it covers memory. |
| `T1.5`/`T1.6` baseline eval, 1B Books @8K, ×2 for bar S2 | Lead | 2.0 | at booking rate | Each run recompiles; budget 5–20 min per run on top of ~35 min eval. |
| `T1.7` negative control, acceptance, redacted copy-out | Lead | 0.5 | at booking rate | `check_baseline_acceptance.py` prints the verdict; read the bar first. |

Nominal total ≈ **4 h** on an H100 80 GB (or A100 80 GB). Record the live E2E hourly rate in the
booking row, because COST_MODEL §9.3's E2E figures are from 2026-09-14. The Phase 0.5 cap stays at **$325**.

**Superseded 2026-09-16:** the first revision planned this on JarvisLabs (card-only, unusable)
with a gcloud login on the box. The earlier plan before that was two sessions totalling ≈6.5 h, with a 125M rehearsal first.

## Claiming row #1

Copy this into the table above and fill the blanks **before** the instance starts. It is not
a claim until Start, Owner and Est. h are real.

```
| 1 | YYYY-MM-DDTHH:MMZ | — | <name> | P0-1/T1.4-T1.7 | 4 | — | ₹1,020 | *(in progress)* |
```

On release, replace the Outcome with what ran, what it produced, what broke and how far it
got, and anything the next holder must know. The box is not free until that is written.
