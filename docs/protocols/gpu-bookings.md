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
| **Prerequisite, no GPU:** own GCP billing via UPI prepay; Cloud Shell probe passes; W&B preflight passes | Manas | 0 | ₹500–1,000 GCP prepay (egress uses ~₹60–120) | Jaykay is on leave, and his 2026-09-14 run was metadata only, so no files exist. The download now happens on the box itself, in parallel with `uv sync`. |
| `P0-1`/`T1.4` provision, gcloud login, `bootstrap_gpu_box.sh` (download + env), smoke pass | Manas | 1.0 | ~₹220 | Access is checked in seconds before anything slow; a bad GPU image is caught at Step 3. |
| `T1.5`/`T1.6` baseline eval, 1B Books @8K, **50M tokens**, ×2 for bar S2 | Manas | 1.5 | ~₹330 | Each run recompiles; ~20–25 min per pass on an A100 (`TOLERANCE.md` §8, second note). |
| `T1.7` negative control, acceptance, redacted copy-out, destroy box | Manas | 0.5 | ~₹110 | `check_baseline_acceptance.py` prints the verdict; read the bar first. |

Nominal total ≈ **3 h on an E2E A100 80 GB at $2.10/h ≈ ₹650 including GST** (bad day 4 h ≈ ₹870).
Rates checked on e2enetworks.com on 2026-09-16; record the rate actually charged in the booking row.
The Phase 0.5 cap stays at **$325**.

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
