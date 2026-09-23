# GPU booking record



> The queue. Rules: [`gpu-queue.md`](gpu-queue.md). Append in chronological order.

> Claim the row **before** the instance starts; fill in Outcome on release.

> A row with no task ID, no owner or no estimate is not a booking.



**No box has been provisioned. No booking has been made. Nothing has been spent.**



| # | Start (UTC) | End (UTC) | Owner | Task | Est. h | Actual h | Budget | Outcome |

|---|---|---|---|---|---|---|---|---|

| — | — | — | — | — | — | — | — | *(no bookings yet)* |



## 2026-09-22: session 1 needs no booking

Session 1 now runs on **125M on Manas's own laptop GPU** (RTX 3070 Ti, 8 GB, WSL2), per
`PREREGISTERED.md`'s revision 2026-09-22 and `gpu-session-1-runbook.md` Part B-local.
**Nothing is rented, so there is no row to claim, and the empty table above is correct.**
The only spend is about ₹15 of GCS egress for the Colab fetch, billed to whoever runs it.
If C1 falls back to the free Kaggle TPU, that isn't a booking either. The 1B plan below
is deferred, not cancelled.

## Expected first entries

Not bookings. This is a plan, recorded so the first claim is quick to make. **Revised 2026-09-21:**
the session runs on **GCP, inside Jaykay's project** (A100 80GB quota granted), one-and-done
through the 001 verdict and the Phase 2 gate. It is operated by Jaykay from
`gpu-session-1-runbook.md` and the shared "Session 1 Field Protocol" page, with Manas remote.

| Task | Owner | Est. h | Budget | Notes |
|---|---|---|---|---|
| Create `a2-ultragpu-1g`, bootstrap (download + env), fingerprint | Jaykay | 0.8 | GCP credit | Step 3 catches a bad GPU image; stop rule is delete within 30 min. |
| `T1.5`-`T1.7` 000 baseline (smoke, 2 runs for S2, control, acceptance) | Jaykay | 1.5 | GCP credit | `run_gpu_session.sh`, `DEADLINE_HOURS=2.5`. A FAIL stops the session. |
| C1b prepare + C2 timing | Jaykay | 0.5 | GCP credit | C2 sets `--max-iters` for C3; agree it with Manas. |
| C3 001 spike, C5 gate, C4 sequence (if time) | Jaykay | 3-4 | GCP credit | Crash-safe: rerun the same command to resume. Copy out after C3. |

Nominal total ≈ **6-7 h at ~$4-5/h ≈ $25-35 of GCP credit**; the console's estimate and the bill
are the real numbers. Record the rate actually charged in the booking row.

**Superseded 2026-09-21:** the 2026-09-16 plan (E2E Networks over UPI, with a teammate's
service-account key for the GCS download). Both constraints behind it are gone.

## Claiming row #1



Copy this into the table above and fill the blanks **before** the instance starts. It is not

a claim until Start, Owner and Est. h are real.



```

| 1 | YYYY-MM-DDTHH:MMZ | — | Jaykay (box), Manas (remote) | T1.4-T1.7 + 001 C2-C5 | 7 | — | GCP credit (~$35) | *(in progress)* |

```



On release, replace the Outcome with what ran, what it produced, what broke and how far it

got, and anything the next holder must know. The box is not free until that is written.

