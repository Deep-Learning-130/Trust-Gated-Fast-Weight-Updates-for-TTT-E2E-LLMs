#!/usr/bin/env bash
# One-command GCS probe. Settles every open question in task-allocation T1.3.
#
# STANDALONE BY DESIGN. It reads nothing from this repo and writes nothing into
# it, so it can be pasted straight into Google Cloud Shell, where `gsutil` is
# preinstalled and already authenticated. That is the whole point: the person
# holding the billing account should not have to clone a private repo, install
# the Cloud SDK or run `gcloud auth login` to answer four questions.
#
#   Cloud Shell:  paste the file, then  bash probe_gcs_access.sh
#   Local:        GCP_BILLING_PROJECT=my-proj bash scripts/probe_gcs_access.sh
#
# METADATA ONLY. `du` and `ls` are Class A operations against object metadata.
# They do not transfer object contents. There is no `cp` anywhere in this file
# and there is no flag that makes it transfer anything -- if you want bytes on
# disk, that is scripts/fetch_checkpoints.sh, which is a separate decision with
# its own egress guard.
#
# Cost: a few thousand Class A operations, under $0.05. See COST_MODEL.md 2.3.
#
# Exit codes:  0 all probes answered   1 setup problem   3 a probe failed
set -uo pipefail

# ---------------------------------------------------------------- constants --
CKPT_BUCKET="gs://ttt-e2e-checkpoints"
CKPT_1B="1b_ttt_e2e_finetune_books_8k_1x_cc"     # ADR-004/005: the 1B target, deferred
CKPT_125M="125m_ttt_e2e_finetune_books_8k_1x_cc" # the session target (PREREGISTERED.md 2026-09-22)
DATA_BUCKET="gs://llama3-books3"

# COST_MODEL.md 2.2: int32 token ids, Blosc/zstd, assume ~2.5x => ~1.6 B stored
# per token. The plausible band written down in August, before any measurement.
BYTES_PER_TOKEN="1.6"
BAND_LO=50000000      # 50 M tokens
BAND_HI=1000000000    # 1 B tokens
EGRESS_USD_PER_GB="0.12"

OUT="${OUT:-gcs-probe-$(date -u +%Y%m%dT%H%M%SZ).txt}"

# ------------------------------------------------------------- preflight -----
say() { printf '%s\n' "$*"; }
hr()  { printf '%s\n' "------------------------------------------------------------"; }

if ! command -v gsutil >/dev/null 2>&1; then
  say "ERROR: gsutil not found."
  say "  In Google Cloud Shell it is preinstalled -- you are probably not in Cloud Shell."
  say "  Locally: install the Google Cloud SDK, then 'gcloud auth login'."
  exit 1
fi

PROJECT="${GCP_BILLING_PROJECT:-$(gcloud config get-value project 2>/dev/null | tr -d '[:space:]')}"
if [ -z "$PROJECT" ] || [ "$PROJECT" = "(unset)" ]; then
  say "ERROR: no billing project."
  say "  Set one:  export GCP_BILLING_PROJECT=your-project-id"
  say "  Or:       gcloud config set project your-project-id"
  say "  It is the project ID (like 'my-project-479312'), not the project NAME."
  exit 1
fi

ACCOUNT="$(gcloud config get-value account 2>/dev/null | tr -d '[:space:]')"

say ""
say "TTT-E2E GCS access probe -- metadata only, transfers nothing"
hr
say "  billing project : $PROJECT"
say "  account         : ${ACCOUNT:-unknown}"
say "  buckets         : $CKPT_BUCKET  $DATA_BUCKET"
say "  both are REQUESTER-PAYS: the Class A operations below bill to $PROJECT"
hr
say ""

# ----------------------------------------------------------------- probes ----
FAILED=0
declare -a LINES=()

record() { LINES+=("$1"); }

# du_bytes <gs-path> -> echoes byte count, or empty on failure (stderr captured)
LAST_ERR=""
du_bytes() {
  local path="$1" out
  LAST_ERR=""
  if ! out="$(gsutil -u "$PROJECT" du -s "$path" 2>&1)"; then
    LAST_ERR="$out"
    return 1
  fi
  printf '%s\n' "$out" | awk 'NF{v=$1} END{print v+0}'
}

human_gb() { awk -v b="$1" 'BEGIN{printf "%.2f", b/1000000000}'; }

# Sets the global PROBE_BYTES rather than echoing it. Command substitution would
# run this in a subshell, where both the progress output and the `record` appends
# to LINES would be lost -- the appends silently, which is the dangerous one.
PROBE_BYTES=""
probe_size() {  # <label> <path> <key>
  local label="$1" path="$2" key="$3" bytes
  PROBE_BYTES=""
  printf '  %-28s ' "$label"
  if bytes="$(du_bytes "$path")" && [ "${bytes:-0}" -gt 0 ]; then
    say "$(human_gb "$bytes") GB   ($bytes bytes)"
    record "${key}_bytes: $bytes"
    PROBE_BYTES="$bytes"
    return 0
  fi
  say "FAILED"
  say "      ${LAST_ERR:-no bytes returned -- treat the path as unconfirmed}"
  record "${key}_bytes: FAILED"
  record "${key}_error: ${LAST_ERR:-empty}"
  FAILED=1
  return 1
}

say "1. Checkpoints"
probe_size "1B books@8K (deferred)" "$CKPT_BUCKET/$CKPT_1B" "ckpt_1b"
CKPT_1B_BYTES="$PROBE_BYTES"
probe_size "125M books@8K (the target)" "$CKPT_BUCKET/$CKPT_125M" "ckpt_125m"
CKPT_125M_BYTES="$PROBE_BYTES"
say ""

say "2. Dataset -- /val is the one number that can move the cost model"
probe_size "books3 /val" "$DATA_BUCKET/val" "val"
VAL_BYTES="$PROBE_BYTES"
say ""

say "3. /train/zarr.json -- without it the run dies during setup, AFTER billing starts"
printf '  %-28s ' "train/zarr.json"
if ZJ="$(gsutil -u "$PROJECT" ls "$DATA_BUCKET/train/zarr.json" 2>&1)"; then
  say "PRESENT"
  record "train_zarr_json: PRESENT"
else
  say "MISSING OR UNREADABLE"
  say "      $ZJ"
  say "      This is COST_MODEL 1.1's trap. Do not start a paid run until it resolves."
  record "train_zarr_json: MISSING"
  record "train_zarr_json_error: $ZJ"
  FAILED=1
fi
say ""

# ------------------------------------------------------------- derived -------
say "4. What the numbers mean"
hr
if [ -n "$VAL_BYTES" ]; then
  VAL_TOKENS="$(awk -v b="$VAL_BYTES" -v r="$BYTES_PER_TOKEN" 'BEGIN{printf "%d", b/r}')"
  say "  /val estimated tokens : ~$(awk -v t="$VAL_TOKENS" 'BEGIN{printf "%.0f M", t/1000000}')"
  say "    (COST_MODEL 2.2 assumes ~${BYTES_PER_TOKEN} stored bytes per int32 token"
  say "     after Blosc/zstd. The estimate inherits that assumption.)"
  record "val_tokens_estimated: $VAL_TOKENS"
  if [ "$VAL_TOKENS" -lt "$BAND_LO" ] || [ "$VAL_TOKENS" -gt "$BAND_HI" ]; then
    say ""
    say "  >> OUT OF BAND. COST_MODEL 2.2 assumed 50 M - 1 B tokens."
    say "  >> This is 6's first listed way the model blows up. The egress is still"
    say "  >> under a dollar; it is WALL-CLOCK that moves (3). Re-derive before booking."
    record "val_band_check: OUT_OF_BAND (assumed ${BAND_LO}-${BAND_HI})"
  else
    say "  >> Inside the assumed 50 M - 1 B band. COST_MODEL 3 timings stand."
    record "val_band_check: IN_BAND"
  fi
fi
if [ -n "$CKPT_125M_BYTES" ]; then
  say "  125M rehearsal ckpt   : $(human_gb "$CKPT_125M_BYTES") GB -- the cheap way to"
  say "                          exercise the whole path before spending on 1B"
fi
if [ -n "$CKPT_1B_BYTES" ] && [ -n "$VAL_BYTES" ]; then
  TOT=$((CKPT_1B_BYTES + VAL_BYTES))
  say "  total egress if fetched: $(human_gb "$TOT") GB  ~\$$(awk -v g="$TOT" -v p="$EGRESS_USD_PER_GB" 'BEGIN{printf "%.2f", (g/1000000000)*p}') off-GCP, \$0 same-region"
  record "total_egress_bytes: $TOT"
fi
hr
say ""

# -------------------------------------------------------------- record ------
{
  echo "# TTT-E2E GCS access probe -- task-allocation T1.3"
  echo "# Metadata only. Nothing was transferred."
  echo "probed_utc:        $(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "probed_by:         ${ACCOUNT:-unknown}"
  echo "billing_project:   $PROJECT"
  echo "checkpoint_bucket: $CKPT_BUCKET"
  echo "data_bucket:       $DATA_BUCKET"
  if [ "${#LINES[@]}" -gt 0 ]; then for l in "${LINES[@]}"; do echo "$l"; done; fi
} > "$OUT"

say "Written to: $OUT"
say ""
say "=============== COPY EVERYTHING BELOW THIS LINE AND SEND IT BACK ==============="
cat "$OUT"
say "=============== COPY EVERYTHING ABOVE THIS LINE ==============================="
say ""

if [ "$FAILED" -ne 0 ]; then
  say "One or more probes failed. Send the block above anyway -- the failure text"
  say "is the useful part. Common causes, in order:"
  say "  1. the account has no billing enabled on $PROJECT"
  say "  2. $PROJECT is a project NAME, not a project ID"
  say "  3. the Cloud Storage API is not enabled on the project"
  exit 3
fi

say "All probes answered. Nothing was downloaded and nothing was charged beyond"
say "a few cents of Class A operations."
exit 0
