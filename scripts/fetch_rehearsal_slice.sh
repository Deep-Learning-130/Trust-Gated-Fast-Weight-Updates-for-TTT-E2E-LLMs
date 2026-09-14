#!/usr/bin/env bash
# Fetch the smallest useful real slice of the TTT-E2E release: the 125M Books@8K
# checkpoint plus ONE chunk of books3 /val. About 1.08 GB, about $0.13 of egress.
#
# STANDALONE BY DESIGN, like scripts/probe_gcs_access.sh. It reads nothing from
# this repo and writes nothing into it, so it can be pasted straight into Google
# Cloud Shell, where `gsutil` is preinstalled and already authenticated. The
# person holding the billing account should not have to clone anything.
#
#   Cloud Shell:  bash fetch_rehearsal_slice.sh
#   Local:        GCP_BILLING_PROJECT=my-proj bash scripts/fetch_rehearsal_slice.sh
#
# WHY ONE CHUNK. /val is uncompressed zarr v3: 2,000,168,321 uint32 tokens in 21
# chunks of exactly 100,000,000 tokens (400,000,000 bytes) each -- measured
# 2026-09-14, COST_MODEL.md 2.2. Absent chunks read back as the fill value, so a
# single chunk is a usable 100 M-token array on its own. Taking one instead of
# all 21 turns $1.01 of egress into $0.05 and 8.4 GB of transfer into 0.4 GB.
#
# THIS SPENDS MONEY. Unlike probe_gcs_access.sh, which is metadata only, this
# script transfers object contents and the bucket is REQUESTER-PAYS: the egress
# bills to whoever runs it. It therefore probes sizes FIRST, prints the total,
# and refuses to exceed MAX_BYTES without an explicit override.
#
# Exit codes:  0 fetched   1 setup problem   2 over cap, nothing fetched
#              3 a fetch failed
set -uo pipefail

# ---------------------------------------------------------------- constants --
CKPT_BUCKET="gs://ttt-e2e-checkpoints"
CKPT="${CKPT:-125m_ttt_e2e_finetune_books_8k_1x_cc}"   # 0.68 GB, measured
DATA="gs://llama3-books3/data.zarr"                    # store root is INSIDE the bucket
DEST="${DEST:-$HOME/ttt-handoff}"

# 1.5 GB. Sized for ckpt + one chunk with headroom, NOT for the 1B checkpoint
# (5.35 GB) or the full /val (8.40 GB). Raise it deliberately, never reflexively.
MAX_BYTES="${MAX_BYTES:-1500000000}"

say() { printf '%s\n' "$*"; }
hr()  { printf '%s\n' "------------------------------------------------------------"; }
human_gb() { awk -v b="$1" 'BEGIN{printf "%.2f", b/1000000000}'; }
usd() { awk -v b="$1" 'BEGIN{printf "%.2f", (b/1000000000)*0.12}'; }

# ------------------------------------------------------------- preflight -----
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
  say "  It is the project ID (like 'my-project-479312'), not the project NAME."
  exit 1
fi

ACCOUNT="$(gcloud config get-value account 2>/dev/null | tr -d '[:space:]')"

say ""
say "TTT-E2E rehearsal slice -- 125M checkpoint + one /val chunk"
hr
say "  billing project : $PROJECT"
say "  account         : ${ACCOUNT:-unknown}"
say "  destination     : $DEST"
say "  cap             : $(human_gb "$MAX_BYTES") GB"
say ""
say "  THIS TRANSFERS DATA AND BILLS EGRESS TO $PROJECT."
say "  Sizes are probed first and printed before anything moves."
hr
say ""

# ------------------------------------------------------- discover the chunk --
# The chunk key layout is NOT assumed. zarr v3 writes 'c/0' with the default
# separator and 'c.0' with a dot separator, and a v2-style store writes a bare
# '0'. Guessing produces a 404 that reads like an access failure, so list first.
say "1. Locating the first /val chunk (listing is metadata only, free)"
VAL_LS="$(gsutil -u "$PROJECT" ls "$DATA/val/" 2>&1)"
if [ $? -ne 0 ]; then
  say "   FAILED to list $DATA/val/"
  say "   $VAL_LS"
  say ""
  say "   Most likely: the account has no billing enabled on $PROJECT, or"
  say "   $PROJECT is a project NAME rather than a project ID."
  exit 1
fi
printf '%s\n' "$VAL_LS" | sed 's/^/     /'

CHUNK=""
if printf '%s\n' "$VAL_LS" | grep -q "$DATA/val/c/$"; then
  # Default zarr v3 layout: chunks live under a 'c/' prefix.
  CHUNK="$(gsutil -u "$PROJECT" ls "$DATA/val/c/" 2>/dev/null | head -1)"
else
  # Flat layout: take the first object that is not the metadata document.
  CHUNK="$(printf '%s\n' "$VAL_LS" | grep -v 'zarr.json$' | grep -v '/$' | head -1)"
fi
CHUNK="$(printf '%s' "$CHUNK" | tr -d '[:space:]')"

if [ -z "$CHUNK" ]; then
  say ""
  say "   Could not identify a chunk object under $DATA/val/."
  say "   Send the listing above back rather than guessing a path."
  exit 1
fi
say ""
say "   first chunk: $CHUNK"
say ""

# ------------------------------------------------------------ probe sizes ----
# `du -s` is a Class A operation against object metadata. It does not transfer
# object contents, so this whole section costs a fraction of a cent.
du_bytes() {
  local out
  out="$(gsutil -u "$PROJECT" du -s "$1" 2>&1)" || { printf '0\n'; return 1; }
  printf '%s\n' "$out" | awk 'NF{v=$1} END{print v+0}'
}

say "2. Probing sizes before transferring anything"
printf '  %-18s ' "checkpoint"
CKPT_BYTES="$(du_bytes "$CKPT_BUCKET/$CKPT")"
say "$(human_gb "$CKPT_BYTES") GB   ($CKPT_BYTES bytes)   $CKPT"

printf '  %-18s ' "one /val chunk"
CHUNK_BYTES="$(du_bytes "$CHUNK")"
say "$(human_gb "$CHUNK_BYTES") GB   ($CHUNK_BYTES bytes)"

# A failed probe reports 0, which would sail under the cap and then fetch an
# unknown quantity. Treat an unmeasurable object as a stop, not as free.
if [ "${CKPT_BYTES:-0}" -le 0 ] || [ "${CHUNK_BYTES:-0}" -le 0 ]; then
  say ""
  say "REFUSING: at least one object could not be sized."
  say "  A zero here means the probe failed, not that the object is empty, so the"
  say "  total below would be fiction and the cap would not be protecting anything."
  say "  Nothing was transferred. Send this output back rather than overriding."
  exit 1
fi

TOTAL=$((CKPT_BYTES + CHUNK_BYTES))
say ""
say "  total: $(human_gb "$TOTAL") GB  ~\$$(usd "$TOTAL") of egress, billed to $PROJECT"
hr

if [ "$TOTAL" -gt "$MAX_BYTES" ]; then
  say ""
  say "REFUSING: $(human_gb "$TOTAL") GB exceeds the $(human_gb "$MAX_BYTES") GB cap."
  say "Nothing was transferred and nothing was charged beyond the probe above."
  say "If this is expected, rerun with:  MAX_BYTES=$TOTAL bash $0"
  exit 2
fi
say ""

# ----------------------------------------------------------------- fetch -----
mkdir -p "$DEST/val/c"
FAILED=0

# -n skips objects already on disk, so a re-run after a failure resumes rather
# than paying for the same bytes twice.
say "3. Fetching (this is the part that costs money)"

say "   checkpoint ..."
if ! gsutil -u "$PROJECT" -m cp -n -r "$CKPT_BUCKET/$CKPT" "$DEST/"; then
  say "   FAILED: checkpoint"
  FAILED=1
fi

# Both metadata documents are a few hundred bytes each and the store is unusable
# without them. train/zarr.json is included because train.py:125 builds the train
# iterator before the eval branch returns, so its absence kills a vendor-loop run
# after billing has already started.
say "   zarr metadata ..."
gsutil -u "$PROJECT" cp -n "$DATA/zarr.json"       "$DEST/"       || FAILED=1
gsutil -u "$PROJECT" cp -n "$DATA/val/zarr.json"   "$DEST/val/"   || FAILED=1
mkdir -p "$DEST/train"
gsutil -u "$PROJECT" cp -n "$DATA/train/zarr.json" "$DEST/train/" || say "   (train/zarr.json unavailable -- not fatal for a direct-zarr read)"

say "   /val chunk ..."
if ! gsutil -u "$PROJECT" cp -n "$CHUNK" "$DEST/val/c/"; then
  say "   FAILED: chunk"
  FAILED=1
fi
say ""

# ---------------------------------------------------------------- verify -----
say "4. What landed on disk"
hr
ON_DISK="$(du -sb "$DEST" 2>/dev/null | awk '{print $1}')"
say "  $DEST  =  $(human_gb "${ON_DISK:-0}") GB"
find "$DEST" -type f | sort | sed 's/^/    /' | head -40
say ""
say "  checkpoint sha256 (this is the fingerprint to report back):"
find "$DEST/$CKPT" -type f -print0 2>/dev/null \
  | sort -z | xargs -0 sha256sum 2>/dev/null | sha256sum | awk '{print "    " $1}'
hr
say ""

if [ "$FAILED" -ne 0 ]; then
  say "One or more fetches failed. Re-running is safe and resumes (cp -n)."
  exit 3
fi

# ------------------------------------------------------------- handoff -------
say "5. Getting it off this box"
say ""
say "  Cloud Shell has no GPU, so nothing can be evaluated here. Pack it:"
say ""
say "    tar -czf ~/ttt-handoff.tgz -C \"$(dirname "$DEST")\" \"$(basename "$DEST")\""
say ""
say "  Then download it: the three-dot menu at the top right of the Cloud Shell"
say "  window -> Download -> enter  ttt-handoff.tgz"
say ""
say "  Note the checkpoint is a public release from the TTT-E2E authors, so"
say "  passing it along redistributes nothing that is not already public."
say ""
say "Done. $(human_gb "$TOTAL") GB transferred, ~\$$(usd "$TOTAL") of egress."
exit 0
