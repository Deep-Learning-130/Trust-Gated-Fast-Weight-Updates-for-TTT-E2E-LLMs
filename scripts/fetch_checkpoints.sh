#!/usr/bin/env bash
# Fetch TTT-E2E checkpoints from GCS (Requester-Pays -- you pay egress).
# Requires a GCP billing project. Checkpoints are git-ignored.
set -euo pipefail

: "${GCP_BILLING_PROJECT:?Set GCP_BILLING_PROJECT to your billing project id}"

# Start at 1B: fits an 80GB card comfortably and answers the attack question
# without needing 3B. Override SIZE=3b only for the 128k-context runs.
SIZE="${SIZE:-1b}"
DEST="${DEST:-checkpoints/${SIZE}}"

# NOTE: confirm the exact bucket path against the vendor README before running
# -- this is a template, not a verified path.
BUCKET="${TTT_BUCKET:-gs://REPLACE_WITH_VENDOR_BUCKET}/${SIZE}"

if [[ "$BUCKET" == *REPLACE_WITH_VENDOR_BUCKET* ]]; then
  echo "ERROR: set TTT_BUCKET to the real vendor GCS path (see vendor/ttt-e2e/README.md)."
  exit 1
fi

mkdir -p "$DEST"
echo "==> Fetching $SIZE checkpoint from $BUCKET (billing: $GCP_BILLING_PROJECT)"
gsutil -u "$GCP_BILLING_PROJECT" -m cp -r "$BUCKET/*" "$DEST/"
echo "==> Done -> $DEST"
