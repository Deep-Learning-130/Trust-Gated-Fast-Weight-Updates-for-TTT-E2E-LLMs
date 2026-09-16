#!/usr/bin/env bash
# Fetch a released TTT-E2E checkpoint from GCS and fingerprint it.
#
# The bucket is REQUESTER-PAYS: we pay the egress. So this script always runs a
# metadata-only size probe FIRST and prints the byte count BEFORE transferring
# anything, and refuses to exceed MAX_BYTES without an explicit override.
#
# Checkpoint layout is FLAT NAMED DIRECTORIES, not ${BUCKET}/${SIZE}:
#   gs://ttt-e2e-checkpoints/1b_ttt_e2e_finetune_books_8k_1x_cc
# See vendor/ttt-e2e/README.md "Model Checkpoints".
#
# NOT YET VERIFIED AGAINST THE LIVE BUCKET. Every path here is transcribed from
# the vendor README; nothing has been listed, probed or fetched, because that
# needs `gcloud` authenticated against a billing project and we do not have that
# yet. The first real run of this script is also the first test of it.
#
# Usage:
#   GCP_BILLING_PROJECT=my-proj bash scripts/fetch_checkpoints.sh
#   GCP_BILLING_PROJECT=my-proj PROBE_ONLY=1 bash scripts/fetch_checkpoints.sh
#   GCP_BILLING_PROJECT=my-proj CKPT=3b_ttt_e2e_finetune_books_8k_3x_cc bash scripts/...
#
# Idempotent: re-running after a partial failure resumes the copy (`gsutil -m
# cp -n`) and rewrites the fingerprint from whatever is on disk.
set -euo pipefail

: "${GCP_BILLING_PROJECT:?Set GCP_BILLING_PROJECT to the billing project id that pays egress}"

BUCKET="${BUCKET:-gs://ttt-e2e-checkpoints}"

# Plan of record: the 1B DCLM-pretrained, Books-extension checkpoint at 8K.
# ADR-004 fixes this as the baseline target; ADR-005 records why, and what the
# alternatives cost. Other released names (vendor README):
#   125m_ttt_e2e_pretrain_dclm_8k_1x_cc     1b_ttt_e2e_pretrain_dclm_8k_1x_cc
#   3b_ttt_e2e_pretrain_dclm_8k_3x_cc       125m_ttt_e2e_finetune_books_8k_1x_cc
#   1b_ttt_e2e_finetune_books_8k_1x_cc      3b_ttt_e2e_finetune_books_8k_3x_cc
#   3b_ttt_e2e_finetune_books_128k_3x_cc
CKPT="${CKPT:-1b_ttt_e2e_finetune_books_8k_1x_cc}"

SRC="${BUCKET}/${CKPT}"
DEST="${DEST:-checkpoints/${CKPT}}"          # checkpoints/ is git-ignored

# Egress guard. Default from experiments/000-repro-baseline/COST_MODEL.md.
# Raise deliberately, never reflexively: this is the only thing standing between
# a typo'd path and a surprise bill.
MAX_BYTES="${MAX_BYTES:-40000000000}"        # 40 GB

RESULTS="${RESULTS:-experiments/000-repro-baseline/results}"
PROBE_ONLY="${PROBE_ONLY:-0}"

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

command -v gsutil >/dev/null 2>&1 || {
  echo "ERROR: gsutil not found. Install the Google Cloud SDK and run 'gcloud auth login'." >&2
  exit 1
}

human() {  # bytes -> human readable, for the log only
  awk -v b="$1" 'BEGIN{
    split("B KB MB GB TB",u," "); i=1
    while (b>=1024 && i<5) { b/=1024; i++ }
    printf (i==1 ? "%d %s" : "%.2f %s"), b, u[i]
  }'
}

# ---------------------------------------------------------------- probe -----
# Metadata only. `du` and `ls` are Class A operations against object metadata;
# they do not transfer object contents, so this costs essentially nothing and
# tells us the price of the transfer before we commit to it.
echo "==> Probing $SRC (metadata only, no transfer)"
echo "    billing project: $GCP_BILLING_PROJECT"

if ! du_out="$(gsutil -u "$GCP_BILLING_PROJECT" du -s "$SRC" 2>&1)"; then
  echo "ERROR: probe failed. Output:" >&2
  echo "$du_out" >&2
  echo >&2
  echo "Check, in order:" >&2
  echo "  1. 'gcloud auth login' has been run and the account can read the bucket" >&2
  echo "  2. GCP_BILLING_PROJECT is a project you can bill to (requester-pays)" >&2
  echo "  3. the path exists:  gsutil -u \$GCP_BILLING_PROJECT ls $BUCKET/" >&2
  exit 1
fi

BYTES="$(printf '%s\n' "$du_out" | awk 'NF{v=$1} END{print v+0}')"
if [[ "$BYTES" -le 0 ]]; then
  echo "ERROR: probe returned no bytes for $SRC -- treat the path as unconfirmed." >&2
  echo "       Raw output: $du_out" >&2
  echo "       List what actually exists:  gsutil -u \$GCP_BILLING_PROJECT ls $BUCKET/" >&2
  exit 1
fi

OBJECTS="$(gsutil -u "$GCP_BILLING_PROJECT" ls -r "$SRC/**" 2>/dev/null | grep -c . || true)"

echo
echo "    ================ SIZE BEFORE TRANSFER ================"
echo "    checkpoint : $CKPT"
echo "    bytes      : $BYTES  ($(human "$BYTES"))"
echo "    objects    : $OBJECTS"
echo "    cap        : $MAX_BYTES  ($(human "$MAX_BYTES"))"
echo "    ======================================================"
echo

mkdir -p "$RESULTS"
PROBE_LOG="$RESULTS/checkpoint-probe-${CKPT}.txt"
{
  echo "checkpoint:        $CKPT"
  echo "source:            $SRC"
  echo "bytes:             $BYTES"
  echo "human:             $(human "$BYTES")"
  echo "objects:           $OBJECTS"
  echo "billing_project:   $GCP_BILLING_PROJECT"
  echo "probed_utc:        $(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "probed_by:         ${USER:-unknown}@$(hostname)"
} > "$PROBE_LOG"
echo "==> Probe recorded -> $PROBE_LOG"

if [[ "$BYTES" -gt "$MAX_BYTES" ]]; then
  echo >&2
  echo "REFUSING TO TRANSFER: $BYTES bytes exceeds MAX_BYTES=$MAX_BYTES." >&2
  echo "  The byte count above is now known -- take it to COST_MODEL.md and decide." >&2
  echo "  To proceed anyway:  MAX_BYTES=$BYTES bash scripts/fetch_checkpoints.sh" >&2
  exit 2
fi

if [[ "$PROBE_ONLY" != "0" ]]; then
  echo "==> PROBE_ONLY set. No transfer. Byte count above is what COST_MODEL.md needs."
  exit 0
fi

# ------------------------------------------------------------- transfer -----
# -n skips objects already present, which is what makes a re-run after a
# partial failure cheap instead of a second full egress bill.
echo "==> Fetching -> $DEST"
mkdir -p "$DEST"
gsutil -u "$GCP_BILLING_PROJECT" -m cp -n -r "$SRC/*" "$DEST/"

# ---------------------------------------------------------- fingerprint -----
# A checkpoint is a directory of many files, so one hash is not enough: a sorted
# per-file manifest plus a hash OF that manifest (TEAM_PLAN P0-5). The logic lives
# in fingerprint_checkpoint.sh so a side-loaded checkpoint gets the same record.
DEST="$DEST" CKPT="$CKPT" RESULTS="$RESULTS" SOURCE="$SRC" REMOTE_BYTES="$BYTES" \
  bash "$repo_root/scripts/fingerprint_checkpoint.sh"

echo
echo "==> Done."
echo "    checkpoint:      $DEST"
echo
echo "    NOTE: $RESULTS/ is git-ignored. Copy the hash above into the tracked"
echo "          experiment record, or this run leaves no evidence."
