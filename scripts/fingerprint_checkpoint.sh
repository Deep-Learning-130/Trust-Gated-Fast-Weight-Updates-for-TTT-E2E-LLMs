#!/usr/bin/env bash
# Fingerprint a checkpoint directory already on local disk, or verify it against
# a manifest written earlier. Needs no network and no GCP account.
#
# Split out of fetch_checkpoints.sh so a checkpoint that arrived by any route --
# gsutil, scp, rsync, a USB disk -- gets the same sha256 record (TEAM_PLAN P0-5).
#
# Usage:
#   DEST=checkpoints/<ckpt> CKPT=<ckpt> bash scripts/fingerprint_checkpoint.sh
#   DEST=... CKPT=... VERIFY=path/to/checkpoint-sha256-<ckpt>.txt bash scripts/fingerprint_checkpoint.sh
#
# Optional: RESULTS (where the manifest is written), SOURCE and REMOTE_BYTES
# (recorded in the header when the caller knows them).
#
# With VERIFY set, every per-file hash in that manifest must match the bytes on
# disk and no file may be missing or extra. A mismatch exits 3: a checkpoint that
# is not the one the manifest names must not be evaluated.
set -euo pipefail

: "${DEST:?Set DEST to the local checkpoint directory}"
: "${CKPT:?Set CKPT to the checkpoint name}"

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RESULTS="${RESULTS:-$repo_root/experiments/000-repro-baseline/results}"
SOURCE="${SOURCE:-local disk}"
REMOTE_BYTES="${REMOTE_BYTES:-unknown}"

[[ -d "$DEST" ]] || { echo "ERROR: $DEST is not a directory" >&2; exit 1; }

if command -v sha256sum >/dev/null 2>&1; then
  sha256() { sha256sum "$@"; }
elif command -v shasum >/dev/null 2>&1; then
  sha256() { shasum -a 256 "$@"; }
else
  echo "ERROR: no sha256sum or shasum available; cannot fingerprint." >&2
  exit 1
fi

TMP="$(mktemp)"
trap 'rm -f "$TMP" "$TMP.want"' EXIT

# LC_ALL=C so the sort order is byte order and the manifest hash is portable.
# A read loop rather than `xargs -d` / `find -printf`, both of which are GNU-only.
echo "==> Hashing $DEST (this reads every byte; ~1 min per 5 GB)"
# The sed normalises binary-mode output: Git Bash on Windows prints `hash *./file`
# where Linux prints `hash  ./file`, so a manifest written on one would falsely
# mismatch on the other.
( cd "$DEST" && find . -type f | LC_ALL=C sort | while IFS= read -r f; do
    sha256 "$f"
  done ) | sed 's/^\([0-9a-f]\{64\}\) \*/\1  /' > "$TMP"

FILE_COUNT="$(grep -c . "$TMP" || true)"
[[ "$FILE_COUNT" -gt 0 ]] || { echo "ERROR: $DEST contains no files" >&2; exit 1; }
MANIFEST_SHA="$(sha256 < "$TMP" | awk '{print $1}')"
LOCAL_BYTES="$( cd "$DEST" && find . -type f -exec cat {} + | wc -c | tr -d ' ' )"

if [[ -n "${VERIFY:-}" ]]; then
  [[ -f "$VERIFY" ]] || { echo "ERROR: VERIFY manifest $VERIFY not found" >&2; exit 1; }
  grep -v '^#' "$VERIFY" | grep . | tr -d '\r' | sed 's/^\([0-9a-f]\{64\}\) \*/\1  /' > "$TMP.want" || true
  if ! diff -q "$TMP.want" "$TMP" >/dev/null; then
    echo "ERROR: $DEST does not match $VERIFY" >&2
    diff "$TMP.want" "$TMP" | head -20 >&2 || true   # diff exits 1 here; keep exit 3
    exit 3
  fi
  echo "==> Verified: all $FILE_COUNT files match $VERIFY"
  echo "    manifest_sha256: $MANIFEST_SHA"
  exit 0
fi

mkdir -p "$RESULTS"
MANIFEST="$RESULTS/checkpoint-sha256-${CKPT}.txt"
{
  echo "# Checkpoint fingerprint -- TEAM_PLAN P0-5"
  echo "# checkpoint:      $CKPT"
  echo "# source:          $SOURCE"
  echo "# dest:            $DEST"
  echo "# remote_bytes:    $REMOTE_BYTES"
  echo "# local_bytes:     $LOCAL_BYTES"
  echo "# files:           $FILE_COUNT"
  echo "# manifest_sha256: $MANIFEST_SHA   <-- quote THIS as the checkpoint hash"
  echo "# vendor_sha:      $(git -C "$repo_root/vendor/ttt-e2e" rev-parse HEAD 2>/dev/null || echo 'submodule not initialised')"
  echo "# fingerprinted:   $(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "# by:              ${USER:-unknown}@$(hostname)"
  echo "#"
  echo "# Per-file sha256, paths relative to $DEST, sorted (LC_ALL=C):"
  cat "$TMP"
} > "$MANIFEST"

if [[ "$REMOTE_BYTES" != "unknown" && "$LOCAL_BYTES" != "$REMOTE_BYTES" ]]; then
  echo "WARNING: local bytes ($LOCAL_BYTES) != remote bytes ($REMOTE_BYTES)."
  echo "         The transfer may be incomplete. Re-run the fetch -- 'cp -n' will"
  echo "         fetch only what is missing."
fi

echo "==> Fingerprint written"
echo "    manifest:        $MANIFEST   (git-ignored -- transcribe the hash into a tracked file)"
echo "    manifest_sha256: $MANIFEST_SHA"
