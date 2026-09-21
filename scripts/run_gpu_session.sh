#!/usr/bin/env bash
# Run GPU session 1 end to end, unattended, after bootstrap_gpu_box.sh succeeded.
#
#   reshape to smoke -> smoke pass -> reshape to VAL_TOKENS
#   -> baseline run 1 -> baseline run 2 (bar S2) -> negative control (bar S3)
#   -> collect + redact -> acceptance check
#
# Run it INSIDE tmux. It stops at the first failure and says where to look.
# Each phase's artefacts are copied out as soon as that phase ends, because run 2
# writes to the same log_dir as run 1 and would overwrite its .npy.
#
# Usage:
#   export WANDB_KEY=...                      # same shell as the bootstrap
#   bash scripts/run_gpu_session.sh
#
# Overrides:
#   EXP_DIR=/mnt/runs          must match the bootstrap
#   DEADLINE_HOURS=4           drop order kicks in to stay inside this (runbook s4)
#   OOM_RETRY=1                on a smoke OOM, retry once at eval batch 4 (records it)
#   START_AT=eval-1            resume from a phase: smoke | eval-1 | eval-2 | control | collect
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
EXP_DIR="${EXP_DIR:-$HOME/ttt-runs}"
BOOT="$EXP_DIR/bootstrap"
DEADLINE_HOURS="${DEADLINE_HOURS:-4}"
OOM_RETRY="${OOM_RETRY:-1}"
START_AT="${START_AT:-smoke}"

die() { printf '\nFATAL: %s\n' "$*" >&2; exit 1; }

# The bootstrap installs uv (and possibly the Cloud SDK) with an `export PATH`
# that dies with its own process, so the shell running this may not see them.
# Every phase below shells out to `uv run`; find it here, not in phase one.
export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$HOME/google-cloud-sdk/bin:$PATH"
command -v uv >/dev/null 2>&1 || die "uv not on PATH. The bootstrap installs it to ~/.local/bin; re-run the bootstrap."

[[ -f "$BOOT/session.env" ]] || die "$BOOT/session.env not found. Run scripts/bootstrap_gpu_box.sh first (with the same EXP_DIR)."
# shellcheck disable=SC1091
source "$BOOT/session.env"
: "${WANDB_KEY:?export WANDB_KEY in this shell first}"

SMOKE_CMD="$BOOT/1-smoke-${CKPT}.sh"
REAL_CMD="$BOOT/2-eval-${CKPT}.sh"
CTRL_CMD="$BOOT/3-dummy-control-${CKPT}.sh"
for f in "$SMOKE_CMD" "$REAL_CMD" "$CTRL_CMD" "$BOOT/reshape-to-smoke.sh" "$BOOT/reshape-to-real.sh"; do
  [[ -f "$f" ]] || die "$f missing. Re-run the bootstrap."
done

OUT="$EXP_DIR/collected"
mkdir -p "$OUT/logs"
SUMMARY="$OUT/SESSION_SUMMARY.txt"
record() { printf '%s\n' "$*" | tee -a "$SUMMARY"; }

if [[ -z "${TMUX:-}" && -z "${STY:-}" ]]; then
  echo "WARNING: not inside tmux/screen. If SSH drops, this session dies. Ctrl-C now and run: tmux new -s ttt"
  sleep 10
fi

session_start="$(date +%s)"
record "# GPU session summary"
record "started_utc:   $(date -u +%Y-%m-%dT%H:%M:%SZ)"
record "checkpoint:    $CKPT"
record "val_tokens:    $VAL_TOKENS"
record "deadline_h:    $DEADLINE_HOURS"
record "overlay_sha:   $(git -C "$repo_root" rev-parse HEAD)"

# ---- GPU memory sampler: COST_MODEL s9.4.1 asks for the measured peak.
MEMLOG="$OUT/gpu-memory-mib.log"
( while true; do
    printf '%s %s\n' "$(date -u +%H:%M:%S)" \
      "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | head -1 | tr -d ' ')"
    sleep 5
  done ) >> "$MEMLOG" 2>/dev/null &
MEMPID=$!
finish() {
  kill "$MEMPID" 2>/dev/null || true
  local peak
  peak="$(awk '$2 ~ /^[0-9]+$/ && $2 > m {m = $2} END {print m + 0}' "$MEMLOG" 2>/dev/null || echo 0)"
  record "peak_gpu_mib:  $peak   (sampled every 5 s; COST_MODEL s9.4.1)"
  record "ended_utc:     $(date -u +%Y-%m-%dT%H:%M:%SZ)"
  record "elapsed_min:   $(( ($(date +%s) - session_start) / 60 ))"
}
trap finish EXIT

elapsed() { echo $(( $(date +%s) - session_start )); }
deadline_s="$(awk -v h="$DEADLINE_HOURS" 'BEGIN { printf "%d", h * 3600 }')"   # fractional hours allowed

phase_order=(smoke eval-1 eval-2 control collect)
should_run() {  # phase -> 0 if at or after START_AT
  local want="$1" seen=0 p
  for p in "${phase_order[@]}"; do
    [[ "$p" == "$START_AT" ]] && seen=1
    [[ "$p" == "$want" ]] && { [[ "$seen" == "1" ]]; return; }
  done
  return 1
}
printf '%s\n' "${phase_order[@]}" | grep -qx "$START_AT" || die "START_AT=$START_AT is not one of: ${phase_order[*]}"

# run_phase <name> <script> <exp_name>  -- sets PHASE_SECONDS
run_phase() {
  local name="$1" script="$2" exp_name="$3"
  local npy="$EXP_DIR/demo/$exp_name/train_holdout_token_nll_loss.npy"
  local t0 rc
  rm -f "$npy"   # never collect a stale curve from an earlier run
  record ""
  record "== $name: $(date -u +%H:%M:%S)  (session elapsed $(( $(elapsed) / 60 )) min)"
  record "   command: $script  EVAL_GBS=${EVAL_GBS:-2} EVAL_BATCH_SIZE=${EVAL_BATCH_SIZE:-8}"
  t0="$(date +%s)"
  set +e
  RUN_LOG="$OUT/logs/$name.log" bash "$script"
  rc=$?
  set -e
  PHASE_SECONDS=$(( $(date +%s) - t0 ))
  record "   exit: $rc   wall: $(( PHASE_SECONDS / 60 )) min $(( PHASE_SECONDS % 60 )) s"
  if [[ "$rc" == "0" ]]; then
    record "   $(grep 'Eval -- train_holdout/loss:' "$OUT/logs/$name.log" | tail -1 | sed 's/.*\(Eval -- \)/\1/')"
    mkdir -p "$OUT/$name"
    if [[ -f "$npy" ]]; then
      cp "$npy" "$OUT/$name/"
    else
      record "   WARNING: $npy not written"
    fi
  fi
  return "$rc"
}

# XLA's allocator failure reads "RESOURCE_EXHAUSTED: Out of memory while trying to allocate".
is_oom() { grep -qE 'RESOURCE_EXHAUSTED|[Oo]ut of memory' "$1"; }

fail_phase() {
  record "   FAILED. Last 40 lines of $OUT/logs/$1.log:"
  tail -40 "$OUT/logs/$1.log" | sed 's/^/     | /' | tee -a "$SUMMARY"
  record ""
  record "STOPPED at $1. Nothing downstream is attributable (runbook s5)."
  record "Resume after a fix with: START_AT=$1 bash scripts/run_gpu_session.sh"
  exit 1
}

# ------------------------------------------------------------------ smoke ---
if should_run smoke; then
  bash "$BOOT/reshape-to-smoke.sh" | tee -a "$SUMMARY"
  if ! run_phase smoke "$SMOKE_CMD" "smoke-${CKPT}"; then
    if [[ "$OOM_RETRY" == "1" ]] && is_oom "$OUT/logs/smoke.log"; then
      record "   OOM at eval batch 8. Retrying once at eval batch 4 (runbook s3)."
      record "   RUN CONDITION CHANGED: training.global_batch_size=1 training.eval_batch_size=4"
      mv "$OUT/logs/smoke.log" "$OUT/logs/smoke-oom-batch8.log"
      export EVAL_GBS=1 EVAL_BATCH_SIZE=4
      echo "EVAL_GBS=1" >> "$BOOT/session.env"
      echo "EVAL_BATCH_SIZE=4" >> "$BOOT/session.env"
      run_phase smoke "$SMOKE_CMD" "smoke-${CKPT}" || fail_phase smoke
    else
      fail_phase smoke
    fi
  fi
fi

# A resumed session must keep the batch the smoke pass settled on.
export EVAL_GBS="${EVAL_GBS:-2}" EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-8}"

# ------------------------------------------------------------- baseline -----
run1_seconds=0
if should_run eval-1; then
  bash "$BOOT/reshape-to-real.sh" "$VAL_TOKENS" | tee -a "$SUMMARY"
  run_phase eval-1 "$REAL_CMD" "eval-${CKPT}" || fail_phase eval-1
  run1_seconds=$PHASE_SECONDS
fi

# Drop order (runbook s4): run 2 first, then the control. The control reads 2^25
# tokens, ~22% of a 150M run, plus its own compile; estimate it that way.
control_estimate=$(( run1_seconds * 25 / 100 + 900 ))

if should_run eval-2; then
  if (( $(elapsed) + run1_seconds + control_estimate > deadline_s )); then
    record ""
    record "== eval-2: SKIPPED by drop order -- would pass DEADLINE_HOURS=$DEADLINE_HOURS. Bar S2 is UNVERIFIED."
  else
    run_phase eval-2 "$REAL_CMD" "eval-${CKPT}" || fail_phase eval-2
  fi
fi

if should_run control; then
  if (( $(elapsed) + control_estimate > deadline_s )); then
    record ""
    record "== control: SKIPPED by drop order -- would pass DEADLINE_HOURS=$DEADLINE_HOURS. Bar S3 is UNVERIFIED."
  else
    run_phase control "$CTRL_CMD" "dummy-${CKPT}" || fail_phase control
  fi
fi

# --------------------------------------------------------------- collect ----
# Never dropped. Copy-out and redaction, then the acceptance check on the copy.
record ""
record "== collect + acceptance: $(date -u +%H:%M:%S)"
finish; trap - EXIT   # write the peak memory into the summary before it is copied

cp "$BOOKS3_LOCAL/val-subset-manifest.json" "$OUT/" 2>/dev/null || record "   WARNING: no val-subset manifest"
cp "$BOOT/env-record.txt" "$OUT/" 2>/dev/null || true
cp "$CKPT_MANIFEST" "$OUT/" 2>/dev/null || record "   WARNING: no checkpoint fingerprint"
cp "$BOOT"/bootstrap-*.log "$OUT/logs/" 2>/dev/null || true
cp "$BOOT"/*.sh "$OUT/logs/" 2>/dev/null || true

DEST="$repo_root/experiments/000-repro-baseline/results/session-$(date -u +%Y%m%dT%H%M%SZ)"
python3 "$repo_root/scripts/collect_results.py" --src "$OUT" --dest "$DEST" \
  || die "redaction check failed -- see above. Do NOT copy $OUT off the box as-is."

set +e
# The vendor env, because S1 reads the .npy and the system python3 may lack numpy.
# That env has no trustgate package at all, which is itself part of S4.
( cd "$repo_root/vendor/ttt-e2e" && uv run --exact python "$repo_root/scripts/check_baseline_acceptance.py" \
    --collected "$DEST" --out "$DEST/ACCEPTANCE.txt" )
verdict=$?
set -e

cat <<EOF

==> Session finished. Redacted results: $DEST

  Pull them to your laptop BEFORE releasing the box (run this on the laptop):
    scp -r <user>@<box-ip>:$DEST  experiments/000-repro-baseline/results/

  Then fill the Outcome column in docs/protocols/gpu-bookings.md.
EOF
exit "$verdict"
