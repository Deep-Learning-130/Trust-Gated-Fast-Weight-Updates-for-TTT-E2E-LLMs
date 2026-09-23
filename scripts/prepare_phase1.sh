#!/usr/bin/env bash
# Prepare the box for runbook C2-C5 (001 spike, sequence arms, Phase 2 gate).
#
# Run ONCE, after scripts/run_gpu_session.sh has returned PASS for 000, in the
# same tmux session and with the same EXP_DIR:
#
#   export GCP_BILLING_PROJECT=<project>     # or TRAIN_DIR=<side-loaded train-zarr>, below
#   bash scripts/prepare_phase1.sh
#   source "$EXP_DIR/phase1.env"          # defines $TG, $T, $SIZE, $CKPT_DEST, $CKPT_MANIFEST
#
# TRAIN_DIR: a directory holding train/zarr.json and train chunk 0, fetched
# elsewhere (scripts/colab/fetch_checkpoint_colab.ipynb, cell 5). With it, no
# billing project or gsutil is needed.
#
# Why this exists: nothing on the box had an interpreter that could run the
# checkpoint CLI. `trustgate` is deliberately not installed anywhere (it runs
# from PYTHONPATH=src, see experiments/003-smoke-125m/setup.sh), the vendor env
# lacks the fluency scorer's `tokenizers`/`safetensors`, the reference model was
# never fetched, and the .npy token files the CLI reads did not exist. Each of
# those would have surfaced as a crash minutes into a billed run.
#
# Every step is idempotent. It ends with "Phase 1 prep complete" or stops at the
# first failure.
#
# Note: run_gpu_session.sh's acceptance step uses `uv run --exact`, which strips
# packages not in the vendor lock. Re-running 000 after this script removes the
# two fluency packages again; re-run this script afterwards.
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
EXP_DIR="${EXP_DIR:-$HOME/ttt-runs}"
die() { printf '\nFATAL: %s\n' "$*" >&2; exit 1; }
say() { printf '\n== %s\n' "$*"; }

# Same reason as run_gpu_session.sh: the bootstrap's PATH export for uv and the
# Cloud SDK does not survive into this shell.
export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$HOME/google-cloud-sdk/bin:$PATH"
command -v uv >/dev/null 2>&1 || die "uv not on PATH; the bootstrap installs it to ~/.local/bin"
if [[ -n "${TRAIN_DIR:-}" ]]; then
  [[ -f "$TRAIN_DIR/train/zarr.json" ]] || die "TRAIN_DIR=$TRAIN_DIR has no train/zarr.json (expected the Colab notebook's train-zarr)."
else
  command -v gsutil >/dev/null 2>&1 || die "gsutil not on PATH; dump_tokens.py needs it to fetch /train chunk 0 (or set TRAIN_DIR)"
fi

[[ -f "$EXP_DIR/bootstrap/session.env" ]] || die "$EXP_DIR/bootstrap/session.env missing: run bootstrap_gpu_box.sh first (same EXP_DIR)."
# shellcheck disable=SC1091
source "$EXP_DIR/bootstrap/session.env"
[[ -n "${TRAIN_DIR:-}" ]] || : "${GCP_BILLING_PROJECT:?export GCP_BILLING_PROJECT (needed to fetch /train chunk 0), or set TRAIN_DIR to a side-loaded copy}"

# --size for every C2-C5 command follows the checkpoint, as +experiment= does in
# the bootstrap, so a 125M checkpoint is never restored into a 1B skeleton.
SIZE="${CKPT%%_*}"
case "$SIZE" in
  125m|350m|760m|1b|3b) ;;
  *) die "cannot derive --size from CKPT='$CKPT'" ;;
esac

VPY="$repo_root/vendor/ttt-e2e/.venv/bin/python"
[[ -x "$VPY" ]] || die "$VPY not found: the vendor env is missing (bootstrap Step 2)."
cd "$repo_root"

say "1. fluency scorer dependencies into the vendor env"
uv pip install --python "$VPY" "safetensors>=0.4" "tokenizers>=0.19"

say "2. one interpreter sees the vendor, trustgate and the GPU"
PYTHONPATH="$repo_root/src" "$VPY" - <<'PY'
import jax
import safetensors, tokenizers  # noqa: F401
import ttt.model.transformer  # noqa: F401
import trustgate.eval.cli, trustgate.eval.gate_eval  # noqa: F401
devs = jax.devices()
print("   devices:", devs)
if all(d.platform == "cpu" for d in devs):
    raise SystemExit("JAX sees no GPU")
PY

say "3. independent fluency reference (GPT-2 small + Llama-3 decoder)"
"$VPY" scripts/fetch_reference_model.py
PYTHONPATH="$repo_root/src" "$VPY" -m trustgate.eval.cli --fluency-selftest

say "4. token files"
T="$EXP_DIR/tokens"
if [[ -f "$T/tokens-manifest.json" ]]; then
  echo "   already written: $T"
elif [[ -n "${TRAIN_DIR:-}" ]]; then
  "$VPY" scripts/dump_tokens.py --books3 "$BOOKS3_LOCAL" --train-dir "$TRAIN_DIR" --out "$T"
else
  "$VPY" scripts/dump_tokens.py --books3 "$BOOKS3_LOCAL" \
    --billing "$GCP_BILLING_PROJECT" --out "$T"
fi

say "5. gate smoke: random-init 125m, the real vendor forward (~minutes, no verdict)"
SMOKE="$EXP_DIR/gate-smoke"
PYTHONPATH="$repo_root/src" "$VPY" -m trustgate.eval.cli \
  --objective degrade --strategy select --gate-eval \
  --random-init --size 125m --stream-tokens 8192 --seeds 0 1 \
  --overhead-repeats 3 --out "$SMOKE" > "$SMOKE.log" 2>&1 \
  || { tail -40 "$SMOKE.log"; die "gate smoke failed; log: $SMOKE.log"; }
PYTHONPATH="$repo_root/src" "$VPY" - "$SMOKE/gate-result.pkl" <<'PY'
import math, pickle, sys
r = pickle.load(open(sys.argv[1], "rb"))
c = r.calibration
top = max(c.divergences)
print(f"   calibration: {len(c.divergences)} scorable, {c.unscorable_steps} unscorable, max divergence {top:.6g}")
if c.unscorable_steps or not top > 0 or not math.isfinite(top):
    raise SystemExit("probe forward is broken: divergence is zero, nan or unscorable. Do NOT run C5.")
print(f"   overhead (125m, 3 repeats, indicative only): {r.overhead.overhead_fraction * 100:+.1f}%")
PY

say "6. writing $EXP_DIR/phase1.env"
mkdir -p "$repo_root/experiments/001-attack-spike/results/logs"
cat > "$EXP_DIR/phase1.env" <<EOF
# Source this before C2-C5: source "$EXP_DIR/phase1.env"
# Every path here is absolute, so C2-C5 work from any directory.
export EXP_DIR="$EXP_DIR"
export R="$repo_root/experiments/001-attack-spike/results"
export T="$T"
export SIZE="$SIZE"
# Empty = the vendor default (bf16). Every C2-C5 command takes it as $DTYPE_FLAG.
export DTYPE_FLAG="${COMPUTE_DTYPE:+--compute-dtype $COMPUTE_DTYPE}"
export CKPT_DEST="$CKPT_DEST"
export CKPT_MANIFEST="$CKPT_MANIFEST"
# Unbuffered, so the timing lines reach the tee'd log as they happen.
export TG="env PYTHONPATH=$repo_root/src PYTHONUNBUFFERED=1 XLA_PYTHON_CLIENT_PREALLOCATE=false $VPY -m trustgate.eval.cli"
EOF
cat "$EXP_DIR/phase1.env"

say "Phase 1 prep complete"
