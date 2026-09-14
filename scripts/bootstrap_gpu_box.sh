#!/usr/bin/env bash
# Cold-start a fresh GPU box: nothing -> ready to run the baseline eval.
#
#   vendor submodule at the pinned SHA
#   vendor environment installed (its own uv.lock)
#   `import ttt` succeeds
#   1B checkpoint on local disk, sha256 recorded
#   books3 /val on local disk (the loader cannot read gs://)
#   training.exp_dir created OUTSIDE this repo
#   a ready-to-paste eval command written out, with every path filled in
#   everything above logged to a file
#
# ---------------------------------------------------------------------------
# THIS SCRIPT HAS NEVER BEEN RUN. It cannot be validated without hardware: there
# is no GPU, no `gcloud` auth and no CUDA stack on any machine we have. It has
# been shell-parsed (`bash -n`) and its guard clauses exercised, nothing more.
# Its real test is the first booking, and it should be read before it is trusted.
# Every step is idempotent, so re-running after a failure is the intended repair.
# ---------------------------------------------------------------------------
#
# Usage:
#   export GCP_BILLING_PROJECT=... WANDB_ENTITY=... WANDB_PROJECT=... WANDB_KEY=...
#   bash scripts/bootstrap_gpu_box.sh
#
# Useful overrides:
#   CKPT=125m_ttt_e2e_finetune_books_8k_1x_cc   # the cheap rehearsal (do this first)
#   DATA_ROOT=/mnt/data  EXP_DIR=/mnt/runs
#   SKIP_DATA=1                                 # checkpoint only
#   ALLOW_SHA_DRIFT=1                           # accept a vendor SHA != the pin
set -euo pipefail

CKPT="${CKPT:-1b_ttt_e2e_finetune_books_8k_1x_cc}"
BUCKET="${BUCKET:-gs://ttt-e2e-checkpoints}"
# The zarr store root is data.zarr/ INSIDE the bucket, not the bucket root --
# listed 2026-09-14. LocalStore() is pointed at this, so every path below and
# deploy_paths.data.books3 must resolve to the directory holding zarr.json.
DATA_BUCKET="${DATA_BUCKET:-gs://llama3-books3/data.zarr}"
PINNED_SHA="a4fc4788ace38e29b5067916d4f4be33da894085"

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

# Both default outside the repo. exp_dir especially: the vendor's default is
# ./experiments, which from the repo root writes into our own tracked tree.
DATA_ROOT="${DATA_ROOT:-$HOME/ttt-data}"
EXP_DIR="${EXP_DIR:-$HOME/ttt-runs}"
BOOKS3_LOCAL="$DATA_ROOT/llama3-books3/data.zarr"   # mirrors the remote store root
CKPT_DEST="$repo_root/checkpoints/$CKPT"     # checkpoints/ is git-ignored

say()  { printf '\n==> %s\n' "$*"; }
skip() { printf '    [already done] %s\n' "$*"; }
die()  { printf '\nFATAL: %s\n' "$*" >&2; exit 1; }

# ------------------------------------------------- containment check FIRST ---
# This runs before anything creates a directory. The log itself lives under
# EXP_DIR, so validating EXP_DIR after opening the log would already have
# written into the tree this check exists to protect.
abspath() {  # normalise without requiring the path to exist
  if command -v realpath >/dev/null 2>&1; then realpath -m "$1" 2>/dev/null && return; fi
  case "$1" in
    /*) printf '%s\n' "$1" ;;
    *)  printf '%s/%s\n' "$PWD" "$1" ;;
  esac
}
inside_repo() {
  local p; p="$(abspath "$1")"
  case "$p" in "$repo_root"|"$repo_root"/*) return 0 ;; *) return 1 ;; esac
}

inside_repo "$EXP_DIR" && die "EXP_DIR ($EXP_DIR) is inside the repo.
       The vendor writes run output there and it would land in our tracked tree --
       .gitignore covers experiments/*/results/ but not experiments/demo/.
       Set EXP_DIR to a path outside $repo_root."
inside_repo "$DATA_ROOT" && die "DATA_ROOT ($DATA_ROOT) is inside the repo.
       Datasets are large and must not enter the tree.
       Set DATA_ROOT to a path outside $repo_root."

# ------------------------------------------------------------------ logging --
mkdir -p "$EXP_DIR/bootstrap"
LOG="$EXP_DIR/bootstrap/bootstrap-$(date -u +%Y%m%dT%H%M%SZ).log"
exec > >(tee -a "$LOG") 2>&1

say "Bootstrap starting"
echo "    repo      : $repo_root"
echo "    checkpoint: $CKPT"
echo "    data root : $DATA_ROOT"
echo "    exp_dir   : $EXP_DIR"
echo "    log       : $LOG"

# ----------------------------------------------------------------- preflight --
say "Step 0 — preflight"

: "${GCP_BILLING_PROJECT:?Set GCP_BILLING_PROJECT -- both buckets are requester-pays}"
: "${WANDB_ENTITY:?Set WANDB_ENTITY -- W&B is mandatory, log_wandb=false does not avoid it (ADR-004 s3)}"
: "${WANDB_PROJECT:?Set WANDB_PROJECT}"
: "${WANDB_KEY:?Set WANDB_KEY -- a real key. This is procurement (P0-10), not configuration}"

missing=()
for t in git uv gsutil; do command -v "$t" >/dev/null 2>&1 || missing+=("$t"); done
[[ ${#missing[@]} -eq 0 ]] || die "missing tools: ${missing[*]}"
echo "    tools     : git uv gsutil present"

if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader \
    | sed 's/^/    gpu       : /'
else
  echo "    gpu       : WARNING - nvidia-smi not found. The vendor needs CUDA 12.8 / cuDNN 9.8."
fi

# Free space: checkpoint + dataset + wheels. 100 GB is comfortable, not tight.
avail_kb="$(df -Pk "$HOME" | awk 'NR==2{print $4}')"
echo "    disk      : $((avail_kb / 1024 / 1024)) GB available under $HOME"
[[ "$avail_kb" -gt 52428800 ]] || echo "    disk      : WARNING - under 50 GB free."

# ------------------------------------------------------------- 1. submodule --
say "Step 1 — vendor submodule at the pinned SHA"
git submodule update --init --recursive vendor/ttt-e2e
actual="$(git -C vendor/ttt-e2e rev-parse HEAD)"
if [[ "$actual" != "$PINNED_SHA" ]]; then
  if [[ "${ALLOW_SHA_DRIFT:-0}" == "1" ]]; then
    echo "    WARNING: vendor at $actual, pin is $PINNED_SHA (ALLOW_SHA_DRIFT=1)"
  else
    die "vendor at $actual, expected $PINNED_SHA.
       The interceptor monkeypatch targets MetaModel.inner_loop_step at the pin, and every
       file-and-line citation in EVAL_ENTRYPOINT.md is against it. Re-read the vendor source
       and run tests/test_interceptor.py before accepting a bump.
       To proceed anyway: ALLOW_SHA_DRIFT=1 bash scripts/bootstrap_gpu_box.sh"
  fi
else
  echo "    vendor at $actual (matches pin)"
fi

# ADR-002: the vendor tree has no licence and must never be edited.
if [[ -n "$(git -C vendor/ttt-e2e status --porcelain)" ]]; then
  git -C vendor/ttt-e2e status --short
  die "vendor/ttt-e2e is dirty. ADR-002: it has no licence and must stay pristine.
       Revert those changes before continuing."
fi
echo "    vendor tree clean (ADR-002)"

# ----------------------------------------------------------- 2. environment --
# `uv sync --frozen` is itself idempotent; re-running it after a partial install
# resumes rather than restarting.
say "Step 2 — vendor environment (uv sync --frozen)"
( cd vendor/ttt-e2e && uv sync --frozen )

say "Step 3 — import check"
( cd vendor/ttt-e2e && uv run --exact python -c "import ttt; print('    import ttt OK')" )

# ------------------------------------------------------------ 4. checkpoint --
# fetch_checkpoints.sh probes before transferring and refuses to exceed MAX_BYTES,
# so a wrong CKPT name costs a metadata call rather than an egress bill.
say "Step 4 — checkpoint"
if [[ -d "$CKPT_DEST" ]] && [[ -n "$(ls -A "$CKPT_DEST" 2>/dev/null)" ]]; then
  skip "checkpoint present at $CKPT_DEST"
  echo "    (delete it to force a refetch)"
else
  CKPT="$CKPT" BUCKET="$BUCKET" DEST="$CKPT_DEST" bash scripts/fetch_checkpoints.sh
fi

# --------------------------------------------------------------- 5. dataset --
# The loader is LocalStore-only (lm_dataset.py:14) -- there is no gs:// path, so
# this copy is mandatory, not an optimisation.
#
# We copy /val plus BOTH arrays' metadata. /train's metadata is required because
# _make_train_iterator (train.py:125) opens the train array before the eval
# branch returns; with zarr v3 its absent chunks read as fill value, so no /train
# chunk data is needed. See COST_MODEL.md s1.1.
say "Step 5 — dataset (books3 /val -> local disk)"
if [[ "${SKIP_DATA:-0}" == "1" ]]; then
  echo "    SKIP_DATA=1 -- skipping. The eval will fail without it."
elif [[ -d "$BOOKS3_LOCAL/val" ]] && [[ -f "$BOOKS3_LOCAL/train/zarr.json" ]]; then
  skip "books3 /val present at $BOOKS3_LOCAL"
else
  mkdir -p "$BOOKS3_LOCAL/train"
  g() { gsutil -u "$GCP_BILLING_PROJECT" "$@"; }

  echo "    probing $DATA_BUCKET/val (metadata only, no transfer)"
  if val_du="$(g du -s "$DATA_BUCKET/val" 2>&1)"; then
    echo "    /val bytes: $val_du"
    printf '%s\n' "$val_du" > "$EXP_DIR/bootstrap/books3-val-size.txt"
  else
    echo "    WARNING: could not probe /val. Layout may differ from the assumption."
    echo "$val_du"
  fi

  # Group metadata. zarr v3 puts zarr.json at the store root; v2 uses .zgroup.
  g cp "$DATA_BUCKET/zarr.json" "$BOOKS3_LOCAL/" 2>/dev/null \
    || g cp "$DATA_BUCKET/.zgroup" "$BOOKS3_LOCAL/" 2>/dev/null \
    || echo "    note: no root group metadata found (may be a bare array store)"

  # /train metadata only -- deliberately no chunks.
  g cp "$DATA_BUCKET/train/zarr.json" "$BOOKS3_LOCAL/train/" 2>/dev/null \
    || g cp "$DATA_BUCKET/train/.zarray" "$BOOKS3_LOCAL/train/" 2>/dev/null \
    || echo "    note: no /train metadata found -- if the eval crashes opening /train, this is why"

  echo "    copying /val (this is the transfer)"
  g -m cp -r "$DATA_BUCKET/val" "$BOOKS3_LOCAL/"
fi

if [[ -d "$BOOKS3_LOCAL" ]]; then
  echo "    local dataset: $BOOKS3_LOCAL"
  ls -1 "$BOOKS3_LOCAL" 2>/dev/null | sed 's/^/      /'
fi

# --------------------------------------------------------------- 6. exp_dir --
say "Step 6 — exp_dir outside the repo"
mkdir -p "$EXP_DIR"
echo "    $EXP_DIR"

# ---------------------------------------------------------- 7. env for P1-3 --
say "Step 7 — environment record (for P1-3)"
ENVREC="$EXP_DIR/bootstrap/env-record.txt"
{
  echo "recorded_utc:   $(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "host:           $(hostname)"
  echo "overlay_sha:    $(git rev-parse HEAD)"
  echo "vendor_sha:     $actual"
  echo "checkpoint:     $CKPT"
  echo "uv:             $(uv --version 2>/dev/null || echo n/a)"
  echo "gsutil:         $(gsutil version 2>/dev/null | head -1 || echo n/a)"
  echo "nvidia-smi:     $(nvidia-smi --query-gpu=name,driver_version --format=csv,noheader 2>/dev/null | head -1 || echo n/a)"
  echo "nvcc:           $(nvcc --version 2>/dev/null | tail -1 || echo n/a)"
  ( cd vendor/ttt-e2e && uv run --exact python -c \
      "import jax; print('jax:            '+jax.__version__); print('jax_devices:    '+str(jax.devices()))" \
    ) 2>/dev/null || echo "jax:            n/a"
} > "$ENVREC"
cat "$ENVREC" | sed 's/^/    /'

# --------------------------------------------------- 8. the eval command -----
# Written to a file rather than only printed, so the first session runs a
# reviewed command instead of one retyped at 2am.
say "Step 8 — writing the eval command"
CMD="$EXP_DIR/bootstrap/eval-command-${CKPT}.sh"
cat > "$CMD" <<EOF
#!/usr/bin/env bash
# Baseline eval for $CKPT -- generated $(date -u +%Y-%m-%dT%H:%M:%SZ) by bootstrap_gpu_box.sh.
# Derivation and citations: experiments/000-repro-baseline/EVAL_ENTRYPOINT.md
# Decision record:          docs/adr/ADR-004-baseline-eval-invocation.md
# Pre-registered bar:       experiments/000-repro-baseline/TOLERANCE.md
#
# The gate must NOT be installed for this run: is_installed() == False throughout.
set -euo pipefail
cd "$repo_root/vendor/ttt-e2e"

uv run --exact train \\
  +deploy=interactive \\
  +experiment=1b/extension/ext-1b-e2e-32K \\
  training.eval_mode=true \\
  training.seq_length=8192 \\
  training.global_batch_size=2 \\
  training.exp_name=eval-${CKPT} \\
  training.load_part=params \\
  checkpoint.resume_checkpoint_dir=$CKPT_DEST \\
  deploy_paths.data.books3=$BOOKS3_LOCAL \\
  training.exp_dir=$EXP_DIR \\
  training.wandb_entity=$WANDB_ENTITY \\
  training.wandb_project=$WANDB_PROJECT \\
  training.wandb_key=\$WANDB_KEY

# Result: "Eval -- train_holdout/loss: <value>"  (mean CE, nats/token)
# Per-token curve: $EXP_DIR/demo/eval-${CKPT}/train_holdout_token_nll_loss.npy
EOF
chmod +x "$CMD"
echo "    $CMD"
echo
echo "    NOTE: --experiment is hardcoded to the 1B config. For the 125M or 3B"
echo "          checkpoints, edit +experiment= and training.seq_length to match."

# ------------------------------------------------------------------- done ----
say "Bootstrap complete"
cat <<EOF

  Next, in order:

  1. Read $CMD before running it. It has never been executed.
  2. Run it. Expect the first launch to fail; the four known ways are in
     COST_MODEL.md s4 (W&B, eval batch size, exp_dir, the zarr /train layout).
  3. Copy results out -- results/ and *.npy are BOTH git-ignored, so a run that
     is not deliberately transcribed leaves no record:
       cp $EXP_DIR/demo/eval-${CKPT}/train_holdout_token_nll_loss.npy \\
          $repo_root/experiments/000-repro-baseline/results/
  4. Compare against experiments/000-repro-baseline/TOLERANCE.md. Read the bar
     BEFORE reading the number -- it is pre-registered for a reason (Rule 5).

  Log: $LOG
EOF
