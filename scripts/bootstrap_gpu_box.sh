#!/usr/bin/env bash
# Cold-start a fresh GPU box: nothing -> ready to run the baseline eval.
#
#   uv and the gcloud SDK installed if the image lacks them (JarvisLabs images do)
#   vendor submodule at the pinned SHA
#   vendor environment installed (its own uv.lock)
#   `import ttt` succeeds
#   1B checkpoint on local disk, sha256 recorded
#   a TRUNCATED books3 /val on local disk, with a manifest recording what was cut
#   training.exp_dir created OUTSIDE this repo
#   three ready-to-paste commands written out -- smoke, baseline, negative control
#   a shared XLA compilation cache, so the smoke pass pays the compile once
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
#   VAL_TOKENS=150000000                        # eval length; THE wall-clock knob
#   SMOKE_TOKENS=131073                         # 16 sequences = 2 eval batches
#   CKPT=125m_ttt_e2e_finetune_books_8k_1x_cc   # +experiment= follows this automatically
#   DATA_ROOT=/mnt/data  EXP_DIR=/mnt/runs      # point these at persistent storage
#   SKIP_DATA=1                                 # checkpoint only
#   AUTO_INSTALL=0                              # refuse rather than install uv/gcloud
#   ALLOW_SHA_DRIFT=1                           # accept a vendor SHA != the pin
#
# On /val size: the 2026-09-14 probe measured it at 2,000,168,321 tokens (8.4 GB).
# A full pass is ~7.5 GPU-hours at the 1B rate and TOLERANCE.md s5 bar S2 wants two
# of them. VAL_TOKENS is how that fits a booking; scripts/make_val_subset.py records
# what was cut. Resize later with `make_val_subset.py reshape` -- it transfers nothing.
set -euo pipefail

CKPT="${CKPT:-1b_ttt_e2e_finetune_books_8k_1x_cc}"
BUCKET="${BUCKET:-gs://ttt-e2e-checkpoints}"
DATA_BUCKET="${DATA_BUCKET:-gs://llama3-books3}"
PINNED_SHA="a4fc4788ace38e29b5067916d4f4be33da894085"

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

# Both default outside the repo. exp_dir especially: the vendor's default is
# ./experiments, which from the repo root writes into our own tracked tree.
DATA_ROOT="${DATA_ROOT:-$HOME/ttt-data}"
EXP_DIR="${EXP_DIR:-$HOME/ttt-runs}"
BOOKS3_LOCAL="$DATA_ROOT/llama3-books3"
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

# A JarvisLabs (or any non-GCP) image ships neither `uv` nor the gcloud SDK, and
# both are one command away. Dying here used to mean the box billed while someone
# read a README. AUTO_INSTALL=0 restores the old refuse-and-exit behaviour.
AUTO_INSTALL="${AUTO_INSTALL:-1}"

ensure_tool() {  # name, human-readable installer description, installer command
  local tool="$1" what="$2" installer="$3"
  if command -v "$tool" >/dev/null 2>&1; then
    echo "    tools     : $tool present"
    return 0
  fi
  if [[ "$AUTO_INSTALL" != "1" ]]; then
    die "$tool not found and AUTO_INSTALL=0. Install $what and re-run."
  fi
  echo "    tools     : $tool missing -- installing $what"
  eval "$installer" || die "failed to install $tool. Install it by hand and re-run."
  command -v "$tool" >/dev/null 2>&1 || die "$tool still not on PATH after install.
       Open a new shell (or re-source your profile) and re-run; this script is idempotent."
  echo "    tools     : $tool installed"
}

command -v git >/dev/null 2>&1 || die "git not found -- this is a git checkout, so something is very wrong"
echo "    tools     : git present"

ensure_tool uv "astral.sh/uv"   'curl -LsSf https://astral.sh/uv/install.sh | sh && export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"'

ensure_tool gsutil "the Google Cloud SDK"   'curl -sSL https://sdk.cloud.google.com | bash -s -- --disable-prompts && export PATH="$HOME/google-cloud-sdk/bin:$PATH"'

# gsutil exists but is useless without credentials, and the failure otherwise
# lands mid-transfer rather than here.
if ! gcloud auth list --filter=status:ACTIVE --format="value(account)" 2>/dev/null | grep -q .; then
  die "gcloud has no active account. Run:  gcloud auth login
       Both buckets are requester-pays; nothing downstream can start without it.
       This is procurement (T1.3), not configuration -- it is not fixable on a billing clock."
fi
echo "    gcloud    : $(gcloud auth list --filter=status:ACTIVE --format='value(account)' 2>/dev/null | head -1)"

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
say "Step 5 — dataset (books3 /val subset -> local disk)"
# The 2026-09-14 probe measured /val at 2,000,168,321 tokens / 8.4 GB. A full
# pass is ~7.5 GPU-hours at the 1B rate, and TOLERANCE.md s5 bar S2 wants the
# identical command run twice. So we take a subset and record exactly what was
# taken: scripts/make_val_subset.py, which also copies /train's metadata
# (COST_MODEL s1.1 -- without it the run dies after billing starts).
#
# VAL_TOKENS is the one knob that sets eval wall-clock. It can be changed later
# with `reshape`, which transfers nothing.
VAL_TOKENS="${VAL_TOKENS:-150000000}"
SMOKE_TOKENS="${SMOKE_TOKENS:-131073}"   # 16 sequences = exactly 2 eval batches

if [[ "${SKIP_DATA:-0}" == "1" ]]; then
  echo "    SKIP_DATA=1 -- skipping. The eval will fail without it."
else
  # Plain python3, not the vendor env: make_val_subset.py imports nothing outside
  # the standard library, precisely so it runs before `uv sync` has finished.
  export GCP_BILLING_PROJECT
  python3 "$repo_root/scripts/make_val_subset.py"     --dest "$BOOKS3_LOCAL" fetch --tokens "$VAL_TOKENS"
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

# --------------------------------------------------- 8. the eval commands ----
# Written to files rather than only printed, so the first session runs reviewed
# commands instead of ones retyped at 2am. Three of them, because the session
# needs three and "edit this one to make the others" is how bookings get lost.
say "Step 8 — writing the eval commands"

# +experiment follows the checkpoint. It used to be hardcoded to 1B with a note
# telling the reader to edit it; that note was a defect, not documentation.
SIZE="${CKPT%%_*}"
case "$SIZE" in
  125m|350m|760m|1b|3b) EXPERIMENT="${SIZE}/extension/ext-${SIZE}-e2e-32K" ;;
  *) die "cannot derive +experiment from CKPT='$CKPT' (size prefix '$SIZE').
       Known sizes: 125m 350m 760m 1b 3b. Set EXPERIMENT= by hand if this is deliberate." ;;
esac
EXPERIMENT="${EXPERIMENT_OVERRIDE:-$EXPERIMENT}"
echo "    checkpoint: $CKPT"
echo "    experiment: $EXPERIMENT"

# One XLA cache, shared by every command below. Batch size and sequence length
# are identical across them -- only the number of batches differs, which is a
# Python-level loop count -- so the smoke pass compiles the 24-layer scan once
# and the real runs start hot. This is worth 20-60 minutes of a 4-hour booking.
CACHE_DIR="$EXP_DIR/jax-cache"
mkdir -p "$CACHE_DIR"
echo "    xla cache : $CACHE_DIR"

write_eval_cmd() {  # path, exp_name, extra hydra overrides..., preceded by a header comment
  local path="$1" exp_name="$2" header="$3"; shift 3
  cat > "$path" <<EOF
#!/usr/bin/env bash
# $header
# Generated $(date -u +%Y-%m-%dT%H:%M:%SZ) by bootstrap_gpu_box.sh. NEVER EXECUTED before this session.
#
# Derivation and citations: experiments/000-repro-baseline/EVAL_ENTRYPOINT.md
# Decision record:          docs/adr/ADR-004-baseline-eval-invocation.md
# Pre-registered bar:       experiments/000-repro-baseline/TOLERANCE.md
#
# The gate must NOT be installed for this run: is_installed() == False throughout,
# and no trustgate import appears anywhere in it (T1.6).
set -euo pipefail

export JAX_COMPILATION_CACHE_DIR="$CACHE_DIR"
export JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS=1

cd "$repo_root/vendor/ttt-e2e"

uv run --exact train \
  +deploy=interactive \
  +experiment=$EXPERIMENT \
  training.eval_mode=true \
  training.seq_length=8192 \
  training.global_batch_size=2 \
  training.exp_name=$exp_name \
  training.load_part=params \
  checkpoint.resume_checkpoint_dir=$CKPT_DEST \
  deploy_paths.data.books3=$BOOKS3_LOCAL \
  training.exp_dir=$EXP_DIR \
  training.wandb_entity=$WANDB_ENTITY \
  training.wandb_project=$WANDB_PROJECT \
  training.wandb_key=\$WANDB_KEY $*

# Result: "Eval -- train_holdout/loss: <value>"  (mean CE, nats/token)
# Per-token curve: $EXP_DIR/demo/$exp_name/train_holdout_token_nll_loss.npy
EOF
  chmod +x "$path"
  echo "    $path"
}

SMOKE_CMD="$EXP_DIR/bootstrap/1-smoke-${CKPT}.sh"
REAL_CMD="$EXP_DIR/bootstrap/2-eval-${CKPT}.sh"
CTRL_CMD="$EXP_DIR/bootstrap/3-dummy-control-${CKPT}.sh"

write_eval_cmd "$SMOKE_CMD" "smoke-${CKPT}" \
  "SMOKE PASS -- two eval batches. Shakes out the launch and warms the XLA cache. NOT a result."
write_eval_cmd "$REAL_CMD" "eval-${CKPT}" \
  "BASELINE EVAL for $CKPT. Run it TWICE unchanged -- TOLERANCE.md s5 bar S2 wants 4-decimal agreement."
write_eval_cmd "$CTRL_CMD" "dummy-${CKPT}" \
  "NEGATIVE CONTROL (TOLERANCE.md s5 bar S3). Random tokens; the loss must land FAR ABOVE the band." \
  "training.dummy_dataset=true"

# The subset commands sit beside them, because the smoke-to-real transition is a
# reshape and nothing else -- no refetch, no cache invalidation.
RESHAPE_SMOKE="$EXP_DIR/bootstrap/reshape-to-smoke.sh"
RESHAPE_REAL="$EXP_DIR/bootstrap/reshape-to-real.sh"
cat > "$RESHAPE_SMOKE" <<EOF
#!/usr/bin/env bash
set -euo pipefail
python3 "$repo_root/scripts/make_val_subset.py" --dest "$BOOKS3_LOCAL" reshape --tokens $SMOKE_TOKENS
EOF
cat > "$RESHAPE_REAL" <<EOF
#!/usr/bin/env bash
# Resize the eval without refetching. Measure tokens/sec on the smoke pass first,
# then set --tokens so that TWO passes fit the time left in the booking.
set -euo pipefail
python3 "$repo_root/scripts/make_val_subset.py" --dest "$BOOKS3_LOCAL" reshape --tokens "\${1:-$VAL_TOKENS}"
EOF
chmod +x "$RESHAPE_SMOKE" "$RESHAPE_REAL"
echo "    $RESHAPE_SMOKE"
echo "    $RESHAPE_REAL"

# ------------------------------------------------------------------- done ----
say "Bootstrap complete"
cat <<EOF

  Runbook: docs/protocols/gpu-session-1-runbook.md -- read it, it is one page.

  In order:

  1. Shrink to the smoke size, then run the smoke pass. It has never executed;
     expect it to fail. The four known ways are in COST_MODEL.md s4 (W&B, eval
     batch size, exp_dir, the zarr /train layout).
       bash $RESHAPE_SMOKE
       bash $SMOKE_CMD

  2. Note tokens/sec from the smoke pass. Size the real run so TWO passes fit
     the time you have left, then reshape:
       bash $RESHAPE_REAL <tokens>

  3. Run the baseline TWICE, unchanged (bar S2 wants 4-decimal agreement):
       bash $REAL_CMD && bash $REAL_CMD

  4. Negative control (bar S3):
       bash $CTRL_CMD

  5. Copy results out. results/, *.npy and wandb/ are ALL git-ignored, so a run
     that is not deliberately transcribed leaves no record:
       cp $EXP_DIR/demo/eval-${CKPT}/train_holdout_token_nll_loss.npy \
          $repo_root/experiments/000-repro-baseline/results/
       cp $BOOKS3_LOCAL/val-subset-manifest.json \
          $repo_root/experiments/000-repro-baseline/results/
       cp $ENVREC $repo_root/experiments/000-repro-baseline/results/

     The manifest is not optional: it is what lets the number name the tokens it
     was computed over.

  6. Compare against experiments/000-repro-baseline/TOLERANCE.md. Read the bar
     BEFORE reading the number -- it is pre-registered for a reason (Rule 5).

  7. Fill in the Outcome column in docs/protocols/gpu-bookings.md. The box is not
     free until that is done.

  Log: $LOG
EOF
