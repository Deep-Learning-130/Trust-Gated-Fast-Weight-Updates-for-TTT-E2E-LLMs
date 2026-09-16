#!/usr/bin/env bash
# Cold-start a fresh GPU box: nothing -> ready to run the baseline eval.
#
#   uv installed if the image lacks it
#   the NVIDIA driver checked, and JAX proven to see and run on the GPU
#   vendor submodule at the pinned SHA, vendor environment installed
#   1B checkpoint on local disk and sha256-fingerprinted (or verified)
#   a TRUNCATED books3 /val on local disk, with a manifest recording what was cut
#   training.exp_dir created OUTSIDE this repo
#   three eval commands written out -- smoke, baseline, negative control --
#   each logging to a file and failing loudly if no loss line is produced
#   everything above logged to a file
#
# DATA SOURCE. GCS is used ONLY if the checkpoint or the /val subset is missing
# from local disk. If both were side-loaded (scp, rsync, a disk -- any route),
# no Google Cloud account, gsutil or billing project is needed or touched.
#
# ---------------------------------------------------------------------------
# THIS SCRIPT HAS NOT RUN ON A GPU. It has been shell-parsed and rehearsed end to
# end against stub `nvidia-smi`/`uv` binaries with fake data (no GPU, no CUDA).
# Its real test is the first booking. Every step is idempotent, so re-running
# after a failure is the intended repair.
# ---------------------------------------------------------------------------
#
# Usage:
#   export WANDB_ENTITY=... WANDB_PROJECT=... WANDB_KEY=...
#   export DATA_ROOT=/mnt/data EXP_DIR=/mnt/runs
#   bash scripts/bootstrap_gpu_box.sh
#
# Useful overrides:
#   VAL_TOKENS=50000000                         # eval length; THE wall-clock knob (TOLERANCE s8)
#   SMOKE_TOKENS=131073                         # 16 sequences = 2 eval batches
#   CKPT=125m_ttt_e2e_finetune_books_8k_1x_cc   # +experiment= follows this automatically
#   CKPT_DIR=/mnt/data/<ckpt>                   # where the checkpoint lives (default: repo checkpoints/)
#   CKPT_SHA_MANIFEST=path                      # verify a side-loaded checkpoint against this manifest
#   GCP_BILLING_PROJECT=...                     # only needed if data must be fetched from GCS
#   SKIP_DATA=1                                 # checkpoint only
#   AUTO_INSTALL=0                              # refuse rather than install uv/gcloud
#   ALLOW_SHA_DRIFT=1                           # accept a vendor SHA != the pin
#   ALLOW_CKPT_LAYOUT=1                         # accept a checkpoint without an orbax step dir
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
CKPT_DEST="${CKPT_DIR:-$repo_root/checkpoints/$CKPT}"     # checkpoints/ is git-ignored
RESULTS="$repo_root/experiments/000-repro-baseline/results"
VAL_TOKENS="${VAL_TOKENS:-50000000}"      # 6,103 sequences; one 100M-token chunk to fetch
SMOKE_TOKENS="${SMOKE_TOKENS:-131073}"   # 16 sequences = exactly 2 eval batches

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

# Generated scripts and the eval command line are space-separated; a path with a
# space would split into two Hydra overrides and fail after billing starts.
for v in "$EXP_DIR" "$DATA_ROOT" "$CKPT_DEST" "$repo_root"; do
  [[ "$v" != *" "* ]] || die "path '$v' contains a space. Use a path without spaces."
done

# ------------------------------------------------------------------ logging --
mkdir -p "$EXP_DIR/bootstrap" "$DATA_ROOT"
LOG="$EXP_DIR/bootstrap/bootstrap-$(date -u +%Y%m%dT%H%M%SZ).log"
exec > >(tee -a "$LOG") 2>&1

say "Bootstrap starting"
echo "    repo      : $repo_root"
echo "    checkpoint: $CKPT"
echo "    ckpt dir  : $CKPT_DEST"
echo "    data root : $DATA_ROOT"
echo "    exp_dir   : $EXP_DIR"
echo "    log       : $LOG"

# ----------------------------------------------------------------- preflight --
say "Step 0 — preflight"

: "${WANDB_ENTITY:?Set WANDB_ENTITY -- W&B is mandatory, log_wandb=false does not avoid it (ADR-004 s3)}"
: "${WANDB_PROJECT:?Set WANDB_PROJECT}"
: "${WANDB_KEY:?Set WANDB_KEY -- a real key, proven off-box first with scripts/preflight_wandb.py}"

if [[ -z "${TMUX:-}" && -z "${STY:-}" ]]; then
  echo "    WARNING   : not inside tmux/screen. A dropped SSH connection kills a"
  echo "                multi-hour eval and loses its number. Start one: tmux new -s ttt"
fi

command -v git >/dev/null 2>&1 || die "git not found -- this is a git checkout, so something is very wrong"
command -v python3 >/dev/null 2>&1 || die "python3 not found -- scripts/make_val_subset.py needs it (stdlib only)"
echo "    tools     : git, python3 present"

# --- Does this session need GCS at all? Decided by what is on disk, not asked.
ckpt_present=0
if [[ -d "$CKPT_DEST" ]] && [[ -n "$(ls -A "$CKPT_DEST" 2>/dev/null)" ]]; then
  ckpt_present=1
fi
data_present=0
if [[ "${SKIP_DATA:-0}" == "1" ]] || python3 "$repo_root/scripts/make_val_subset.py" --dest "$BOOKS3_LOCAL" covers --tokens "$VAL_TOKENS"; then
  data_present=1
fi
need_gcs=0
[[ "$ckpt_present" == "1" && "$data_present" == "1" ]] || need_gcs=1
echo "    checkpoint: $([[ $ckpt_present == 1 ]] && echo "present at $CKPT_DEST" || echo MISSING)"
echo "    val subset: $([[ $data_present == 1 ]] && echo "present, covers $VAL_TOKENS tokens" || echo "MISSING or too small for VAL_TOKENS=$VAL_TOKENS")"

# A non-GCP image ships neither `uv` nor the gcloud SDK, and both are one command
# away. AUTO_INSTALL=0 restores refuse-and-exit.
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

ensure_tool uv "astral.sh/uv"   'curl -LsSf https://astral.sh/uv/install.sh | sh && export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"'

if [[ "$need_gcs" == "1" ]]; then
  echo "    gcs       : NEEDED -- something above is missing from local disk"
  : "${GCP_BILLING_PROJECT:?Checkpoint or val subset is missing locally, and fetching it needs GCP_BILLING_PROJECT (requester-pays). Side-load both instead to avoid GCS entirely}"
  ensure_tool gsutil "the Google Cloud SDK"   'curl -sSL https://sdk.cloud.google.com | bash -s -- --disable-prompts && export PATH="$HOME/google-cloud-sdk/bin:$PATH"'
  if ! gcloud auth list --filter=status:ACTIVE --format="value(account)" 2>/dev/null | grep -q .; then
    die "gcloud has no active account. Run:  gcloud auth login --no-launch-browser
       Both buckets are requester-pays; the fetch cannot start without it."
  fi
  echo "    gcloud    : $(gcloud auth list --filter=status:ACTIVE --format='value(account)' 2>/dev/null | head -1)"
  # Seconds, metadata only. The likeliest failure -- the account cannot bill to
  # the project, or a path is wrong -- must surface HERE, not after `uv sync` has
  # spent half an hour of GPU billing and the background download reports it.
  if [[ "$ckpt_present" != "1" ]]; then
    gsutil -u "$GCP_BILLING_PROJECT" du -s "$BUCKET/$CKPT" >/dev/null 2>"$EXP_DIR/bootstrap/gcs-access-check.err" \
      || die "cannot read $BUCKET/$CKPT billed to $GCP_BILLING_PROJECT:
$(sed 's/^/       /' "$EXP_DIR/bootstrap/gcs-access-check.err")
       Check the project id, that billing is enabled on it, and that this account may use it."
  fi
  if [[ "$data_present" != "1" ]]; then
    gsutil -u "$GCP_BILLING_PROJECT" ls "$DATA_BUCKET/val/zarr.json" >/dev/null 2>&1 \
      || gsutil -u "$GCP_BILLING_PROJECT" ls "$DATA_BUCKET/data.zarr/val/zarr.json" >/dev/null 2>"$EXP_DIR/bootstrap/gcs-access-check.err" \
      || die "cannot read the books3 /val metadata under $DATA_BUCKET billed to $GCP_BILLING_PROJECT:
$(sed 's/^/       /' "$EXP_DIR/bootstrap/gcs-access-check.err")"
  fi
  echo "    gcs access: checkpoint and /val readable, billed to $GCP_BILLING_PROJECT"
else
  echo "    gcs       : not needed -- checkpoint and val subset both on local disk"
fi

# --- GPU and driver. The vendor lock resolves CUDA 12.8 / cuDNN 9.8 pip wheels,
# which carry the toolkit themselves; the host supplies only the driver.
command -v nvidia-smi >/dev/null 2>&1 || die "nvidia-smi not found. This is not a working GPU image.
       Stop rule: do not debug a broken image on a billing clock -- release it and pick another."
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader \
  | sed 's/^/    gpu       : /'
driver="$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1 | tr -d ' ')"
driver_major="${driver%%.*}"
if ! [[ "$driver_major" =~ ^[0-9]+$ ]]; then
  die "could not parse the NVIDIA driver version from '$driver'"
elif (( driver_major < 525 )); then
  die "driver $driver is too old for any CUDA 12 JAX wheel (needs >= 525; 570+ for CUDA 12.8).
       Release this box and pick an image whose nvidia-smi shows CUDA Version >= 12.8."
elif (( driver_major < 570 )); then
  echo "    WARNING   : driver $driver predates CUDA 12.8 (570+). JAX may run under minor-version"
  echo "                compatibility, or may fail at the GPU check in Step 3. If Step 3 fails, release the box."
fi
mem_total_mib="$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1 | tr -d ' ')"
if [[ "$mem_total_mib" =~ ^[0-9]+$ ]] && (( mem_total_mib < 70000 )); then
  echo "    WARNING   : ${mem_total_mib} MiB GPU memory. The 1B eval is estimated at 36-49 GB"
  echo "                (COST_MODEL s9.4.1); an 80 GB card was the recommendation."
fi

# Free space where the bytes actually go.
avail_kb="$(df -Pk "$DATA_ROOT" | awk 'NR==2{print $4}')"
echo "    disk      : $((avail_kb / 1024 / 1024)) GB available under $DATA_ROOT"
[[ "$avail_kb" -gt 31457280 ]] || echo "    disk      : WARNING - under 30 GB free (vendor env ~10 GB, ckpt 5.4 GB, val 0.8 GB)."

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

# ADR-002: the vendor tree has no licence and must never be edited. Untracked
# files are listed but tolerated only if the vendor's own .gitignore covers them,
# which `--porcelain` already respects.
if [[ -n "$(git -C vendor/ttt-e2e status --porcelain)" ]]; then
  git -C vendor/ttt-e2e status --short
  die "vendor/ttt-e2e is dirty. ADR-002: it has no licence and must stay pristine.
       If the only entry is an untracked checkpoints/ directory, an earlier eval ran
       without training.checkpoint_path pointed outside the tree; delete it and re-run."
fi
echo "    vendor tree clean (ADR-002)"

# ------------------------------------------------ 1b. download, in background --
# Billing is per minute, so nothing waits on anything it does not depend on. The
# GCS download needs neither the vendor env nor the GPU, and `uv sync` needs
# neither the checkpoint nor the data -- they run side by side, and Step 4 waits.
FETCH_PID=""
if [[ "$need_gcs" == "1" ]]; then
  FETCH_LOG="$EXP_DIR/bootstrap/fetch-$(date -u +%Y%m%dT%H%M%SZ).log"
  say "Step 1b — GCS download started in the background (log: $FETCH_LOG)"
  export GCP_BILLING_PROJECT
  (
    set -euo pipefail
    if [[ "$ckpt_present" != "1" ]]; then
      # fetch_checkpoints.sh probes before transferring and refuses to exceed
      # MAX_BYTES, so a wrong CKPT name costs a metadata call, not an egress bill.
      mkdir -p "$(dirname "$CKPT_DEST")"
      CKPT="$CKPT" BUCKET="$BUCKET" DEST="$CKPT_DEST" RESULTS="$RESULTS" bash scripts/fetch_checkpoints.sh
    fi
    if [[ "$data_present" != "1" ]]; then
      python3 "$repo_root/scripts/make_val_subset.py" --dest "$BOOKS3_LOCAL" fetch --bucket "$DATA_BUCKET" --tokens "$VAL_TOKENS"
    fi
  ) > "$FETCH_LOG" 2>&1 &
  FETCH_PID=$!
  # If a later step dies (e.g. the GPU check), do not leave a download running.
  trap '[[ -n "$FETCH_PID" ]] && kill "$FETCH_PID" 2>/dev/null || true' EXIT
  echo "    watch it from another tmux window:  tail -f $FETCH_LOG"
fi

# ----------------------------------------------------------- 2. environment --
# `uv sync --frozen` is itself idempotent; re-running it after a partial install
# resumes rather than restarting.
say "Step 2 — vendor environment (uv sync --frozen)"
( cd vendor/ttt-e2e && uv sync --frozen )

# Not just `import ttt`: prove JAX sees a CUDA device AND can compile and run a
# kernel on it. This is where a driver/CUDA mismatch surfaces, in minutes, rather
# than an hour in at the first eval batch.
say "Step 3 — import check and GPU compute check"
( cd vendor/ttt-e2e && uv run --exact python - <<'PY'
import ttt  # noqa: F401
import jax
import jax.numpy as jnp

devices = jax.devices()
print(f"    import ttt OK; jax {jax.__version__}; devices: {devices}")
if not any(d.platform in ("gpu", "cuda") for d in devices):
    raise SystemExit("FATAL: JAX sees no GPU. The CUDA wheels did not load against this driver. Release the box.")
x = jnp.ones((1024, 1024), dtype=jnp.float32)
value = float(jax.jit(lambda a: (a @ a).sum())(x))
assert value == 1024.0 ** 3, value
print("    GPU compute check OK (jit matmul on device)")
PY
) || die "Step 3 failed: JAX cannot run on this GPU. Stop rule: release the box inside 30 minutes."

# ------------------------------------------------------------ 4. checkpoint --
say "Step 4 — checkpoint"
if [[ -n "$FETCH_PID" ]]; then
  echo "    waiting for the background download (started in Step 1b)..."
  if ! wait "$FETCH_PID"; then
    FETCH_PID=""
    tail -30 "$FETCH_LOG" | sed 's/^/      | /'
    die "GCS download failed; full log: $FETCH_LOG
       Re-running this script resumes it (gsutil cp -n, and fetched chunks are skipped)."
  fi
  FETCH_PID=""
  echo "    download finished"
elif [[ "$ckpt_present" == "1" ]]; then
  skip "checkpoint present at $CKPT_DEST"
fi
[[ -d "$CKPT_DEST" ]] || die "checkpoint directory $CKPT_DEST does not exist"

# orbax's CheckpointManager (ttt/infra/checkpoint.py:97) looks for integer step
# directories; without one the eval dies at load with "No checkpoints found".
# Catch a wrongly nested side-load here, before the vendor env spends a launch on it.
step_dirs="$(find "$CKPT_DEST" -mindepth 1 -maxdepth 1 -type d -regex '.*/[0-9]+' | wc -l | tr -d ' ')"
if [[ "$step_dirs" == "0" ]]; then
  echo "    contents of $CKPT_DEST:"; ls -la "$CKPT_DEST" | sed 's/^/      /'
  if [[ "${ALLOW_CKPT_LAYOUT:-0}" == "1" ]]; then
    echo "    WARNING: no integer step directory (ALLOW_CKPT_LAYOUT=1)"
  else
    die "no integer step directory directly under $CKPT_DEST.
       The vendor's orbax loader needs <ckpt>/<step>/model_weights/. If the side-load
       nested it one level deeper (e.g. $CKPT_DEST/$CKPT/<step>), point CKPT_DIR at the
       level that holds the step directory. ALLOW_CKPT_LAYOUT=1 overrides."
  fi
else
  echo "    orbax step dir(s): $(find "$CKPT_DEST" -mindepth 1 -maxdepth 1 -type d -regex '.*/[0-9]+' -exec basename {} \; | tr '\n' ' ')"
fi

CKPT_MANIFEST="$RESULTS/checkpoint-sha256-${CKPT}.txt"
if [[ -n "${CKPT_SHA_MANIFEST:-}" ]]; then
  DEST="$CKPT_DEST" CKPT="$CKPT" VERIFY="$CKPT_SHA_MANIFEST" bash scripts/fingerprint_checkpoint.sh \
    || die "checkpoint does not match $CKPT_SHA_MANIFEST. Do not evaluate a checkpoint you cannot name."
  mkdir -p "$RESULTS"
  [[ -f "$CKPT_MANIFEST" ]] || cp "$CKPT_SHA_MANIFEST" "$CKPT_MANIFEST"
elif [[ -f "$CKPT_MANIFEST" ]]; then
  skip "fingerprint present at $CKPT_MANIFEST"
else
  DEST="$CKPT_DEST" CKPT="$CKPT" RESULTS="$RESULTS" SOURCE="side-loaded to $CKPT_DEST" \
    bash scripts/fingerprint_checkpoint.sh
fi

# --------------------------------------------------------------- 5. dataset --
# The loader is LocalStore-only (lm_dataset.py:14) -- there is no gs:// path, so
# a local store is mandatory. /train's metadata must be in it too: _make_train_iterator
# (train.py:125) opens the train array before the eval branch returns (COST_MODEL s1.1).
say "Step 5 — dataset (books3 /val subset on local disk)"
if [[ "${SKIP_DATA:-0}" == "1" ]]; then
  echo "    SKIP_DATA=1 -- skipping. The eval will fail without it."
elif python3 "$repo_root/scripts/make_val_subset.py" --dest "$BOOKS3_LOCAL" covers --tokens "$VAL_TOKENS"; then
  echo "    val subset at $BOOKS3_LOCAL covers $VAL_TOKENS tokens"
else
  die "val subset at $BOOKS3_LOCAL does not cover VAL_TOKENS=$VAL_TOKENS after the download step.
       Re-run this script to resume the fetch."
fi

if [[ "${SKIP_DATA:-0}" != "1" ]]; then
  [[ -f "$BOOKS3_LOCAL/zarr.json" ]] || die "$BOOKS3_LOCAL/zarr.json (group metadata) is missing.
       The side-loaded store must be the whole store root: zarr.json, val/, train/zarr.json."
  [[ -f "$BOOKS3_LOCAL/train/zarr.json" ]] || die "$BOOKS3_LOCAL/train/zarr.json is missing.
       train.py:125 opens /train before the eval branch returns; without its metadata the run
       dies during setup (COST_MODEL s1.1). Copy it alongside val/."
  python3 "$repo_root/scripts/make_val_subset.py" --dest "$BOOKS3_LOCAL" status | sed 's/^/    /'
fi

# ---------------------------------------------------------- 6. env for P1-3 --
say "Step 6 — environment record (for P1-3)"
ENVREC="$EXP_DIR/bootstrap/env-record.txt"
{
  echo "recorded_utc:   $(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "host:           $(hostname)"
  echo "overlay_sha:    $(git rev-parse HEAD)"
  echo "vendor_sha:     $actual"
  echo "checkpoint:     $CKPT"
  echo "ckpt_manifest:  $(grep -m1 'manifest_sha256' "$CKPT_MANIFEST" 2>/dev/null | awk '{print $3}' || echo n/a)"
  echo "uv:             $(uv --version 2>/dev/null || echo n/a)"
  echo "nvidia-smi:     $(nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader 2>/dev/null | head -1 || echo n/a)"
  ( cd vendor/ttt-e2e && uv run --exact python -c \
      "import jax; print('jax:            '+jax.__version__); print('jax_devices:    '+str(jax.devices()))" \
    ) 2>/dev/null || echo "jax:            n/a"
} > "$ENVREC"
sed 's/^/    /' "$ENVREC"

# --------------------------------------------------- 7. the eval commands ----
# Written to files rather than only printed, so the session runs reviewed commands
# instead of ones retyped at 2am.
say "Step 7 — writing the eval commands"

# +experiment follows the checkpoint.
SIZE="${CKPT%%_*}"
case "$SIZE" in
  125m|350m|760m|1b|3b) EXPERIMENT="${SIZE}/extension/ext-${SIZE}-e2e-32K" ;;
  *) die "cannot derive +experiment from CKPT='$CKPT' (size prefix '$SIZE').
       Known sizes: 125m 350m 760m 1b 3b. Set EXPERIMENT_OVERRIDE= by hand if this is deliberate." ;;
esac
EXPERIMENT="${EXPERIMENT_OVERRIDE:-$EXPERIMENT}"
echo "    experiment: $EXPERIMENT"

# The vendor's main() (train.py:275-278) sets jax_compilation_cache_dir from
# backend.compilation_cache_dir, overriding JAX_COMPILATION_CACHE_DIR, so the
# override has to be a Hydra one. Note what this cache does NOT do: each eval is a
# fresh process and re-traces and re-compiles the model. Budget the compile per run.
CACHE_DIR="$EXP_DIR/jax-cache"
SCRATCH_CKPT="$EXP_DIR/ckpt-scratch"
mkdir -p "$CACHE_DIR" "$SCRATCH_CKPT" "$EXP_DIR/logs"

write_eval_cmd() {  # path, exp_name, header, extra hydra overrides...
  local path="$1" exp_name="$2" header="$3"; shift 3
  cat > "$path" <<EOF
#!/usr/bin/env bash
# $header
# Generated $(date -u +%Y-%m-%dT%H:%M:%SZ) by bootstrap_gpu_box.sh.
#
# Derivation and citations: experiments/000-repro-baseline/EVAL_ENTRYPOINT.md
# Decision record:          docs/adr/ADR-004-baseline-eval-invocation.md
# Pre-registered bar:       experiments/000-repro-baseline/TOLERANCE.md
#
# The gate must NOT be installed for this run: no trustgate import appears anywhere in it (T1.6).
#
# Overrides beyond ADR-004's command, and why:
#   training.checkpoint_path  -- the vendor builds a SAVING Checkpointer even in eval mode
#                                (train.py:116), default ./checkpoints, which would create an
#                                untracked dir inside vendor/ttt-e2e (ADR-002)
#   backend.compilation_cache_dir -- train.py:278 ignores JAX_COMPILATION_CACHE_DIR
#
# Env:
#   WANDB_KEY        required, read at run time so the key is not written into this file
#   RUN_LOG          log path (default: $EXP_DIR/logs/$exp_name-<utc>.log)
#   EVAL_GBS / EVAL_BATCH_SIZE   default 2 / 8. The OOM fallback is 1 / 4 -- both are
#                    needed because train.py:211 takes max(eval_batch_size, gbs*4).
set -euo pipefail
: "\${WANDB_KEY:?export WANDB_KEY in this shell first}"

RUN_LOG="\${RUN_LOG:-$EXP_DIR/logs/$exp_name-\$(date -u +%Y%m%dT%H%M%SZ).log}"
mkdir -p "\$(dirname "\$RUN_LOG")"
export PYTHONUNBUFFERED=1

cd "$repo_root/vendor/ttt-e2e"

echo "run log: \$RUN_LOG"
uv run --exact train \\
  +deploy=interactive \\
  +experiment=$EXPERIMENT \\
  training.eval_mode=true \\
  training.seq_length=8192 \\
  training.global_batch_size=\${EVAL_GBS:-2} \\
  training.eval_batch_size=\${EVAL_BATCH_SIZE:-8} \\
  training.exp_name=$exp_name \\
  training.load_part=params \\
  checkpoint.resume_checkpoint_dir=$CKPT_DEST \\
  deploy_paths.data.books3=$BOOKS3_LOCAL \\
  training.exp_dir=$EXP_DIR \\
  training.checkpoint_path=$SCRATCH_CKPT \\
  backend.compilation_cache_dir=$CACHE_DIR \\
  training.wandb_entity=$WANDB_ENTITY \\
  training.wandb_project=$WANDB_PROJECT \\
  training.wandb_key="\$WANDB_KEY" $* 2>&1 | tee "\$RUN_LOG"

# A zero exit with no loss line is not a result.
if ! grep -q "Eval -- train_holdout/loss:" "\$RUN_LOG"; then
  echo "FATAL: the run exited without printing 'Eval -- train_holdout/loss:'. See \$RUN_LOG" >&2
  exit 4
fi
grep "Eval -- train_holdout/loss:" "\$RUN_LOG" | tail -1
# Per-token curve: $EXP_DIR/demo/$exp_name/train_holdout_token_nll_loss.npy
EOF
  chmod +x "$path"
  echo "    $path"
}

SMOKE_CMD="$EXP_DIR/bootstrap/1-smoke-${CKPT}.sh"
REAL_CMD="$EXP_DIR/bootstrap/2-eval-${CKPT}.sh"
CTRL_CMD="$EXP_DIR/bootstrap/3-dummy-control-${CKPT}.sh"

write_eval_cmd "$SMOKE_CMD" "smoke-${CKPT}" \
  "SMOKE PASS -- two eval batches. Shakes out the launch, measures memory. NOT a result."
write_eval_cmd "$REAL_CMD" "eval-${CKPT}" \
  "BASELINE EVAL for $CKPT. Run it TWICE unchanged -- TOLERANCE.md s5 bar S2 wants 4-decimal agreement."
write_eval_cmd "$CTRL_CMD" "dummy-${CKPT}" \
  "NEGATIVE CONTROL (TOLERANCE.md s5 bar S3). Random tokens; the loss must land FAR ABOVE the band." \
  "training.dummy_dataset=true"

RESHAPE_SMOKE="$EXP_DIR/bootstrap/reshape-to-smoke.sh"
RESHAPE_REAL="$EXP_DIR/bootstrap/reshape-to-real.sh"
cat > "$RESHAPE_SMOKE" <<EOF
#!/usr/bin/env bash
set -euo pipefail
python3 "$repo_root/scripts/make_val_subset.py" --dest "$BOOKS3_LOCAL" reshape --tokens $SMOKE_TOKENS --no-digest
EOF
cat > "$RESHAPE_REAL" <<EOF
#!/usr/bin/env bash
# Resize the eval without refetching. Refuses a size beyond the chunks on disk.
set -euo pipefail
python3 "$repo_root/scripts/make_val_subset.py" --dest "$BOOKS3_LOCAL" reshape --tokens "\${1:-$VAL_TOKENS}"
EOF
chmod +x "$RESHAPE_SMOKE" "$RESHAPE_REAL"
echo "    $RESHAPE_SMOKE"
echo "    $RESHAPE_REAL"

# Recorded for run_gpu_session.sh, so it runs against exactly what was bootstrapped.
cat > "$EXP_DIR/bootstrap/session.env" <<EOF
CKPT=$CKPT
CKPT_DEST=$CKPT_DEST
BOOKS3_LOCAL=$BOOKS3_LOCAL
VAL_TOKENS=$VAL_TOKENS
SMOKE_TOKENS=$SMOKE_TOKENS
CKPT_MANIFEST=$CKPT_MANIFEST
EOF

# ------------------------------------------------------------------- done ----
say "Bootstrap complete"
cat <<EOF

  Runbook: docs/protocols/gpu-session-1-runbook.md

  Next, inside tmux, one command runs the whole session unattended
  (smoke -> baseline x2 -> negative control -> collect -> acceptance):

    bash scripts/run_gpu_session.sh

  Log: $LOG
EOF
