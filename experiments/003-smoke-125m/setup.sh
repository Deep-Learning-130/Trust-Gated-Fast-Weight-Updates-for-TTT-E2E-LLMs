#!/usr/bin/env bash
# Provision a GPU box (Colab cell or WSL2 shell) for experiments/003-smoke-125m.
#
# Deliberately NOT scripts/bootstrap_gpu_box.sh. That script fetches a
# checkpoint and a dataset over requester-pays GCS and is sized for the paid 1B
# session. This one touches no bucket, needs no gcloud auth and spends nothing:
# the smoke run builds a random-init model (train.py:198) on dummy data
# (training.dummy_dataset).
#
#   bash experiments/003-smoke-125m/setup.sh
#
# Idempotent. Safe to re-run after a Colab runtime reset.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

echo "== repo: $REPO_ROOT"

# --- 1. Vendor submodule -----------------------------------------------------
# ADR-002: the tree is unlicensed. We check it out to run against it; we never
# edit it, and CI never fetches it.
if [ ! -f vendor/ttt-e2e/pyproject.toml ]; then
  echo "== fetching vendor submodule"
  git submodule update --init --recursive vendor/ttt-e2e
else
  echo "== vendor submodule already present"
fi

EXPECTED_SHA="a4fc4788ace38e29b5067916d4f4be33da894085"
ACTUAL_SHA="$(git -C vendor/ttt-e2e rev-parse HEAD)"
if [ "$ACTUAL_SHA" != "$EXPECTED_SHA" ]; then
  echo "!! vendor pin drift: expected $EXPECTED_SHA, got $ACTUAL_SHA"
  echo "!! tests/test_interceptor.py is the tripwire for this. Re-read"
  echo "!! vendor/ttt-e2e/ttt/model/transformer.py around inner_loop_step"
  echo "!! before trusting src/trustgate/eval/vendor_bind.py against it."
fi

# --- 2. Python version -------------------------------------------------------
PYVER="$(python -c 'import sys; print("%d.%d" % sys.version_info[:2])')"
echo "== python $PYVER (vendor requires >= 3.12)"

# --- 3. Vendor deps ----------------------------------------------------------
# jax[cuda12]<0.6 comes from the vendor's own pyproject, so the CUDA build is
# pinned by the submodule rather than by us.
echo "== installing vendor tree (editable)"
pip install -q -e vendor/ttt-e2e

# trustgate itself is NOT installed. `PYTHONPATH=src` is the documented
# invocation; installing a second copy into site-packages would make it
# ambiguous which `trustgate` was imported, and `eval/prereg.py` resolves
# PREREGISTERED.md relative to its own __file__.

# --- 4. Verify the accelerator ----------------------------------------------
echo "== jax devices"
python - <<'PY'
import jax
devs = jax.devices()
print(f"   {len(devs)} device(s): {devs}")
if all(d.platform == "cpu" for d in devs):
    print("   !! CPU ONLY. On Windows there are no jax[cuda12] wheels --")
    print("   !! use WSL2 or Colab. The smoke run will report this and stop.")
else:
    d = devs[0]
    print(f"   device_kind: {getattr(d, 'device_kind', '?')}")
    print("   !! If this is a Tesla T4 (Turing), it has NO native bf16 and the")
    print("   !! vendor default compute_dtype is bf16. Pass --compute-dtype fp32.")
PY

cat <<'EOF'

== ready. Next:

   PYTHONPATH=src python experiments/003-smoke-125m/run_smoke.py --seq-length 8192

   On a T4 add:   --compute-dtype fp32
   If it OOMs:    --seq-length 4096   (still 4 inner steps at mini_batch 1024)

This run spends nothing, downloads no checkpoint and renders no verdict.
EOF
