#!/usr/bin/env bash
# A live demo of experiment 002, for a room. About 40 seconds, CPU only.
#
#   bash scripts/demo.sh
#
# WHY THIS IS RUNNABLE IN FRONT OF PEOPLE. No GPU, no download, no GCS, no
# credentials, no network. The victim is a ~0.1M-parameter byte-level stand-in
# and the corpus is a committed file, so the only thing that has to work is
# numpy and jax on CPU.
#
# WHAT IT DEMONSTRATES, in order:
#   1. the corpus is content-addressed -- the digest is checked, live
#   2. the attack runs end to end and produces real numbers
#   3. the control-vs-control noise floor is printed next to the effect
#
# Point 3 is the one worth pausing on. It is printed by the experiment itself,
# not added here for the demo, because a result that does not carry its own
# noise floor cannot be read.
#
# This renders NO verdict. See experiments/002-pilot-tiny-ttt/README.md.
set -uo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."

# The venv lives in a different place on Windows and on Unix; find it rather
# than making the presenter remember which machine they are on.
if   [ -x .venv/Scripts/python.exe ]; then PY=.venv/Scripts/python.exe
elif [ -x .venv/bin/python ];         then PY=.venv/bin/python
elif command -v python3 >/dev/null;   then PY=python3
else PY=python
fi

rule() { printf '%s\n' "--------------------------------------------------------------------"; }

rule
printf '  TrustGate -- experiment 002, scaled-down pilot\n'
printf '  Does a benign-looking stream corrupt a model that learns at inference?\n'
rule
printf '  victim   ~0.1M-parameter byte-level TTT stand-in (NOT TTT-E2E)\n'
printf '  corpus   committed snapshot, digest checked below\n'
printf '  cost     $0, CPU only, no network\n'
printf '  verdict  none -- the realism bar is unscored by construction\n'
rule
echo

# --- 1. the corpus is content-addressed ------------------------------------
# Until 2026-09-15 this was a live glob of the repository's own markdown, so
# every commit silently changed the experiment. The digest is the fix, and
# showing it is the point: a number that cannot name its inputs is not a result.
printf '1. Verifying the corpus digest\n'
"$PY" - <<'PY'
import hashlib, pathlib, re, sys

src = pathlib.Path("experiments/002-pilot-tiny-ttt/run_pilot.py").read_text(encoding="utf-8")
pinned = re.search(r'CORPUS_SHA256 = "([0-9a-f]{64})"', src).group(1)
raw = pathlib.Path("experiments/002-pilot-tiny-ttt/corpus.txt").read_bytes()
got = hashlib.sha256(raw).hexdigest()

print(f"   pinned in source : {pinned[:32]}...")
print(f"   corpus.txt       : {got[:32]}...")
print(f"   {len(raw):,} bytes")
print("   MATCH -- the run below is reproducible" if got == pinned
      else "   MISMATCH -- refusing", flush=True)
sys.exit(0 if got == pinned else 1)
PY
[ $? -eq 0 ] || { echo; echo "Corpus digest failed. Nothing below would mean anything."; exit 1; }
echo

# --- 2. run it -------------------------------------------------------------
printf '2. Running the pilot (about 35 seconds)\n\n'
PYTHONPATH=src "$PY" experiments/002-pilot-tiny-ttt/run_pilot.py
status=$?
echo

rule
if [ "$status" -ne 0 ]; then
  printf '  The run failed (exit %s). That output is the useful part.\n' "$status"
  rule
  exit "$status"
fi
printf '  Read the two d values together, not separately.\n'
printf '  The first is poison vs control. The second is control vs control,\n'
printf '  where the true effect is zero by construction.\n'
printf '\n'
printf '  Next: scripts/demo.sh is the shallow variant. run_deep.py meta-trains\n'
printf '  the base weights first and takes about 6 minutes; run_null.py draws the\n'
printf '  null 20 times and takes about 35 seconds.\n'
rule
