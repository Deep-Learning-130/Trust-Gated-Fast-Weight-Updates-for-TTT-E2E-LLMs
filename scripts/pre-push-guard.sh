#!/usr/bin/env bash
# Pre-push guard: make pushing to a NEW remote a deliberate act.
#
# Install:  cp scripts/pre-push-guard.sh .git/hooks/pre-push && chmod +x .git/hooks/pre-push
#
# WHAT CHANGED AND WHY (2026-08-08)
# --------------------------------
# This guard used to refuse any push to a public-looking host and demand you type
# "private". That made sense while the repo was confidential. It does not now:
# `origin` has been PUBLIC since 2026-08-01 and, as of 2026-08-02, deliberately so
# (DISCLOSURE.md). The old guard fired on every single push to a remote we had
# already decided about, which trains people to type the magic word without
# reading -- and a guard everyone reflexively dismisses protects nothing.
#
# Its job is now narrower and still real: `origin` is a settled decision, but a
# DIFFERENT remote is a DIFFERENT decision. The Deep-Learning-130 org repo is a
# separate surface and has not been decided. So this guard is silent for remotes
# with a recorded decision, and blocks on anything else.
#
# Adding a remote below IS the decision. It is a reviewed change to a tracked
# file -- which is the point.
set -euo pipefail

remote_name="${1:-}"
remote_url="${2:-}"

# Remotes with a recorded, deliberate decision. ERE patterns, matched against the URL.
DECIDED=(
  # Manas-Maahir/Trust-Gated-...  -- PUBLIC since 2026-08-01; decision 2026-08-02.
  'github\.com[:/]+Manas-Maahir/Trust-Gated-Fast-Weight-Updates-for-TTT-E2E-LLMs(\.git)?$'

  # Deep-Learning-130/Trust-Gated-...  -- PRIVATE mirror; recorded 2026-08-08.
  # Verified that day: 404 unauthenticated (not publicly readable), but `git ls-remote`
  # resolves and its main was at the same commit as origin -- it already holds the
  # identical history, so pushing discloses nothing that is not already public.
  'github\.com[:/]+Deep-Learning-130/Trust-Gated-Fast-Weight-Updates-for-TTT-E2E-LLMs(\.git)?$'
)

for pat in "${DECIDED[@]}"; do
  if [[ "$remote_url" =~ $pat ]]; then
    echo "pre-push: '$remote_name' has a recorded disclosure decision (DISCLOSURE.md). Proceeding."
    exit 0
  fi
done

cat <<EOF
==============================================================
 PRE-PUSH GUARD -- DISCLOSURE.md
 Remote : $remote_name
 URL    : $remote_url

 This remote has NO recorded disclosure decision.

 Pushing here publishes this work to a surface nobody has
 decided about. \`origin\` being public does not decide it --
 that was one decision about one remote.

 Before proceeding:
   1. Is publishing to THIS remote a decision someone made?
   2. Does this push carry anything covering a FUTURE invention
      not already disclosed? The US grace period (to ~2027-08-01)
      runs from the 2026-08-01 disclosure, not from this push.
   3. Any credentials in the diff -- W&B keys, GCP project ids, .env?

 If it is intended, record it: add the URL to DECIDED[] in
 scripts/pre-push-guard.sh and note it in DISCLOSURE.md. Then
 this prompt stops firing, for everyone, for a stated reason.
==============================================================
EOF

if [[ -t 1 ]] && [[ -e /dev/tty ]]; then
  read -r -p " Type 'new remote' to proceed anyway: " ans < /dev/tty
  [[ "$ans" == "new remote" ]] || { echo "Aborted."; exit 1; }
  echo " Proceeding. Record this remote in DISCLOSURE.md."
else
  echo " Non-interactive shell; refusing. Push from a terminal if this is intended."
  exit 1
fi
