#!/usr/bin/env bash
# Pre-push guard: refuse to push to a non-private-looking remote while the
# provisional is unfiled. An accidental public push is unrecoverable for foreign
# patent rights (DISCLOSURE.md).
#
# Install:  cp scripts/pre-push-guard.sh .git/hooks/pre-push && chmod +x .git/hooks/pre-push
#
# This is a coarse net (it blocks github.com/gitlab.com public hosts and asks for
# confirmation), NOT a substitute for confirming remote visibility in the host UI.
set -euo pipefail

remote_url="${2:-}"

if [[ "$remote_url" =~ github\.com|gitlab\.com|bitbucket\.org ]]; then
  echo "=============================================================="
  echo " PRE-PUSH GUARD — DISCLOSURE.md"
  echo " Pushing to: $remote_url"
  echo " Confirm this remote is PRIVATE. A public push before the"
  echo " provisional is filed forfeits foreign patent rights."
  echo "=============================================================="
  if [[ -t 1 ]]; then
    read -r -p " Type 'private' to proceed: " ans < /dev/tty
    [[ "$ans" == "private" ]] || { echo "Aborted."; exit 1; }
  else
    echo " Non-interactive shell; refusing. Push manually if intended."
    exit 1
  fi
fi
