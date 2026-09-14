# Pre-push guard (PowerShell): make pushing to a NEW remote a deliberate act.
#
# Install: Copy-Item scripts/pre-push-guard.ps1 .git/hooks/pre-push.ps1
# Then add to .git/hooks/pre-push (bash wrapper):
#   #!/bin/sh
#   powershell.exe -ExecutionPolicy Bypass -File .git/hooks/pre-push.ps1 "$@"
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

param(
    [string]$remote_name = "",
    [string]$remote_url = ""
)

# Remotes with a recorded, deliberate decision. Regex patterns, matched against the URL.
$DECIDED = @(
    # Manas-Maahir/Trust-Gated-...  -- PUBLIC since 2026-08-01; decision 2026-08-02.
    'github\.com[:/]+Manas-Maahir/Trust-Gated-Fast-Weight-Updates-for-TTT-E2E-LLMs(\.git)?$',

    # Deep-Learning-130/Trust-Gated-...  -- PRIVATE mirror; recorded 2026-08-08.
    # Verified that day: 404 unauthenticated (not publicly readable), but `git ls-remote`
    # resolves and its main was at the same commit as origin -- it already holds the
    # identical history, so pushing discloses nothing that is not already public.
    'github\.com[:/]+Deep-Learning-130/Trust-Gated-Fast-Weight-Updates-for-TTT-E2E-LLMs(\.git)?$'
)

foreach ($pat in $DECIDED) {
    if ($remote_url -match $pat) {
        Write-Host "pre-push: '$remote_name' has a recorded disclosure decision (DISCLOSURE.md). Proceeding."
        exit 0
    }
}

Write-Host @"
==============================================================
 PRE-PUSH GUARD -- DISCLOSURE.md
 Remote : $remote_name
 URL    : $remote_url

 This remote has NO recorded disclosure decision.

 Pushing here publishes this work to a surface nobody has
 decided about. `origin` being public does not decide it --
 that was one decision about one remote.

 Before proceeding:
   1. Is publishing to THIS remote a decision someone made?
   2. Does this push carry anything covering a FUTURE invention
      not already disclosed? The US grace period (to ~2027-08-01)
      runs from the 2026-08-01 disclosure, not from this push.
   3. Any credentials in the diff -- W&B keys, GCP project ids, .env?

 If it is intended, record it: add the URL to `$DECIDED` in
 scripts/pre-push-guard.sh (and .ps1) and note it in DISCLOSURE.md.
 Then this prompt stops firing, for everyone, for a stated reason.
==============================================================
"@

$ans = Read-Host " Type 'new remote' to proceed anyway"
if ($ans -eq "new remote") {
    Write-Host " Proceeding. Record this remote in DISCLOSURE.md."
    exit 0
} else {
    Write-Host "Aborted."
    exit 1
}
