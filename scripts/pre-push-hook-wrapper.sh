#!/bin/sh
# Pre-push hook wrapper for Windows (Git Bash -> PowerShell).
#
# Install: Copy this file to .git/hooks/pre-push and make executable.
# It calls the PowerShell guard script with the same arguments.

powershell.exe -ExecutionPolicy Bypass -File .git/hooks/pre-push.ps1 "$@"
