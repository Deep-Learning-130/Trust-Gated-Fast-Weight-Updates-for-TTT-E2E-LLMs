---
description: Constraints for testing git hooks, specifically pre-push
---
# Git Hook Testing Rules

When interacting with custom git hooks (like the `pre-push` guard in this repository):

1. **Do not invoke hooks manually:** Do not manually execute scripts via `bash .git/hooks/<hook>` or `sh`. On Windows, bare `bash` invocations can mistakenly route to WSL instead of Git for Windows bash.
2. **Use Git operations for testing:** Always verify hooks by triggering the actual git operation (e.g., using `git push --dry-run origin <branch>`).
3. **Do not modify environment:** Do not install WSL distributions, modify the hook script itself to fix Windows execution issues, or change the Git configuration to resolve execution errors from manual invocations.
