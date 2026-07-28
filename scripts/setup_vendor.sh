#!/usr/bin/env bash
# Initialise the vendored TTT-E2E submodule at its pinned SHA and install its
# own environment. The vendor tree is READ-ONLY (no licence -- see ADR-002);
# never edit inside vendor/ttt-e2e.
set -euo pipefail

PINNED_SHA="a4fc4788ace38e29b5067916d4f4be33da894085"

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

echo "==> Initialising submodule vendor/ttt-e2e"
git submodule update --init --recursive vendor/ttt-e2e

echo "==> Verifying pinned SHA"
actual="$(git -C vendor/ttt-e2e rev-parse HEAD)"
if [[ "$actual" != "$PINNED_SHA" ]]; then
  echo "WARNING: vendor at $actual, expected $PINNED_SHA"
  echo "         The interceptor monkeypatch is pinned to $PINNED_SHA."
  echo "         Run tests/test_interceptor.py after any intentional bump."
fi

echo "==> Installing vendor environment (its own uv.lock, needs CUDA 12.8 / cuDNN 9.8)"
echo "    (run manually on the GPU box:)"
echo "    cd vendor/ttt-e2e && uv sync --frozen"

echo "==> Done. Vendor ready at $actual"
