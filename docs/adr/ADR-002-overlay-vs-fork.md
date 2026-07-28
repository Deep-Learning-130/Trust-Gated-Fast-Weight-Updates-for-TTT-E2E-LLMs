# ADR-002 — Overlay package, not a fork; vendor as a pinned submodule

**Status:** Accepted **Date:** 2026-07-28 **Deciders:** project owner (+ patent counsel on the licence question)

## Context

F1 builds a defense *on top of* TTT-E2E ([test-time-training/e2e](https://github.com/test-time-training/e2e)).
We need the vendor's model, sliding-window attention, inner-loop fast-weight
updater, Hydra config, and Submitit launcher. Two facts constrain how we take it:

1. **The vendor repo ships no LICENSE file.** Verified at commit
   `a4fc4788ace38e29b5067916d4f4be33da894085` (2026-07-28): root contains
   `configs/`, `hydra_plugins/`, `ttt/`, `.gitignore`, `README.md`,
   `pyproject.toml`, `uv.lock` — no licence, no patent grant, no CONTRIBUTING.
   Under default copyright this is **all rights reserved**: we may use it
   privately for research, but we may **not distribute a derivative work**.
2. We intend to file a patent on our gate. The claim must read on **our**
   interceptor, cleanly separated from their updater, or the boundary between
   prior art and invention is muddy.

## Decision

- Vendor the upstream repo as a **git submodule** pinned to an explicit SHA
  (`vendor/ttt-e2e`), never a copied-in snapshot.
- All our code lives in a **separate overlay package** (`src/trustgate/`) that
  imports and wraps the vendor. The vendor tree is **read-only**; we never edit
  a file inside it.
- Integration with the vendor's sole commit point (`MetaModel.inner_loop_step`)
  is done by **runtime monkeypatch** from `trustgate.vendor_patch`, confined to
  one module, explicit and reversible. See ADR-003 for why not subclassing.

## Consequences

- **Legal boundary is clean.** We distribute only `src/trustgate/`. Anyone
  reproducing our work fetches the vendor themselves via its own terms. We never
  redistribute unlicensed code.
- **Claim boundary is clean.** The patent figures can show the interceptor as a
  distinct block between the (prior-art) updater and the committed weights.
- **Fragility:** the monkeypatch is pinned to a vendor SHA. A submodule bump can
  move the hook. Mitigated by `tests/test_interceptor.py`, which asserts the
  call-site shape; if it fails after a bump, re-read the vendor source first.
- **We should still ask upstream for a licence.** If they add a permissive one,
  some of this caution relaxes — but the overlay structure is good engineering
  regardless, so we keep it either way.
- Not legal advice; confirm the licence posture with counsel before any
  disclosure or distribution.
