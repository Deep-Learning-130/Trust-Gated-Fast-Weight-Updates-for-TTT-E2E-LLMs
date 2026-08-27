"""trustgate -- trust-gated fast-weight updates for TTT-E2E LLMs.

An overlay package. It never modifies the vendored TTT-E2E tree
(`vendor/ttt-e2e/`, no licence -- see ADR-002); it imports and wraps it.

Public since 2026-08-01 by decision; no patent has been filed. Read
`DISCLOSURE.md` before pushing to a remote you have not pushed to before.

Entry point:

    from trustgate.vendor_patch import install_gate
    install_gate()                      # pass-through, provable no-op
    install_gate(my_gate)               # with a policy
"""

__all__ = ["__version__"]

__version__ = "0.0.1"
