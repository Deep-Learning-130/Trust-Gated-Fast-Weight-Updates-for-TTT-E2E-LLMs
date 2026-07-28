"""trustgate -- trust-gated fast-weight updates for TTT-E2E LLMs.

An overlay package. It never modifies the vendored TTT-E2E tree
(`vendor/ttt-e2e/`, no licence -- see ADR-002); it imports and wraps it.

Confidential: the provisional patent application is not yet filed. Read
`DISCLOSURE.md` before pushing, publishing, or presenting any of this.

Entry point:

    from trustgate.vendor_patch import install_gate
    install_gate()                      # pass-through, provable no-op
    install_gate(my_gate)               # with a policy
"""

__all__ = ["__version__"]

__version__ = "0.0.1"
