"""Install the trust gate into the unmodified vendor tree.

Why monkeypatching, and not something cleaner
---------------------------------------------
Subclassing does not work here. `MetaModel.loss_for_sequence` invokes the step
through the *class*, not the instance (vendor `ttt/model/transformer.py:703`):

    new_model, inner_opt_state, state_tuple, metrics = MetaModel.inner_loop_step(
        model_inner, inner_opt_state, state_tuple, suffix_chunk, prefix_chunk
    )

A `GatedMetaModel(MetaModel)` override would never be called. The three options
were: (a) patch the class attribute, (b) subclass -- does not work, (c) copy
`loss_for_sequence` into our tree and call our own step.

(c) means copying ~60 lines of a repository that ships **no licence**, which
ADR-002 rules out. So (a) it is. We keep it honest by confining the patch to
this one module, making it explicit and reversible, and refusing to double-apply.

This is a real fragility: it is pinned to vendor SHA a4fc478, and
`tests/test_interceptor.py` asserts the call site still looks the way we expect.
If that test fails after a submodule bump, the hook moved -- re-read the vendor
source before touching anything else.
"""

from __future__ import annotations

from typing import Callable

from trustgate.interceptor import make_gated_inner_loop_step, passthrough_gate
from trustgate.types import Gate

#: Vendor commit this patch was written against.
PINNED_VENDOR_SHA = "a4fc4788ace38e29b5067916d4f4be33da894085"

_original_inner_loop_step: Callable | None = None


def install_gate(gate: Gate = passthrough_gate) -> None:
    """Replace `MetaModel.inner_loop_step` with the gated version.

    Idempotent in the sense that re-installing swaps the policy rather than
    wrapping twice -- double-wrapping would apply the gate to its own output and
    silently double-count drift.
    """
    from ttt.model.transformer import MetaModel

    global _original_inner_loop_step

    if _original_inner_loop_step is None:
        _original_inner_loop_step = MetaModel.inner_loop_step

    MetaModel.inner_loop_step = make_gated_inner_loop_step(
        _original_inner_loop_step, gate
    )


def uninstall_gate() -> None:
    """Restore the vendor's original method. Safe to call when not installed."""
    from ttt.model.transformer import MetaModel

    global _original_inner_loop_step

    if _original_inner_loop_step is not None:
        MetaModel.inner_loop_step = _original_inner_loop_step
        _original_inner_loop_step = None


def is_installed() -> bool:
    return _original_inner_loop_step is not None
