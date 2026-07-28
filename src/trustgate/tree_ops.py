"""Pytree arithmetic over fast weights.

All operations are jit-safe and operate only on inexact array leaves. Non-array
leaves (Equinox stores static config, callables and `None` in its pytrees) pass
through untouched -- getting this wrong is the most common way to break an
`eqx.Module` round-trip.
"""

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp

from trustgate.types import FastWeights, Updates


def _is_inexact(leaf) -> bool:
    return eqx.is_inexact_array(leaf)


def tree_sub(a: FastWeights, b: FastWeights) -> Updates:
    """Elementwise `a - b` over inexact leaves. Structure of `a` is preserved."""
    return jax.tree.map(
        lambda x, y: (x - y) if _is_inexact(x) else x,
        a,
        b,
    )


def tree_add(a: FastWeights, b: Updates) -> FastWeights:
    """Elementwise `a + b` over inexact leaves."""
    return jax.tree.map(
        lambda x, y: (x + y) if _is_inexact(x) else x,
        a,
        b,
    )


def tree_select(pred: jnp.ndarray, on_true: FastWeights, on_false: FastWeights) -> FastWeights:
    """Traced-safe pytree-wide `where`.

    `pred` is a scalar boolean *array*, so this compiles to a select rather than
    a Python branch. Both operands are evaluated -- that is unavoidable inside a
    scan, and it is why rejecting an update costs the same compute as accepting
    one. The saving from a reject is in *drift*, not in FLOPs.
    """
    return jax.tree.map(
        lambda t, f: jnp.where(pred, t, f) if _is_inexact(t) else t,
        on_true,
        on_false,
    )


def tree_global_norm(tree) -> jnp.ndarray:
    """L2 norm over all inexact leaves, as a float32 scalar.

    Used as the default drift metric: `||delta_theta||_2`. Computed in float32
    regardless of parameter dtype, because bf16 accumulation of a sum of squares
    over a 1B-parameter subtree loses too much precision to threshold against.
    """
    leaves = [leaf for leaf in jax.tree.leaves(tree) if _is_inexact(leaf)]
    if not leaves:
        return jnp.float32(0.0)
    squares = [jnp.sum(jnp.square(leaf.astype(jnp.float32))) for leaf in leaves]
    return jnp.sqrt(jnp.sum(jnp.stack(squares)))


def tree_zeros_like(tree) -> FastWeights:
    """Zeros for inexact leaves, passthrough otherwise."""
    return jax.tree.map(
        lambda x: jnp.zeros_like(x) if _is_inexact(x) else x,
        tree,
    )
