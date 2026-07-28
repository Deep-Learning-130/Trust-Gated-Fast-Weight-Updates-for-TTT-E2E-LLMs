"""Shared fixtures. Force JAX onto CPU before it is imported anywhere.

These tests must run on any machine, including CI without a GPU. Setting the
platform here, before the first `import jax`, keeps a stray CUDA install from
being selected and turning a logic test into an environment test.
"""

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "0")

import jax.numpy as jnp
import pytest


@pytest.fixture
def tiny_fast_weights():
    """A minimal pytree standing in for `MetaModel.inner_parameters()`.

    A dict of arrays plus one non-array leaf, to prove the tree ops leave static
    leaves untouched -- the failure mode that breaks eqx.Module round-trips.
    """
    return {
        "layer0": {
            "w": jnp.arange(6, dtype=jnp.float32).reshape(2, 3),
            "b": jnp.zeros(3, dtype=jnp.float32),
        },
        "layer1": {
            "w": jnp.ones((3, 2), dtype=jnp.float32),
        },
        "meta": "not-an-array",  # static leaf, must pass through untouched
    }
