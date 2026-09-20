#!/usr/bin/env python
"""Write a small orbax checkpoint so the loading path can be rehearsed for free.

`build_from_checkpoint` has one failure mode that matters more than the rest:
`unify_dict_with_eqx_module` only *warns* about paths it could not find
(`infra/checkpoint.py:221-224`) and `train.py:171` discards the warning, so a
partial restore yields a model that is part checkpoint and part random init.
That model scores like a real result in every metric this project computes,
and it wears no banner.

`build_from_checkpoint` refuses such a restore. **This script exists so that
refusal is proven to fire before a billing instance trusts it with real
weights** -- runbook step A6. It also exercises the two cheaper guards: the
integer-step-directory layout orbax requires, and the shape assertion in
`fetch_from_eqx_module`.

Needs orbax, so it runs on the box (or any machine with the vendor environment),
not in the CPU test suite. The CPU suite covers the same logic through
`unexpected_missing_paths`, which is pure pytree work.

Usage:
    python scripts/make_dummy_checkpoint.py DEST [--step 12000] [--partial]

    --partial writes a checkpoint that is MISSING one weight, which is the case
    the loader must refuse. Use it to prove the refusal, not to produce a
    checkpoint anyone runs.
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path


def build(dest: Path, step: int, partial: bool) -> int:
    import equinox as eqx
    import jax
    import jax.numpy as jnp
    import orbax.checkpoint as ocp

    class Tiny(eqx.Module):
        """Two inexact leaves and one integer leaf.

        The integer leaf is the point of the fixture: `MetaModel.weights()` is
        `eqx.filter(self, eqx.is_inexact_array)`, so non-inexact leaves are
        legitimately absent from every healthy checkpoint. A loader that
        flagged them would refuse good weights.
        """

        w: jnp.ndarray
        b: jnp.ndarray
        counter: jnp.ndarray

    model = Tiny(
        w=jnp.arange(12, dtype=jnp.float32).reshape(3, 4),
        b=jnp.ones((4,), dtype=jnp.float32),
        counter=jnp.asarray([7], dtype=jnp.int32),
    )
    weights = eqx.filter(model, eqx.is_inexact_array)

    if partial:
        # Drop `b`. This is what a size mismatch or a truncated upload looks
        # like from the loader's side, and it is the case that must be refused
        # rather than warned about.
        weights = eqx.tree_at(lambda t: t.b, weights, replace=None, is_leaf=lambda x: x is None)

    if dest.exists():
        shutil.rmtree(dest)
    dest.mkdir(parents=True)

    registry = ocp.DefaultCheckpointHandlerRegistry()
    registry.add("model_weights", ocp.args.StandardSave, ocp.StandardCheckpointHandler)
    registry.add("model_weights", ocp.args.StandardRestore, ocp.StandardCheckpointHandler)

    with ocp.CheckpointManager(
        dest.resolve(), handler_registry=registry
    ) as manager:
        manager.save(
            step=step,
            args=ocp.args.Composite(model_weights=ocp.args.StandardSave(weights)),
        )
        manager.wait_until_finished()

    written = sorted(c.name for c in dest.iterdir())
    leaves = len(jax.tree_util.tree_leaves(weights))
    print(f"wrote {dest}/ step={step} entries={written} saved_leaves={leaves}")
    if partial:
        print(
            "PARTIAL on purpose: `b` is missing. build_from_checkpoint must "
            "REFUSE this rather than warn. If it loads, the guard is broken."
        )
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dest", type=Path)
    parser.add_argument("--step", type=int, default=12000)
    parser.add_argument(
        "--partial",
        action="store_true",
        help="omit one weight, to prove the loader refuses a partial restore",
    )
    args = parser.parse_args(argv)

    try:
        return build(args.dest, args.step, args.partial)
    except ImportError as exc:
        raise SystemExit(
            f"needs orbax and equinox from the vendor environment; this script "
            f"runs on the box, not in the CPU test suite. Original error: {exc}"
        ) from exc


if __name__ == "__main__":
    sys.exit(main())
