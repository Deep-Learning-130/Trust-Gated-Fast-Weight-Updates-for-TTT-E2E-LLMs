#!/usr/bin/env python
"""Prove the W&B credentials work the way the vendor uses them -- on a laptop, for free.

Run BEFORE renting a GPU. The vendor contacts W&B before it loads a single weight
(ADR-004 s3), and `training.log_wandb=false` does not avoid it:

    ttt/infra/wandb_utils.py:61   wandb.login(key=wandb_key)
    ttt/infra/wandb_utils.py:62   api = wandb.Api(api_key=wandb_key)
    ttt/infra/wandb_utils.py:63   runs = api.runs(f"{entity}/{project}", filters={"display_name": exp_name})
    ttt/infra/wandb_utils.py:64   num_existing = len(runs)

The `len(runs)` query is the risky one. Against a project that does not exist yet
the public API can raise instead of returning zero, and the first launch would die
there with the box billing. So this script makes the project exist (one tiny run
named `preflight`, only if the query fails) and then repeats the vendor's exact
calls. Exit 0 means the box will get past W&B.

Pin the vendor's wandb version so the check exercises the same client code:

    WANDB_ENTITY=... WANDB_PROJECT=... WANDB_KEY=... \\
        uv run --no-project --with wandb==0.19.9 python scripts/preflight_wandb.py
"""

from __future__ import annotations

import os
import sys

EXP_NAMES = (
    "smoke-1b_ttt_e2e_finetune_books_8k_1x_cc",
    "eval-1b_ttt_e2e_finetune_books_8k_1x_cc",
    "dummy-1b_ttt_e2e_finetune_books_8k_1x_cc",
)


def vendor_query(wandb, entity: str, project: str, key: str, exp_name: str) -> int:
    """wandb_utils.py:61-64, verbatim in effect."""
    wandb.login(key=key)
    api = wandb.Api(api_key=key)
    runs = api.runs(f"{entity}/{project}", filters={"display_name": exp_name})
    return len(runs)


def main() -> int:
    missing = [v for v in ("WANDB_ENTITY", "WANDB_PROJECT", "WANDB_KEY") if not os.environ.get(v)]
    if missing:
        print(f"FATAL: set {', '.join(missing)}", file=sys.stderr)
        return 1
    entity, project, key = (os.environ[v] for v in ("WANDB_ENTITY", "WANDB_PROJECT", "WANDB_KEY"))

    import wandb

    print(f"wandb client {wandb.__version__} (vendor lock: 0.19.9)")
    if wandb.__version__ != "0.19.9":
        print("WARNING: not the vendor's version; run with `uv run --no-project --with wandb==0.19.9`")

    try:
        n = vendor_query(wandb, entity, project, key, EXP_NAMES[0])
        print(f"vendor query OK against existing project {entity}/{project} ({n} matching runs)")
    except Exception as exc:  # the failure the box must not meet
        print(f"vendor query failed: {type(exc).__name__}: {exc}")
        print(f"creating {entity}/{project} with one tiny run named 'preflight', then retrying")
        run = wandb.init(entity=entity, project=project, name="preflight", settings=wandb.Settings(api_key=key))
        run.log({"preflight": 1})
        run.finish()

    try:
        for name in EXP_NAMES:
            n = vendor_query(wandb, entity, project, key, name)
            print(f"  {name}: {n} existing run(s)" + ("  (the vendor will RESUME it)" if n else ""))
    except Exception as exc:
        print(f"FATAL: vendor query still fails: {type(exc).__name__}: {exc}", file=sys.stderr)
        print("Check the entity name (your username or team), the key, and project permissions.", file=sys.stderr)
        return 1

    print("\nPREFLIGHT OK: the box will get past wandb_utils.py:61-64 with these credentials.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
