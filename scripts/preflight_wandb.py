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

How the key travels, exactly as the generated eval scripts pass it: through
WANDB_API_KEY and a private netrc, with the vendor's `wandb_key` EMPTY. wandb 0.19.9
(the vendor lock) raises "API key must be 40 characters long" inside
`wandb.login(key=...)`, and W&B now issues 86-character `wandb_v1_` keys; with an
empty key that version skips the check and reads the environment, and
`wandb.init`'s backend reads the netrc. So this checks the route the box uses,
for old and new keys alike.

Pin the vendor's wandb version so the check exercises the same client code:

    WANDB_ENTITY=... WANDB_PROJECT=... WANDB_KEY=... \\
        uv run --no-project --with wandb==0.19.9 python scripts/preflight_wandb.py
"""

from __future__ import annotations

import os
import sys
import tempfile

EXP_NAMES = (
    "smoke-1b_ttt_e2e_finetune_books_8k_1x_cc",
    "eval-1b_ttt_e2e_finetune_books_8k_1x_cc",
    "dummy-1b_ttt_e2e_finetune_books_8k_1x_cc",
)


#: What the generated eval scripts pass as `training.wandb_key`.
VENDOR_KEY_ARG = ""


def vendor_query(wandb, entity: str, project: str, key: str, exp_name: str) -> int:
    """wandb_utils.py:61-64, verbatim in effect, with the box's empty key argument."""
    wandb.login(key=VENDOR_KEY_ARG)
    api = wandb.Api(api_key=VENDOR_KEY_ARG)
    runs = api.runs(f"{entity}/{project}", filters={"display_name": exp_name})
    return len(runs)


def main() -> int:
    missing = [v for v in ("WANDB_ENTITY", "WANDB_PROJECT", "WANDB_KEY") if not os.environ.get(v)]
    if missing:
        print(f"FATAL: set {', '.join(missing)}", file=sys.stderr)
        return 1
    entity, project, key = (os.environ[v] for v in ("WANDB_ENTITY", "WANDB_PROJECT", "WANDB_KEY"))

    # The box's route: environment + private netrc (see the module docstring).
    os.environ["WANDB_API_KEY"] = key
    fd, netrc = tempfile.mkstemp(prefix="wandb-preflight-", suffix=".netrc")
    with os.fdopen(fd, "w") as fh:
        fh.write(f"machine api.wandb.ai\n  login user\n  password {key}\n")
    os.environ["NETRC"] = netrc
    print(f"key: {len(key)} characters" + (" (new wandb_v1_ format)" if key.startswith("wandb_v1_") else ""))
    try:
        return _check(entity, project, key)
    finally:
        os.remove(netrc)


def _check(entity: str, project: str, key: str) -> int:
    import wandb

    print(f"wandb client {wandb.__version__} (vendor lock: 0.19.9)")
    if wandb.__version__ != "0.19.9":
        print("WARNING: not the vendor's version; run with `uv run --no-project --with wandb==0.19.9`")

    try:
        n = vendor_query(wandb, entity, project, key, EXP_NAMES[0])
        print(f"vendor query OK against existing project {entity}/{project} ({n} matching runs)")
    except Exception as exc:  # the failure the box must not meet
        print(f"vendor query failed: {type(exc).__name__}: {exc}")
        print(f"creating {entity}/{project} with the run below, then retrying")

    # Always, not only on a missing project: the box's first 000 run calls
    # wandb.init with the same empty key, and that is the path that needs the
    # netrc. One tiny run named `preflight`; its local files go to a temp dir.
    try:
        run = wandb.init(entity=entity, project=project, name="preflight", dir=tempfile.gettempdir(),
                         settings=wandb.Settings(api_key=VENDOR_KEY_ARG, entity=entity, project=project))
        run.log({"preflight": 1})
        run.finish()
        print("wandb.init OK (the path the box's first 000 run takes)")
    except Exception as exc:
        print(f"FATAL: wandb.init fails: {type(exc).__name__}: {exc}", file=sys.stderr)
        print("Check the key (wandb.ai/authorize) and that the entity is yours.", file=sys.stderr)
        return 1

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
