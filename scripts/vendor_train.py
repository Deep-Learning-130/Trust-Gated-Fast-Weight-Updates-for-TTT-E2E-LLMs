#!/usr/bin/env python
"""Run the vendor's `train` entry point with a declared instrument patch installed.

Exists for one case: C1 on a GPU the vendor's cuDNN attention cannot run on (a
Kaggle T4, 2026-09-27). The generated eval scripts call this instead of the
`train` console script when the bootstrap ran with BLOCKED_ATTENTION=1. It installs
`trustgate.attention_patch`, prints one marker line that
`check_baseline_acceptance.py` looks for, and then calls `ttt.train:main` -- the
same function the `train` console script calls (vendor pyproject.toml), with the
same Hydra overrides from the command line.

Nothing else from trustgate is imported: not the gate, not the interceptor, not
vendor_patch. TOLERANCE.md S4 (revised 2026-09-27) still fails on any of those.

    uv run --no-sync python scripts/vendor_train.py <hydra overrides...>
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

#: Printed verbatim; check_baseline_acceptance.py matches it. Keep them in sync.
MARKER = "INSTRUMENT PATCH: trustgate.attention_patch installed"


def main() -> None:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
    from trustgate import attention_patch

    attention_patch.install_blocked_attention()
    print(
        f"{MARKER} -- blocked exact attention serves every jax.nn.dot_product_attention "
        f"call (the vendor's cuDNN kernel cannot run on this GPU). TOLERANCE.md S4 "
        f"revision 2026-09-27; measured against cuDNN in experiments/003-smoke-125m.",
        flush=True,
    )
    os.environ["TRUSTGATE_INSTRUMENT_PATCH"] = "attention"

    from ttt.train import main as vendor_main

    vendor_main()


if __name__ == "__main__":
    main()
