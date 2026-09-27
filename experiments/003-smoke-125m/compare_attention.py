#!/usr/bin/env python
"""Measure `trustgate.attention_patch` against the vendor's cuDNN attention on real hardware.

Needs an Ampere-or-newer GPU, where the vendor's cuDNN kernel runs -- the laptop's
RTX 3070 Ti. The point is to put a number on what running C1 on a Kaggle T4 (which
cannot run that kernel, 2026-09-27) costs in fidelity, before anyone reads a result
produced that way.

Two levels, both at the session's real geometry (seq 8192, 12 heads x 64, bf16):

1. kernel: one attention call, cuDNN vs blocked, for the prefix (causal) and the
   suffix sliding-window call shapes.
2. model: the full random-init 125M `loss_for_sequence` -- prefix pass plus 8
   chunks of inner-loop adaptation -- per-chunk loss, cuDNN vs blocked. Any kernel
   difference compounds through the adaptation here, which is what the 001
   measurements actually sit on.

Renders no verdict. Writes a JSON report.

    PYTHONPATH=src vendor/ttt-e2e/.venv/bin/python experiments/003-smoke-125m/compare_attention.py
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.90")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")


def kernel_level(seq: int = 8192, heads: int = 12, dim: int = 64) -> dict:
    import jax
    import jax.numpy as jnp

    from trustgate.attention_patch import blocked_attention

    ks = jax.random.split(jax.random.PRNGKey(0), 3)
    q, k, v = (jax.random.normal(x, (seq, heads, dim), jnp.bfloat16) for x in ks)
    out = {}
    for name, kw in (
        ("prefix causal (attention.py:214)", dict(is_causal=True)),
        ("sliding window (attention.py:312)", dict(is_causal=True, local_window_size=(seq - 1, 0))),
    ):
        cudnn = jax.jit(lambda q, k, v: jax.nn.dot_product_attention(q, k, v, implementation="cudnn", **kw))
        blocked = jax.jit(lambda q, k, v: blocked_attention(q, k, v, **kw))
        # No plain-XLA column: at T = 8192 it needs one 3.2 GB score matrix and OOMs
        # the 8 GB card -- the very cost the blocked version exists to avoid.
        a, b = (f(q, k, v).astype(jnp.float32) for f in (cudnn, blocked))
        out[name] = {
            "max_abs_blocked_vs_cudnn": float(jnp.max(jnp.abs(b - a))),
            "mean_abs_blocked_vs_cudnn": float(jnp.mean(jnp.abs(b - a))),
            "mean_abs_output": float(jnp.mean(jnp.abs(a))),
        }
        print(name, json.dumps(out[name]))
    return out


def model_level(seq: int = 8192) -> dict:
    import jax
    import jax.numpy as jnp

    from trustgate import attention_patch
    from trustgate.eval import vendor_bind
    from trustgate.eval.model_build import build_config, build_model, dummy_tokens
    from ttt.model.transformer import MetaModel

    cfg = build_config(size="125m", seq_length=seq, exp_dir="/tmp/trustgate-attn-compare")
    model, state, mesh = build_model(cfg)
    batch = vendor_bind.make_batch(dummy_tokens(seq + 1, seed=0), bos_token_id=cfg.model.bos_token_id)

    def per_chunk():
        with mesh:
            _, metrics = model.loss_for_sequence(batch, state)
        return [float(x) for x in jnp.asarray(metrics[MetaModel.MetricType.loss]).ravel()]

    attention_patch.uninstall_blocked_attention()
    cudnn_a = per_chunk()
    cudnn_b = per_chunk()  # same process, same kernel: the in-process noise floor
    attention_patch.install_blocked_attention()
    try:
        blocked = per_chunk()
    finally:
        attention_patch.uninstall_blocked_attention()

    gap = [b - a for a, b in zip(cudnn_a, blocked)]
    res = {
        "cudnn_per_chunk": cudnn_a,
        "cudnn_repeat_identical": cudnn_a == cudnn_b,
        "blocked_per_chunk": blocked,
        "per_chunk_gap": gap,
        "max_abs_gap_nats": max(abs(g) for g in gap),
        "chunk0_gap_nats": abs(gap[0]),
    }
    print(json.dumps({k: v for k, v in res.items() if "per_chunk" not in k}))
    print("per-chunk gap:", ", ".join(f"{g:+.2e}" for g in gap))
    return res


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default=str(Path(__file__).parent / "results" / "compare-attention.json"))
    args = ap.parse_args()

    import jax
    from jax._src.lib import cuda_versions as cv

    dev = jax.devices()[0]
    print(f"device {dev.device_kind} sm {getattr(dev, 'compute_capability', '?')} "
          f"| jax {jax.__version__} | cudnn {cv.cudnn_get_version()}")
    report = {
        "device": dev.device_kind,
        "compute_capability": getattr(dev, "compute_capability", None),
        "cudnn": cv.cudnn_get_version(),
        "kernel": kernel_level(),
        "model": model_level(),
        "context": (
            "Measured, not a verdict. For scale: the vendor's own loss moved 2.0e-4 nats per "
            "chunk between processes on this laptop (003 smoke, 2026-09-27); TOLERANCE.md's "
            "band is 0.491 nats; the random-init inner loop falls ~2.0 nats over 8 chunks."
        ),
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
