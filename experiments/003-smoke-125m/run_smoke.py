#!/usr/bin/env python
"""First execution of the vendor MetaModel and the carry overlay on real hardware.

**This renders no verdict.** See `README.md`. It is instrument validation: a
random-init 125M model has nothing worth corrupting, so nothing here is a
measurement of anything. What it establishes is that the wiring executes and
that the invariants the experiment depends on actually hold against the vendor
rather than against a CPU stand-in.

Needs: a GPU, an importable vendor tree. Needs **no** checkpoint (`load_part`
defaults to `none`, so `train.py:198` builds a fresh model), **no** dataset
(`training.dummy_dataset`), **no** GCS auth and **no** W&B key.

    python experiments/003-smoke-125m/run_smoke.py --seq-length 8192

Each check is independent and reports its own result. A failure does not abort
the run, because the point of a rehearsal is to find *all* the first-launch
failures in one session rather than one per session.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import traceback
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
VENDOR_CONFIGS = ROOT / "vendor" / "ttt-e2e" / "configs"
RESULTS = Path(__file__).resolve().parent / "results"

sys.path.insert(0, str(ROOT / "src"))

# JAX otherwise grabs 75% of the card at import, which on an 8GB laptop card
# leaves nothing for the 1024-token logit tensor (1024 x 128256 x 4B = 525 MB
# per chunk, and value_and_grad holds more than one).
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

# This run is single-device by construction: n_data_parallel=1 and
# n_state_parallel=1 below, and `ModelSharding.__init__` (sharding.py:33-35)
# asserts their product equals `jax.device_count()`. On any box with more than
# one accelerator -- Kaggle's T4 x2, a multi-GPU rental -- that assertion fires
# during model construction, after the vendor stack has already been imported.
#
# The vendor's own knobs do NOT prevent this. `backend.local_device_ids` and
# `backend.num_devices` are read only inside `if distributed_config.distributed:`
# (jax_utils.py:44-48), and this runner sets `backend.distributed=false`, so both
# are dead config here. CUDA_VISIBLE_DEVICES is the only lever that actually
# changes what JAX enumerates, and it has to be set before jax is imported.
#
# setdefault, so an explicit `CUDA_VISIBLE_DEVICES=1` to pick a different card
# still wins. Wanting several devices means changing n_data_parallel too, which
# is a different experiment from this one.
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")


@dataclass
class Check:
    name: str
    why: str
    """What failing this check would mean -- not what it does. Several of these
    are routes to a *false STOP*, which the numbers alone cannot reveal."""

    passed: bool | None = None
    detail: str = ""
    observed: dict = field(default_factory=dict)

    def ok(self, detail: str, **observed):
        self.passed, self.detail, self.observed = True, detail, observed
        return self

    def fail(self, detail: str, **observed):
        self.passed, self.detail, self.observed = False, detail, observed
        return self


CHECKS: list[Check] = []


def check(name: str, why: str):
    """Register a check that reports rather than raises."""

    def decorate(fn):
        def run(*args, **kwargs):
            c = Check(name=name, why=why)
            CHECKS.append(c)
            try:
                fn(c, *args, **kwargs)
            except Exception as exc:  # noqa: BLE001 - a failed check is data
                c.fail(f"{type(exc).__name__}: {exc}")
                c.observed["traceback"] = traceback.format_exc()
            return c

        return run

    return decorate


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


def build_config(args):
    """Compose the vendor's own hydra config, overridden for one small GPU.

    Composed rather than hand-built so the smoke run exercises the same config
    path `train.py` does. Every override below has a reason; see README.
    """
    from hydra import compose, initialize_config_dir
    from ttt.config import register_configs

    register_configs()

    overrides = [
        # `+` is required: neither group is in config.yaml's defaults list.
        # EVAL_ENTRYPOINT.md section 1 documents this, and the vendor README
        # (:61-66) uses the same form.
        "+experiment=125m/extension/ext-125m-e2e-32K",
        "+deploy=interactive",
        # `dataset_path` interpolates `deploy_paths.data[...]`, which ships as
        # `???` in interactive.yaml:9. Nothing reads it under dummy_dataset, but
        # OmegaConf resolves it eagerly, so it needs *a* value. No `+` here --
        # the key already exists, it is merely MISSING.
        "deploy_paths.data.books3=/dev/null",
        "training.dummy_dataset=true",
        "training.eval_mode=true",
        "training.log_wandb=false",
        f"training.seq_length={args.seq_length}",
        # Note: these do NOT give an eval batch of 1. `train.py:212` takes
        # max(eval_batch_size, global_batch_size // accum_steps * 4), so the
        # floor is 4 (EVAL_ENTRYPOINT.md section 5, gotcha 1). It does not
        # matter here -- this runner never calls `evaluator.eval_fn`, it drives
        # `loss_for_sequence` directly -- but it will matter for Phase 0.5.
        "training.global_batch_size=1",
        "training.eval_batch_size=1",
        "training.n_data_parallel=1",
        "training.n_state_parallel=1",
        f"training.exp_dir={args.exp_dir}",
        "backend.distributed=false",
        "backend.num_devices=1",
        'backend.local_device_ids="0"',
    ]
    if args.compute_dtype:
        overrides.append(f"model.compute_dtype={args.compute_dtype}")
    if args.param_dtype:
        # `qk_norm` defaults True (config.py:115) and q_norm/k_norm are built at
        # param_dtype (attention.py:100-101), which defaults to fp32. Normalising
        # a bf16 xq against an fp32 weight promotes the result back to fp32, and
        # the prefix path then hands it to a cuDNN kernel that takes fp16/bf16
        # only. Setting compute_dtype alone may therefore not be enough.
        overrides.append(f"model.param_dtype={args.param_dtype}")

    with initialize_config_dir(config_dir=str(VENDOR_CONFIGS), version_base=None):
        cfg = compose(config_name="config", overrides=overrides)

    # train.py:123 -- the model reads its length from here, not from training.
    cfg.model.seq_len = cfg.training.seq_length
    return cfg


def build_model(cfg):
    """`train.py:127-132`, minus the sharding of a single device."""
    import equinox as eqx
    import jax
    from ttt.model.sharding import ModelSharding
    from ttt.model.transformer import MetaModel
    from ttt.utils.jax_utils import set_random_seed

    key = set_random_seed(cfg.training.model_seed)
    model_sharding = ModelSharding(cfg)

    @eqx.filter_jit
    def create():
        model, state = eqx.nn.make_with_state(MetaModel)(cfg, key=key)
        state = jax.device_put(
            state, jax.NamedSharding(model_sharding.mesh, jax.sharding.PartitionSpec())
        )
        return model_sharding.shard_params(model), state

    with model_sharding.mesh:
        model, state = create()
    return model, state, model_sharding.mesh


def dummy_tokens(n: int, seed: int, vocab_lo: int = 0, vocab_hi: int = 20):
    """Match `DummyDataset.__getitem__` (`lm_dataset.py:36`): ids in [0, 20).

    Note what this means for the loss bar below. The *targets* only span 20
    ids, but a random-init model spreads its mass over all 128256, so CE starts
    at ln(128256). It would fall toward ln(20) only after training -- which is
    the tell that this model has learned the dummy distribution and not
    language.
    """
    import numpy as np

    rng = np.random.default_rng(seed)
    return np.asarray(rng.integers(vocab_lo, vocab_hi, n, dtype=np.int32))


# ---------------------------------------------------------------------------
# Tier 0 -- the vendor runs at all
# ---------------------------------------------------------------------------


@check(
    "gpu-present",
    "A silent CPU fallback would make every timing and memory number below "
    "meaningless while still passing.",
)
def check_gpu(c):
    import jax

    devices = jax.devices()
    kinds = sorted({d.platform for d in devices})
    if "gpu" in kinds or "cuda" in kinds or any(d.platform != "cpu" for d in devices):
        c.ok(f"{len(devices)} device(s): {devices}", devices=[str(d) for d in devices])
    else:
        c.fail(
            f"only CPU devices found: {devices}. On Windows this usually means "
            "jax[cuda12] is not installed -- there are no Windows CUDA wheels, "
            "so the vendor stack needs WSL2 or Colab.",
            devices=[str(d) for d in devices],
        )


@check(
    "random-init-loss",
    "If CE at init is not ln(vocab), the loss, the masking or the dtype is "
    "wrong, and nothing downstream of it can be trusted.",
)
def check_random_init_loss(c, cfg, model, state, mesh, tol=0.05):
    import jax.numpy as jnp

    from trustgate.eval import vendor_bind

    expected = math.log(cfg.model.vocab_size)
    seq = vendor_bind.make_batch(
        dummy_tokens(cfg.training.seq_length + 1, seed=0),
        bos_token_id=cfg.model.bos_token_id,
    )

    with mesh:
        loss, _metrics = model.loss_for_sequence(seq, state)
    observed = float(jnp.asarray(loss))

    detail = f"CE {observed:.4f} nats vs ln({cfg.model.vocab_size}) = {expected:.4f}"
    if abs(observed - expected) <= tol:
        c.ok(detail, observed_nats=observed, expected_nats=expected)
    else:
        c.fail(detail + " -- outside tolerance", observed_nats=observed, expected_nats=expected)


@check(
    "param-count",
    "A mismatch means the config that built this model is not the config the "
    "analytic cost model was priced against (COST_MODEL.md section 2.1).",
)
def check_param_count(c, model, binding):
    """Counted against `binding.model_split`, not against `model`.

    A freshly built `MetaModel` has **no `suffix_blocks` attribute at all**, so
    `model.inner_parameters()` cannot match `spec_inner` and raises rather than
    returning zero. `BlockCollectionSplit` is constructed *inside*
    `loss_for_sequence` (`transformer.py:662-668`) and grafted onto the tree at
    `:676`, on every call -- it is not part of model construction. Counting the
    fast weights therefore requires a split model, which is exactly what
    `vendor_bind.bind` already produces.

    `trainable_parameters()` is safe on the unsplit tree: it filters on
    `spec_outer`, which does not mention `suffix_blocks`.
    """
    import jax

    trainable = sum(
        x.size for x in jax.tree_util.tree_leaves(model.trainable_parameters())
    )
    inner = sum(
        x.size
        for x in jax.tree_util.tree_leaves(binding.model_split.inner_parameters())
    )
    c.ok(
        f"{trainable:,} trainable, {inner:,} inner (fast) weights",
        trainable_params=int(trainable),
        inner_params=int(inner),
    )
    if inner == 0:
        c.fail(
            "zero inner parameters: the inner spec matched nothing, so there "
            "are no fast weights to adapt and any attack result would be a "
            "structural null.",
            trainable_params=int(trainable),
            inner_params=0,
        )


@check(
    "determinism",
    "A non-deterministic forward pass means the poison/control difference is "
    "not attributable to the stream, which invalidates the comparison.",
)
def check_determinism(c, cfg, mesh):
    import jax.numpy as jnp

    from trustgate.eval import vendor_bind

    seq = vendor_bind.make_batch(
        dummy_tokens(cfg.training.seq_length + 1, seed=0),
        bos_token_id=cfg.model.bos_token_id,
    )
    losses = []
    for _ in range(2):
        model, state, _ = build_model(cfg)
        with mesh:
            loss, _ = model.loss_for_sequence(seq, state)
        losses.append(float(jnp.asarray(loss)))

    if losses[0] == losses[1]:
        c.ok(f"bit-identical across two builds: {losses[0]:.6f}", losses=losses)
    else:
        c.fail(f"differed: {losses[0]:.8f} vs {losses[1]:.8f}", losses=losses)


# ---------------------------------------------------------------------------
# Tier 1 -- the carry overlay against the real model. The reason for the run.
# ---------------------------------------------------------------------------


@check(
    "inner-lr-saturated",
    "An unsaturated inner LR runs a near-frozen inner loop and produces a null "
    "for a reason unrelated to the attack -- a false STOP (ADR-006).",
)
def check_inner_lr(c, binding):
    from trustgate.eval.carry import SATURATED_INNER_LR_MULTIPLIER

    m = binding.inner_lr_multiplier
    detail = f"multiplier {m} (expected {SATURATED_INNER_LR_MULTIPLIER})"
    if abs(m - SATURATED_INNER_LR_MULTIPLIER) <= 1e-6:
        # Do not let this pass be over-read. At this config the ramp is a
        # constant, so the check is vacuous here and is NOT evidence that the
        # guard works where `ilr_warmup_steps > 0`.
        c.ok(
            detail
            + " -- NOTE: training/125m/ext.yaml leaves ilr_warmup_steps at 0, "
            "so get_ilr_multiplier returns a hard 1.0 (transformer.py:565-567) "
            "and this check cannot fail at this config. It is not evidence the "
            "guard works where the ramp is live.",
            multiplier=m,
        )
    else:
        c.fail(detail, multiplier=m)


@check(
    "carry-is-non-trivial",
    "THE check this run exists for. If fast weights come back unchanged, the "
    "vendor's scan-carry discard (transformer.py:712) is still in force and "
    "every attack measurement would be eval noise -- a STOP indistinguishable "
    "from a real one, bought with a paid GPU run (ADR-006).",
)
def check_carry(c, cfg, binding, mesh, condition, stream, out: dict):
    import jax
    import jax.numpy as jnp

    from trustgate.eval.harness import run_stream

    initial = binding.fast_weights()
    with mesh:
        carry = run_stream(None, stream, condition, binding=binding)

    deltas = [
        float(jnp.linalg.norm(jnp.asarray(a) - jnp.asarray(b)))
        for a, b in zip(
            jax.tree_util.tree_leaves(carry.fast_weights),
            jax.tree_util.tree_leaves(initial),
        )
    ]
    total = float(sum(deltas))

    detail = f"||adapted - initial|| = {total:.6e} over {int(carry.n_steps)} inner steps"
    if total > 0.0:
        out["carry"] = carry
        c.ok(detail, delta_norm=total, n_steps=int(carry.n_steps))
    else:
        c.fail(
            detail + " -- the carry is NOT carrying; see ADR-006",
            delta_norm=0.0,
            n_steps=int(carry.n_steps),
        )


@check(
    "adaptation-changes-benign-loss",
    "The other half of the same question. Fast weights could move and still be "
    "discarded before measurement -- that is precisely the two-call harness "
    "ADR-006 rules out.",
)
def check_eval_differs(c, binding, mesh, carry, eval_tokens, condition):
    from trustgate.eval.harness import eval_benign_curve

    with mesh:
        fresh = eval_benign_curve(binding, binding.init_carry(), eval_tokens, condition)
        carried = eval_benign_curve(binding, carry, eval_tokens, condition)

    delta = carried.mean_loss - fresh.mean_loss
    detail = (
        f"fresh {fresh.mean_loss:.6f} vs carried {carried.mean_loss:.6f} "
        f"(delta {delta:+.6e}); carried curve decay {carried.decay:+.6e}"
    )
    observed = dict(
        fresh_mean=fresh.mean_loss,
        carried_mean=carried.mean_loss,
        delta=delta,
        fresh_curve=list(fresh.per_chunk_loss),
        carried_curve=list(carried.per_chunk_loss),
        carried_decay=carried.decay,
    )
    if delta != 0.0:
        c.ok(detail, **observed)
    else:
        c.fail(
            detail + " -- identical, so the carry did not reach the measurement",
            **observed,
        )


# ---------------------------------------------------------------------------


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--seq-length",
        type=int,
        default=8192,
        help="Must be a multiple of mini_batch_size (1024). 8192 gives 8 inner "
        "steps -- the granularity the pre-registration is scoped to.",
    )
    ap.add_argument(
        "--stream-sequences",
        type=int,
        default=2,
        help="Sequences in the adaptation stream. >1 exercises cross-sequence "
        "carry, which is the capability the vendor does not have at all.",
    )
    ap.add_argument(
        "--compute-dtype",
        default=None,
        help="Override model.compute_dtype (vendor default bf16). NOTE: fp32 "
        "cannot work -- the prefix path forces a cuDNN kernel that takes "
        "fp16/bf16 only. Recorded in the results either way.",
    )
    ap.add_argument(
        "--param-dtype",
        default=None,
        help="Override model.param_dtype (vendor default fp32). Try bf16 if the "
        "attention kernel still reports float32 with compute_dtype at bf16: "
        "qk_norm runs at param_dtype and promotes xq back up. Recorded.",
    )
    ap.add_argument("--exp-dir", default="/tmp/trustgate-smoke")
    ap.add_argument("--out", default=str(RESULTS / "smoke.json"))
    args = ap.parse_args()

    print(__doc__.strip().split("\n\n")[0])
    print(f"\nseq_length={args.seq_length} stream_sequences={args.stream_sequences}\n")

    # Fail here, legibly, rather than three frames deep in hydra. Without the
    # vendor tree there is nothing to smoke-test, and the overwhelmingly likely
    # cause is running on the CPU dev box instead of the GPU box.
    try:
        import hydra  # noqa: F401
        import ttt  # noqa: F401
    except ImportError as exc:
        print(
            f"cannot import the vendor stack ({exc}).\n"
            "Run `bash experiments/003-smoke-125m/setup.sh` first, on a machine "
            "with a GPU.\n"
            "The CPU dev environment deliberately installs none of hydra/einops/"
            "zarr/grain/orbax/wandb -- that is what keeps the test suite and CI "
            "network-free.\n"
            "Nothing is wrong with this checkout; it is the wrong box.",
            file=sys.stderr,
        )
        return 2

    check_gpu()
    if not CHECKS[-1].passed:
        print(f"\n[FAIL] gpu-present: {CHECKS[-1].detail}", file=sys.stderr)
        print(
            "Refusing to continue: every number below would be a CPU number.",
            file=sys.stderr,
        )
        return 2

    cfg = build_config(args)
    model, state, mesh = build_model(cfg)

    check_random_init_loss(cfg, model, state, mesh)
    check_determinism(cfg, mesh)

    from trustgate.eval import vendor_bind
    from trustgate.eval.harness import RunCondition, eval_tokens_digest

    with mesh:
        binding = vendor_bind.bind(model, state)

    # After bind, not before: the fast weights do not exist as an addressable
    # subtree until the block split has happened. See check_param_count.
    check_param_count(model, binding)

    check_inner_lr(binding)

    # Disjoint slices. Measuring on the stream itself is a pre-registered
    # invalidating condition, so the eval tokens are drawn from a different
    # seed than the stream and the two never overlap.
    n_stream = args.seq_length * args.stream_sequences
    stream = dummy_tokens(n_stream, seed=1)
    eval_tokens = dummy_tokens(args.seq_length + 1, seed=2)

    condition = RunCondition(
        seed=0,
        stream_tokens=n_stream,
        mini_batch_size=binding.mini_batch_size,
        seq_length=args.seq_length,
        checkpoint="random-init-125m (NO CHECKPOINT -- not a measurement)",
        benign_eval_split="dummy-eval",
        span_tokens=binding.mini_batch_size,
        n_chunks=n_stream // binding.mini_batch_size,
        valid_tokens=n_stream,
        dtype=str(stream.dtype),
        stream_corpus_split="dummy-stream",
        eval_tokens_sha256=eval_tokens_digest(eval_tokens[:-1]),
        inner_lr_multiplier=binding.inner_lr_multiplier,
    )

    # Adapting the stream is the expensive step; hand the carry straight to
    # the next check rather than paying for it twice.
    adapted: dict = {}
    check_carry(cfg, binding, mesh, condition, stream, adapted)
    if "carry" in adapted:
        check_eval_differs(binding, mesh, adapted["carry"], eval_tokens, condition)

    # ---- report ----
    print("\n" + "=" * 72)
    failed = [c for c in CHECKS if not c.passed]
    for c in CHECKS:
        mark = "PASS" if c.passed else "FAIL"
        print(f"[{mark}] {c.name}: {c.detail}")
        if not c.passed:
            print(f"       why it matters: {c.why}")

    print("=" * 72)
    print(f"{len(CHECKS) - len(failed)}/{len(CHECKS)} checks passed")
    print(
        "\nThis run renders NO verdict and touches no pre-registered bar. "
        "A random-init\n125M model has nothing worth corrupting; these are "
        "instrument checks only."
    )

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(
        json.dumps(
            {
                "disclaimer": (
                    "Instrument validation only. Random-init 125M, dummy data, "
                    "no checkpoint. Renders no PROCEED/STOP and touches no "
                    "pre-registered threshold. Not comparable to TOLERANCE.md."
                ),
                "config": {
                    "seq_length": args.seq_length,
                    "stream_sequences": args.stream_sequences,
                    "mini_batch_size": binding.mini_batch_size,
                    "compute_dtype": args.compute_dtype or "bf16 (config default)",
                    "param_dtype": args.param_dtype or "fp32 (config default)",
                    "suffix_len": binding.suffix_len,
                },
                "checks": [
                    {
                        "name": c.name,
                        "passed": c.passed,
                        "detail": c.detail,
                        "why": c.why,
                        "observed": c.observed,
                    }
                    for c in CHECKS
                ],
            },
            indent=2,
            default=str,
        ),
        encoding="utf-8",
    )
    print(f"\nWrote {args.out}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
