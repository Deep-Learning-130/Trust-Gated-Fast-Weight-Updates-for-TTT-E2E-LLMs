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
import gc
import json
import math
import os
import sys
import traceback
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RESULTS = Path(__file__).resolve().parent / "results"

sys.path.insert(0, str(ROOT / "src"))

# JAX otherwise grabs 75% of the card at import, which on an 8GB laptop card
# leaves nothing for the 1024-token logit tensor (1024 x 128256 x 4B = 525 MB
# per chunk, and value_and_grad holds more than one).
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

# Preallocation off does NOT lift JAX's cap: the allocator still stops at 75% of
# the card, 6.0 GiB of 8. Observed 2026-09-27 on the laptop: carry-is-non-trivial
# OOMed on the vendor's own loss at that cap, and passed at 0.90 with a 5.82 GiB
# peak. 0.90 rather than more because Windows holds ~680 MiB of the same card.
os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.90")

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
            c.observed.update(gpu_memory())
            return c

        return run

    return decorate


def gpu_memory() -> dict:
    """Device memory after a check, so an OOM names the check that caused it.

    `peak_bytes_in_use` is the process-lifetime peak, not per-check: a check
    that raises it is the one that set a new high. Empty on backends that do not
    report (CPU).
    """
    try:
        import jax

        stats = jax.devices()[0].memory_stats() or {}
    except Exception:  # noqa: BLE001 - never fail a check over its telemetry
        return {}
    keys = ("peak_bytes_in_use", "bytes_in_use", "bytes_limit")
    return {f"gpu_{k}": int(stats[k]) for k in keys if k in stats}


# ---------------------------------------------------------------------------
# Config and model
# ---------------------------------------------------------------------------
# These used to live here. They now live in `trustgate.eval.model_build`,
# because `trustgate.eval.cli` needs the same victim and two copies of a config
# this fiddly would drift. The overrides and their reasons moved with them.

from trustgate.eval.model_build import (  # noqa: E402
    build_config as _build_config,
    build_model,
    dummy_tokens,
)


def build_config(args):
    """Adapt this runner's argparse namespace to `model_build.build_config`."""
    return _build_config(
        size="125m",
        seq_length=args.seq_length,
        compute_dtype=args.compute_dtype,
        param_dtype=args.param_dtype,
        exp_dir=args.exp_dir,
    )


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
    "If the FIRST chunk's CE is not ln(vocab), the loss, the masking or the "
    "dtype is wrong, and nothing downstream of it can be trusted.",
)
def check_random_init_loss(c, cfg, model, state, mesh, tol=0.05):
    import jax.numpy as jnp

    from trustgate.eval import vendor_bind

    # ln(vocab) is the CE of a *uniform* predictive distribution, and a
    # random-init model is not uniform. With tied embeddings the logits are
    # h @ E.T, where the final RMSNorm leaves h at unit second moment and E is
    # initialised N(0, initializer_range^2). So each logit has variance
    # hidden_size * initializer_range^2, and for large V
    #
    #     E[CE] = E[logsumexp(z)] - E[z_target] ~= ln(V) + sigma^2 / 2
    #
    # At this config that is 11.7618 + 0.1536 = 11.9154 against an observed
    # 11.8858 -- a residual of -0.030, inside the existing tolerance.
    #
    # This correction was added 2026-09-15, AFTER seeing 11.8858 fail against a
    # bare ln(V). It is a derivation from two config values rather than a bar
    # widened to fit: it predicts a value 0.030 ABOVE what was observed, and it
    # would fail just as loudly if the loss were wrong in the other direction.
    # Nothing pre-registered is touched -- this is an instrument check, and
    # Standing Rule 5 governs PREREGISTERED.md, not this file.
    uniform = math.log(cfg.model.vocab_size)
    init_logit_var = cfg.model.hidden_size * cfg.model.initializer_range**2
    expected = uniform + init_logit_var / 2
    seq = vendor_bind.make_batch(
        dummy_tokens(cfg.training.seq_length + 1, seed=0),
        bos_token_id=cfg.model.bos_token_id,
    )

    with mesh:
        loss, metrics = model.loss_for_sequence(seq, state)

    # The bar is the FIRST chunk, not the mean. `loss_for_sequence` returns
    # `metrics[M.loss].mean()` in meta mode (`transformer.py:720`), and each
    # chunk is scored after the inner loop has already adapted on the ones
    # before it. Only chunk 0 is measured at initialisation; the mean is a
    # partly-adapted quantity and is *expected* to sit below ln(vocab).
    #
    # Checking the mean against ln(vocab) -- as this did until 2026-09-15 --
    # reports a working inner loop as a failure, and invites someone to "fix"
    # the model until it matches a bar that does not describe it.
    from ttt.model.transformer import MetaModel

    per_chunk = [float(x) for x in jnp.asarray(metrics[MetaModel.MetricType.loss]).ravel()]
    observed = per_chunk[0]
    mean = float(jnp.asarray(loss))

    curve = ", ".join(f"{v:.3f}" for v in per_chunk)
    drop = per_chunk[0] - min(per_chunk)
    detail = (
        f"chunk 0 CE {observed:.4f} vs expected {expected:.4f} "
        f"(= ln {cfg.model.vocab_size} + init logit var/2 = {uniform:.4f} + "
        f"{init_logit_var / 2:.4f}); mean over {len(per_chunk)} chunks "
        f"{mean:.4f}, falling {drop:.3f} nats; curve [{curve}]"
    )
    obs = dict(
        observed_nats=observed,
        expected_nats=expected,
        uniform_nats=uniform,
        init_logit_var=init_logit_var,
        mean_nats=mean,
        per_chunk_nats=per_chunk,
        inner_loop_drop_nats=drop,
    )
    if abs(observed - expected) <= tol:
        c.ok(detail, **obs)
    else:
        c.fail(detail + " -- chunk 0 outside tolerance", **obs)


@check(
    "chunked-ce-equivalence",
    "trustgate.memory_patch replaces the vendor's lm_loss so the 8 GB card fits. "
    "If it does not reproduce the vendor's numbers on the real model, every "
    "later measurement is of a different computation.",
)
def check_chunked_ce(c, cfg, model, state, mesh, tol_nats=1e-3):
    """Run the whole 8-chunk meta-learning pass twice on one sequence: vendor
    `lm_loss`, then the sliced one. The inner loop trains on this loss, so a
    difference in its gradients would show up as drift across the chunks.

    The bar, 1e-3 nats per chunk, was set before this ran on a GPU. It is ~500x
    below TOLERANCE.md's 0.491-nat band. On CPU the two agree to ~1e-7
    (tests/test_memory_patch.py); what is left on a GPU is GEMM tiling of a
    128-row vs a 1024-row bf16 matmul. If this fails, find out why -- do not
    widen it.
    """
    import jax.numpy as jnp

    from trustgate import memory_patch
    from trustgate.eval import vendor_bind
    from ttt.model.transformer import MetaModel

    M = MetaModel.MetricType
    seq = vendor_bind.make_batch(
        dummy_tokens(cfg.training.seq_length + 1, seed=0),
        bos_token_id=cfg.model.bos_token_id,
    )

    def run():
        with mesh:
            _, metrics = model.loss_for_sequence(seq, state)
        return (
            jnp.asarray(metrics[M.loss], dtype=jnp.float32).ravel(),
            jnp.asarray(metrics[M.token_nll_loss], dtype=jnp.float32).ravel(),
        )

    memory_patch.uninstall_chunked_ce()
    vendor_loss, vendor_nll = run()
    memory_patch.install_chunked_ce()
    sliced_loss, sliced_nll = run()

    loss_gap = float(jnp.max(jnp.abs(sliced_loss - vendor_loss)))
    nll_gap = float(jnp.max(jnp.abs(sliced_nll - vendor_nll)))
    detail = (
        f"max |sliced - vendor| per-chunk loss {loss_gap:.3e} nats, "
        f"per-token NLL {nll_gap:.3e} (bar {tol_nats:g} on the per-chunk loss)"
    )
    obs = dict(
        per_chunk_loss_gap=loss_gap,
        per_token_nll_gap=nll_gap,
        vendor_per_chunk=[float(x) for x in vendor_loss],
        sliced_per_chunk=[float(x) for x in sliced_loss],
    )
    if loss_gap <= tol_nats:
        c.ok(detail, **obs)
    else:
        c.fail(detail + " -- outside the bar", **obs)


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
        # Drop the build before the next one. Holding both costs a second
        # fp32 copy of the parameters -- 737 MB at 125M -- which is enough to
        # push an 8 GB card into OOM on a run that otherwise fits. `float()`
        # above has already pulled the only value we need back to the host.
        del model, state, loss
        gc.collect()

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


def code_revision() -> str:
    """Identify the checked-out revision, so every run says what it ran.

    Three sessions of this experiment were spent on a stale checkout: the output
    was byte-identical each time and read as a persistent technical failure
    rather than an un-applied `git pull`. A result that cannot name its own code
    is not reproducible, and more immediately it cannot be told apart from the
    run before it.
    """
    import subprocess

    def _git(*args: str) -> subprocess.CompletedProcess:
        return subprocess.run(
            ["git", "-C", str(ROOT), *args],
            capture_output=True,
            text=True,
            timeout=10,
        )

    try:
        head = _git("rev-parse", "--short", "HEAD")
        if head.returncode != 0:
            return "unknown (git failed: " + head.stderr.strip()[:80] + ")"
        # -c core.fileMode=false: this repo is routinely checked out on a
        # Windows filesystem and run from WSL, where git sees different mode
        # bits and reports every tracked file as modified. Without this the
        # dirty marker fires on a clean tree, which is worse than not having
        # it -- a marker that is always on carries no information.
        dirty = _git("-c", "core.fileMode=false", "status", "--porcelain").stdout.strip()
        return head.stdout.strip() + (" +local-changes" if dirty else "")
    except Exception as exc:  # noqa: BLE001 - never block the run on this
        return "unknown (" + type(exc).__name__ + ")"


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
    ap.add_argument(
        "--chunked-ce",
        action="store_true",
        help="Check trustgate.memory_patch against the vendor's lm_loss on this "
        "model (chunked-ce-equivalence), then keep it for the checks after. Off by "
        "default: the vendor's own loss fits 8 GB at a 0.90 memory fraction.",
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

    print("runner revision: " + code_revision())
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

    # On the vendor's own lm_loss: the reference every later number rests on.
    check_random_init_loss(cfg, model, state, mesh)
    if args.chunked_ce:
        # Compares the two on this model and leaves the patch installed, before
        # `bind` traces anything, for every check below.
        check_chunked_ce(cfg, model, state, mesh)

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

    # Determinism runs LAST, and deliberately. It is the only check that builds
    # a second model, so on a card where memory is tight it is the one most
    # likely to fail -- and it is also the least important. Running it before
    # the carry check meant an 8 GB box spent its last free bytes proving the
    # forward pass was reproducible and then had nothing left for the check the
    # whole experiment exists for.
    #
    # And it runs with nothing else resident. Observed 2026-09-27 on the laptop:
    # after the carry checks, 5.70 GiB was still held by the model, the binding
    # and the adapted carry, and determinism's own build OOMed on top of it. It
    # needs none of them -- it builds its own models from `cfg`.
    mini_batch_size, suffix_len = binding.mini_batch_size, binding.suffix_len
    del model, state, binding, adapted
    gc.collect()
    # `del` alone left 4.75 GiB in use on the same box: compiled executables in
    # jax's caches keep the arrays they closed over alive.
    import jax

    jax.clear_caches()
    gc.collect()
    check_determinism(cfg, mesh)

    # ---- report ----
    print("\n" + "=" * 72)
    failed = [c for c in CHECKS if not c.passed]
    for c in CHECKS:
        mark = "PASS" if c.passed else "FAIL"
        print(f"[{mark}] {c.name}: {c.detail}")
        if "gpu_peak_bytes_in_use" in c.observed:
            print(
                f"       gpu peak so far {c.observed['gpu_peak_bytes_in_use'] / 2**30:.2f} GiB, "
                f"in use {c.observed.get('gpu_bytes_in_use', 0) / 2**30:.2f} GiB"
            )
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
                    "mini_batch_size": mini_batch_size,
                    "compute_dtype": args.compute_dtype or "bf16 (config default)",
                    "param_dtype": args.param_dtype or "fp32 (config default)",
                    "chunked_ce": bool(args.chunked_ce),
                    "code_revision": code_revision(),
                    "suffix_len": suffix_len,
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
