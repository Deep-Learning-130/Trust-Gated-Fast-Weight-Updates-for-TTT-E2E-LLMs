"""Build a real vendor `MetaModel` from the vendor's own hydra config.

Two callers need this and one copy should own it:
`experiments/003-smoke-125m/run_smoke.py`, which validates the carry binding on
hardware, and `trustgate.eval.cli`, which drives the Phase 1 spike. Before this
module the logic lived only in the smoke runner, so the CLI had no way to obtain
a victim at all.

Two victim paths live here and they are deliberately interchangeable.
`build_random_init` takes `train.py:198`, which random-initialises when
`load_part` is left at its default; `build_from_checkpoint` takes the
`load_part=params` branch above it. Both return `(cfg, model, state, mesh)`, so
`vendor_bind.bind` and every caller downstream is indifferent to which ran.

A random-init model has learned nothing, so it has nothing worth corrupting:
anything measured against one is instrument validation, never a result. Callers
say so in whatever they emit -- see `cli.write_report` and the banner it
applies. `build_from_checkpoint` carries the opposite obligation: it must refuse
a *partial* restore rather than warn about one, or a half-random model reaches
the same metrics wearing no banner at all.

The config is *composed*, not hand-built, so both callers exercise the same
hydra path `train.py` does. Every override below has a reason, recorded beside
it; `experiments/003-smoke-125m/README.md` has the long form.

Every vendor import is deliberately inside a function. This module has to be
importable on a machine with no vendor tree -- that is what lets the CPU test
suite exercise the argument handling without `hydra`, `zarr`, `grain` or
`orbax` installed.
"""

from __future__ import annotations

import os
from pathlib import Path

#: The vendor's config tree, resolved the way `prereg.PREREGISTERED_PATH` is.
VENDOR_CONFIGS = Path(__file__).resolve().parents[3] / "vendor" / "ttt-e2e" / "configs"

#: Model sizes with an `extension` config in the vendor tree. Checked against
#: `configs/experiment/*/extension/` -- a typo here is otherwise a hydra stack
#: trace five frames deep.
SIZES = ("125m", "350m", "760m", "1b", "3b")

#: `transformer.py:681` asserts `seq_length % mini_batch_size == 0`. At the
#: project's standard 8192 this gives exactly 8 inner steps, which is the
#: granularity `PREREGISTERED.md` is scoped to (ADR-004, ADR-007).
MINI_BATCH_SIZE = 1024
DEFAULT_SEQ_LENGTH = 8192


def _require_vendor() -> None:
    """Fail legibly rather than three frames deep in hydra.

    A bare `ModuleNotFoundError: hydra` is unhelpful: on a CPU box it is
    *expected*, and on a GPU box it means the install is wrong. Distinguish the
    two, the way `vendor_bind._require_vendor` does for the model surface.
    """
    try:
        import hydra  # noqa: F401
        import ttt.config  # noqa: F401
        import ttt.model.transformer  # noqa: F401
    except ImportError as exc:
        raise ImportError(
            "trustgate.eval.model_build requires the vendor tree to be "
            "importable (`uv pip install -e vendor/ttt-e2e`, Python >= 3.12, "
            "jax[cuda12]<0.6). This is EXPECTED to fail in the CPU test "
            "environment, which deliberately installs none of einops/zarr/"
            "grain/orbax/wandb/hydra. On a GPU box it means the install did "
            "not finish -- run `bash experiments/003-smoke-125m/setup.sh`. "
            f"Original error: {exc}"
        ) from exc


def experiment_for(size: str) -> str:
    """Hydra experiment path for a model size."""
    if size not in SIZES:
        raise ValueError(f"unknown size {size!r}; expected one of {', '.join(SIZES)}")
    return f"{size}/extension/ext-{size}-e2e-32K"


def prepare_jax_env() -> None:
    """Stop JAX grabbing the whole card at import.

    JAX preallocates 75% of VRAM on first use. At 1024 x 128256 x 4 B = 525 MB
    per chunk of logits -- and `value_and_grad` holds more than one live -- that
    preallocation is the difference between running and an OOM that looks like a
    model problem. Set before the first `import jax` anywhere in the process.
    """
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")


def build_config(
    *,
    size: str = "125m",
    seq_length: int = DEFAULT_SEQ_LENGTH,
    compute_dtype: str | None = None,
    exp_dir: str = "/tmp/trustgate-model",
    num_devices: int = 1,
):
    """Compose the vendor hydra config for a checkpoint-free, dataset-free run.

    Args:
        size: one of `SIZES`. Selects `+experiment=<size>/extension/...`.
        seq_length: must be a multiple of 1024. 8192 gives 8 inner steps.
        compute_dtype: override `model.compute_dtype`. Pass `"fp32"` on Turing
            (T4) -- it has no native bf16 and the vendor default is bf16.
        exp_dir: vendor run output. Must be outside the repo; the vendor's own
            default is `./experiments`, which from the repo root writes straight
            into our tracked tree.
        num_devices: single-device is the only configuration this project runs.
    """
    if seq_length % MINI_BATCH_SIZE != 0:
        raise ValueError(
            f"seq_length {seq_length} is not a multiple of mini_batch_size "
            f"{MINI_BATCH_SIZE}; transformer.py:681 asserts divisibility. "
            f"Nearest valid: {round(seq_length / MINI_BATCH_SIZE) * MINI_BATCH_SIZE}."
        )
    experiment = experiment_for(size)
    _require_vendor()

    from hydra import compose, initialize_config_dir
    from ttt.config import register_configs

    register_configs()

    overrides = [
        # `+` is required: neither group is in config.yaml's defaults list.
        # EVAL_ENTRYPOINT.md §1 documents this; the vendor README uses the same
        # form.
        f"+experiment={experiment}",
        "+deploy=interactive",
        # `dataset_path` interpolates `deploy_paths.data[...]`, which ships as
        # `???` in interactive.yaml:9. Nothing reads it under dummy_dataset, but
        # OmegaConf resolves eagerly, so it needs *a* value. No `+` -- the key
        # exists, it is merely MISSING.
        "deploy_paths.data.books3=/dev/null",
        "training.dummy_dataset=true",
        "training.eval_mode=true",
        "training.log_wandb=false",
        f"training.seq_length={seq_length}",
        # These do NOT give an eval batch of 1. train.py:211 takes
        # max(eval_batch_size, global_batch_size // accum_steps * 4), so the
        # floor is 4 (EVAL_ENTRYPOINT.md §5 gotcha 1). It does not matter to
        # callers that drive `loss_for_sequence` directly rather than
        # `evaluator.eval_fn`, which is both of ours.
        "training.global_batch_size=1",
        "training.eval_batch_size=1",
        "training.n_data_parallel=1",
        "training.n_state_parallel=1",
        f"training.exp_dir={exp_dir}",
        "backend.distributed=false",
        f"backend.num_devices={num_devices}",
        'backend.local_device_ids="0"',
    ]
    if compute_dtype:
        overrides.append(f"model.compute_dtype={compute_dtype}")

    with initialize_config_dir(config_dir=str(VENDOR_CONFIGS), version_base=None):
        cfg = compose(config_name="config", overrides=overrides)

    # train.py:123 -- the model reads its length from here, not from training.
    cfg.model.seq_len = cfg.training.seq_length
    return cfg


def _sharded_creator(cfg):
    """`train.py:127-132` as a reusable pair: the jitted creator and its mesh.

    Both victim paths need the *same* creator. The checkpoint path additionally
    hands it to `eval_shape_and_sharding`, which is how orbax learns where to
    place each restored array (`train.py:161`) -- so a second lookalike creator
    could let the abstract shardings and the real ones drift apart silently.
    """
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

    return create, model_sharding.mesh


def build_model(cfg):
    """`train.py:127-132`, minus the sharding of a single device.

    Returns `(model, state, mesh)`. The model is **randomly initialised** --
    there is no checkpoint load on this path by design. See
    `build_from_checkpoint` for the path that does load weights.
    """
    create, mesh = _sharded_creator(cfg)
    with mesh:
        model, state = create()
    return model, state, mesh


def build_random_init(
    *,
    size: str = "125m",
    seq_length: int = DEFAULT_SEQ_LENGTH,
    compute_dtype: str | None = None,
    exp_dir: str = "/tmp/trustgate-model",
):
    """`build_config` then `build_model`, the pairing every caller wants.

    Returns `(cfg, model, state, mesh)`.
    """
    prepare_jax_env()
    cfg = build_config(
        size=size,
        seq_length=seq_length,
        compute_dtype=compute_dtype,
        exp_dir=exp_dir,
    )
    model, state, mesh = build_model(cfg)
    return cfg, model, state, mesh


def validate_checkpoint_source(
    checkpoint: str, *, allow_layout_mismatch: bool = False
) -> str:
    """Check a checkpoint location before importing anything from the vendor.

    Runs before `_require_vendor` for the reason `experiment_for` does: on a CPU
    box a bad path should say "bad path", not "no module named orbax", and on a
    GPU box it should fail in the second before the instance bills rather than
    five frames into orbax.

    `gs://` URIs pass through unchecked -- verifying one needs network and
    credentials, and `infra/checkpoint.py:83-84` hands them to orbax verbatim.
    Local paths are checked for the layout orbax actually requires:
    `CheckpointManager` wants the directory that *contains* integer-named step
    directories. Pointed at a step directory itself it reports "No checkpoints
    found", which reads like a missing checkpoint rather than a wrong path --
    `scripts/bootstrap_gpu_box.sh:329-344` makes the same check on the box.
    """
    if not checkpoint or not checkpoint.strip():
        raise ValueError(
            "checkpoint path is empty; pass the directory containing the step "
            "directories, or use --random-init for a weightless victim"
        )

    if checkpoint.startswith("gs://"):
        return checkpoint

    path = Path(checkpoint).expanduser().resolve()
    if not path.is_dir():
        raise ValueError(
            f"checkpoint directory {path} does not exist or is not a directory. "
            f"Pass the directory that CONTAINS the step directories, not a file "
            f"and not the step directory itself."
        )

    if allow_layout_mismatch:
        return str(path)

    steps = [c.name for c in path.iterdir() if c.is_dir() and c.name.isdigit()]
    if not steps:
        found = sorted(c.name for c in path.iterdir())[:6] or ["nothing"]
        raise ValueError(
            f"checkpoint directory {path} contains no integer-named step "
            f"directory, so orbax's CheckpointManager will report 'No "
            f"checkpoints found' -- which reads like a missing checkpoint "
            f"rather than a wrong path. Found instead: {found}. Point at the "
            f"parent of the step directory, or pass allow_layout_mismatch=True "
            f"if the layout is deliberate."
        )
    return str(path)


def unexpected_missing_paths(not_found_paths, model) -> list[str]:
    """Filter `unify_dict_with_eqx_module`'s report down to the real failures.

    `MetaModel.weights()` is `eqx.filter(self, eqx.is_inexact_array)`
    (`transformer.py:747-751`), so every non-inexact leaf -- integer counters,
    boolean flags -- is *legitimately* absent from the saved dict and comes back
    in the not-found list on every healthy restore. Anything outside that set is
    a weight the checkpoint should have carried and did not.

    This matters more than it looks. `unify_dict_with_eqx_module` only
    `warnings.warn`s (`infra/checkpoint.py:221-224`) and `train.py:171` discards
    the list, so a partial restore produces a model that is part checkpoint and
    part random init. That is indistinguishable from a real result in every
    metric this project computes, which is the failure the whole `NO CHECKPOINT`
    banner discipline exists to prevent.
    """
    import equinox as eqx
    import jax

    expected_absent = {
        jax.tree_util.keystr(path)
        for path, leaf in jax.tree_util.tree_flatten_with_path(model)[0]
        if not eqx.is_inexact_array(leaf)
    }
    return [path for path in not_found_paths if path not in expected_absent]


def build_from_checkpoint(
    *,
    checkpoint: str,
    size: str = "125m",
    seq_length: int = DEFAULT_SEQ_LENGTH,
    compute_dtype: str | None = None,
    exp_dir: str = "/tmp/trustgate-model",
    step: int | None = None,
    allow_layout_mismatch: bool = False,
):
    """`build_random_init`'s counterpart: same 4-tuple, real weights.

    Returns `(cfg, model, state, mesh)` -- deliberately the shape
    `build_random_init` returns, because `vendor_bind.bind` reads its config off
    `model.config` and consumes only `(MetaModel, eqx.nn.State)`. Nothing
    downstream needs to know which path built the victim.

    Mirrors the `load_part=params` branch of `train.py:147-194`, minus the wandb
    resume path (which forces `load_part=all` off a preexisting run) and minus
    the optimizer and dataset-iterator state, which eval does not need and the
    released checkpoints do not ship. `state` comes fresh from the creator, as
    it does at `train.py:175` -- only the weights are restored.
    """
    checkpoint = validate_checkpoint_source(
        checkpoint, allow_layout_mismatch=allow_layout_mismatch
    )
    if step is not None and step < 0:
        raise ValueError(f"step must be non-negative, got {step}")

    prepare_jax_env()
    cfg = build_config(
        size=size,
        seq_length=seq_length,
        compute_dtype=compute_dtype,
        exp_dir=exp_dir,
    )

    from ttt.config import TrainingConfig
    from ttt.infra.checkpoint import Checkpointer, unify_dict_with_eqx_module
    from ttt.utils.jax_utils import eval_shape_and_sharding

    # train.py:152-156. Left alone, `resume_checkpoint_dir` interpolates to
    # `./checkpoints/demo/` -- inside our own tracked tree.
    cfg.training.load_part = TrainingConfig.LoadPart.params
    cfg.checkpoint.resume_checkpoint_dir = checkpoint
    cfg.training.resume_step = step
    # Nothing here constructs a `for_saving=True` Checkpointer, but the same
    # interpolation points the save directory into the repo. Move it.
    cfg.checkpoint.checkpoint_dir = f"{exp_dir}/unused-save-dir"

    create, mesh = _sharded_creator(cfg)

    with mesh:
        # train.py:161. The shardings attached here are what orbax uses to place
        # each restored array, which is why this runs under the mesh and before
        # the restore rather than after it.
        abstract_weights = eval_shape_and_sharding(lambda: create()[0].weights())

        restored = Checkpointer(config=cfg, for_saving=False).load_checkpoint(
            step=step,
            targets={"model_weights": abstract_weights},
            restore=TrainingConfig.LoadPart.params,
        )

        model, state = create()
        loaded, not_found = unify_dict_with_eqx_module(restored["model_weights"], model)

    unexpected = unexpected_missing_paths(not_found, model)
    if unexpected:
        shown = unexpected[:5]
        more = f" (and {len(unexpected) - 5} more)" if len(unexpected) > 5 else ""
        raise ValueError(
            f"checkpoint restore left {len(unexpected)} model weight(s) at "
            f"their random-init values: {shown}{more}. The vendor only warns "
            f"about this and train.py discards the warning, but a part-"
            f"checkpoint part-random model scores like a real result in every "
            f"metric this project computes. Check that size={size!r} matches "
            f"the checkpoint's architecture and that {checkpoint} is the "
            f"intended run."
        )

    return cfg, loaded, state, mesh


def dummy_tokens(n: int, seed: int, vocab_lo: int = 0, vocab_hi: int = 20):
    """Match `DummyDataset.__getitem__` (`lm_dataset.py:36`): ids in [0, 20).

    Note what this means for a loss bar. The *targets* span only 20 ids, but a
    random-init model spreads its mass over all 128256, so CE starts at
    ln(128256). It would fall toward ln(20) only after training -- which is the
    tell that the model has learned the dummy distribution and not language.
    """
    import numpy as np

    rng = np.random.default_rng(seed)
    return np.asarray(rng.integers(vocab_lo, vocab_hi, n, dtype=np.int32))
