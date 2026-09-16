"""Build a real vendor `MetaModel` from the vendor's own hydra config.

Two callers need this and one copy should own it:
`experiments/003-smoke-125m/run_smoke.py`, which validates the carry binding on
hardware, and `trustgate.eval.cli`, which drives the Phase 1 spike. Before this
module the logic lived only in the smoke runner, so the CLI had no way to obtain
a victim at all.

**No checkpoint is loaded.** `train.py:198` random-initialises when `load_part`
is left at its default, and that is the path taken here. A random-init model has
learned nothing, so it has nothing worth corrupting: anything measured against
one is instrument validation, never a result. Callers are responsible for saying
so in whatever they emit -- see `cli.write_report` and the banner it applies.

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


def build_model(cfg):
    """`train.py:127-132`, minus the sharding of a single device.

    Returns `(model, state, mesh)`. The model is **randomly initialised** --
    there is no checkpoint load on this path by design.
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

    with model_sharding.mesh:
        model, state = create()
    return model, state, model_sharding.mesh


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
