"""Argument handling for the checkpoint-free model builder.

CPU only. The vendor tree is not importable here by design, so what this file
can cover is everything that happens *before* the first vendor import: size
validation, the `seq_length % mini_batch_size` rule the vendor asserts on, and
the legibility of the failure when the vendor is absent.

It cannot cover that the composed hydra config is correct -- only a box with the
vendor installed can, and `experiments/003-smoke-125m/run_smoke.py` is what does
that. Same division as `tests/test_vendor_bind.py`, for the same reason.
"""

import pytest

from trustgate.eval import model_build


# ------------------------------------------------------------------- sizes ---


@pytest.mark.parametrize("size", model_build.SIZES)
def test_every_declared_size_maps_to_a_vendor_config_path(size):
    assert model_build.experiment_for(size) == f"{size}/extension/ext-{size}-e2e-32K"


@pytest.mark.parametrize("size", model_build.SIZES)
def test_every_declared_size_actually_exists_in_the_vendor_tree(size):
    """SIZES is a promise about files on disk; check it against them.

    A size that composes a path hydra cannot resolve fails five frames deep,
    on a box that is billing.
    """
    config = model_build.VENDOR_CONFIGS / "experiment" / f"{size}" / "extension"
    if not model_build.VENDOR_CONFIGS.exists():
        pytest.skip("vendor submodule not checked out")
    assert (config / f"ext-{size}-e2e-32K.yaml").is_file()


def test_unknown_size_is_refused_by_name():
    with pytest.raises(ValueError) as excinfo:
        model_build.experiment_for("7b")
    message = str(excinfo.value)
    assert "7b" in message
    # The message must list what IS valid, or the reader has to go read source.
    for size in model_build.SIZES:
        assert size in message


# -------------------------------------------------------------- seq_length ---


@pytest.mark.parametrize("seq_length", [1024, 4096, 8192, 32768])
def test_valid_seq_lengths_pass_the_divisibility_check(seq_length):
    """Reaching the vendor import means the arithmetic guard let it through."""
    with pytest.raises(ImportError):
        model_build.build_config(seq_length=seq_length)


def test_seq_length_not_divisible_by_mini_batch_is_refused_before_the_vendor():
    """`transformer.py:681` asserts this. Catching it here costs nothing;
    catching it there costs a launch."""
    with pytest.raises(ValueError) as excinfo:
        model_build.build_config(seq_length=5000)
    message = str(excinfo.value)
    assert "5000" in message
    assert "1024" in message
    assert "5120" in message  # names the nearest valid value, not just the rule


def test_unknown_size_is_refused_before_the_vendor_import():
    """Order matters: a bad size on a CPU box should say 'bad size', not
    'no module named hydra'."""
    with pytest.raises(ValueError):
        model_build.build_config(size="7b")


# ------------------------------------------------------- absent vendor tree ---


def test_missing_vendor_explains_which_situation_this_is():
    """On CPU this failure is expected; on a GPU box it means a broken install.
    A bare ImportError cannot tell the reader which one they are in."""
    with pytest.raises(ImportError) as excinfo:
        model_build.build_config()
    message = str(excinfo.value)
    assert "EXPECTED to fail in the CPU test environment" in message
    assert "setup.sh" in message  # what to run if this is the GPU-box case


# -------------------------------------------------------------- jax preamble --


def test_prepare_jax_env_disables_preallocation(monkeypatch):
    """JAX otherwise takes 75% of the card at import, which is the difference
    between running and an OOM that looks like a model problem."""
    monkeypatch.delenv("XLA_PYTHON_CLIENT_PREALLOCATE", raising=False)
    model_build.prepare_jax_env()
    import os

    assert os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] == "false"


def test_prepare_jax_env_does_not_override_a_deliberate_setting(monkeypatch):
    monkeypatch.setenv("XLA_PYTHON_CLIENT_PREALLOCATE", "true")
    model_build.prepare_jax_env()
    import os

    assert os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] == "true"


# ------------------------------------------------------------ dummy tokens ---


def test_dummy_tokens_match_the_vendor_dummy_dataset_range():
    """`DummyDataset.__getitem__` (`lm_dataset.py:36`) emits ids in [0, 20).

    The bar that depends on this is subtle: targets spanning 20 ids while a
    random-init model spreads mass over 128256 is why CE starts at ln(128256)
    and not ln(20).
    """
    tokens = model_build.dummy_tokens(4096, seed=0)
    assert tokens.min() >= 0
    assert tokens.max() < 20
    assert tokens.dtype.name == "int32"


def test_dummy_tokens_are_deterministic_per_seed():
    a = model_build.dummy_tokens(256, seed=3)
    b = model_build.dummy_tokens(256, seed=3)
    c = model_build.dummy_tokens(256, seed=4)
    assert (a == b).all()
    assert not (a == c).all()


def test_dummy_tokens_never_emit_bos():
    """A BOS inside a span would be masked out of the loss (`lm_dataset.py:52`)
    and silently shorten the stream the condition claims to have run."""
    from trustgate.tokens import BOS_TOKEN_ID

    wide = model_build.dummy_tokens(8192, seed=1, vocab_lo=10, vocab_hi=50_000)
    assert BOS_TOKEN_ID not in set(wide.tolist())
