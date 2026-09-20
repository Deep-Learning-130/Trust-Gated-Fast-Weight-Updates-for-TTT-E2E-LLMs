"""Argument handling for both victim builders.

CPU only. The vendor tree is not importable here by design, so what this file
can cover is everything that happens *before* the first vendor import: size
validation, the `seq_length % mini_batch_size` rule the vendor asserts on, and
the legibility of the failure when the vendor is absent.

The checkpoint path is covered the same way: every guard that runs before
the first vendor import, plus the partial-restore filter, which is pure
pytree work and needs no orbax at all.

It cannot cover that the composed hydra config is correct -- only a box with the
vendor installed can, and `experiments/003-smoke-125m/run_smoke.py` is what does
that. Same division as `tests/test_vendor_bind.py`, for the same reason.
"""

import equinox as eqx
import jax.numpy as jnp
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


# ------------------------------------------------------ checkpoint sources ---


def test_gs_uri_passes_through_without_touching_the_filesystem():
    """Verifying a bucket needs network and credentials; orbax takes the URI
    verbatim (`infra/checkpoint.py:83-84`), so this layer must not pretend to
    check it."""
    uri = "gs://ttt-e2e-checkpoints/1b-books"
    assert model_build.validate_checkpoint_source(uri) == uri


@pytest.mark.parametrize("empty", ["", "   "])
def test_empty_checkpoint_is_refused_and_names_the_alternative(empty):
    with pytest.raises(ValueError) as excinfo:
        model_build.validate_checkpoint_source(empty)
    assert "--random-init" in str(excinfo.value)


def test_missing_checkpoint_directory_is_refused_by_path(tmp_path):
    absent = tmp_path / "not-there"
    with pytest.raises(ValueError) as excinfo:
        model_build.validate_checkpoint_source(str(absent))
    assert "not-there" in str(excinfo.value)


def test_a_file_is_not_a_checkpoint_directory(tmp_path):
    handle = tmp_path / "ckpt.msgpack"
    handle.write_bytes(b"")
    with pytest.raises(ValueError):
        model_build.validate_checkpoint_source(str(handle))


def test_directory_with_a_step_subdir_is_accepted(tmp_path):
    (tmp_path / "12000").mkdir()
    assert model_build.validate_checkpoint_source(str(tmp_path)) == str(
        tmp_path.resolve()
    )


def test_directory_without_a_step_subdir_is_refused_before_orbax_sees_it(tmp_path):
    """Orbax reports 'No checkpoints found' here, which reads like a missing
    checkpoint rather than the wrong path. Say which it is."""
    (tmp_path / "checkpoint_12000").mkdir()
    with pytest.raises(ValueError) as excinfo:
        model_build.validate_checkpoint_source(str(tmp_path))
    message = str(excinfo.value)
    assert "integer-named step directory" in message
    assert "checkpoint_12000" in message  # what it found, not just what it wanted


def test_layout_mismatch_can_be_overridden_deliberately(tmp_path):
    """`bootstrap_gpu_box.sh` has the same escape hatch, for the same reason:
    the check is a guess about layout, not a law."""
    (tmp_path / "checkpoint_12000").mkdir()
    assert model_build.validate_checkpoint_source(
        str(tmp_path), allow_layout_mismatch=True
    ) == str(tmp_path.resolve())


def test_checkpoint_source_is_validated_before_the_vendor_import(tmp_path):
    """A bad path on a CPU box should say 'bad path', not 'no module named
    orbax' -- and on a GPU box it should fail before the instance bills."""
    with pytest.raises(ValueError):
        model_build.build_from_checkpoint(checkpoint=str(tmp_path / "absent"))


def test_size_and_seq_length_are_still_checked_on_the_checkpoint_path(tmp_path):
    (tmp_path / "0").mkdir()
    with pytest.raises(ValueError) as excinfo:
        model_build.build_from_checkpoint(checkpoint=str(tmp_path), size="7b")
    assert "7b" in str(excinfo.value)

    with pytest.raises(ValueError) as excinfo:
        model_build.build_from_checkpoint(checkpoint=str(tmp_path), seq_length=5000)
    assert "5000" in str(excinfo.value)


def test_negative_step_is_refused(tmp_path):
    (tmp_path / "0").mkdir()
    with pytest.raises(ValueError) as excinfo:
        model_build.build_from_checkpoint(checkpoint=str(tmp_path), step=-1)
    assert "-1" in str(excinfo.value)


def test_valid_checkpoint_args_reach_the_vendor_import(tmp_path):
    """Reaching ImportError means every pre-vendor guard let it through."""
    (tmp_path / "12000").mkdir()
    with pytest.raises(ImportError) as excinfo:
        model_build.build_from_checkpoint(checkpoint=str(tmp_path))
    assert "EXPECTED to fail in the CPU test environment" in str(excinfo.value)


# --------------------------------------------- partial-restore detection ---


class _Leafy(eqx.Module):
    """Stand-in with one inexact leaf and one integer leaf.

    `MetaModel.weights()` is `eqx.filter(self, eqx.is_inexact_array)`, so the
    integer leaf is legitimately absent from every saved checkpoint and must not
    be reported as a gap.
    """

    w: jnp.ndarray
    counter: jnp.ndarray


def _model():
    return _Leafy(w=jnp.zeros((2, 2)), counter=jnp.asarray([0], dtype=jnp.int32))


def test_non_inexact_leaves_are_not_treated_as_missing_weights():
    """Every healthy restore reports these. Flagging them would make the guard
    fire on a perfectly good checkpoint."""
    missing = model_build.unexpected_missing_paths([".counter"], _model())
    assert missing == []


def test_a_missing_inexact_weight_is_reported():
    missing = model_build.unexpected_missing_paths([".w"], _model())
    assert missing == [".w"]


def test_a_clean_restore_reports_nothing():
    assert model_build.unexpected_missing_paths([], _model()) == []


def test_real_gaps_survive_alongside_expected_ones():
    """The filter must not be fooled by a list that mixes both."""
    missing = model_build.unexpected_missing_paths([".counter", ".w"], _model())
    assert missing == [".w"]
