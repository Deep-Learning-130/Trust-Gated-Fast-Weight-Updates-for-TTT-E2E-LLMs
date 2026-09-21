"""`vendor_bind` against a stand-in for the vendor surface.

The vendor tree is not importable in this environment by design -- no `einops`,
`zarr`, `grain`, `orbax`, `wandb` or `hydra` -- which is what keeps CI
network-free and dependency-free. `tests/test_interceptor.py` and
`tests/test_carry.py` already test against stand-ins for the same reason.

**What this file can and cannot establish.** It covers the seam logic: that the
step function narrows a returned `MetaModel` back to the inner subtree, that
outer parameters come from the frozen reference rather than the adapting tree,
that the carry survives chunks and sequences, and that the stream/chunk
arithmetic refuses ragged input rather than padding it.

It cannot establish that the transcription of `transformer.py:660-714` is
faithful -- only a real `MetaModel` can, and that is exactly what
`experiments/003-smoke-125m/run_smoke.py` exists to check on hardware. Do not
read a green run here as evidence the binding is correct against the vendor.
"""

import sys
import types

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from trustgate.eval import carry as carry_mod


# ---------------------------------------------------------------------------
# A stand-in for the six vendor names `_require_vendor` imports.
# ---------------------------------------------------------------------------


class FakeBatch(eqx.Module):
    input_ids: jnp.ndarray
    target_tokens: jnp.ndarray
    loss_masks: jnp.ndarray


def fake_tree_slice(tree, i):
    return jax.tree.map(lambda x: x[i], tree)


class FakeInnerLoopStepResult(eqx.Module):
    new_model: object
    new_optimizer_state: object
    new_state: object
    metrics: dict


class _LossKey:
    """Stands in for `MetaModel.MetricType.loss` -- matched on `.name`."""

    name = "loss"


LOSS = _LossKey()


class FakeModel(eqx.Module):
    """Two leaves: one 'inner' (adapts), one 'outer' (must never move)."""

    inner: jnp.ndarray
    outer: jnp.ndarray


class FakeMetaModel:
    @staticmethod
    def inner_loop_step(model, opt_state, state_tuple, seq, prefix_outputs):
        # Move the inner leaf by a fixed amount per step, and deliberately try
        # to move the outer leaf too. A correct `combine` overwrites the latter
        # from the frozen reference on the next step, so it can never drift.
        new = FakeModel(inner=model.inner + 1.0, outer=model.outer + 100.0)
        return FakeInnerLoopStepResult(
            new_model=new,
            new_optimizer_state=opt_state + 1,
            new_state=state_tuple,
            metrics={LOSS: jnp.asarray(float(model.inner.sum()))},
        )


def _install_fake_vendor(monkeypatch):
    """Inject the six names `_require_vendor` imports into `sys.modules`."""
    mods = {
        "ttt": types.ModuleType("ttt"),
        "ttt.model": types.ModuleType("ttt.model"),
        "ttt.model.data": types.ModuleType("ttt.model.data"),
        "ttt.model.transformer": types.ModuleType("ttt.model.transformer"),
        "ttt.utils": types.ModuleType("ttt.utils"),
        "ttt.utils.filter_utils": types.ModuleType("ttt.utils.filter_utils"),
        "ttt.utils.jax_utils": types.ModuleType("ttt.utils.jax_utils"),
    }
    mods["ttt.model.data"].Batch = FakeBatch
    mods["ttt.model.data"].tree_slice = fake_tree_slice
    mods["ttt.model.transformer"].MetaModel = FakeMetaModel
    mods["ttt.model.transformer"].BlockCollectionSplit = object
    mods["ttt.utils.filter_utils"].get_filter_spec = lambda *a, **k: None
    mods["ttt.utils.jax_utils"].clone_pytree = lambda t: t

    for name, mod in mods.items():
        monkeypatch.setitem(sys.modules, name, mod)
    # These fakes read concrete values (`float(...)`), so they run eagerly.
    # The compiled path has its own trace-safe fake in tests/test_gate_eval.py.
    monkeypatch.setenv("TRUSTGATE_NO_JIT", "1")


@pytest.fixture
def vendor_bind(monkeypatch):
    _install_fake_vendor(monkeypatch)
    from trustgate.eval import vendor_bind as vb

    return vb


def make_binding(vendor_bind, mini_batch_size=4, seq_length=8):
    """A `VendorBinding` built by hand, bypassing `bind()`.

    `bind()` itself needs a real `MetaModel` (it calls `substate`, `split_state`
    and `tree_at` on a specific tree shape); the *step* logic does not, and that
    is what is worth testing here.
    """
    model = FakeModel(inner=jnp.zeros(3), outer=jnp.ones(3))
    # `spec_inner` selects the `inner` leaf only.
    spec = FakeModel(inner=True, outer=False)
    return vendor_bind.VendorBinding(
        model_split=model,
        model_outer=model,
        spec_inner=spec,
        state_all=None,
        state_prefix=None,
        state_suffix=None,
        mini_batch_size=mini_batch_size,
        seq_length=seq_length,
        suffix_len=1,
        inner_lr_multiplier=1.0,
        state_dtype="fp32",
    )


# ---------------------------------------------------------------------------
# The seam: narrowing, and the frozen outer reference.
# ---------------------------------------------------------------------------


def test_step_fn_narrows_the_returned_model_to_the_inner_subtree(vendor_bind):
    """`inner_loop_step` hands back a whole MetaModel via `filter_apply_updates`.

    The carry must hold fast weights, not a model -- otherwise every downstream
    norm, drift and gate computation is silently over the slow weights too.
    """
    binding = make_binding(vendor_bind)
    step_fn = vendor_bind.make_step_fn(binding)

    result = step_fn(binding.fast_weights(), 0, None, (None, None))

    assert result.fast_weights.outer is None, (
        "the outer leaf survived the narrowing; the carry is holding a model"
    )
    np.testing.assert_allclose(result.fast_weights.inner, jnp.ones(3))


def test_outer_params_come_from_the_frozen_reference_not_the_adapting_tree(
    vendor_bind,
):
    """`transformer.py:697-698` takes outer params off the dtype-cast `model`.

    The fake's `inner_loop_step` tries to move the outer leaf by +100 every
    step. If `combine` sourced outer params from the adapting tree, that drift
    would compound -- and a "fast-weight" attack would actually be editing slow
    weights, which is a different experiment.
    """
    binding = make_binding(vendor_bind)
    step_fn = vendor_bind.make_step_fn(binding)

    carry = carry_mod.init_carry(
        fast_weights=binding.fast_weights(), opt_state=0, state_tuple=None
    )
    carry, _ = carry_mod.run_chunks(carry, [(None, None)] * 5, step_fn)

    recombined = binding.combine(carry.fast_weights)
    np.testing.assert_allclose(
        recombined.outer, jnp.ones(3), err_msg="outer parameters drifted"
    )
    np.testing.assert_allclose(recombined.inner, jnp.full(3, 5.0))


def test_the_carry_actually_accumulates_across_chunks(vendor_bind):
    """The one thing the vendor cannot do (ADR-006), on the fake surface."""
    binding = make_binding(vendor_bind)
    step_fn = vendor_bind.make_step_fn(binding)

    carry = carry_mod.init_carry(
        fast_weights=binding.fast_weights(), opt_state=0, state_tuple=None
    )
    before = carry.fast_weights.inner
    carry, metrics = carry_mod.run_chunks(carry, [(None, None)] * 3, step_fn)

    assert int(carry.n_steps) == 3
    assert float(jnp.linalg.norm(carry.fast_weights.inner - before)) > 0.0
    # Losses are read before each update, so they must be strictly increasing
    # here -- evidence the carry fed forward rather than restarting.
    losses = [float(m[LOSS]) for m in metrics]
    assert losses == sorted(losses) and losses[0] < losses[-1]


# ---------------------------------------------------------------------------
# Stream and chunk arithmetic. Pure -- no vendor needed beyond `tree_slice`.
# ---------------------------------------------------------------------------


def test_split_stream_gives_each_sequence_one_extra_token_for_the_shift(
    vendor_bind,
):
    """`_to_batch` drops one token to the input/target shift (lm_dataset.py:51)."""
    seqs = vendor_bind.split_stream(jnp.arange(24), seq_length=8)

    assert len(seqs) == 3
    assert all(s.shape[0] == 9 for s in seqs)
    # Sequences overlap by exactly the shift token, so no position is lost.
    assert int(seqs[0][-1]) == int(seqs[1][0])


def test_split_stream_repeats_only_the_final_target(vendor_bind):
    """The last sequence has no following position to borrow a target from."""
    seqs = vendor_bind.split_stream(jnp.arange(16), seq_length=8)
    assert int(seqs[-1][-1]) == int(seqs[-1][-2]) == 15


def test_split_stream_refuses_a_ragged_stream(vendor_bind):
    with pytest.raises(ValueError, match="not a multiple of seq_length"):
        vendor_bind.split_stream(jnp.arange(17), seq_length=8)


def test_chunk_sequence_refuses_a_ragged_sequence(vendor_bind):
    """`transformer.py:681` asserts this; we reproduce it with the real reason.

    Padding would be masked straight out of the loss and would break
    valid-token matching between arms while looking perfectly fine.
    """
    binding = make_binding(vendor_bind, mini_batch_size=4)
    seq = FakeBatch(
        input_ids=jnp.arange(10),
        target_tokens=jnp.arange(10),
        loss_masks=jnp.ones(10, dtype=bool),
    )
    with pytest.raises(ValueError, match="divisible by mini_batch_size"):
        vendor_bind.chunk_sequence(binding, seq, jnp.zeros((10, 2)))


def test_chunk_sequence_keeps_tokens_and_prefix_output_aligned(vendor_bind):
    """A misalignment here silently trains each chunk on another's context."""
    binding = make_binding(vendor_bind, mini_batch_size=4)
    seq = FakeBatch(
        input_ids=jnp.arange(8),
        target_tokens=jnp.arange(8),
        loss_masks=jnp.ones(8, dtype=bool),
    )
    prefix = jnp.arange(8).reshape(8, 1).astype(jnp.float32)

    chunks = vendor_bind.chunk_sequence(binding, seq, prefix)

    assert len(chunks) == 2
    for suffix_chunk, prefix_chunk in chunks:
        np.testing.assert_allclose(
            np.asarray(suffix_chunk.input_ids, dtype=np.float32),
            np.asarray(prefix_chunk).ravel(),
        )


def test_make_batch_masks_bos_out_of_the_loss(vendor_bind):
    """`loss_masks = targets != bos` (lm_dataset.py:53).

    `RunCondition.valid_tokens` counts this mask, not the raw length, because
    the inner objective's denominator is the chunk's valid count.
    """
    tokens = jnp.asarray([5, 128000, 7, 9], dtype=jnp.int32)
    batch = vendor_bind.make_batch(tokens, bos_token_id=128000)

    assert batch.input_ids.shape[0] == 3
    np.testing.assert_array_equal(np.asarray(batch.loss_masks), [False, True, True])


def test_make_batch_refuses_a_stream_too_short_to_form_a_pair(vendor_bind):
    with pytest.raises(ValueError, match="at least 2 tokens"):
        vendor_bind.make_batch(jnp.asarray([1]), bos_token_id=128000)


def test_split_stream_with_lookahead_uses_the_real_final_target(vendor_bind):
    """A CraftedStream is `length_tokens + 1`; its last token is real text."""
    seqs = vendor_bind.split_stream(jnp.arange(17), seq_length=8, lookahead=True)
    assert [s.shape[0] for s in seqs] == [9, 9]
    assert int(seqs[-1][-1]) == 16, "the lookahead token, not a repeat"
    assert int(seqs[0][-1]) == int(seqs[1][0])


def test_split_stream_with_lookahead_refuses_the_wrong_length(vendor_bind):
    with pytest.raises(ValueError, match="k \+ 1"):
        vendor_bind.split_stream(jnp.arange(16), seq_length=8, lookahead=True)
