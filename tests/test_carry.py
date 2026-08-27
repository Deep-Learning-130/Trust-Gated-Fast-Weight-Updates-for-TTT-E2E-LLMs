"""Fast-weight carry across chunks and sequences.

Tested against a lightweight stand-in for the vendor inner step, so no
checkpoint and no GPU are needed -- the same approach, and for the same reason,
as `tests/test_interceptor.py`: the vendor is not importable in this
environment (no `einops`, `zarr`, `grain`, `orbax`, `wandb`, `hydra`).

`test_step_receives_previous_output_not_initial_weights` is the one that matters.
The vendor discards its scan carry (`transformer.py:712`, ADR-006), so a harness
built the obvious way re-adapts from the *initial* weights every step and
measures nothing. That failure is silent and produces a false STOP, which ends
the project. This asserts the carry actually threads.
"""

import jax
import jax.numpy as jnp
import pytest

from trustgate.eval.carry import (
    SATURATED_INNER_LR_MULTIPLIER,
    CarryState,
    InnerStepResult,
    assert_saturated_inner_lr,
    chunk_tokens,
    init_carry,
    run_chunks,
    run_sequences,
)


def _weights(value=0.0):
    return {"w": jnp.full((2, 2), value, dtype=jnp.float32)}


def make_recording_step(delta=1.0):
    """A stand-in inner step that adds `delta` and records what it was given.

    Recording the *incoming* fast weights is the whole point: it is the only way
    to distinguish a threaded carry from a reset one, because both produce a
    plausible-looking final number.
    """
    seen = []

    def step(fast_weights, opt_state, state_tuple, chunk):
        seen.append(fast_weights)
        return InnerStepResult(
            fast_weights={k: v + delta for k, v in fast_weights.items()},
            opt_state=(opt_state or 0) + 1,
            state_tuple=state_tuple,
            metrics={"loss": jnp.float32(1.0)},
        )

    return step, seen


def test_step_receives_previous_output_not_initial_weights():
    step, seen = make_recording_step(delta=1.0)
    carry, _ = run_chunks(init_carry(_weights(0.0)), [None, None, None], step)

    # Each step must see its predecessor's output: 0.0, then 1.0, then 2.0.
    assert [float(s["w"][0, 0]) for s in seen] == [0.0, 1.0, 2.0]
    # And the carry must hold the accumulated result, not the initial weights.
    assert float(carry.fast_weights["w"][0, 0]) == 3.0


def test_carry_survives_a_sequence_boundary():
    step, seen = make_recording_step(delta=1.0)
    # Two sequences of 4 tokens each, one chunk per 2 tokens => 2 chunks per seq.
    sequences = [jnp.arange(4), jnp.arange(4)]
    carry, _ = run_sequences(init_carry(_weights(0.0)), sequences, step, mini_batch_size=2)

    # Four steps total, and the first step of sequence 2 sees the state left by
    # sequence 1 -- this is the capability the vendor does not have at all.
    assert [float(s["w"][0, 0]) for s in seen] == [0.0, 1.0, 2.0, 3.0]
    assert float(carry.fast_weights["w"][0, 0]) == 4.0


def test_n_steps_counts_every_chunk_across_sequences():
    step, _ = make_recording_step()
    sequences = [jnp.arange(8), jnp.arange(8), jnp.arange(8)]
    carry, _ = run_sequences(init_carry(_weights()), sequences, step, mini_batch_size=4)
    assert int(carry.n_steps) == 6  # 3 sequences x 2 chunks


def test_opt_state_threads_through():
    step, _ = make_recording_step()
    carry, _ = run_chunks(init_carry(_weights(), opt_state=0), [None] * 3, step)
    assert carry.opt_state == 3


def test_metrics_are_collected_one_per_step():
    step, _ = make_recording_step()
    _, metrics = run_chunks(init_carry(_weights()), [None] * 5, step)
    assert len(metrics) == 5
    assert all("loss" in m for m in metrics)


def test_no_chunks_leaves_the_carry_untouched():
    step, seen = make_recording_step()
    carry, metrics = run_chunks(init_carry(_weights(7.0)), [], step)
    assert seen == []
    assert metrics == []
    assert int(carry.n_steps) == 0
    assert float(carry.fast_weights["w"][0, 0]) == 7.0


def test_carry_state_is_a_registered_pytree():
    # A plain dataclass would be an opaque leaf and every tree_ops helper would
    # silently no-op on it -- the bug an earlier version of the interceptor
    # tests hid.
    carry = init_carry(_weights(3.0))
    shapes = [leaf.shape for leaf in jax.tree.leaves(carry) if hasattr(leaf, "shape")]
    assert (2, 2) in shapes


def test_saturated_inner_lr_passes():
    assert_saturated_inner_lr(init_carry(_weights()))


def test_unsaturated_inner_lr_raises():
    # ilr_init, not the saturated multiplier: a near-frozen inner loop, which
    # produces a null result for a reason unrelated to the attack.
    carry = init_carry(_weights(), inner_lr_multiplier=0.01)
    with pytest.raises(ValueError, match="inner LR multiplier"):
        assert_saturated_inner_lr(carry)


def test_saturated_constant_matches_vendor_ramp_target():
    # transformer.py:564-573 ramps toward 1.0; train.py:224 pins step_index high
    # so eval mode reaches it.
    assert SATURATED_INNER_LR_MULTIPLIER == 1.0


def test_chunk_tokens_splits_on_mini_batch_size():
    chunks = chunk_tokens(jnp.arange(12), mini_batch_size=4)
    assert len(chunks) == 3
    assert all(len(c) == 4 for c in chunks)
    assert int(chunks[1][0]) == 4


def test_chunk_tokens_refuses_a_ragged_tail():
    # Padding would be BOS-masked out of the loss (lm_dataset.py:53) and would
    # break valid-token matching while looking fine.
    with pytest.raises(ValueError, match="not a multiple"):
        chunk_tokens(jnp.arange(10), mini_batch_size=4)


def test_chunk_tokens_rejects_nonpositive_mini_batch_size():
    with pytest.raises(ValueError, match="must be positive"):
        chunk_tokens(jnp.arange(10), mini_batch_size=0)


def test_init_carry_starts_at_zero_steps():
    carry = init_carry(_weights())
    assert int(carry.n_steps) == 0
    assert isinstance(carry, CarryState)
