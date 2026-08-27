"""Carry adapted fast weights across chunks *and* across sequences.

Why this module has to exist
---------------------------
The vendor throws the adapted fast weights away. In
`ttt/model/transformer.py:712` the meta branch binds the scan carry:

    carry, metrics = scan_remat_chunk(
        eqx.filter_checkpoint(process_suffix_chunk, prevent_cse=False),
        (model, inner_opt_state, (state_all, state_suffix)),
        ...
    )

`carry` holds `new_model` -- the adapted prime MLP -- and is then **never used
again**; line 745 returns `loss, metrics` only. Two further resets compound it:
line 685 rebuilds `model` from outer parameters with a dtype cast on *every*
call, line 686 re-initialises `inner_opt_state`, and `loop.py:160` vmaps the
whole thing over the batch so each element gets an independent trajectory.

So `run_stream(...) -> adapted fast weights` cannot be implemented by calling
the vendor. The obvious harness -- one call to adapt on the poison stream, a
second call to evaluate benign loss -- would silently reset the fast weights
between the two and measure eval noise. That is a *structurally guaranteed
null*: a STOP verdict indistinguishable from a real one, bought with a paid GPU
run. See `docs/adr/ADR-006-fast-weight-carry.md`.

This is not a small gap to route around. At `seq_length=8192` with
`mini_batch_size=1024` (ADR-004; `ext-1b-e2e-32K.yaml:23`) one sequence is only
**8 inner SGD steps**, while `AttackSpec.stream_tokens` ("large relative to a
single window") and the pre-registration's secondary metric ("number of stream
tokens needed to reach threshold") both presuppose accumulation across many
sequences. Cross-sequence carry is a precondition of the experiment existing,
not an optimisation.

Why a Python loop and not `lax.scan`
------------------------------------
The vendor scans for memory: `eqx.filter_checkpoint` per chunk keeps activations
bounded so an *outer* gradient can flow back through the inner loop during
meta-training. The attack spike takes no outer gradient -- it adapts forward and
measures -- so there is nothing to rematerialise for, and an explicit loop is
both sufficient and far easier to assert over. `craft_stream`'s SOFT strategy
would need gradients through the stream; that is T1.9 and out of scope here.

What is ours and what is injected
---------------------------------
Everything in this module -- the chunk loop, the carry threading, the optimiser
state threading, the step accounting -- is trustgate code and CPU-testable
today. `step_fn` is the single injected seam standing in for
`MetaModel.inner_loop_step`. The vendor is not importable in the CPU test
environment (no `einops`, `zarr`, `grain`, `orbax`, `wandb`, `hydra`), which is
the same reason `tests/test_interceptor.py` tests against stand-ins rather than
the real `MetaModel`.
"""

from __future__ import annotations

from typing import Any, Iterable, Protocol, Sequence

import equinox as eqx
import jax.numpy as jnp

from trustgate.types import FastWeights

#: `get_ilr_multiplier` (vendor `transformer.py:564-573`) ramps the inner
#: learning rate from `ilr_init` to 1.0 as a function of `state[step_index]`.
#: `train.py:224` pins `step_index = INT32_MAX - 100` in eval mode precisely so
#: the ramp is fully saturated. A hand-rolled harness that leaves `step_index`
#: at 0 gets `ilr_init` instead -- a near-frozen inner loop, and a second route
#: to a guaranteed false STOP. Nothing in the vendor asserts this for us.
SATURATED_INNER_LR_MULTIPLIER = 1.0


class InnerStepResult(eqx.Module):
    """What one inner step returns.

    Mirrors the field names of the vendor's `MetaModel.InnerLoopStepResult`
    (`transformer.py:584-591`) with `new_model` narrowed to `fast_weights`,
    because the partition/combine of inner against outer parameters is the
    overlay's job, not the step's.
    """

    fast_weights: FastWeights
    opt_state: Any
    state_tuple: Any
    metrics: dict


class InnerStep(Protocol):
    """The injected seam. Stands in for `MetaModel.inner_loop_step`."""

    def __call__(
        self,
        fast_weights: FastWeights,
        opt_state: Any,
        state_tuple: Any,
        chunk: Any,
    ) -> InnerStepResult: ...


class CarryState(eqx.Module):
    """The carry the vendor discards.

    An `eqx.Module` rather than a plain dataclass so it is a registered pytree
    and the `tree_ops` helpers descend into it -- the same reason
    `tests/test_interceptor.py` uses `eqx.Module` stand-ins.
    """

    fast_weights: FastWeights
    opt_state: Any
    state_tuple: Any

    n_steps: jnp.ndarray
    """int32. Inner SGD steps applied so far, across every sequence. This is the
    "attack slowness" axis the pre-registration reports as a secondary metric,
    and it is only meaningful because the carry survives sequence boundaries."""

    inner_lr_multiplier: jnp.ndarray
    """float32. Recorded, not assumed -- see `SATURATED_INNER_LR_MULTIPLIER`."""


def init_carry(
    fast_weights: FastWeights,
    opt_state: Any = None,
    state_tuple: Any = None,
    inner_lr_multiplier: float = SATURATED_INNER_LR_MULTIPLIER,
) -> CarryState:
    """Fresh carry at zero steps."""
    return CarryState(
        fast_weights=fast_weights,
        opt_state=opt_state,
        state_tuple=state_tuple,
        n_steps=jnp.int32(0),
        inner_lr_multiplier=jnp.float32(inner_lr_multiplier),
    )


def assert_saturated_inner_lr(carry: CarryState, tolerance: float = 1e-6) -> None:
    """Fail loudly unless the inner-LR ramp is fully saturated.

    Called before any measurement. An unsaturated multiplier means the inner
    loop is running at `ilr_init` and the fast weights barely move, which
    produces a null result for a reason that has nothing to do with the attack.
    Cheap to check, and the failure it prevents is a false STOP.
    """
    multiplier = float(carry.inner_lr_multiplier)
    if abs(multiplier - SATURATED_INNER_LR_MULTIPLIER) > tolerance:
        raise ValueError(
            f"inner LR multiplier is {multiplier}, expected "
            f"{SATURATED_INNER_LR_MULTIPLIER} (saturated). The vendor pins "
            f"step_index = INT32_MAX-100 in eval mode (train.py:224) to reach "
            f"this; a harness that leaves step_index at 0 runs a near-frozen "
            f"inner loop and will produce a null result for the wrong reason."
        )


def chunk_tokens(tokens, mini_batch_size: int) -> list:
    """Split a flat token array into the vendor's inner-loop chunks.

    One inner SGD step happens per `mini_batch_size` tokens
    (`transformer.py:709-718`), so this is the granularity at which fast weights
    actually move. Refuses a ragged tail rather than padding it: BOS-padding
    would be masked straight out of the loss (`lm_dataset.py:53`) and would
    corrupt the valid-token matching between poison and control while looking
    perfectly fine.
    """
    if mini_batch_size <= 0:
        raise ValueError(f"mini_batch_size must be positive, got {mini_batch_size}")

    n = len(tokens)
    if n % mini_batch_size != 0:
        raise ValueError(
            f"token count {n} is not a multiple of mini_batch_size "
            f"{mini_batch_size}; refusing to pad (padding is masked out of the "
            f"loss and would break valid-token matching)"
        )

    return [tokens[i : i + mini_batch_size] for i in range(0, n, mini_batch_size)]


def run_chunks(
    carry: CarryState,
    chunks: Iterable[Any],
    step_fn: InnerStep,
) -> tuple[CarryState, list[dict]]:
    """Thread one sequence of chunks through the inner step.

    Returns `(carry, metrics)` in the same shape as the vendor's
    `carry, metrics = scan_remat_chunk(...)` -- except that here the caller
    actually receives the carry.
    """
    metrics: list[dict] = []
    n_steps = carry.n_steps

    fast_weights = carry.fast_weights
    opt_state = carry.opt_state
    state_tuple = carry.state_tuple

    for chunk in chunks:
        result = step_fn(fast_weights, opt_state, state_tuple, chunk)
        fast_weights = result.fast_weights
        opt_state = result.opt_state
        state_tuple = result.state_tuple
        metrics.append(result.metrics)
        n_steps = n_steps + jnp.int32(1)

    return (
        CarryState(
            fast_weights=fast_weights,
            opt_state=opt_state,
            state_tuple=state_tuple,
            n_steps=n_steps,
            inner_lr_multiplier=carry.inner_lr_multiplier,
        ),
        metrics,
    )


def run_sequences(
    carry: CarryState,
    sequences: Sequence[Any],
    step_fn: InnerStep,
    mini_batch_size: int,
) -> tuple[CarryState, list[dict]]:
    """Thread many sequences through the inner step, carrying fast weights across.

    This is the capability the vendor does not have. Each sequence is chunked
    and run in order, and the carry out of sequence *i* is the carry into
    sequence *i+1* -- so a stream longer than one context window accumulates,
    which is what the threat model in dossier section 1 ("a slow, benign-looking
    sequence") actually describes.
    """
    all_metrics: list[dict] = []

    for tokens in sequences:
        carry, metrics = run_chunks(
            carry, chunk_tokens(tokens, mini_batch_size), step_fn
        )
        all_metrics.extend(metrics)

    return carry, all_metrics
