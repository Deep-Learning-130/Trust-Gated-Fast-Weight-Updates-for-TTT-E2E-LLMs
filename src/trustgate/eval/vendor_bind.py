"""Bind `trustgate.eval.carry` to the real vendor `MetaModel`.

This is the seam the whole Phase 1 harness was written around and the last
structural gap in it. `carry.py` implements the chunk loop, the carry threading,
the optimiser-state threading and the step accounting, and CPU-tests all four
against a stand-in `step_fn`. This module supplies the real `step_fn` -- and
nothing else.

Why the vendor cannot be called directly
----------------------------------------
`MetaModel.loss_for_sequence` (`transformer.py:647-745`) already does exactly
the right computation and then throws the answer away. Its meta branch binds

    carry, metrics = scan_remat_chunk(process_suffix_chunk, (model, opt, state), ...)

at `:712` and returns `loss, metrics` at `:745`. `carry` -- which holds the
adapted prime MLP -- is never read. Three further resets compound it:

- `:685` rebuilds `model` from outer parameters with a dtype cast on every call,
- `:686` re-initialises `inner_opt_state` from scratch,
- `loop.py:160` vmaps the whole thing per batch element.

So "adapt on a stream, then measure benign loss" cannot be expressed as two
vendor calls. It would silently reset the fast weights between them and measure
eval noise -- a null indistinguishable from a real STOP, bought with a paid GPU
run. ADR-006.

What this module reproduces, and why it is a transcription rather than a fork
----------------------------------------------------------------------------
`bind()` performs the same setup as `loss_for_sequence:660-691` -- the block
split, the state split, the dtype-cast outer reference, the inner optimiser
init, the prefix pass -- and `make_step_fn` is `process_suffix_chunk`
(`:694-714`) with one change: it *returns* the carry instead of feeding it to a
scan that discards it.

Every line is transcribed against a cited vendor line rather than imported,
because ADR-002 forbids editing `vendor/ttt-e2e/` and there is no seam in
`loss_for_sequence` to inject into. Where the vendor is callable, it is called:
`MetaModel.inner_loop_step` is invoked unbound, exactly as `:700` does.

Why a Python loop and not `scan_remat_chunk`
--------------------------------------------
The vendor scans for memory -- `eqx.filter_checkpoint` per chunk bounds
activations so an *outer* gradient can flow back through the inner loop during
meta-training. The attack spike takes no outer gradient; it adapts forward and
measures. There is nothing to rematerialise for. See `carry.py`.

Import discipline
-----------------
**Every vendor import in this file is lazy** (inside a function body), matching
`eval/fluency.py`. The CPU test suite has no `einops`, `zarr`, `grain`,
`orbax`, `wandb` or `hydra` by design -- that is what keeps CI network-free --
so importing `ttt.model.transformer` at module scope would break 197 passing
tests on every machine that is not a provisioned GPU box.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import equinox as eqx
import jax
import jax.numpy as jnp

from trustgate.eval.carry import (
    CarryState,
    InnerStepResult,
    init_carry,
)

#: `train.py:224` pins `step_index` to this before `evaluator.eval_fn` in eval
#: mode, so `get_ilr_multiplier` (`transformer.py:564-573`) is fully saturated.
#: A harness that leaves `step_index` at 0 gets `ilr_init` instead: a
#: near-frozen inner loop, and a null result for a reason unrelated to the
#: attack. We pin it for the same reason, rather than inheriting whatever the
#: caller's state happened to hold.
SATURATING_STEP_INDEX = int(jnp.iinfo(jnp.int32).max) - 100


def _require_vendor():
    """Import the vendor tree, or explain precisely what is missing.

    A bare ImportError here is unhelpful: on a CPU box it is *expected*, and on
    a GPU box it means the install is wrong. Distinguish the two.
    """
    try:
        from ttt.model.data import Batch, tree_slice
        from ttt.model.transformer import BlockCollectionSplit, MetaModel
        from ttt.utils.filter_utils import get_filter_spec
        from ttt.utils.jax_utils import clone_pytree
    except ImportError as exc:  # pragma: no cover - requires the vendor install
        raise ImportError(
            "trustgate.eval.vendor_bind requires the vendor tree to be "
            "importable (`uv pip install -e vendor/ttt-e2e`, Python >= 3.12, "
            "jax[cuda12]<0.6). This is expected to fail in the CPU test "
            "environment, which deliberately installs none of einops/zarr/"
            "grain/orbax/wandb/hydra -- see tests/test_interceptor.py for why "
            "the CPU suite tests against stand-ins instead. Original error: "
            f"{exc}"
        ) from exc

    return (
        Batch,
        tree_slice,
        BlockCollectionSplit,
        MetaModel,
        get_filter_spec,
        clone_pytree,
    )


@dataclass(frozen=True)
class VendorBinding:
    """Everything `loss_for_sequence` sets up before its scan, kept addressable.

    Constructed once per model and reused across every sequence of a stream.
    The vendor rebuilds all of this on every call (`transformer.py:685-691`),
    which is the reset ADR-006 exists to defeat.
    """

    model_split: Any
    """The MetaModel with `BlockCollectionSplit` installed at
    `language_model.model.h` (`transformer.py:661-676`). The inner spec
    `language_model.**.suffix_blocks.feed_forward_prime.**` only matches after
    this split, so every filter operation must run against *this* tree and not
    against the model the caller passed in."""

    model_outer: Any
    """`jax.tree.map(astype(state_dtype), model_split)` -- `transformer.py:685`.
    Source of the outer (frozen) parameters recombined on every inner step. The
    dtype cast is load-bearing, which is why `RunCondition.dtype` is a matched
    field."""

    spec_inner: Any
    """Filter spec pytree from `get_filter_spec(model_split, spec_inner)`."""

    state_all: Any
    state_prefix: Any
    state_suffix: Any

    mini_batch_size: int
    seq_length: int
    suffix_len: int
    inner_lr_multiplier: float
    state_dtype: str

    def fast_weights(self, model=None):
        """Partition the inner (fast) parameters out of a full model."""
        model = self.model_split if model is None else model
        inner_params, _ = eqx.partition(model, self.spec_inner)
        return inner_params

    def combine(self, fast_weights):
        """Recombine fast weights with the frozen outer parameters.

        `transformer.py:696-698` verbatim: inner params come from the adapting
        tree, outer params from the dtype-cast reference -- *not* from the
        adapting tree, so nothing outside the inner spec can drift.
        """
        _, outer_params = eqx.partition(self.model_outer, self.spec_inner)
        return eqx.combine(fast_weights, outer_params)

    def init_opt_state(self):
        """Fresh inner optimiser state -- `transformer.py:686`.

        Re-initialised per *stream*, not per chunk. The vendor re-inits per
        call; carrying it is the point of ADR-006, and `RunCondition.n_chunks`
        is matched across arms precisely because the first chunks run with an
        unwarmed optimiser.
        """
        return self.model_outer.inner_optimizer(self.state_all).init(
            self.model_outer.inner_parameters()
        )

    def init_carry(self) -> CarryState:
        return init_carry(
            fast_weights=self.fast_weights(),
            opt_state=self.init_opt_state(),
            state_tuple=(self.state_all, self.state_suffix),
            inner_lr_multiplier=self.inner_lr_multiplier,
        )


def bind(model, state, *, pin_step_index: bool = True) -> VendorBinding:
    """Reproduce `loss_for_sequence`'s meta-branch setup, keeping the pieces.

    Transcribes `transformer.py:660-691`. `model` is a `MetaModel` and `state`
    its `equinox.nn.State`, exactly as `train.py` holds them.

    Args:
        pin_step_index: set `state[step_index]` to `INT32_MAX - 100` first, as
            `train.py:224` does in eval mode. Leave True unless you are
            deliberately testing the unsaturated path -- `RunCondition` rejects
            an unsaturated multiplier at construction, so a False here surfaces
            as a loud failure rather than a quiet null.
    """
    (
        _Batch,
        _tree_slice,
        BlockCollectionSplit,
        _MetaModel,
        get_filter_spec,
        clone_pytree,
    ) = _require_vendor()

    cfg = model.config

    if cfg.training.train_mode != "meta":
        raise ValueError(
            f"train_mode is {cfg.training.train_mode!r}; only 'meta' has an "
            "inner loop. In 'pretrain' mode there are no fast weights to "
            "poison and the attack is vacuously null -- PREREGISTERED.md names "
            "'meta only' and calls the pretrain arm a vacuous null."
        )

    if pin_step_index:
        # train.py:224. Do this *before* the optimiser is built: the inner LR
        # multiplier is read off the state at `inner_optimizer()` time.
        state = state.set(
            model.step_index, jnp.array(SATURATING_STEP_INDEX, dtype=jnp.int32)
        )

    # transformer.py:661-666 -- install the prefix/suffix split.
    block_collection = model.language_model.model.h.blocks
    prime_storage = (
        model.language_model.model.h.prime_storage if cfg.model.prime else None
    )
    if prime_storage is None:
        raise ValueError(
            "model.prime is False: there is no prime (fast-weight) MLP to "
            "adapt, so there is nothing for an attack to corrupt. The "
            "extension configs set `prime: True`."
        )

    new_collection = BlockCollectionSplit(
        cfg.model,
        block_collection=block_collection,
        prime_storage=prime_storage,
        key=jax.random.PRNGKey(0),
    )

    # transformer.py:668-671 -- split the block state into prefix and suffix.
    state_prefix_suffix = state.substate(model.language_model.model.h.blocks)
    state_prefix, state_suffix = BlockCollectionSplit.split_state(
        state_prefix_suffix, cfg.model.suffix_len
    )
    state_all = clone_pytree(state)

    # transformer.py:676
    model_split = eqx.tree_at(lambda m: m.language_model.model.h, model, new_collection)

    # transformer.py:685 -- the dtype-cast outer reference.
    model_outer = jax.tree.map(
        lambda p: p.astype(model_split.state_dtype), model_split
    )

    # transformer.py:695. Computed once: the tree structure does not change
    # across chunks, and the vendor only recomputes it because it is inside a
    # scanned closure.
    spec_inner = get_filter_spec(
        model_split, cfg.training.spec_inner, "inner parameters"
    )

    multiplier = float(
        model_outer.get_ilr_multiplier(state_all.get(model.step_index))
    )

    return VendorBinding(
        model_split=model_split,
        model_outer=model_outer,
        spec_inner=spec_inner,
        state_all=state_all,
        state_prefix=state_prefix,
        state_suffix=state_suffix,
        mini_batch_size=int(cfg.model.mini_batch_size),
        seq_length=int(cfg.training.seq_length),
        suffix_len=int(cfg.model.suffix_len),
        inner_lr_multiplier=multiplier,
        state_dtype=str(cfg.model.state_dtype),
    )


def make_step_fn(binding: VendorBinding) -> Callable:
    """The injected seam `carry.run_chunks` expects, backed by the real vendor.

    This is `process_suffix_chunk` (`transformer.py:694-714`) with the carry
    returned instead of discarded. Chunks are `(suffix_chunk, prefix_chunk)`
    pairs, as the vendor's scan `xs` supplies them at `:713`.
    """
    _Batch, _tree_slice, _Split, MetaModel, _spec, _clone = _require_vendor()

    def step_fn(fast_weights, opt_state, state_tuple, chunk) -> InnerStepResult:
        suffix_chunk, prefix_chunk = chunk

        # transformer.py:696-698.
        model_inner = binding.combine(fast_weights)

        # transformer.py:700-702 -- called unbound, exactly as the vendor does.
        result = MetaModel.inner_loop_step(
            model_inner, opt_state, state_tuple, suffix_chunk, prefix_chunk
        )

        return InnerStepResult(
            # `inner_loop_step` returns a whole MetaModel via
            # `filter_apply_updates` (`filter_utils.py:189`); narrow it back to
            # the inner subtree so the carry holds fast weights and not a model.
            fast_weights=binding.fast_weights(result.new_model),
            opt_state=result.new_optimizer_state,
            state_tuple=result.new_state,
            metrics=result.metrics,
        )

    return step_fn


def make_batch(tokens, *, bos_token_id: int) -> Any:
    """Build a vendor `Batch` from `seq_len + 1` flat token ids.

    Transcribes `lm_dataset._to_batch` (`lm_dataset.py:43-54`). The off-by-one
    is real and load-bearing: a `Batch` of length `T` consumes `T + 1` source
    tokens, so a stream of `n` tokens yields `n - 1` loss-bearing positions per
    sequence. `RunCondition.valid_tokens` counts the mask, not the raw length,
    because the inner objective's denominator is that chunk's valid count
    (`loss.py:15,26`) and a mismatch rescales every surviving gradient.
    """
    Batch, _tree_slice, _Split, _MetaModel, _spec, _clone = _require_vendor()

    tokens = jnp.asarray(tokens, dtype=jnp.int32)
    if tokens.ndim != 1:
        raise ValueError(f"expected a flat token array, got shape {tokens.shape}")
    if tokens.shape[0] < 2:
        raise ValueError("need at least 2 tokens to form one input/target pair")

    return Batch(
        input_ids=tokens[:-1],
        target_tokens=tokens[1:],
        loss_masks=(tokens[1:] != bos_token_id),
    )


def prefix_pass(binding: VendorBinding, seq) -> jnp.ndarray:
    """Run the frozen prefix blocks over a whole sequence.

    `transformer.py:688-691`. Deliberately *outside* the chunk loop and
    deliberately not adapted: only the last `suffix_len` layers carry a prime
    MLP, so the prefix contributes no fast weights, and re-running it per chunk
    would be both wrong (it needs full-sequence context) and wasteful.
    """
    xt_embed = binding.model_split.language_model.wte_call(seq.input_ids)
    return binding.model_split.language_model.prefix_call(
        binding.model_split.language_model.model.h.prefix_blocks,
        xt_embed,
        binding.state_prefix,
        seq,
    ).last_hidden_state


def chunk_sequence(binding: VendorBinding, seq, prefix_output) -> list:
    """Split a sequence and its prefix output into aligned inner-loop chunks.

    The vendor does this with `tree_rearrange(..., "(chunk token) ... -> chunk
    token ...")` at `:710-711` because it feeds a scan. A Python loop wants a
    list, and the assertion at `:681` is reproduced here so the failure names
    the real constraint rather than surfacing as a reshape error.
    """
    _Batch, tree_slice, _Split, _MetaModel, _spec, _clone = _require_vendor()

    seqlen = seq.input_ids.shape[0]
    tokens_per_chunk = binding.mini_batch_size

    if seqlen % tokens_per_chunk != 0:
        raise ValueError(
            f"seqlen {seqlen} must be divisible by mini_batch_size "
            f"{tokens_per_chunk} (transformer.py:681). Refusing to pad: "
            "BOS padding is masked straight out of the loss (lm_dataset.py:53) "
            "and would break valid-token matching between arms while looking "
            "perfectly fine."
        )

    chunks = []
    for start in range(0, seqlen, tokens_per_chunk):
        window = slice(start, start + tokens_per_chunk)
        chunks.append((tree_slice(seq, window), prefix_output[window]))
    return chunks


def sequence_chunks(binding: VendorBinding, tokens, *, bos_token_id: int) -> list:
    """`tokens` (flat, `seq_len + 1` ids) -> the chunk list for one sequence."""
    seq = make_batch(tokens, bos_token_id=bos_token_id)
    prefix_output = prefix_pass(binding, seq)
    return chunk_sequence(binding, seq, prefix_output)


def split_stream(tokens, seq_length: int) -> list:
    """Cut a flat stream into sequences of `seq_length` loss-bearing positions.

    Each sequence needs `seq_length + 1` source tokens (`_to_batch` drops one to
    the input/target shift), and consecutive sequences overlap by exactly that
    one token so no position is lost at the seam. Cross-sequence carry is what
    makes this meaningful at all: one 8K sequence at `mini_batch_size=1024` is
    only 8 inner steps, while `AttackSpec.stream_tokens` presupposes
    accumulation over many. ADR-006.
    """
    tokens = jnp.asarray(tokens, dtype=jnp.int32)
    n = tokens.shape[0]

    if n % seq_length != 0:
        raise ValueError(
            f"stream of {n} tokens is not a multiple of seq_length "
            f"{seq_length}; refusing to pad (see chunk_sequence)"
        )

    sequences = []
    for start in range(0, n, seq_length):
        end = start + seq_length
        # One extra token for the input/target shift. The final sequence has no
        # following position to borrow a target from, so it repeats its last
        # token. That costs one masked-in position out of `seq_length`,
        # identically in both arms, and `RunCondition.valid_tokens` records the
        # result rather than assuming it.
        if end + 1 <= n:
            sequences.append(tokens[start : end + 1])
        else:
            sequences.append(jnp.concatenate([tokens[start:end], tokens[end - 1 :]]))
    return sequences
