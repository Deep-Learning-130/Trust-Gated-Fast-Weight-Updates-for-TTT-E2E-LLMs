"""A tiny TTT-shaped model, so the Phase 1 pipeline can be exercised end to end.

What this is for
----------------
The pre-registered experiment needs the released 1B checkpoint, a requester-pays
GCS bucket, a W&B key and an 80GB accelerator. None of that is available yet, so
nothing downstream of `run_stream` has ever executed against a real model. This
module supplies the smallest thing that is honestly *TTT-shaped* -- fast weights
updated by next-token prediction at inference, carried across chunks, with the
base weights frozen -- so the crafted-stream builders, the matching discipline,
the corruption metric and the report renderer all run against real numbers.

**It is not the victim.** See `README.md` in this directory for the full list of
what the pilot cannot claim.

The shape, and why it is the shape it is
----------------------------------------
Slow (base) weights: token embedding, a causal local mixer, and an output head.
Frozen at inference, exactly as in the threat model -- the attacker never writes
to them.

Fast weights: a two-layer MLP applied position-wise, whose output is added to the
mixed hidden state. These are the analogue of the vendor's "prime MLP", and they
are the only thing the inner loop touches. One inner SGD step per
`mini_batch_size` tokens, on the chunk's own next-token cross-entropy -- the same
self-supervised objective the vendor uses (`transformer.py:709-718`).

The causal mixer is a cumulative-mean over a fixed window rather than attention.
Attention would make the pilot a small language model with a TTT bolt-on; a
weak mixer forces the fast weights to carry most of the context, which is the
part under study. It also keeps a training run inside a couple of minutes on CPU.

Interface
---------
`make_step_fn` returns a callable satisfying `trustgate.eval.carry.InnerStep`, so
`carry.run_chunks` and `carry.run_sequences` -- already implemented and CPU-tested
-- drive this model unchanged. That is the whole point of ADR-006's injected seam:
the overlay was written against a Protocol precisely so a stand-in could be
substituted for `MetaModel.inner_loop_step`.
"""

from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np
import optax

from trustgate.eval.carry import InnerStepResult

# Byte-level. 256 byte values plus one BOS, which `TokenCorpus` masks out of the
# loss the same way the vendor masks Llama-3's BOS (`lm_dataset.py:53`).
VOCAB_SIZE = 257
PILOT_BOS_ID = 256


@dataclass(frozen=True)
class PilotConfig:
    d_model: int = 128
    d_ff: int = 256
    window: int = 16
    inner_lr: float = 0.05
    mini_batch_size: int = 256
    span_tokens: int = 64


# --------------------------------------------------------------------------
# Parameters
# --------------------------------------------------------------------------
def init_params(key, cfg: PilotConfig):
    """Slow (base) and fast parameters, returned separately.

    They are kept apart at construction rather than partitioned later because
    the split *is* the claim boundary: the attacker moves `fast`, never `slow`.
    """
    k_emb, k_self, k_mix, k_head, k_w1, k_w2 = jax.random.split(key, 6)
    d, f = cfg.d_model, cfg.d_ff

    slow = {
        "embed": jax.random.normal(k_emb, (VOCAB_SIZE, d)) * 0.02,
        # The current token gets its own projection. Folding it into the window
        # mean instead would bury it at 1/window of the signal, and a model that
        # cannot see the token it is predicting from never leaves uniform.
        "self": jax.random.normal(k_self, (d, d)) * (1.0 / np.sqrt(d)),
        "mix": jax.random.normal(k_mix, (d, d)) * (1.0 / np.sqrt(d)),
        "head": jax.random.normal(k_head, (d, VOCAB_SIZE)) * (1.0 / np.sqrt(d)),
    }
    # `w2` is zero-initialised, so the fast branch contributes exactly nothing
    # before the first inner step. This matters more than it looks: with a random
    # `w2` the branch injects noise that base training then learns to compensate
    # for, and any subsequent inner update moves the model *away* from the state
    # it was trained in -- adaptation measurably hurts. A zero-initialised
    # residual branch is also how the vendor's prime MLP enters the residual
    # stream, so this is the faithful choice as well as the working one.
    fast = {
        "w1": jax.random.normal(k_w1, (d, f)) * (1.0 / np.sqrt(d)),
        "b1": jnp.zeros((f,)),
        "w2": jnp.zeros((f, d)),
        "b2": jnp.zeros((d,)),
    }
    return slow, fast


def _causal_window_mean(h, window: int):
    """Cumulative mean over the previous `window` positions, causal by construction.

    Position t sees positions max(0, t-window+1)..t and nothing after, so no
    information leaks backwards from the targets.
    """
    csum = jnp.cumsum(h, axis=0)
    padded = jnp.concatenate([jnp.zeros((window, h.shape[1])), csum], axis=0)
    lagged = padded[:-window][: h.shape[0]]
    counts = jnp.minimum(jnp.arange(1, h.shape[0] + 1), window)[:, None]
    return (csum - lagged) / counts


def forward(slow, fast, tokens, cfg: PilotConfig):
    """Logits for every position. `tokens` is 1-D int32."""
    e = slow["embed"][tokens]
    h = e @ slow["self"] + _causal_window_mean(e, cfg.window) @ slow["mix"]
    h = jnp.tanh(h)
    # The fast MLP -- the only thing the inner loop moves.
    delta = jnp.tanh(h @ fast["w1"] + fast["b1"]) @ fast["w2"] + fast["b2"]
    return (h + delta) @ slow["head"]


def sequence_loss(slow, fast, tokens, cfg: PilotConfig):
    """Mean next-token CE in nats over valid (non-BOS-target) positions.

    Deliberately the same quantity the victim reports, and computed the same way
    `trustgate.eval.lm_loss.masked_mean_ce` computes it: shift by one, mask the
    targets that are BOS, mean over valid positions only. The attacker optimises
    the number the defender reads.
    """
    logits = forward(slow, fast, tokens[:-1], cfg)
    targets = tokens[1:]
    valid = (targets != PILOT_BOS_ID).astype(jnp.float32)
    logp = jax.nn.log_softmax(logits, axis=-1)
    nll = -jnp.take_along_axis(logp, targets[:, None], axis=-1)[:, 0]
    return jnp.sum(nll * valid) / jnp.maximum(jnp.sum(valid), 1.0)


def per_token_nll(slow, fast, tokens, cfg: PilotConfig):
    """Per-position NLL, for the structural check that context is actually used."""
    logits = forward(slow, fast, tokens[:-1], cfg)
    targets = tokens[1:]
    logp = jax.nn.log_softmax(logits, axis=-1)
    return -jnp.take_along_axis(logp, targets[:, None], axis=-1)[:, 0]


# --------------------------------------------------------------------------
# The inner step -- the seam `trustgate.eval.carry` drives
# --------------------------------------------------------------------------
def make_step_fn(slow, cfg: PilotConfig):
    """Return an `InnerStep` closed over the frozen base weights.

    One SGD step on the chunk's own next-token CE. Plain SGD rather than an
    optimiser with state, because `opt_state` threading is already exercised by
    `carry.run_chunks` and adding momentum here would only make the pilot's
    dynamics harder to reason about.
    """
    grad_fn = jax.value_and_grad(sequence_loss, argnums=1)

    @jax.jit
    def _update(fast, chunk):
        loss, grads = grad_fn(slow, fast, chunk, cfg)
        new_fast = jax.tree.map(lambda p, g: p - cfg.inner_lr * g, fast, grads)
        return new_fast, loss

    def step_fn(fast_weights, opt_state, state_tuple, chunk):
        new_fast, loss = _update(fast_weights, chunk)
        return InnerStepResult(
            fast_weights=new_fast,
            opt_state=opt_state,
            state_tuple=state_tuple,
            metrics={"inner_loss": float(loss)},
        )

    return step_fn


# --------------------------------------------------------------------------
# Base-weight training
# --------------------------------------------------------------------------
def train_base(slow, fast, corpus_tokens, cfg: PilotConfig, *, steps=700,
               lr=5e-3, batch=16, seq_len=256, seed=0):
    """Train the slow weights only, with the fast weights held at init.

    The fast weights are frozen here so that whatever the inner loop achieves at
    inference is attributable to the inner loop rather than to having been
    trained in. Returns the trained slow weights and the final training loss.
    """
    grad_fn = jax.jit(jax.value_and_grad(
        lambda s, f, batch_tokens: jnp.mean(
            jax.vmap(lambda t: sequence_loss(s, f, t, cfg))(batch_tokens)
        )
    ))
    # Adam rather than SGD: `optax` is already a core dependency, and plain SGD
    # at this scale spends the whole budget escaping the uniform-prediction
    # plateau at ln(257) = 5.549 nats.
    opt = optax.adam(lr)
    opt_state = opt.init(slow)

    @jax.jit
    def _step(slow, opt_state, batch_tokens):
        loss, grads = grad_fn(slow, fast, batch_tokens)
        updates, opt_state = opt.update(grads, opt_state, slow)
        return optax.apply_updates(slow, updates), opt_state, loss

    rng = np.random.default_rng(seed)
    n = len(corpus_tokens)
    loss = float("nan")

    for step in range(steps):
        offs = rng.integers(0, n - seq_len - 1, size=batch)
        batch_tokens = jnp.stack(
            [jnp.asarray(corpus_tokens[o : o + seq_len + 1]) for o in offs]
        )
        slow, opt_state, loss = _step(slow, opt_state, batch_tokens)
        if step % 100 == 0:
            print("    base step %4d  loss %.4f" % (step, float(loss)))

    return slow, float(loss)
