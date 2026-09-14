"""GPT-2's forward pass, on the jax stack the project already has.

Why implemented rather than imported
------------------------------------
`transformers` **5.x dropped Flax models entirely** -- `FlaxGPT2LMHeadModel` no
longer exists, and importing `GPT2LMHeadModel` warns "PyTorch was not found.
Models won't be available". That left three options: pin a deprecated
`transformers` 4.x Flax path, add torch as a second array framework alongside
jax, or implement the forward pass. GPT-2 is a small, completely specified
architecture and this repo already prefers transcription over dependency where
the transcription is auditable -- `eval/lm_loss.py` does the same thing with the
vendor's cross-entropy, for the same reason.

The upside is control over the reduction, which is not incidental here. The
fluency bar is a *ratio*, and a ratio is only trustworthy if both arms are scored
with identical tokenization, identical windowing and identical masking. Owning
those three things is worth more than the convenience of a one-line model load.

The risk is implementing it wrong, and the mitigation is the thing T3.2 asks for
anyway: reproduce a published perplexity. See
`experiments/001-attack-spike/FLUENCY_REFERENCE.md`.

Correctness details that matter
-------------------------------
- HF's GPT-2 uses `Conv1D`, whose weight is stored `[in_features, out_features]`.
  So every projection is `x @ W + b` with **no transpose**. Getting this backwards
  produces a model that runs and returns garbage.
- The activation is `gelu_new`, the tanh approximation, not exact GELU.
  `jax.nn.gelu(x, approximate=True)` is that same function. Exact GELU changes
  perplexity measurably.
- LayerNorm epsilon comes from `config.json` (1e-5), not from a jax default.
- The output head is **tied** to the token embedding.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from functools import partial
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np


@dataclass(frozen=True)
class GPT2Config:
    n_layer: int
    n_head: int
    n_embd: int
    n_positions: int
    vocab_size: int
    layer_norm_epsilon: float

    @property
    def head_dim(self) -> int:
        if self.n_embd % self.n_head:
            raise ValueError(
                f"n_embd {self.n_embd} is not divisible by n_head {self.n_head}"
            )
        return self.n_embd // self.n_head

    @classmethod
    def from_json(cls, path: Path) -> GPT2Config:
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(
            n_layer=raw["n_layer"],
            n_head=raw["n_head"],
            n_embd=raw["n_embd"],
            n_positions=raw["n_positions"],
            vocab_size=raw["vocab_size"],
            layer_norm_epsilon=raw.get("layer_norm_epsilon", 1e-5),
        )


def load_weights(path: Path) -> dict[str, jnp.ndarray]:
    """Read safetensors into a flat dict of float32 arrays.

    Key prefixes are normalised: the released `gpt2` checkpoint stores
    `h.0.attn.c_attn.weight` while an LM-head export stores
    `transformer.h.0...`. Both are accepted. Attention buffers (`attn.bias`,
    `attn.masked_bias`) are dropped -- they are the causal mask, which is
    recomputed here.
    """
    from safetensors.numpy import load_file

    raw = load_file(str(path))
    weights: dict[str, jnp.ndarray] = {}

    for key, value in raw.items():
        name = key[len("transformer.") :] if key.startswith("transformer.") else key
        if name.endswith((".attn.bias", ".attn.masked_bias")):
            continue
        if name.startswith("lm_head."):
            continue  # tied to wte
        weights[name] = jnp.asarray(value, dtype=jnp.float32)

    required = {"wte.weight", "wpe.weight", "ln_f.weight", "ln_f.bias"}
    missing = required - set(weights)
    if missing:
        raise ValueError(
            f"safetensors at {path} is missing {sorted(missing)}; this does not "
            f"look like a GPT-2 checkpoint"
        )
    return weights


def _layer_norm(x, weight, bias, eps: float):
    mean = jnp.mean(x, axis=-1, keepdims=True)
    variance = jnp.var(x, axis=-1, keepdims=True)
    return (x - mean) / jnp.sqrt(variance + eps) * weight + bias


def _attention(x, weights, prefix: str, config: GPT2Config):
    seq_len = x.shape[0]

    # Conv1D: [in, out], so no transpose.
    qkv = x @ weights[f"{prefix}.attn.c_attn.weight"] + weights[f"{prefix}.attn.c_attn.bias"]
    q, k, v = jnp.split(qkv, 3, axis=-1)

    def split_heads(t):
        return t.reshape(seq_len, config.n_head, config.head_dim).transpose(1, 0, 2)

    q, k, v = split_heads(q), split_heads(k), split_heads(v)

    scores = jnp.einsum("hqd,hkd->hqk", q, k) / jnp.sqrt(config.head_dim).astype(jnp.float32)

    causal = jnp.tril(jnp.ones((seq_len, seq_len), dtype=bool))
    scores = jnp.where(causal, scores, jnp.finfo(jnp.float32).min)

    attended = jnp.einsum("hqk,hkd->hqd", jax.nn.softmax(scores, axis=-1), v)
    merged = attended.transpose(1, 0, 2).reshape(seq_len, config.n_embd)

    return merged @ weights[f"{prefix}.attn.c_proj.weight"] + weights[f"{prefix}.attn.c_proj.bias"]


def _mlp(x, weights, prefix: str):
    hidden = x @ weights[f"{prefix}.mlp.c_fc.weight"] + weights[f"{prefix}.mlp.c_fc.bias"]
    # gelu_new -- the tanh approximation GPT-2 was trained with.
    hidden = jax.nn.gelu(hidden, approximate=True)
    return hidden @ weights[f"{prefix}.mlp.c_proj.weight"] + weights[f"{prefix}.mlp.c_proj.bias"]


@partial(jax.jit, static_argnums=(2,))
def forward(weights: dict[str, jnp.ndarray], input_ids: jnp.ndarray, config: GPT2Config):
    """Logits for one sequence. `input_ids` is `[seq]`, output is `[seq, vocab]`."""
    seq_len = input_ids.shape[0]

    hidden = weights["wte.weight"][input_ids] + weights["wpe.weight"][:seq_len]

    for layer in range(config.n_layer):
        prefix = f"h.{layer}"
        residual = hidden
        hidden = _layer_norm(
            hidden,
            weights[f"{prefix}.ln_1.weight"],
            weights[f"{prefix}.ln_1.bias"],
            config.layer_norm_epsilon,
        )
        hidden = residual + _attention(hidden, weights, prefix, config)

        residual = hidden
        hidden = _layer_norm(
            hidden,
            weights[f"{prefix}.ln_2.weight"],
            weights[f"{prefix}.ln_2.bias"],
            config.layer_norm_epsilon,
        )
        hidden = residual + _mlp(hidden, weights, prefix)

    hidden = _layer_norm(
        hidden, weights["ln_f.weight"], weights["ln_f.bias"], config.layer_norm_epsilon
    )

    # Tied output head.
    return hidden @ weights["wte.weight"].T


class GPT2Reference:
    """A loaded GPT-2, exposing exactly one operation: logits for token ids.

    Deliberately narrow. This object must never be handed anything that could
    make it the victim model, and it does not know how to adapt, train or carry
    state -- it only scores.
    """

    def __init__(self, model_dir: Path):
        model_dir = Path(model_dir)
        self.config = GPT2Config.from_json(model_dir / "config.json")
        self.weights = load_weights(model_dir / "model.safetensors")
        self._tokenizer_path = model_dir / "tokenizer.json"

    @property
    def max_length(self) -> int:
        return self.config.n_positions

    def tokenizer(self):
        from tokenizers import Tokenizer

        return Tokenizer.from_file(str(self._tokenizer_path))

    def logits(self, input_ids) -> jnp.ndarray:
        ids = jnp.asarray(np.asarray(input_ids, dtype=np.int32))
        if ids.ndim != 1:
            raise ValueError(f"expected a 1-D id sequence, got shape {ids.shape}")
        if ids.shape[0] > self.max_length:
            raise ValueError(
                f"sequence of {ids.shape[0]} exceeds the model's {self.max_length} "
                f"positions; window it before calling (see eval/fluency.py)"
            )
        return forward(self.weights, ids, self.config)
