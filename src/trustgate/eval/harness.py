"""Phase 1 evaluation harness: run poison and control streams under matched conditions.

Status: orchestration is specified and the model-touching steps are **bound**.
`run_stream` and `eval_benign` drive the real `MetaModel.inner_loop_step`
through `trustgate.eval.vendor_bind`, so they need an importable vendor tree
and a GPU -- but **not** a checkpoint: `train.py:198` creates a fresh
random-init model when `load_part=none`, which is what
`experiments/003-smoke-125m/` exercises. A *verdict* still needs the released
1B checkpoint and the Phase 0.5 baseline; execution no longer does.

Matching discipline
-------------------
Poison and control runs must differ in *exactly one* thing: the content of the
stream. Same seed, same length in tokens, same chunking, same dtype, same
benign eval set, same model revision. Every one of these is a plausible
confound, and fast weights drift on any input at all -- so an unmatched control
would make a null result look like a positive one. `RunCondition` exists to make
the matching explicit and assertable rather than a matter of care.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Callable

import numpy as np

from trustgate.attack.objectives import AttackSpec
from trustgate.attack.stream import (
    CraftedStream,
    StreamStrategy,
    assert_stream_not_from_eval_split,
    assert_streams_matched,
)
from trustgate.eval import carry as carry_mod
from trustgate.eval.carry import SATURATED_INNER_LR_MULTIPLIER
from trustgate.eval.metrics import CorruptionResult, corruption_metric


def eval_tokens_digest(tokens) -> str:
    """sha256 over the benign eval tokens.

    `benign_eval_split` is a *name*, not the data: two runs can evaluate
    different windows of the same split under the same string and
    `assert_matches` would see nothing wrong. The vendor's own eval loader is
    `seed=0, shuffle=False, repeat=False` (`loop.py:63-74`), so its windows are
    deterministic -- but a hand-rolled harness loses that guarantee, and the
    poison/control comparison is only meaningful if both arms were measured on
    identical held-out tokens.
    """
    return hashlib.sha256(np.ascontiguousarray(tokens, dtype=np.int32).tobytes()).hexdigest()


@dataclass(frozen=True)
class RunCondition:
    """Everything that must match between poison and control runs.

    Every field here is a plausible confound, and fast weights drift on any
    input at all -- so an unmatched control would make a null result look like a
    positive one. `PREREGISTERED.md` names control mismatch as an invalidating
    condition, which is why this is assertable rather than a matter of care.
    """

    seed: int
    stream_tokens: int
    mini_batch_size: int
    seq_length: int
    checkpoint: str
    benign_eval_split: str

    span_tokens: int
    """ADR-007. A STOP verdict is scoped to the granularity that produced it, so
    the granularity has to travel with the result."""

    n_chunks: int
    """Inner SGD steps the stream drives. `inner_opt_state` is re-initialised per
    call (`transformer.py:686`), so the first chunks run with an unwarmed
    optimiser -- identical across arms only if the chunk counts match."""

    valid_tokens: int
    """Loss-bearing positions, not raw length. The inner objective's denominator
    is that chunk's valid count (`loss.py:15,26`), so a difference here rescales
    every surviving token's gradient."""

    dtype: str
    """Pre-registered as guarded and previously absent. `PREREGISTERED.md`
    lists "control not length/seed/**dtype**-matched" as invalidating and says
    `assert_matches` guards it; it did not. `transformer.py:685` casts the whole
    model to `state_dtype` per call and the outer params come from that cast
    copy, so dtype is load-bearing. Adding this field satisfies a pre-registered
    claim rather than moving a bar."""

    stream_corpus_split: str
    """Which split the stream was drawn from. Must differ from
    `benign_eval_split` -- see `assert_stream_not_from_eval_split`."""

    eval_tokens_sha256: str
    """Identity of the benign eval tokens themselves. See `eval_tokens_digest`."""

    inner_lr_multiplier: float
    """`get_ilr_multiplier` (`transformer.py:564-573`) ramps the inner LR off
    `state[step_index]`; `train.py:224` pins it high in eval mode to saturate
    the ramp. A harness leaving `step_index` at 0 runs a near-frozen inner loop
    and returns a null for a reason unrelated to the attack."""

    def __post_init__(self) -> None:
        """Reject a condition that cannot be a valid measurement.

        Validated at construction rather than at comparison time: two runs that
        both left a field unset would *match* each other, which is precisely the
        failure this class exists to prevent.
        """
        if self.mini_batch_size <= 0 or self.span_tokens <= 0:
            raise ValueError(
                f"mini_batch_size and span_tokens must be positive, got "
                f"{self.mini_batch_size} and {self.span_tokens}"
            )
        if self.mini_batch_size % self.span_tokens != 0:
            raise ValueError(
                f"span_tokens {self.span_tokens} must divide mini_batch_size "
                f"{self.mini_batch_size} (ADR-007)"
            )
        expected_chunks = self.stream_tokens // self.mini_batch_size
        if self.n_chunks != expected_chunks:
            raise ValueError(
                f"n_chunks {self.n_chunks} does not match "
                f"stream_tokens // mini_batch_size = {expected_chunks}"
            )
        if self.valid_tokens > self.stream_tokens:
            raise ValueError(
                f"valid_tokens {self.valid_tokens} exceeds stream_tokens "
                f"{self.stream_tokens}"
            )
        if len(self.eval_tokens_sha256) != 64:
            raise ValueError(
                "eval_tokens_sha256 must be a full sha256 hex digest; an unset "
                "or truncated value would match between two runs and defeat the "
                "eval-identity check. Use `eval_tokens_digest`."
            )
        if abs(self.inner_lr_multiplier - SATURATED_INNER_LR_MULTIPLIER) > 1e-6:
            raise ValueError(
                f"inner_lr_multiplier is {self.inner_lr_multiplier}, expected "
                f"{SATURATED_INNER_LR_MULTIPLIER}. An unsaturated inner LR means "
                f"a near-frozen inner loop (transformer.py:564-573) and a null "
                f"result for the wrong reason."
            )

    @classmethod
    def from_stream(
        cls,
        stream: CraftedStream,
        *,
        seed: int,
        seq_length: int,
        checkpoint: str,
        benign_eval_split: str,
        eval_tokens_sha256: str,
        inner_lr_multiplier: float = SATURATED_INNER_LR_MULTIPLIER,
    ) -> RunCondition:
        """Derive the stream-side fields from the stream itself.

        Hand-transcribing seven numbers into two conditions is exactly how an
        unmatched control gets built by accident.
        """
        return cls(
            seed=seed,
            stream_tokens=stream.length_tokens,
            mini_batch_size=stream.mini_batch_size,
            seq_length=seq_length,
            checkpoint=checkpoint,
            benign_eval_split=benign_eval_split,
            span_tokens=stream.span_tokens,
            n_chunks=stream.n_chunks,
            valid_tokens=stream.valid_tokens,
            dtype=str(stream.tokens.dtype),
            stream_corpus_split=stream.corpus_name,
            eval_tokens_sha256=eval_tokens_sha256,
            inner_lr_multiplier=inner_lr_multiplier,
        )

    def assert_matches(self, other: RunCondition) -> None:
        """Fail loudly on any mismatch except the stream itself."""
        mismatches = [
            f"{f.name}: {getattr(self, f.name)!r} != {getattr(other, f.name)!r}"
            for f in self.__dataclass_fields__.values()
            if getattr(self, f.name) != getattr(other, f.name)
        ]
        if mismatches:
            raise ValueError(
                "poison and control conditions must be identical; mismatched: "
                + "; ".join(mismatches)
            )


@dataclass
class SpikeResult:
    """Outcome of a full attack spike across seeds."""

    corruption: CorruptionResult
    strategy: StreamStrategy

    per_seed_fluency: list[float] = field(default_factory=list)
    """One ratio per seed, not a single aggregate.

    The pre-registration states the fluency bar as one number, but there are
    five streams and T3.6 requires a ratio for each. The two corruption rows are
    explicitly aggregate ("across seeds") because Cohen's *d* cannot be
    per-seed; the fluency rationale is a property of an *artifact*, and there
    are five artifacts. See `gating_fluency_ratio`."""

    attack_success_rate: float | None = None
    per_seed_poisoned: list[float] = field(default_factory=list)
    per_seed_control: list[float] = field(default_factory=list)

    @property
    def gating_fluency_ratio(self) -> float:
        """The **max** across seeds. This is what the verdict turns on.

        Under a mean, one seed at 3.0 is laundered by four at 1.1 while that
        seed's loss still pools into the gated Cohen's *d* -- so the gating
        fluency statistic has to cover every seed contributing to the gated
        corruption statistic. Max is the strictest defensible reading of an
        ambiguous pre-registration, and Rule 5 exists to stop bars being
        *loosened*, so choosing it cannot be self-serving.

        `experiments/001-attack-spike/README.md` pre-commits the escalation
        path: if the verdict turns on this choice, that is a pre-registration
        ambiguity requiring a dated revision note *before* the verdict is
        recorded.
        """
        if not self.per_seed_fluency:
            return float("nan")
        return float(np.max(self.per_seed_fluency))

    @property
    def mean_fluency_ratio(self) -> float:
        """Reported alongside the max, never gating."""
        if not self.per_seed_fluency:
            return float("nan")
        return float(np.mean(self.per_seed_fluency))

    @property
    def worst_fluency_seed_index(self) -> int | None:
        if not self.per_seed_fluency:
            return None
        return int(np.argmax(self.per_seed_fluency))

    @property
    def is_fluency_scored(self) -> bool:
        return bool(self.per_seed_fluency) and not any(
            np.isnan(r) for r in self.per_seed_fluency
        )


@dataclass(frozen=True)
class BenignEvalResult:
    """Benign loss plus the decay curve that the mean alone hides.

    `001-attack-spike/README.md` resolves measurement mode as "adapt in meta,
    **report the decay curve**": measurement stays in meta mode, so the inner
    loop keeps stepping on the eval data and any poison effect decays across
    eval chunks. A poison effect that has decayed to nothing by the end of the
    window is a different finding from one that persists, and a single mean
    cannot tell the two apart.
    """

    mean_loss: float
    per_chunk_loss: tuple[float, ...]

    @property
    def first_chunk(self) -> float:
        """Least-diluted measurement: one inner step of eval data at most."""
        return self.per_chunk_loss[0]

    @property
    def decay(self) -> float:
        """Last minus first. Near zero means the effect persisted across the
        eval window; strongly positive means it washed out."""
        return self.per_chunk_loss[-1] - self.per_chunk_loss[0]


def run_stream(
    model,
    stream_tokens,
    condition: RunCondition,
    *,
    state=None,
    binding=None,
    gate_spec=None,
    gate_inputs=None,
    metrics_out: list | None = None,
):
    """Feed one stream through TTT-E2E, returning the adapted fast weights.

    Runs with `train_mode="meta"` -- the only mode with an inner loop. In
    `"pretrain"` mode there are no fast weights to poison and the attack is
    vacuously null; `vendor_bind.bind` rejects it.

    The threading is `trustgate.eval.carry`; the vendor inner step is bound by
    `trustgate.eval.vendor_bind`. Neither the chunk loop nor the carry is
    reimplemented here -- this function is the wiring between them, plus the
    assertions that make the result admissible.

    Returns the `CarryState`, not a loss. The carry is the artifact: it is what
    the vendor discards at `transformer.py:712` and what `eval_benign` must be
    handed for the measurement to mean anything (ADR-006).

    Args:
        model: a vendor `MetaModel`. Ignored when `binding` is supplied.
        stream_tokens: flat Llama-3 token ids, `condition.stream_tokens` long.
        state: the model's `equinox.nn.State`. Required unless `binding` is.
        binding: a prebuilt `VendorBinding`, so a multi-seed run pays the block
            split and prefix setup once rather than per stream.
        gate_spec, gate_inputs: run with the trust gate in the inner step. See
            `vendor_bind.make_step_fn`; `None` is the vendor step unmodified.
        metrics_out: if given, every inner step's metrics dict is appended to
            it -- how the gate's per-step `gate/*` values reach the caller.
    """
    from trustgate.eval import vendor_bind

    if binding is None:
        if state is None:
            raise ValueError(
                "run_stream needs either `state` (the model's equinox.nn.State) "
                "or a prebuilt `binding`; the block/state split in "
                "vendor_bind.bind cannot be reconstructed from the model alone."
            )
        binding = vendor_bind.bind(model, state)

    if binding.mini_batch_size != condition.mini_batch_size:
        raise ValueError(
            f"binding mini_batch_size {binding.mini_batch_size} != condition "
            f"{condition.mini_batch_size}. The chunk granularity is what fast "
            f"weights actually move at, and ADR-007 makes it travel with the "
            f"verdict rather than be an implementation detail."
        )

    n_tokens = int(np.asarray(stream_tokens).shape[0])
    if n_tokens != condition.stream_tokens:
        raise ValueError(
            f"stream is {n_tokens} tokens, condition declares "
            f"{condition.stream_tokens}. Length is a matched field; an "
            f"unmatched control makes a null look like a positive."
        )

    step_fn = vendor_bind.make_step_fn(binding, gate_spec, gate_inputs)
    carry = binding.init_carry()

    # Asserted *before* anything runs. An unsaturated inner LR means a
    # near-frozen inner loop and a null for a reason unrelated to the attack --
    # cheaper to catch here than after a paid run.
    carry_mod.assert_saturated_inner_lr(carry)

    sequences = vendor_bind.split_stream(stream_tokens, condition.seq_length)

    all_metrics: list[dict] = []
    for tokens in sequences:
        chunks = vendor_bind.sequence_chunks(
            binding, tokens, bos_token_id=model_bos_token_id(binding)
        )
        carry, metrics = carry_mod.run_chunks(carry, chunks, step_fn)
        all_metrics.extend(metrics)

    if int(carry.n_steps) != condition.n_chunks:
        raise ValueError(
            f"ran {int(carry.n_steps)} inner steps, condition declares "
            f"{condition.n_chunks}. `inner_opt_state` starts unwarmed, so the "
            f"step count is only identical across arms if this matches."
        )

    if metrics_out is not None:
        metrics_out.extend(all_metrics)
    return carry


def model_bos_token_id(binding) -> int:
    """BOS id from the bound model's config -- the loss mask depends on it.

    `lm_dataset.py:53` builds `loss_masks` as `targets != bos_token_id`, so
    reading this off the model rather than hardcoding 128000 keeps
    `RunCondition.valid_tokens` honest if the vocabulary ever changes.
    """
    return int(binding.model_split.config.model.bos_token_id)


def eval_benign_curve(
    binding,
    carry,
    eval_tokens,
    condition: RunCondition,
    *,
    gate_spec=None,
    gate_inputs=None,
) -> BenignEvalResult:
    """Benign-task loss for a model carrying adapted fast weights.

    Measurement stays in **meta** mode, per the resolved ambiguity in
    `001-attack-spike/README.md`: freezing the fast weights would mean the
    `pretrain` branch (`transformer.py:722-738`), which uses `self` rather than
    the dtype-cast `model` and which the pre-registration can fairly be read as
    forbidding. The cost is dilution across eval chunks, which is why the curve
    is returned and not only its mean.

    The carry is **not** mutated for the caller: eval adapts on eval data, and
    letting that leak back into the stream's carry would make the second arm's
    starting point depend on the first arm's measurement. Everything here is a
    functional pytree update, so the caller's `carry` is untouched by
    construction -- this note exists so nobody "optimises" that away.
    """
    from trustgate.eval import vendor_bind

    carry_mod.assert_saturated_inner_lr(carry)

    digest = eval_tokens_digest(np.asarray(eval_tokens)[:-1])
    if digest != condition.eval_tokens_sha256:
        raise ValueError(
            "benign eval tokens do not match `condition.eval_tokens_sha256`. "
            "The split *name* is not the data: two runs can evaluate different "
            "windows of the same split under the same string, and the "
            "poison/control comparison is only meaningful on identical "
            "held-out tokens."
        )

    # A gated arm stays gated through eval: a deployed gate does not switch off
    # while the model adapts on the benign task, and the carry it hands over
    # was produced under it.
    step_fn = vendor_bind.make_step_fn(binding, gate_spec, gate_inputs)
    chunks = vendor_bind.sequence_chunks(
        binding, eval_tokens, bos_token_id=model_bos_token_id(binding)
    )

    _, metrics = carry_mod.run_chunks(carry, chunks, step_fn)

    # `inner_loop_step` computes the loss at the *current* parameters and then
    # updates (`transformer.py:617-627`), so each entry is a predict-then-adapt
    # measurement -- which is the quantity the threat model names.
    per_chunk = tuple(float(np.asarray(m[_loss_metric_key(m)]).mean()) for m in metrics)

    return BenignEvalResult(
        mean_loss=float(np.mean(per_chunk)),
        per_chunk_loss=per_chunk,
    )


def _loss_metric_key(metrics: dict):
    """Find the CE-loss key in a vendor metrics dict without importing the enum.

    `MetaModel.MetricType` lives in the vendor tree, which is not importable in
    the CPU test environment. Matching on the member name keeps this module's
    import discipline intact (see `vendor_bind`) and works against test fakes.
    """
    for key in metrics:
        name = getattr(key, "name", None) or str(key)
        if name == "loss":
            return key
    raise KeyError(
        f"no 'loss' metric in inner-step metrics; got {list(metrics)}. "
        "`inner_loop_step` populates MetricType.loss at transformer.py:617."
    )


def eval_benign(
    binding, carry, eval_tokens, condition: RunCondition, *, gate_spec=None, gate_inputs=None
) -> float:
    """Mean benign loss. See `eval_benign_curve` for the decay curve."""
    return eval_benign_curve(
        binding, carry, eval_tokens, condition, gate_spec=gate_spec, gate_inputs=gate_inputs
    ).mean_loss


def make_adapt_and_eval(
    binding,
    eval_tokens,
    *,
    model=None,
    gate_spec=None,
    gate_inputs=None,
    metrics_out: list | None = None,
) -> Callable[[CraftedStream, RunCondition], float]:
    """Build the `adapt_and_eval` callable `run_attack_spike` injects.

    This is the composition the whole module exists to make possible: adapt on
    the stream, keep the carry, measure the benign task with it. Both halves in
    one call is the *only* correct shape -- a two-call harness resets the fast
    weights between them and measures eval noise (ADR-006).
    """

    def adapt_and_eval(stream: CraftedStream, condition: RunCondition) -> float:
        carry = run_stream(
            model,
            stream.tokens,
            condition,
            binding=binding,
            gate_spec=gate_spec,
            gate_inputs=gate_inputs,
            metrics_out=metrics_out,
        )
        return eval_benign(
            binding, carry, eval_tokens, condition, gate_spec=gate_spec, gate_inputs=gate_inputs
        )

    return adapt_and_eval


def run_attack_spike(
    spec: AttackSpec,
    condition: RunCondition,
    seeds: list[int],
    output_dir: Path,
    *,
    build_arms: Callable[[int], tuple[CraftedStream, CraftedStream]],
    adapt_and_eval: Callable[[CraftedStream, RunCondition], float],
    fluency_scorer: Callable[[CraftedStream, CraftedStream], float] | None = None,
) -> SpikeResult:
    """Execute the full pre-registered comparison.

    For each seed: build a poison stream and a length-matched control, run both,
    evaluate benign loss after each. Then compute the corruption metric across
    seeds and compare against the pre-registered thresholds.

    Does **not** decide go/no-go, and does **not** write a report. It reports;
    the threshold lives in PREREGISTERED.md and the comparison is made
    explicitly by the caller, so the bar cannot be quietly relaxed inside the
    code that produces the number. Report generation belongs to the CLI layer.

    Args:
        build_arms: seed -> (poison, control). Injected so the orchestration is
            testable without a corpus, and so `craft_stream` can supply an
            adversarially-ordered poison arm later.
        adapt_and_eval: stream, condition -> benign loss after adapting on that
            stream. The one genuinely model-dependent step. Build it from
            `trustgate.eval.carry.run_sequences` plus the vendor inner step.
        fluency_scorer: poison, control -> perplexity ratio under an
            *independent* reference model. `None` until T3.2 lands, which leaves
            every per-seed ratio `nan` and fails the realism bar safe.
    """
    if not seeds:
        raise ValueError("no seeds: the pre-registration requires 5 per condition")

    poisoned_losses: list[float] = []
    control_losses: list[float] = []
    fluency: list[float] = []
    strategy: StreamStrategy | None = None

    for seed in seeds:
        poison, control = build_arms(seed)
        strategy = poison.strategy

        assert_streams_matched(poison, control)
        assert_stream_not_from_eval_split(poison, condition.benign_eval_split)
        assert_stream_not_from_eval_split(control, condition.benign_eval_split)

        seed_condition = replace(condition, seed=seed)
        poison_condition = RunCondition.from_stream(
            poison,
            seed=seed,
            seq_length=seed_condition.seq_length,
            checkpoint=seed_condition.checkpoint,
            benign_eval_split=seed_condition.benign_eval_split,
            eval_tokens_sha256=seed_condition.eval_tokens_sha256,
            inner_lr_multiplier=seed_condition.inner_lr_multiplier,
        )
        control_condition = replace(poison_condition, stream_corpus_split=control.corpus_name)
        poison_condition.assert_matches(control_condition)

        poisoned_losses.append(float(adapt_and_eval(poison, poison_condition)))
        control_losses.append(float(adapt_and_eval(control, control_condition)))
        fluency.append(
            float("nan") if fluency_scorer is None else float(fluency_scorer(poison, control))
        )

    return SpikeResult(
        corruption=summarize(poisoned_losses, control_losses),
        # Read off the streams that actually ran rather than off the spec.
        # `PREREGISTERED.md` fixes SELECT as the headline and says PARAPHRASE and
        # SOFT "do not substitute for" it, so a report must name what ran.
        strategy=strategy or StreamStrategy.SELECT,
        per_seed_fluency=fluency,
        per_seed_poisoned=poisoned_losses,
        per_seed_control=control_losses,
    )


def summarize(poisoned: list[float], control: list[float]) -> CorruptionResult:
    """Aggregate per-seed losses. Pure -- unit-tested without a model."""
    return corruption_metric(np.asarray(poisoned), np.asarray(control))


if __name__ == "__main__":
    # `python -m trustgate.eval.harness` is the invocation
    # `experiments/001-attack-spike/README.md` documents. The implementation
    # lives in `cli.py` so that threshold loading and report writing stay out of
    # this module -- `run_attack_spike` must not transitively decide the verdict.
    from trustgate.eval.cli import main

    raise SystemExit(main())
