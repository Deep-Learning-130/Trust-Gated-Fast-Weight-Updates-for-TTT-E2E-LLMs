"""Sequence-position evaluation: where in a stream the damage appears.

A sibling of `harness.run_attack_spike`, deliberately **not** a modification of
it. The kill-gate path stays byte-identical; every arm defined here is secondary
and non-gating, recorded in the `PREREGISTERED.md` addendum dated 2026-09-20.

What the frozen spike cannot tell you
-------------------------------------
`run_attack_spike` measures benign loss once, after the whole stream, and
compares poison against a length-matched benign control. Two things are
invisible to it:

1. **The floor.** Fast weights drift on *any* input, so poison-vs-control
   measures harm above a drift floor that is never itself measured. The `floor`
   arm reads nothing at all and evaluates the same held-out probe.
2. **Whether accumulation matters.** The threat model accuses carry across
   windows specifically -- text that is harmless to read but harmful to have
   read. The `*_no_carry` arms reset the carry before every window, so harm that
   survives the reset was never an accumulation effect at all.

Both no-carry arms exist, not just the poison one. An unmatched reference is
precisely the confound `RunCondition` was built to prevent, and it would be
strange to import that discipline for one comparison and drop it for the next.

Why not `carry.run_sequences`
-----------------------------
It is unused in the real path. `harness.run_stream` (`harness.py:342-350`)
bypasses it because `chunk_tokens`' flat slicing cannot build the
`(suffix_chunk, prefix_chunk)` pairs the vendor step consumes -- a prefix pass
over the whole sequence has to happen first. So the chunking arrives here as an
injected `windows_for`, which the vendor path builds from `vendor_bind` and the
002 pilot builds from `chunk_tokens`. Same orchestrator, two backends.

Measuring mid-stream is safe
----------------------------
Only because `eval_benign_curve` discards the carry it computes
(`harness.py:412`). Its docstring asks that nobody "optimise" that away; this
module is the reason that request now has teeth, and
`test_mid_stream_measurement_does_not_contaminate_the_carry` pins it.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Callable

from trustgate.attack.stream import (
    CraftedStream,
    assert_stream_not_from_eval_split,
    assert_streams_matched,
)
from trustgate.eval import carry as carry_mod
from trustgate.eval.carry import CarryState, InnerStep
from trustgate.eval.harness import RunCondition

#: The threaded arms, in report order. Names are load-bearing -- `report.py` and
#: `metrics.accumulation_excess` address arms by these strings.
POISON = "poison"
CONTROL = "control"
POISON_NO_CARRY = "poison_no_carry"
CONTROL_NO_CARRY = "control_no_carry"

ARM_NAMES = (POISON, CONTROL, POISON_NO_CARRY, CONTROL_NO_CARRY)

#: Arms whose carry is reset before each window.
NO_CARRY_ARMS = (POISON_NO_CARRY, CONTROL_NO_CARRY)


@dataclass(frozen=True)
class ArmTrace:
    """One arm's benign loss as a function of position in the stream."""

    name: str
    per_window_loss: tuple[float, ...]
    """One entry per *measured* window. With `eval_every > 1` this is shorter
    than the window count; `window_index` maps an entry back to its window."""

    eval_every: int

    def window_index(self, i: int) -> int:
        """Which window produced measurement `i`. Zero-based, as windows are."""
        return (i + 1) * self.eval_every - 1

    @property
    def final_loss(self) -> float:
        """The end-of-stream value -- what `run_attack_spike` measures."""
        if not self.per_window_loss:
            raise ValueError(f"arm {self.name!r} has no measurements")
        return self.per_window_loss[-1]


@dataclass(frozen=True)
class SequenceResult:
    """Every arm, every seed. Renders no verdict and touches no frozen bar."""

    n_windows: int
    eval_every: int
    floor_loss: float
    """Benign loss having read nothing at all. Seed-independent by construction:
    a fresh carry is a fresh carry."""

    per_seed: dict[int, dict[str, ArmTrace]]

    @property
    def seeds(self) -> tuple[int, ...]:
        return tuple(sorted(self.per_seed))

    def arm(self, seed: int, name: str) -> ArmTrace:
        return self.per_seed[seed][name]

    def final_losses(self, name: str) -> list[float]:
        """Per-seed end-of-stream loss for one arm, in seed order.

        This is the quantity `corruption_metric` consumes, which is what lets
        `test_final_position_equals_the_single_shot_endpoint` hold the two
        orchestrators to the same number.
        """
        return [self.per_seed[s][name].final_loss for s in self.seeds]


def _run_threaded(
    windows,
    step_fn: InnerStep,
    fresh_carry: Callable[[], CarryState],
    evaluate: Callable[[CarryState, RunCondition], float],
    condition: RunCondition,
    eval_every: int,
    name: str,
) -> ArmTrace:
    """Carry survives every window boundary. The ordinary TTT serving picture."""
    carry = fresh_carry()
    losses: list[float] = []

    for i, window in enumerate(windows):
        carry, _ = carry_mod.run_chunks(carry, window, step_fn)
        if (i + 1) % eval_every == 0:
            losses.append(float(evaluate(carry, condition)))

    return ArmTrace(name=name, per_window_loss=tuple(losses), eval_every=eval_every)


def _run_no_carry(
    windows,
    step_fn: InnerStep,
    fresh_carry: Callable[[], CarryState],
    evaluate: Callable[[CarryState, RunCondition], float],
    condition: RunCondition,
    eval_every: int,
    name: str,
) -> ArmTrace:
    """Fresh carry before every window. Accumulation is switched off.

    The stream content is identical to the threaded arm's -- window `i` sees
    exactly the tokens it would have seen there. The *only* difference is what
    it starts from, which is the whole point.
    """
    losses: list[float] = []

    for i, window in enumerate(windows):
        carry, _ = carry_mod.run_chunks(fresh_carry(), window, step_fn)
        if (i + 1) % eval_every == 0:
            losses.append(float(evaluate(carry, condition)))

    return ArmTrace(name=name, per_window_loss=tuple(losses), eval_every=eval_every)


def run_sequence_eval(
    condition: RunCondition,
    seeds: list[int],
    *,
    build_arms: Callable[[int], tuple[CraftedStream, CraftedStream]],
    windows_for: Callable[[CraftedStream], list],
    step_fn: InnerStep,
    fresh_carry: Callable[[], CarryState],
    evaluate: Callable[[CarryState, RunCondition], float],
    eval_every: int = 1,
    ledger=None,
) -> SequenceResult:
    """Run every arm across every seed and return the position curves.

    Decides nothing. Like `run_attack_spike` it reports, and for the same
    reason: a number produced by the same code that judges it can be quietly
    judged more kindly. Unlike `run_attack_spike` there is no bar to judge it
    against, because these arms are non-gating by construction.

    Args:
        condition: the base run condition. Per-arm conditions are derived from
            the streams via `RunCondition.from_stream` rather than transcribed.
        build_arms: seed -> (poison, control). Same contract as
            `run_attack_spike`, so a caller can feed both from one builder.
        windows_for: stream -> list of windows, each a list of inner-step
            chunks. The backend seam; see the module docstring.
        step_fn: the inner step. `vendor_bind.make_step_fn` or the pilot's.
        fresh_carry: () -> a carry at zero steps. Called once per no-carry
            window, so it must be cheap and must not be a captured singleton.
        evaluate: (carry, condition) -> benign loss on the held-out probe. Must
            not mutate the carry it is handed.
        eval_every: measure after every Nth window. The cost knob: four arms
            times `n_windows` evaluations per seed is not free.
        ledger: optional `trustgate.eval.ledger.Ledger`. Each finished arm is
            recorded before the next starts, and a rerun reuses it.

    Returns:
        A `SequenceResult`.
    """
    if not seeds:
        raise ValueError("no seeds: the pre-registration requires 5 per condition")
    if eval_every < 1:
        raise ValueError(f"eval_every must be >= 1, got {eval_every}")

    # The same check `eval_benign_curve` makes, hoisted to before any compute:
    # an unsaturated inner LR produces a null result for a reason that has
    # nothing to do with the attack, and finding that out after four arms have
    # run is finding it out too late.
    carry_mod.assert_saturated_inner_lr(fresh_carry())

    from trustgate.eval.ledger import condition_key, eval_key

    floor_key = "sequence/floor:" + condition_key(condition)
    cached_floor = ledger.get(floor_key) if ledger is not None else None
    if cached_floor is not None:
        floor_loss = float(cached_floor["loss"])
    else:
        floor_loss = float(evaluate(fresh_carry(), condition))
        if ledger is not None:
            ledger.put(floor_key, "sequence/floor", {"loss": floor_loss})

    def arm(run, windows, stream, arm_condition, name):
        key = cached = None
        if ledger is not None:
            key = eval_key("sequence/arm", stream, arm_condition, name,
                           f"eval_every={eval_every}", f"windows={len(windows)}")
            cached = ledger.get(key)
        if cached is not None:
            return ArmTrace(name=name, per_window_loss=tuple(float(x) for x in cached["losses"]),
                            eval_every=eval_every)
        trace = run(windows, step_fn, fresh_carry, evaluate, arm_condition, eval_every, name)
        if ledger is not None:
            ledger.put(key, "sequence/arm",
                       {"losses": list(trace.per_window_loss), "arm": name, "seed": arm_condition.seed})
        return trace

    per_seed: dict[int, dict[str, ArmTrace]] = {}
    n_windows: int | None = None

    for seed in seeds:
        poison, control = build_arms(seed)

        assert_streams_matched(poison, control)
        assert_stream_not_from_eval_split(poison, condition.benign_eval_split)
        assert_stream_not_from_eval_split(control, condition.benign_eval_split)

        poison_condition = RunCondition.from_stream(
            poison,
            seed=seed,
            seq_length=condition.seq_length,
            checkpoint=condition.checkpoint,
            benign_eval_split=condition.benign_eval_split,
            eval_tokens_sha256=condition.eval_tokens_sha256,
            inner_lr_multiplier=condition.inner_lr_multiplier,
        )
        control_condition = replace(
            poison_condition, stream_corpus_split=control.corpus_name
        )
        poison_condition.assert_matches(control_condition)

        poison_windows = windows_for(poison)
        control_windows = windows_for(control)

        # `assert_streams_matched` matches token counts, but `windows_for` is
        # injected and could chunk two equal-length streams differently. A
        # poison curve read against a control curve of another length would be
        # a silent mismatch, so check the thing actually being compared.
        if len(poison_windows) != len(control_windows):
            raise ValueError(
                f"seed {seed}: poison chunked into {len(poison_windows)} windows "
                f"but control into {len(control_windows)}; the arms are not "
                f"comparable position by position"
            )
        if not poison_windows:
            raise ValueError(
                f"seed {seed}: `windows_for` produced no windows; a stream of "
                f"{poison.length_tokens} tokens cannot be measured"
            )

        if n_windows is None:
            n_windows = len(poison_windows)
        elif n_windows != len(poison_windows):
            raise ValueError(
                f"seed {seed} produced {len(poison_windows)} windows but an "
                f"earlier seed produced {n_windows}; per-position curves across "
                f"seeds would not line up"
            )

        per_seed[seed] = {
            POISON: arm(_run_threaded, poison_windows, poison, poison_condition, POISON),
            CONTROL: arm(_run_threaded, control_windows, control, control_condition, CONTROL),
            POISON_NO_CARRY: arm(
                _run_no_carry, poison_windows, poison, poison_condition, POISON_NO_CARRY
            ),
            CONTROL_NO_CARRY: arm(
                _run_no_carry, control_windows, control, control_condition, CONTROL_NO_CARRY
            ),
        }

    return SequenceResult(
        n_windows=int(n_windows or 0),
        eval_every=eval_every,
        floor_loss=floor_loss,
        per_seed=per_seed,
    )


def windows_from_chunks(chunks: list, chunks_per_window: int = 1) -> list:
    """Group a flat chunk list into windows. The trivial `windows_for` helper.

    A window of one chunk is the finest granularity the fast weights actually
    move at (one inner SGD step per `mini_batch_size` tokens), and therefore the
    default.
    """
    if chunks_per_window < 1:
        raise ValueError(f"chunks_per_window must be >= 1, got {chunks_per_window}")

    n = len(chunks)
    if n % chunks_per_window != 0:
        raise ValueError(
            f"{n} chunks is not a multiple of chunks_per_window "
            f"{chunks_per_window}; a ragged final window would be measured at a "
            f"different dose than every other one"
        )

    return [
        chunks[i : i + chunks_per_window] for i in range(0, n, chunks_per_window)
    ]


def make_vendor_windows(binding: Any, condition: RunCondition, *, bos_token_id: int):
    """Build `windows_for` for the real vendor path.

    Mirrors `harness.run_stream` (`harness.py:342-350`) rather than
    `carry.run_sequences`: the stream is split into sequences, each sequence
    gets its prefix pass, and the resulting `(suffix, prefix)` chunk pairs are
    what the inner step consumes. Flat slicing cannot produce those.
    """
    from trustgate.eval import vendor_bind

    def windows_for(stream: CraftedStream) -> list:
        chunks: list = []
        # `stream.tokens` is `length_tokens + 1` by definition (CraftedStream).
        for tokens in vendor_bind.split_stream(
            stream.tokens, condition.seq_length, lookahead=True
        ):
            chunks.extend(
                vendor_bind.sequence_chunks(binding, tokens, bos_token_id=bos_token_id)
            )
        return windows_from_chunks(chunks)

    return windows_for
