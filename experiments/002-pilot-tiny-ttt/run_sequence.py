"""Sequence-position arms on the tiny pilot -- NOT a verdict, NOT the kill-gate.

Everything `run_pilot.py`'s README refuses to claim, this refuses too: the model
is a ~0.1M-parameter byte-level stand-in, the corpus is this repository's own
prose, there is no attention, and the realism bar is unscored by construction.
It renders no verdict and touches no pre-registered bar.

What it is for
--------------
`trustgate.eval.sequence` is orchestration, and orchestration that has only ever
run against fakes has only ever been proven self-consistent. This runs it
against a model that actually adapts, on a real (if tiny) corpus, for $0 on CPU
-- so that the first time the arms meet a GPU, the only thing missing is
weights.

It is also where `--eval-every` gets chosen. Four arms times fifty windows is a
lot of measurement; find out here what the curves look like at stride 1 before
paying for stride 1 on an A100.

Usage
-----
    PYTHONPATH=src python experiments/002-pilot-tiny-ttt/run_sequence.py
"""

from __future__ import annotations

import dataclasses
import json
import sys
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(HERE))

from pilot_model import (  # noqa: E402
    PILOT_BOS_ID,
    init_params,
    make_step_fn,
    sequence_loss,
    train_base,
)
from run_pilot import CFG, load_corpus_bytes, split_corpus  # noqa: E402

from trustgate.attack.corpus import TokenCorpus  # noqa: E402
from trustgate.attack.stream import (  # noqa: E402
    build_benign_control,
    build_select_stream,
)
from trustgate.eval.carry import chunk_tokens, init_carry  # noqa: E402
from trustgate.eval.harness import RunCondition, eval_tokens_digest  # noqa: E402
from trustgate.eval.metrics import (  # noqa: E402
    accumulation_excess,
    drift_floor_excess,
    onset_window,
)
from trustgate.eval.report import write_sequence_report  # noqa: E402
from trustgate.eval.sequence import (  # noqa: E402
    CONTROL,
    POISON,
    POISON_NO_CARRY,
    run_sequence_eval,
    windows_from_chunks,
)

N_WINDOWS = 50
STREAM_TOKENS = N_WINDOWS * CFG.mini_batch_size  # 50 inner steps
EVAL_TOKENS = 4096
SEEDS = [0, 1, 2, 3, 4]
EVAL_EVERY = 1
ONSET_DELTA = 0.005

NOT_A_VERDICT = (
    "PILOT SEQUENCE ARMS -- NOT A VERDICT AND NOT THE KILL-GATE. A "
    "0.1M-parameter byte-level stand-in on this repository's own prose, with "
    "the realism bar unscored by construction. Secondary and non-gating even "
    "on the real model (PREREGISTERED.md, Addendum 2026-09-20)."
)


def main() -> int:
    t0 = time.time()
    print(NOT_A_VERDICT + "\n")

    tokens = load_corpus_bytes()
    # `split_corpus` sizes the attacker slice off `run_pilot.STREAM_TOKENS`,
    # which is shorter than ours. Its floor of 60k bytes covers 50 windows of
    # 256 tokens with room to spare; assert rather than assume.
    train, attack, held_out = split_corpus(tokens)
    if len(attack) < 2 * STREAM_TOKENS:
        raise SystemExit(
            "attacker slice is %d bytes; %d windows needs at least %d"
            % (len(attack), N_WINDOWS, 2 * STREAM_TOKENS)
        )
    print(
        "corpus: %d bytes  (train %d / attacker %d / held-out %d)"
        % (len(tokens), len(train), len(attack), len(held_out))
    )

    key = jax.random.PRNGKey(0)
    slow, fast0 = init_params(key, CFG)
    print("\ntraining base weights (fast weights frozen at init)")
    slow, final = train_base(slow, fast0, train, CFG, seed=0)
    print("    final base loss %.4f nats/token" % final)

    held_j = jnp.asarray(held_out)
    base_loss = float(sequence_loss(slow, fast0, held_j, CFG))

    # ---- Inner LR, chosen on benign text only ---------------------------
    # Same discipline as `run_pilot`: nothing about the poison-vs-control
    # contrast enters the choice, so it cannot become a knob that tunes the
    # result. Reproduced rather than imported because `run_pilot` does it
    # inside `main`.
    print("\ninner-LR selection (benign text only)")
    benign_warm = jnp.asarray(attack[: 16 * CFG.mini_batch_size])
    cfg = CFG
    best_lr, best_loss = None, base_loss
    for lr in [0.003, 0.01, 0.03, 0.1, 0.3, 1.0]:
        probe_cfg = dataclasses.replace(CFG, inner_lr=lr)
        probe = make_step_fn(slow, probe_cfg)
        carry, _ = _run_flat(init_carry(fast0), benign_warm, probe, CFG.mini_batch_size)
        loss = float(sequence_loss(slow, carry.fast_weights, held_j, CFG))
        print(
            "    inner_lr %-6s held-out %.4f  (%+.4f vs no adaptation)"
            % (lr, loss, loss - base_loss)
        )
        if loss < best_loss:
            best_lr, best_loss = lr, loss

    if best_lr is None:
        print("\nno inner LR improved on no adaptation -- the inner loop is inert.")
        print("Stopping: every curve from here would be eval noise, not a null.")
        return 1

    print("    selected inner_lr = %s" % best_lr)
    cfg = dataclasses.replace(CFG, inner_lr=best_lr)
    step_fn = make_step_fn(slow, cfg)

    # ---- The arms -------------------------------------------------------
    corpus = TokenCorpus(attack, name="pilot-attack", bos_token_id=PILOT_BOS_ID)

    def build_arms(seed):
        poison = build_select_stream(
            corpus,
            STREAM_TOKENS,
            seed,
            span_tokens=cfg.span_tokens,
            mini_batch_size=cfg.mini_batch_size,
        )
        control = build_benign_control(
            corpus,
            STREAM_TOKENS,
            seed + 500,
            span_tokens=cfg.span_tokens,
            mini_batch_size=cfg.mini_batch_size,
        )
        return poison, control

    def windows_for(stream):
        # `CraftedStream.tokens` carries one lookahead token beyond
        # `length_tokens`; drop it so the stream chunks exactly. Dropped
        # identically in every arm, so it cannot bias the comparison.
        flat = stream.tokens[:-1] if len(stream.tokens) % cfg.mini_batch_size else stream.tokens
        return windows_from_chunks(chunk_tokens(flat, cfg.mini_batch_size))

    def evaluate(carry, _condition):
        """Held-out loss with the fast weights frozen.

        The same deviation from 001 that `run_pilot.adapt_and_eval` records:
        the 001 design keeps measurement in meta mode because freezing would
        mean the vendor's `pretrain` branch, a constraint the pilot does not
        have. Freezing removes the decay confound. Recorded, not absorbed.
        """
        return float(sequence_loss(slow, carry.fast_weights, held_j, cfg))

    first_poison, _ = build_arms(SEEDS[0])
    condition = RunCondition.from_stream(
        first_poison,
        seed=SEEDS[0],
        seq_length=STREAM_TOKENS,
        checkpoint="pilot-tiny-ttt (NO CHECKPOINT)",
        benign_eval_split="pilot-held-out",
        eval_tokens_sha256=eval_tokens_digest(held_out),
    )

    print(
        "\n%d windows of %d tokens, %d seeds, measuring every %d"
        % (N_WINDOWS, cfg.mini_batch_size, len(SEEDS), EVAL_EVERY)
    )
    result = run_sequence_eval(
        condition,
        SEEDS,
        build_arms=build_arms,
        windows_for=windows_for,
        step_fn=step_fn,
        fresh_carry=lambda: init_carry(fast0),
        evaluate=evaluate,
        eval_every=EVAL_EVERY,
    )

    # ---- Report ---------------------------------------------------------
    out = HERE / "results"
    report_path = write_sequence_report(
        result, out / "sequence.md", onset_delta=ONSET_DELTA
    )

    print("\n%-18s %s" % ("floor (no stream)", "%.4f" % result.floor_loss))
    for name in (POISON, CONTROL, POISON_NO_CARRY):
        finals = result.final_losses(name)
        print("%-18s %.4f  (mean of %d seeds)" % (name, float(np.mean(finals)), len(finals)))

    print("\nper-seed, at the last window:")
    for seed in result.seeds:
        excess = accumulation_excess(
            result.arm(seed, POISON).per_window_loss,
            result.arm(seed, POISON_NO_CARRY).per_window_loss,
        )
        onset = onset_window(
            result.arm(seed, POISON).per_window_loss,
            result.arm(seed, CONTROL).per_window_loss,
            ONSET_DELTA,
        )
        print(
            "    seed %d  accumulation excess %+.4f   onset %s"
            % (
                seed,
                float(excess[-1]),
                "never" if onset is None else result.arm(seed, POISON).window_index(onset),
            )
        )

    payload = {
        "not_a_verdict": NOT_A_VERDICT,
        "n_windows": result.n_windows,
        "eval_every": result.eval_every,
        "onset_delta": ONSET_DELTA,
        "inner_lr": best_lr,
        "base_loss_no_adaptation": base_loss,
        "floor_loss": result.floor_loss,
        "seeds": list(result.seeds),
        "arms": {
            name: {
                str(seed): list(result.arm(seed, name).per_window_loss)
                for seed in result.seeds
            }
            for name in (POISON, CONTROL, POISON_NO_CARRY, "control_no_carry")
        },
        "drift_floor_excess_control_seed0": list(
            drift_floor_excess(
                result.arm(result.seeds[0], CONTROL).per_window_loss,
                result.floor_loss,
            )
        ),
    }
    out.mkdir(parents=True, exist_ok=True)
    (out / "sequence.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print("\nwrote %s" % report_path)
    print("wrote %s" % (out / "sequence.json"))
    print("\n%.1f s" % (time.time() - t0))
    print("\n" + NOT_A_VERDICT)
    return 0


def _run_flat(carry, tokens, step_fn, mini_batch_size):
    """`run_chunks` over a flat token array. The pilot's chunking is flat."""
    from trustgate.eval.carry import run_chunks

    return run_chunks(carry, chunk_tokens(tokens, mini_batch_size), step_fn)


if __name__ == "__main__":
    raise SystemExit(main())
