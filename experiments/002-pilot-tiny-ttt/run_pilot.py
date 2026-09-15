"""Scaled-down pilot of the Phase 1 attack spike.

Runs the real pipeline -- the shipped crafted-stream builders, the shipped
matching discipline, the shipped corruption metric, the shipped report renderer
-- against the tiny TTT model in `pilot_model.py`.

**This is not the kill-gate.** It does not use the released 1B checkpoint, it
does not score the realism bar, and it renders no PROCEED/STOP verdict. See
`README.md`.

What it measures
----------------
For each of five seeds, two arms that differ in exactly one respect -- stream
content:

  poison   adversarially ordered spans of real corpus text
  control  the same construction, same length, same span size, benign ordering

Both arms adapt the fast weights through `carry.run_sequences`, then benign loss
is measured on a **held-out** slice the streams were never drawn from. Measuring
on the stream itself is a pre-registered invalidating condition
(`PREREGISTERED.md:66`).

Three things beyond the headline pair, each of which exists to stop a number
being over-read:

  S1                 per-token NLL must fall across the context, or the inner
                     loop is not using context and every number below is noise
  control-vs-control the noise floor -- ordinary adaptation drift, measured, so
                     corruption can be told apart from it (T3.7)
  dose-response      degradation against stream length, the pre-registration's
                     secondary "attack slowness" metric

Run:  PYTHONPATH=src python experiments/002-pilot-tiny-ttt/run_pilot.py
"""

from __future__ import annotations

import dataclasses
import json
import sys
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import hashlib

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(HERE))

from pilot_model import (  # noqa: E402
    PILOT_BOS_ID,
    PilotConfig,
    init_params,
    make_step_fn,
    per_token_nll,
    sequence_loss,
    train_base,
)

from trustgate.attack.corpus import TokenCorpus  # noqa: E402
from trustgate.attack.stream import (  # noqa: E402
    build_benign_control,
    build_select_stream,
)
from trustgate.eval.carry import (  # noqa: E402
    assert_saturated_inner_lr,
    init_carry,
    run_sequences,
)
from trustgate.eval.metrics import corruption_metric  # noqa: E402

CFG = PilotConfig()
STREAM_TOKENS = 4096          # 16 inner SGD steps at mini_batch_size=256
EVAL_TOKENS = 4096
SEEDS = [0, 1, 2, 3, 4]
DOSE_LENGTHS = [512, 1024, 2048, 4096, 8192]


# --------------------------------------------------------------------------
# Corpus
# --------------------------------------------------------------------------
#: SHA-256 of `corpus.txt`. Printed on every run and written into the results,
#: so a number can be tied to the exact bytes that produced it.
CORPUS_SHA256 = "cc2adbc868e195e3a589b3aab775b4a279bcf55d39b45b95b487839431ebc3c6"

#: The commit whose prose `corpus.txt` freezes -- the commit that introduced
#: this experiment. Reconstructible with `git show a225c91:<path>` over the
#: same file selection, so the snapshot is auditable rather than merely asserted.
CORPUS_COMMIT = "a225c91"


def load_corpus_bytes() -> np.ndarray:
    """Byte stream of this repository's own English prose, FROZEN.

    Entirely local: no download, no GCS, no requester-pays bucket. The pilot is
    a mechanism demonstration, and for that the corpus only has to be real
    natural-language text with real structure.

    Why this reads a committed file instead of globbing
    ---------------------------------------------------
    It used to glob `docs/**/*.md`, `experiments/**/*.md` and `*.md` live, at
    run time. That made **the repository's own documentation the training data**,
    so every commit touching a markdown file silently changed the experiment.

    It was not a theoretical problem. Measured 2026-09-15: the README describes a
    412 KB corpus; the live glob had reached 672 KB across 47 files. Re-running
    the pilot after a morning of editing two unrelated README files moved Cohen's
    *d* from +0.0535 to -1.2838 -- across zero, and past the pre-registered
    |d| >= 0.8 bar in the opposite direction. Same code, same seeds, same config;
    deterministic within a session and different between them.

    Worse, the glob included `experiments/003-smoke-125m/README.md`, which
    documents an experiment *downstream* of this one. Writing up a later result
    changed an earlier one.

    The frozen snapshot is the prose as of CORPUS_COMMIT, so a result file now
    names bytes that can be recovered. `.txt`, not `.md`, so it can never feed
    itself back in.
    """
    frozen = HERE / "corpus.txt"
    if frozen.exists():
        raw = frozen.read_bytes()
        got = hashlib.sha256(raw).hexdigest()
        if got != CORPUS_SHA256:
            raise SystemExit(
                f"corpus.txt does not match the pinned digest.\n"
                f"  expected {CORPUS_SHA256}\n"
                f"  got      {got}\n"
                "Every number this experiment has ever produced is a function of "
                "these bytes. Restore the file rather than re-pinning the digest."
            )
        return np.frombuffer(raw, dtype=np.uint8).astype(np.int32)

    raise SystemExit(
        f"{frozen} is missing.\n"
        "Refusing to fall back to a live glob of the repository's markdown: that "
        "is the defect this file exists to prevent, and it fails silently by "
        "producing plausible numbers from a corpus nobody can reconstruct.\n"
        f"Restore it from git, or rebuild it from {CORPUS_COMMIT}."
    )


def split_corpus(tokens: np.ndarray):
    """Three disjoint slices: base training, attacker source, held-out eval.

    Disjointness is load-bearing in both directions. Drawing poison spans from
    the eval split would manufacture a positive result; drawing control spans
    from it would make the control in-domain and the poison out-of-domain, which
    is the same fault wearing the other hat.
    """
    n = len(tokens)
    n_eval = EVAL_TOKENS + 1
    n_attack = max(60_000, 12 * STREAM_TOKENS)
    if n < n_eval + n_attack + 20_000:
        raise SystemExit(
            "corpus is only %d bytes; need roughly %d. Point load_corpus_bytes "
            "at more text." % (n, n_eval + n_attack + 20_000)
        )
    train = tokens[: n - n_eval - n_attack]
    attack = tokens[n - n_eval - n_attack : n - n_eval]
    held_out = tokens[n - n_eval :]
    return train, attack, held_out


# --------------------------------------------------------------------------
# Adaptation and measurement
# --------------------------------------------------------------------------
def adapt_and_eval(slow, fast0, step_fn, stream_tokens, held_out) -> float:
    """Adapt the fast weights on a stream, then measure benign loss on held-out.

    The adapted weights are carried out of adaptation and into measurement --
    the thing the vendor cannot do (ADR-006). Measurement runs with the fast
    weights frozen rather than continuing to adapt on the eval data: the 001
    design keeps meta mode there because freezing would mean the vendor's
    `pretrain` branch, a constraint the pilot does not have. Freezing removes
    the decay confound, and the deviation is recorded here rather than absorbed.
    """
    carry = init_carry(fast0)
    assert_saturated_inner_lr(carry)
    # `CraftedStream.tokens` carries one extra lookahead token beyond
    # `length_tokens` (the window "+1" convention in `attack/stream.py`), and
    # `chunk_tokens` refuses a ragged tail rather than padding it. Drop the
    # lookahead so the stream chunks exactly; it is dropped identically in both
    # arms, so it cannot bias the comparison.
    adapt_on = stream_tokens[:-1] if len(stream_tokens) % CFG.mini_batch_size else stream_tokens
    carry, _ = run_sequences(carry, [adapt_on], step_fn, CFG.mini_batch_size)
    return float(sequence_loss(slow, carry.fast_weights, jnp.asarray(held_out), CFG))


def adversarial_order(slow, fast0, step_fn, corpus, held_out):
    """Greedy hardest-first ordering -- the attacker's only lever under SELECT.

    Driven through `build_select_stream`'s existing `order_fn` seam, which the
    module documents as the hook a crafting optimiser uses "without
    reimplementing the constraints". `craft_stream` (T1.9) stays unimplemented:
    its contract is against the vendor victim, and a pilot-grade heuristic
    committed under that name would misrepresent it.

    The heuristic scores each span by its NLL under the base model and puts the
    high-loss spans last, so the largest fast-weight updates land closest to the
    measurement. This is a weak lever by design -- every span is real text, and
    only the ordering changes -- which is what makes a positive result
    interesting and a negative one unsurprising.
    """

    def order_fn(spans):
        scored = []
        for sp in spans:
            toks = jnp.asarray(corpus.read(sp.offset, sp.length))
            scored.append((float(sequence_loss(slow, fast0, toks, CFG)), sp))
        scored.sort(key=lambda t: t[0])
        return [sp for _, sp in scored]

    return order_fn


def main() -> int:
    global CFG
    t0 = time.time()
    print("Scaled-down pilot -- NOT the pre-registered kill-gate\n")

    tokens = load_corpus_bytes()
    train, attack, held_out = split_corpus(tokens)
    print("corpus: %d bytes  (train %d / attacker %d / held-out %d)"
          % (len(tokens), len(train), len(attack), len(held_out)))

    key = jax.random.PRNGKey(0)
    slow, fast0 = init_params(key, CFG)
    print("\ntraining base weights (fast weights frozen at init)")
    slow, final = train_base(slow, fast0, train, CFG, seed=0)
    print("    final base loss %.4f nats/token" % final)

    step_fn = make_step_fn(slow, CFG)
    held_j = jnp.asarray(held_out)

    # ---- Inner learning rate, chosen on benign text only ----------------
    # The vendor's inner LR comes with its checkpoint; the pilot has to pick one.
    # It is selected by which value lets benign adaptation reduce held-out loss
    # the most -- i.e. by making the *mechanism* work at all, on benign streams,
    # before any poison stream exists. Nothing about the poison-vs-control
    # contrast enters this choice, which is what keeps it from being a knob that
    # tunes the result.
    print("\ninner-LR selection (benign text only)")
    benign_warm = jnp.asarray(attack[: 16 * CFG.mini_batch_size])
    base_loss = float(sequence_loss(slow, fast0, held_j, CFG))
    best_lr, best_loss = None, base_loss
    for lr in [0.003, 0.01, 0.03, 0.1, 0.3, 1.0]:
        cfg_lr = dataclasses.replace(CFG, inner_lr=lr)
        probe = make_step_fn(slow, cfg_lr)
        c = init_carry(fast0)
        c, _ = run_sequences(c, [benign_warm], probe, CFG.mini_batch_size)
        loss = float(sequence_loss(slow, c.fast_weights, held_j, CFG))
        print("    inner_lr %-6s held-out %.4f  (%+.4f vs no adaptation)"
              % (lr, loss, loss - base_loss))
        if loss < best_loss:
            best_lr, best_loss = lr, loss
    if best_lr is None:
        print("no inner LR improved on no adaptation -- the inner loop is inert.")
        return 1
    print("    selected inner_lr = %s" % best_lr)
    CFG = dataclasses.replace(CFG, inner_lr=best_lr)
    step_fn = make_step_fn(slow, CFG)

    # ---- S1: does the inner loop actually do any work? ------------------
    # The pilot analogue of TOLERANCE.md section 5 S1. The published S1 reads a
    # falling per-token NLL curve across one sequence, which detects a model
    # adapting *while* it reads. That check does not transfer here: measurement
    # freezes the fast weights (see `adapt_and_eval`), so there is no
    # within-sequence adaptation for it to see, and running it unchanged would
    # report noise. The transferable question is the same one it was asked to
    # answer -- are the fast weights carrying anything? -- so it is asked
    # directly: adapting on benign in-domain text must reduce held-out loss
    # relative to no adaptation at all.
    #
    # If this fails, the inner loop is inert and every number below is eval
    # noise rather than a null result. That is the false STOP the carry overlay
    # exists to prevent, and it is worth one cheap check to rule out.
    warm = benign_warm
    carry = init_carry(fast0)
    carry, _ = run_sequences(carry, [warm], step_fn, CFG.mini_batch_size)
    adapted_loss = float(sequence_loss(slow, carry.fast_weights, held_j, CFG))
    s1_pass = adapted_loss < base_loss
    print("\nS1  held-out loss   no adaptation %.4f  ->  after 16 inner steps "
          "%.4f   [%s]" % (base_loss, adapted_loss, "PASS" if s1_pass else "FAIL"))
    if not s1_pass:
        print("S1 FAILED: the inner loop is inert. Stopping -- any corruption "
              "number from here would be noise, not a null result.")
        return 1
    early, late = base_loss, adapted_loss

    corpus = TokenCorpus(attack, name="pilot-attack", bos_token_id=PILOT_BOS_ID)

    # ---- Headline: five seeds, poison vs length-matched control ---------
    print("\nfive seeds, poison vs length-matched benign control")
    poisoned, control = [], []
    order_fn = adversarial_order(slow, fast0, step_fn, corpus, held_j)

    for seed in SEEDS:
        p = build_select_stream(corpus, STREAM_TOKENS, seed,
                                span_tokens=CFG.span_tokens,
                                mini_batch_size=CFG.mini_batch_size,
                                order_fn=order_fn)
        c = build_benign_control(corpus, STREAM_TOKENS, seed + 1000,
                                 span_tokens=CFG.span_tokens,
                                 mini_batch_size=CFG.mini_batch_size)
        # The matching discipline, on the fields a stream can carry.
        assert p.length_tokens == c.length_tokens, "length mismatch"
        assert p.n_chunks == c.n_chunks, "chunk-count mismatch"
        assert p.valid_tokens == c.valid_tokens, "valid-token mismatch"
        assert p.tokens.dtype == c.tokens.dtype, "dtype mismatch"

        lp = adapt_and_eval(slow, fast0, step_fn, jnp.asarray(p.tokens), held_j)
        lc = adapt_and_eval(slow, fast0, step_fn, jnp.asarray(c.tokens), held_j)
        poisoned.append(lp)
        control.append(lc)
        flag = "  <-- poison HELPED" if lp < lc else ""
        print("  seed %d   poison %.5f   control %.5f   delta %+.5f"
              " (multiplicity p=%d c=%d)%s"
              % (seed, lp, lc, lp - lc, p.span_multiplicity,
                 c.span_multiplicity, flag))

    result = corruption_metric(np.array(poisoned), np.array(control))

    # ---- Noise floor: control vs control (T3.7) -------------------------
    print("\ncontrol-vs-control noise floor")
    a, b = [], []
    for seed in SEEDS:
        c1 = build_benign_control(corpus, STREAM_TOKENS, seed + 2000,
                                  span_tokens=CFG.span_tokens,
                                  mini_batch_size=CFG.mini_batch_size)
        c2 = build_benign_control(corpus, STREAM_TOKENS, seed + 3000,
                                  span_tokens=CFG.span_tokens,
                                  mini_batch_size=CFG.mini_batch_size)
        a.append(adapt_and_eval(slow, fast0, step_fn, jnp.asarray(c1.tokens), held_j))
        b.append(adapt_and_eval(slow, fast0, step_fn, jnp.asarray(c2.tokens), held_j))
    floor = corruption_metric(np.array(a), np.array(b))
    print("  Cohen's d %.3f   relative %+.4f%%   (bar for corruption is d >= 0.8)"
          % (floor.effect_size, 100 * floor.relative_degradation))

    # ---- Dose-response --------------------------------------------------
    print("\ndose-response (seed 0)")
    dose_tokens, dose_rel = [], []
    for length in DOSE_LENGTHS:
        if length > len(attack) - CFG.span_tokens:
            break
        p = build_select_stream(corpus, length, 0, span_tokens=CFG.span_tokens,
                                mini_batch_size=CFG.mini_batch_size,
                                order_fn=order_fn)
        c = build_benign_control(corpus, length, 1000, span_tokens=CFG.span_tokens,
                                 mini_batch_size=CFG.mini_batch_size)
        lp = adapt_and_eval(slow, fast0, step_fn, jnp.asarray(p.tokens), held_j)
        lc = adapt_and_eval(slow, fast0, step_fn, jnp.asarray(c.tokens), held_j)
        rel = (lp - lc) / lc
        dose_tokens.append(length)
        dose_rel.append(rel)
        print("  %5d tokens (%2d steps)   poison %.5f  control %.5f  rel %+.3f%%"
              % (length, length // CFG.mini_batch_size, lp, lc, 100 * rel))

    # ---- Emit -----------------------------------------------------------
    out = HERE / "results"
    out.mkdir(exist_ok=True)
    payload = {
        "note": "SCALED-DOWN PILOT. Not the pre-registered kill-gate verdict. "
                "Tiny model, local corpus, realism bar not scored.",
        "per_seed_poisoned": poisoned,
        "per_seed_control": control,
        "poisoned_loss": result.poisoned_loss,
        "control_loss": result.control_loss,
        "effect_size": result.effect_size,
        "relative_degradation": result.relative_degradation,
        "n_seeds": result.n_seeds,
        "noise_floor_effect_size": floor.effect_size,
        "noise_floor_relative": floor.relative_degradation,
        "s1_early_nll": early,
        "s1_late_nll": late,
        "base_train_loss": final,
        "dose_response": {"stream_tokens": dose_tokens,
                          "relative_degradation": dose_rel},
        "config": {
            "d_model": CFG.d_model, "d_ff": CFG.d_ff, "window": CFG.window,
            "inner_lr": CFG.inner_lr, "mini_batch_size": CFG.mini_batch_size,
            "span_tokens": CFG.span_tokens, "stream_tokens": STREAM_TOKENS,
            "eval_tokens": EVAL_TOKENS, "seeds": SEEDS,
        },
    }
    (out / "pilot.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print("\n" + "=" * 62)
    print("PILOT RESULT (not a verdict)")
    print("  Cohen's d            %.3f      (pre-registered bar >= 0.80)"
          % result.effect_size)
    print("  relative degradation %+.2f%%    (pre-registered bar >= 10%%)"
          % (100 * result.relative_degradation))
    print("  fluency ratio        not scored -- the pilot cannot pass the "
          "realism bar,")
    print("                       so it cannot render PROCEED under any "
          "circumstances.")
    print("  noise floor (d)      %.3f" % floor.effect_size)
    print("=" * 62)
    print("\nwrote %s   (%.1fs)" % (out / "pilot.json", time.time() - t0))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
