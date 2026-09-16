"""The null distribution of the gating effect size.

Why
---
`run_deep.py` returned Cohen's $d = 0.831$ against a pre-registered bar of 0.80,
and a single control-vs-control draw at $|d| = 0.557$. That one draw is enough to
distrust the headline, but it is an anecdote: it says the null *can* be large, not
how often.

This run answers the quantitative question. It repeats the control-vs-control arm
`N_DRAWS` times on disjoint seed groups -- every draw compares two benign streams,
so the true effect is exactly zero by construction -- and reports the distribution
of $|d|$ under that null, including the fraction of draws that would have cleared
the 0.80 bar with no attacker present at all.

That fraction is a false-positive rate for a pre-registered criterion, measured
rather than argued. It is an observation about the instrument; it moves no bar
(Standing Rule 5), and a pilot is not the place to move one.

Run:  PYTHONPATH=src python experiments/002-pilot-tiny-ttt/run_null.py [meta_budget_s]
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

from pilot_model import PILOT_BOS_ID, init_params, make_step_fn, sequence_loss  # noqa: E402
from run_deep import CFG, adapt_and_eval, meta_train  # noqa: E402
from run_pilot import load_corpus_bytes, split_corpus  # noqa: E402

from trustgate.attack.corpus import TokenCorpus  # noqa: E402
from trustgate.attack.stream import build_benign_control  # noqa: E402
from trustgate.eval.carry import init_carry, run_sequences  # noqa: E402
from trustgate.eval.metrics import cohens_d  # noqa: E402
from trustgate.eval.prereg import FROZEN  # noqa: E402

N_DRAWS = 20
SEEDS_PER_DRAW = 5
STREAM_TOKENS = 4096
CACHE = HERE / "results" / "slow_weights.npz"


def load_or_train(slow, fast0, train, budget_s):
    """Meta-trained base weights, cached so repeat runs skip the training."""
    if CACHE.exists():
        z = np.load(CACHE)
        print("reusing cached meta-trained base weights (%s)" % CACHE.name)
        return {k: jnp.asarray(z[k]) for k in z.files}, float("nan"), 0
    slow, final, nsteps = meta_train(slow, fast0, train, CFG, budget_s=budget_s)
    CACHE.parent.mkdir(exist_ok=True)
    np.savez(CACHE, **{k: np.asarray(v) for k, v in slow.items()})
    return slow, final, nsteps


def main() -> int:
    budget = float(sys.argv[1]) if len(sys.argv) > 1 else 150.0
    t0 = time.time()
    print("NULL DISTRIBUTION of the gating effect size")
    print("every draw compares two BENIGN streams -- true effect is zero\n")

    tokens = load_corpus_bytes()
    train, attack, held_out = split_corpus(tokens)

    slow, fast0 = init_params(jax.random.PRNGKey(0), CFG)
    slow, final, nsteps = load_or_train(slow, fast0, train, budget)
    if nsteps:
        print("    %d meta steps, final meta loss %.4f" % (nsteps, final))

    half = len(held_out) // 2
    prefix = jnp.asarray(held_out[:half - (half % CFG.mini_batch_size)])
    suffix = jnp.asarray(held_out[half:])

    # Same inner LR the deep run selected, on benign text, before any stream.
    cfg = dataclasses.replace(CFG, inner_lr=0.03)
    step_fn = make_step_fn(slow, cfg)
    corpus = TokenCorpus(attack, name="null-attack", bos_token_id=PILOT_BOS_ID)

    base = float(sequence_loss(slow, fast0, suffix, cfg))
    adapted = adapt_and_eval(slow, fast0, step_fn, None, prefix, suffix, cfg)
    print("S1  suffix loss %.4f -> %.4f after reading the prefix  [%s]"
          % (base, adapted, "PASS" if adapted < base else "FAIL"))
    if adapted >= base:
        print("S1 FAIL -- inner loop inert; a null distribution measured on it "
              "would describe nothing.")
        return 1

    print("\n%d draws x %d seed-pairs, all benign" % (N_DRAWS, SEEDS_PER_DRAW))
    null_ds, null_rels = [], []
    for draw in range(N_DRAWS):
        a, b = [], []
        for k in range(SEEDS_PER_DRAW):
            s = 10_000 + draw * 100 + k
            c1 = build_benign_control(corpus, STREAM_TOKENS, s,
                                      span_tokens=cfg.span_tokens,
                                      mini_batch_size=cfg.mini_batch_size)
            c2 = build_benign_control(corpus, STREAM_TOKENS, s + 50,
                                      span_tokens=cfg.span_tokens,
                                      mini_batch_size=cfg.mini_batch_size)
            a.append(adapt_and_eval(slow, fast0, step_fn,
                                    jnp.asarray(c1.tokens), prefix, suffix, cfg))
            b.append(adapt_and_eval(slow, fast0, step_fn,
                                    jnp.asarray(c2.tokens), prefix, suffix, cfg))
        a, b = np.array(a), np.array(b)
        d = cohens_d(a, b)
        rel = float((a.mean() - b.mean()) / b.mean())
        null_ds.append(float(d))
        null_rels.append(rel)
        print("  draw %2d   d %+7.3f   |d| %5.3f   rel %+.4f%%   %s   (%.0fs)"
              % (draw, d, abs(d), 100 * rel,
                 "<-- would CLEAR the 0.80 bar" if abs(d) >= FROZEN.min_effect_size
                 else "", time.time() - t0), flush=True)

    absd = np.abs(np.array(null_ds))
    bar = FROZEN.min_effect_size
    fpr = float((absd >= bar).mean())
    # Read from results/deep.json rather than pasted in. The literal that used
    # to sit here -- 0.8313 -- outlived the run that produced it: the corpus was
    # a live glob of the repository's own markdown until 2026-09-15, so the deep
    # run's d moved to -0.003 once the corpus was frozen, while this constant
    # did not. A hardcoded comparison value silently keeps quoting a number the
    # code can no longer produce.
    deep_path = HERE / "results" / "deep.json"
    if not deep_path.exists():
        raise SystemExit(
            f"{deep_path} is missing. Run run_deep.py first -- this study's "
            "whole point is to locate that run's d inside this null, so there "
            "is nothing to report without it."
        )
    observed = json.loads(deep_path.read_text())["effect_size"]

    payload = {
        "note": "Null distribution of Cohen's d under control-vs-control. "
                "True effect is zero by construction. Observation about the "
                "instrument; moves no pre-registered bar.",
        "n_draws": N_DRAWS,
        "seeds_per_draw": SEEDS_PER_DRAW,
        "bar": bar,
        "null_effect_sizes": null_ds,
        "null_relative": null_rels,
        "null_abs_mean": float(absd.mean()),
        "null_abs_median": float(np.median(absd)),
        "null_abs_p90": float(np.percentile(absd, 90)),
        "null_abs_max": float(absd.max()),
        "false_positive_rate_at_bar": fpr,
        "observed_effect_size_deep_run": observed,
        "observed_percentile_in_null": float((absd < observed).mean()),
    }
    out = HERE / "results" / "null.json"
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print("\n" + "=" * 66)
    print("NULL DISTRIBUTION OF |d|  (no attacker present in any draw)")
    print("  median %.3f   mean %.3f   p90 %.3f   max %.3f"
          % (np.median(absd), absd.mean(), np.percentile(absd, 90), absd.max()))
    print("  draws clearing the pre-registered bar (|d| >= %.2f): %d/%d = %.0f%%"
          % (bar, int((absd >= bar).sum()), N_DRAWS, 100 * fpr))
    print("  the deep run's observed d = %.3f sits at the %.0fth percentile "
          "of this null" % (observed, 100 * payload["observed_percentile_in_null"]))
    print("=" * 66)
    print("\nwrote %s   (%.0fs)" % (out, time.time() - t0))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
