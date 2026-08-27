# Fluency reference model — choice, reasoning, and acceptance evidence

> **T3.2 / TEAM_PLAN P1-8.** Written 2026-08-27. P1 and P2 both depend on this
> being settled: the fluency ratio is one of the three pre-registered gating
> bars, and until it can be computed no verdict can be PROCEED — an unscored
> stream fails the realism bar safe, so `report.md` says STOP and calls itself
> incomplete regardless of the corruption numbers.

## The model: GPT-2 small (124M), `openai-community/gpt2`

Independent of the victim on every axis that matters:

| | Victim (TTT-E2E 1B) | Reference (GPT-2 small) |
|---|---|---|
| Corpus | DCLM + Books3 | WebText |
| Tokenizer | Llama-3, 128256 | GPT-2 BPE, 50257 |
| Context | 8192 (this experiment) | 1024 |
| Adapts at inference | **yes** — that is the attack surface | no |

Independence is not a formality. `PREREGISTERED.md` names "fluency scored with
the victim model" an invalidating condition, and the reason is mechanical: the
attacker optimises *against* the victim, so the victim assigns low perplexity to
exactly the streams the attack has shaped. A bar scored with it would pass every
attack by construction.

`trustgate.eval.fluency.assert_independent_of_victim` makes the check structural
rather than a matter of care — it refuses any reference model whose vocabulary is
128256, because a model carrying the victim's Llama-3 vocabulary is what the
mistake would look like.

**Why this model and not a larger one.** It scores five poison/control pairs on
CPU in seconds, it has published numbers, and its weights are small enough to
hash and pin. Scaling to `gpt2-medium` is a path change in
`fetch_reference_model.py` — nothing in the scorer assumes the size.

## Implementation: transcribed, not imported

`transformers` 5.x **dropped Flax models entirely** — `FlaxGPT2LMHeadModel` no
longer exists, and `GPT2LMHeadModel` warns "PyTorch was not found. Models won't
be available." That left pinning a deprecated `transformers` 4.x, adding torch as
a second array framework, or implementing the forward pass.

`src/trustgate/eval/gpt2_reference.py` implements it on the jax stack that is
already a core dependency. Consequences:

- The fluency path adds exactly **two** small pure-Python dependencies,
  `safetensors` and `tokenizers`, both lazily imported. No torch, no
  `transformers`, no flax at runtime.
- The CPU test suite exercises the scorer with fakes and needs neither, which is
  what keeps CI network-free.
- We own tokenization, windowing and masking. That is not incidental: the bar is
  a **ratio**, and a ratio is only trustworthy if both arms are scored
  identically.

Three details the implementation had to get right, each of which shifts
perplexity measurably if wrong: HF's `Conv1D` stores weights `[in, out]` so
projections take no transpose; the activation is `gelu_new` (the tanh
approximation), not exact GELU; LayerNorm epsilon is 1e-5 from `config.json`.

## Acceptance evidence

### Primary: equivalence to the reference implementation — **PASS**

`scripts/verify_gpt2_equivalence.py`, run 2026-08-27. Same inputs through our jax
implementation and HuggingFace's `GPT2LMHeadModel`:

| Sequence length | max \|logit diff\| | max \|NLL diff\| | argmax agreement |
|---|---|---|---|
| 1 | 1.373e-04 | 0 | 1.0000 |
| 7 | 1.984e-04 | 6.289e-05 | 1.0000 |
| 64 | 1.907e-04 | 7.823e-05 | 1.0000 |
| 256 | 2.213e-04 | 1.076e-04 | 1.0000 |
| 1024 | 5.798e-04 | 1.027e-04 | 1.0000 |

Worst logit difference 5.8e-04 against a 1e-03 band; worst per-token NLL
difference 1.1e-04. Differences at this magnitude are float32 matmul accumulation
order, not semantics. A transposed `Conv1D`, exact GELU, or a wrong LayerNorm
epsilon all shift logits by O(0.1) — four orders of magnitude outside this band.

`torch` is a **verification-only** dependency. Nothing under `src/` imports it.

### Secondary: the published perplexity — recorded, and *not* used as a gate

`scripts/acceptance_fluency_reference.py`, run 2026-08-27 on the WikiText-103
test split (4358 rows, 1,237,056 chars after the standard WikiText detokenizer),
GPT-2 small, context 1024, stride 512, 528 windows, 825s on CPU:

- **per-BPE-token perplexity: 22.659** (mean NLL 3.12054 nats, 270,724 scored tokens)
- 204,464 whitespace words → 270,725 BPE tokens (1.324 tokens/word)
- **per-word perplexity: 62.290** against a published **37.50**

T3.2 asked for a published perplexity as *the* acceptance check. **It does not
work as one**, and that is the honest finding rather than a tolerance to widen.

GPT-2's reported WikiText perplexities are per *word*, and the word-count
denominator is not specified precisely enough to reproduce. The same measured
total NLL yields:

| Denominator convention | Per-word perplexity |
|---|---|
| 204,464 whitespace words (post-detokenization) | **62.29** |
| 245,569 — WikiText-103's canonical test-set token count | **31.18** |

The published 37.50 sits **between** the two. So the check cannot separate "the
implementation is wrong" from "the denominator convention differs" — which makes
it undecidable as a pass/fail gate on the only question it was meant to answer.
Hence the equivalence check above, which answers it directly.

What the number *does* support: 22.66 per BPE token is squarely in the expected
range for GPT-2 small on WikiText-103, so it is corroborating evidence, just not
a gate.

**Nothing was tuned toward 37.50.** The scripts say so in as many words, and the
reason is that the fluency bar consumes a *ratio* — the absolute cancels — so
fudging an absolute would buy nothing and would hide a real bug.

### Discrimination self-test — **PASS**

`python -m trustgate.eval.harness --fluency-selftest`:

| Text | Perplexity |
|---|---|
| Fluent English (Austen) | 19.98 |
| Same words, scrambled | 1326.84 |

A **66×** separation. Available as one command so it can be run on a fresh box
before any fluency number is trusted.

## A limitation of the bar itself — read this before interpreting a PASS

**Perplexity punishes scrambling but rewards repetition.** Measured, not
theorised: a repeated token pattern scored **1.90**, an order of magnitude
*better* than fluent English at 19.98, because a repetitive stream is trivially
predictable.

So a poison stream that reuses one passage many times would sail through the
≤ 1.5 fluency bar while being obviously not "benign-looking" to a human. The
realism bar cannot detect this on its own.

Two existing mitigations, both already in place, and this is the evidence that
they are load-bearing rather than tidy:

- Spans are drawn **without replacement** by default
  (`corpus.sample_spans(replace=False)`), so SELECT cannot fill every chunk with
  one passage.
- **Span multiplicity is recorded and reported** per arm
  (`CraftedStream.span_multiplicity`), so a repetitive stream is visible in the
  report even when its fluency ratio looks fine.

ADR-007 flagged multiplicity as an unmatched structural variable on reasoning
alone. This measurement is why it needs to stay reported: the fluency bar will
not catch it.

## What is fetched, and where

`python scripts/fetch_reference_model.py` → `TRUSTGATE_REFERENCE_CACHE`
(default `~/.cache/trustgate/reference`), outside the repo. `PROBE_ONLY=1` reports
sizes without fetching bytes.

| File | Size | sha256 (first 16) |
|---|---|---|
| `gpt2/config.json` | 0.001 MB | `0daed7749b4f02b8` |
| `gpt2/tokenizer.json` | 1.4 MB | `8414cab924d8b9b3` |
| `gpt2/model.safetensors` | 548.1 MB | `248dfc3911869ec4` |
| `llama3/tokenizer.json` | 9.1 MB | `e134af98b985517b` |

Full digests in `MANIFEST.json` beside the weights. Nothing is committed.

**The Llama-3 tokenizer is for decoding only.** Crafted streams are Llama-3 token
ids because the vendor ships no tokenizer (`trustgate.tokens`, ADR-007), and the
reference model reads text, so ids are decoded on the way in. An **ungated**
mirror (`NousResearch/Meta-Llama-3-8B`) is used deliberately — the official
`meta-llama` repo is gated, and a gate in the middle of a reproducibility path is
a liability. Vocabulary size is asserted against the vendor's model configs at
load time rather than trusted: a wrong tokenizer decodes to plausible but wrong
text, and nothing downstream could detect it.

## How it reaches the verdict

```python
from trustgate.eval.fluency import load_default_scorer
scorer = load_default_scorer()          # raises with instructions if unfetched
run_attack_spike(..., fluency_scorer=scorer)
```

`StreamFluencyScorer.__call__(poison, control)` decodes both arms, scores both,
writes the perplexities onto the streams so `CraftedStream.is_scored` stops
reporting `nan`, and returns the ratio. `run_attack_spike` collects one ratio per
seed into `SpikeResult.per_seed_fluency`, and the **max** across seeds gates —
see `experiments/001-attack-spike/README.md` for why max and not mean, and for
the pre-committed escalation path if a verdict turns on that choice.

## Open items

- **Stride is 512 against a 1024 context.** Defensible and conventional, and it
  applies identically to both arms so it divides out of the ratio. Not tuned.
- **Only SELECT streams have been scored end to end** (on synthetic corpora and
  on real English in the tests). PARAPHRASE and SOFT do not exist yet (T2.5).
- **`gpt2-medium` was not evaluated.** If the 124M model's perplexity turns out
  too coarse to separate poison from control on real Books3 spans, that is the
  first knob to turn — and it is a path change, not a code change.
