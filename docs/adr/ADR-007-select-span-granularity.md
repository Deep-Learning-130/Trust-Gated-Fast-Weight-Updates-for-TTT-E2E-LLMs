# ADR-007 — SELECT operates on sub-chunk token spans, not sentences or chunks

**Status:** Accepted **Date:** 2026-08-27 **Deciders:** project owner (Lead)
**Relates to:** TEAM_PLAN P1-4/P1-5, task-allocation T2.1/T2.2, `PREREGISTERED.md` §Streams

## Context

`attack/stream.py` specifies SELECT as "choose and order sentences from a benign corpus.
Never synthesises text, so fluency is exact by construction". Two facts make that
under-specified as written.

**There is no tokenizer, so there are no sentence boundaries.** `ttt/config.py:144` is the
only tokenizer reference in the vendor tree, is never read, and names Llama-2 while every
model yaml declares Llama-3 (`trustgate/tokens.py` records the evidence). The data arrives
pre-tokenized and `/train` / `/val` are flat 1-D token-ID streams. Locating a sentence
boundary requires a decoder, which is T3.2 and gated on a network fetch and a
published-perplexity reproduction. Guessing at terminator token ids would be worse than not
cutting on sentences at all.

**The obvious substitute is the inner-loop chunk, and it is a trap.** One inner SGD step
happens per `mini_batch_size` tokens (`transformer.py:709-718`), so the chunk looks like the
natural attacker unit. But `mini_batch_size` is **1024**
(`configs/experiment/1b/extension/ext-1b-e2e-32K.yaml:23`) and `seq_length` is 8192
(ADR-004), so chunk-sized spans give the attacker **8 units of choice per window** — pick 8
passages of ~750 words each and order them.

That is the whole search space. A STOP verdict under an 8-unit lever cannot distinguish
"SELECT cannot corrupt fast weights" from "1024-token granularity cannot", and per dossier
§7 a STOP ends the project. The kill-gate is asymmetric: a false PROCEED costs a wasted
Phase 2, a false STOP costs everything. So granularity is not a neutral implementation
detail.

## Decision

**`span_tokens` is an explicit parameter, decoupled from `mini_batch_size`, constrained by
`mini_batch_size % span_tokens == 0`, defaulting to 64.**

At the default that is 16 spans per inner step and 128 per 8192-token window — a real search
space. Chunk-alignment remains available as the conservative arm by setting
`span_tokens == mini_batch_size`, and the divisibility constraint keeps spans tiling the
chunk exactly so the two arms place their seams at the same points within the gradient.

Three supporting decisions:

1. **Spans are BOS-free.** Any span containing 128000 is rejected, loudly, rather than
   substituted. The reason is stronger than length matching: the inner objective's
   denominator is *that chunk's* valid count (`loss.py:15,26`), so a chunk with fewer valid
   tokens up-weights every surviving token's gradient. BOS is a per-inner-step
   gradient-scale confound.
2. **`terminator_ids` is accepted and unused.** It costs nothing to thread through now and
   enables sentence-level segmentation the moment T3.2 brings a decoder.
3. **Provenance is recorded** — source offset per span, plus span multiplicity and mean
   offset gap — because these are unmatched structural variables between the arms
   (see Consequences).

## Alternatives rejected

**Sentence spans via guessed terminator ids.** A wrong id silently changes the unit of the
attack and there is no way to notice without a decoder. Deferred, not abandoned.

**Chunk-sized spans (`span_tokens = mini_batch_size`).** The 8-unit lever above. Kept
available as a reported conservative arm; rejected as the default.

**A text-side corpus (Gutenberg) with a Llama-3 tokenizer mirror.** Gives real sentence
boundaries today, but adds a dependency and a network fetch, and puts the corpus
out-of-distribution relative to books3/DCLM — which weakens exactly the "benign-looking"
claim the fluency bar exists to establish. Revisit for PARAPHRASE (T2.5), which must edit
text and cannot work in ID space.

## Consequences

- **Spans straddle sentence boundaries**, so concatenation seams carry a small local
  fluency cost. The benign control is built by the identical procedure, so the *baseline*
  seam cost is common to both arms and largely divides out of the fluency ratio. It must
  not be claimed that seams simply "cancel": the poison arm's adjacencies are adversarially
  chosen, so its excess NLL at seams is the larger, and that residual is precisely the
  signal the fluency bar exists to catch. A fluency FAIL is a result, not an artefact.
- **The shared procedure can launder non-fluency.** If concatenation dominates perplexity in
  both arms the ratio tends to 1.0 regardless of how unnatural the poison is, and smaller
  spans make this worse. Mitigated by reporting a third, non-gating **contiguous** arm — one
  unbroken corpus slice — so the procedure's own cost is quantified rather than assumed.
  Reported only; no pre-registered bar moves.
- **Selection bias:** BOS delimits documents, so BOS-free means intra-document only. A real
  attacker submitting a document would start at a boundary. Both arms are equally biased so
  the comparison holds, but the §1 threat-model story is slightly undercut and that belongs
  on the record.
- **A STOP is scoped to the granularity that produced it.** The chosen `span_tokens`,
  `mini_batch_size` and `seq_length` are recorded in `RunCondition` and must appear in the
  report next to the verdict.
- **Difficulty is now the sharpest remaining confound.** Adversarial selection will prefer
  high-entropy, rare-token spans because those produce the largest inner-loop gradients, so
  the arms can differ in intrinsic difficulty rather than in adversarial ordering — a
  distribution shift wearing an attack's clothes. `Span.baseline_nll` and `score_spans`
  exist to build a difficulty-matched control; it is reported alongside, and the
  pre-registered length-matched control remains the one that gates.
