# ADR-006 — The vendor discards the adapted fast weights; carry them in an overlay

**Status:** Accepted **Date:** 2026-08-27 **Deciders:** project owner (Lead)
**Blocks:** `run_stream` / `eval_benign` (TEAM_PLAN P2-1, task-allocation T1.8), and
therefore the whole Phase 1 kill-gate

## Context

`src/trustgate/eval/harness.py` specifies `run_stream(...)` as "feed one stream through
TTT-E2E, returning the adapted fast weights", and `eval_benign(...)` as "benign-task loss
for a model with adapted fast weights". Both were logged as blocked on checkpoints — i.e.
as *hardware* work.

They are not. They are blocked on a gap in the vendor API.

The meta branch of `MetaModel.loss_for_sequence` (`ttt/model/transformer.py:684-720`) binds
the scan carry at line 712:

    carry, metrics = scan_remat_chunk(
        eqx.filter_checkpoint(process_suffix_chunk, prevent_cse=False),
        (model, inner_opt_state, (state_all, state_suffix)),
        (seq, prefix_output),
        ...
    )

    loss = metrics[M.loss].mean()

`carry` holds `new_model` — the adapted prime SwiGLU MLP, which *is* the fast weights — and
is never read again. Line 745 returns `loss, metrics` only. Three further resets compound
it:

- **line 685** rebuilds `model` from outer parameters with a `state_dtype` cast on every
  call, so any adaptation from a previous call is overwritten;
- **line 686** re-initialises `inner_opt_state`, discarding optimiser momentum;
- **`loop.py:160`** vmaps `loss_for_sequence` over the batch, so every batch element runs an
  independent fast-weight trajectory that is also discarded.

The re-splice at lines 698-701 does not help. It takes `inner_params` from the carry and
`outer_params` from the closed-over `model`, which preserves fast weights *within* one
scan and only within one call — that is what it is for.

### Why this mattered enough to stop and write it down

The obvious harness is two calls: one to adapt on the poison stream, a second to measure
benign loss. Under the code above, the second call resets the fast weights, so the
measurement is taken on an unadapted model and the poison/control difference is pure eval
noise. The result is a **structurally guaranteed null** — a STOP verdict indistinguishable
from a genuine one.

Per dossier §7 and `experiments/001-attack-spike/PREREGISTERED.md`, STOP ends the project.
So this defect converts a paid GPU run into a project-ending artefact, and it does so
silently. It was found at the desk for zero spend, which is what deliverables D1–D7 were
for.

A second, independent route to the same false STOP sits next to it: `get_ilr_multiplier`
(`transformer.py:564-573`) ramps the inner learning rate from `ilr_init` to 1.0 as a
function of `state[step_index]`, and `train.py:224` pins `step_index = INT32_MAX - 100` in
eval mode precisely so the ramp is saturated. A hand-rolled harness that leaves
`step_index` at 0 runs a near-frozen inner loop.

### Scale

At `seq_length=8192` with `mini_batch_size=1024` (ADR-004;
`configs/experiment/1b/extension/ext-1b-e2e-32K.yaml:23`), one sequence is **8 inner SGD
steps**. `AttackSpec.stream_tokens` is documented as "large relative to a single window",
and the pre-registration reports "number of stream tokens needed to reach threshold" as a
secondary metric. Both presuppose accumulation *across* sequences, which the vendor does
not provide at all. Cross-sequence carry is a precondition of the experiment existing, not
a performance concern.

## Decision

Carry the fast weights in a **trustgate overlay**: `src/trustgate/eval/carry.py`.

1. **`CarryState`** holds `fast_weights`, `opt_state`, `state_tuple`, `n_steps`, and
   `inner_lr_multiplier`. An `eqx.Module`, so it is a registered pytree and the existing
   `tree_ops` helpers descend into it.
2. **`run_chunks`** threads one sequence of chunks through an inner step, returning
   `(carry, metrics)` — deliberately the same shape as the vendor's line 712, except the
   caller receives the carry.
3. **`run_sequences`** threads many sequences, so the carry out of sequence *i* is the
   carry into sequence *i+1*. This is the capability the vendor lacks.
4. **`step_fn` is injected.** It stands in for `MetaModel.inner_loop_step`. Everything else
   — chunk loop, carry threading, optimiser-state threading, step accounting — is trustgate
   code, and is CPU-tested today with no checkpoint and no GPU.
5. **`assert_saturated_inner_lr`** is called before any measurement, closing the
   `ilr_multiplier` route to a false null.
6. **`chunk_tokens` refuses a ragged tail** rather than padding it. BOS-padding is masked
   straight out of the loss (`lm_dataset.py:53`) and would corrupt valid-token matching
   between poison and control while looking perfectly fine.

This is ADR-002-compatible: an overlay that imports and wraps, with no edit to the
unlicensed vendor tree. It is the same posture as ADR-003 took for the interceptor, and for
the same reason.

## Alternatives rejected

**Copy `loss_for_sequence` into our tree and return the carry.** ~60 lines of a repository
that ships no licence. ADR-002 rules it out, and the reasoning is the same one that made
monkeypatching the interceptor preferable to vendoring the call site.

**Patch `loss_for_sequence` to return the carry.** Larger and more fragile than the
existing `inner_loop_step` monkeypatch: it would have to preserve two `train_mode` branches
and the metrics-flattening tail (line 744), and every vendor SHA bump would re-break it.
The overlay needs nothing from the vendor except one step function.

**Accept one sequence of adaptation (8 inner steps) and drop cross-sequence accumulation.**
Would silently redefine the experiment. The pre-registration's "slow attack" framing and
its stream-length secondary metric both become unmeasurable, and a STOP would be scoped to
8 inner steps without saying so.

**Use `lax.scan` in the overlay to mirror the vendor's memory behaviour.** The vendor's
`eqx.filter_checkpoint` per chunk exists so an *outer* gradient can flow back through the
inner loop during meta-training. The attack spike takes no outer gradient — it adapts
forward and measures — so there is nothing to rematerialise for, and a Python loop is
easier to assert over. This will need revisiting for `craft_stream`'s SOFT strategy (T1.9),
which does differentiate through the stream.

## Consequences

- `run_stream` and `eval_benign` are **no longer blocked on checkpoints**. They are blocked
  on binding `step_fn` to the real `MetaModel.inner_loop_step`, which is desk work.
- The Phase 1 critical path shortens: the wave's riskiest unknown is now resolved and
  tested rather than latent behind a hardware dependency.
- The overlay is a second vendor-coupled surface alongside `vendor_patch.py`, pinned to the
  same SHA `a4fc478`. `tests/test_carry.py` is its tripwire, in the same role
  `tests/test_interceptor.py` plays for the monkeypatch.
- `n_steps` becomes a first-class recorded quantity, which is what makes the
  pre-registration's "attack slowness" secondary metric measurable.
- The claim in `HANDOFF.md` and `task-allocation.md` that the model-touching half of
  `eval/harness.py` is "blocked on checkpoints" was **wrong in kind, not just in timing**.
  Fixed forward here rather than edited away, per the branch-and-review rule on history.
