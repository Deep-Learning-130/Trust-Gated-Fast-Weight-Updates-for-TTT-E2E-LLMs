# Eval entry point for TTT-E2E — resolved

> Deliverable D1 of `docs/superpowers/specs/2026-08-03-phase0-lead-prep-design.md`.
> Written 2026-08-03. Vendor tree read at pinned SHA `a4fc4788ace38e29b5067916d4f4be33da894085`.
> Read-only: nothing under `vendor/ttt-e2e/` was modified.

## Verdict

**An eval-only path exists.** It is not a separate entry point — it is a mode on `train`,
selected by `training.eval_mode=true`, which short-circuits before the training loop and
executes zero optimizer steps.

**P1-1 does not become "write an eval harness." No critical-path scope change.**
`experiments/000-repro-baseline/README.md` step 3 stands as written.

The three prior observations that suggested otherwise were each accurate but insufficient:
`pyproject.toml` really does declare only `train`
([vendor/ttt-e2e/pyproject.toml:31-32](../../vendor/ttt-e2e/pyproject.toml#L31-L32)),
`configs/experiment/1b/` really does contain only `extension/` and `pretrain/`, and the
vendor README really does document only `uv run --exact train`. Eval lives inside the
training path, which is why none of them found it.

---

## 1. The command

```bash
uv run --exact train \
  +deploy=interactive \
  +experiment=1b/extension/ext-1b-e2e-32K \
  training.eval_mode=true \
  training.seq_length=8192 \
  training.global_batch_size=2 \
  training.exp_name=eval-1b-books-8k \
  training.load_part=params \
  checkpoint.resume_checkpoint_dir=/abs/path/to/1b_ttt_e2e_finetune_books_8k_1x_cc \
  deploy_paths.data.books3=/abs/path/to/llama3-books3/data.zarr \
  training.exp_dir=/abs/path/outside/repo/runs \
  training.wandb_entity=<entity> \
  training.wandb_project=<project> \
  training.wandb_key=<key>
```

**This command has never been executed.** There is no GPU and no `gcloud` auth as of writing.
It is derived from source, not from a successful run. Section 5 lists what to expect on first
launch.

### Why each override

| Override | Reason | Citation |
|---|---|---|
| `+deploy=` / `+experiment=` | Neither group is in the `defaults` list, so `+` is required to add them | [configs/config.yaml:1-6](../../vendor/ttt-e2e/configs/config.yaml#L1-L6); matches vendor README:61-66 |
| `training.eval_mode=true` | Selects the eval-only branch | [ttt/config.py:156](../../vendor/ttt-e2e/ttt/config.py#L156), [ttt/train.py:223](../../vendor/ttt-e2e/ttt/train.py#L223) |
| `training.load_part=params` | Released checkpoints ship no optimizer state; also required to enter the load branch at all | vendor README:124; [ttt/train.py:147](../../vendor/ttt-e2e/ttt/train.py#L147) |
| `checkpoint.resume_checkpoint_dir=` | Bypasses the `${deploy_paths.checkpoint}/${exp_folder}/${resume_exp_name}` interpolation, which would otherwise force the checkpoint to sit at `./checkpoints/demo/<name>` | [configs/config.yaml:13](../../vendor/ttt-e2e/configs/config.yaml#L13) |
| `deploy_paths.data.books3=` | Ships as `???` (MISSING); the run cannot start without it | [configs/deploy/interactive.yaml:9](../../vendor/ttt-e2e/configs/deploy/interactive.yaml#L9) |
| `training.seq_length=8192` | The checkpoint is 8K; the only 1B extension config is 32K — see §1.1 | [configs/experiment/1b/extension/ext-1b-e2e-32K.yaml:9](../../vendor/ttt-e2e/configs/experiment/1b/extension/ext-1b-e2e-32K.yaml#L9) |
| `training.global_batch_size=2` | The only way to lower the eval batch — see §5 gotcha 1 | [ttt/train.py:211](../../vendor/ttt-e2e/ttt/train.py#L211) |
| `training.exp_dir=` | Default `./experiments` collides with our own tracked tree — see §5 gotcha 2 | [ttt/config.py:151](../../vendor/ttt-e2e/ttt/config.py#L151) |
| `training.wandb_*` | Mandatory — see §4 | [ttt/config.py:133-135](../../vendor/ttt-e2e/ttt/config.py#L133-L135) |

An equally valid alternative to overriding `checkpoint.resume_checkpoint_dir` is to place
the checkpoint at `./checkpoints/demo/1b_ttt_e2e_finetune_books_8k_1x_cc` and pass
`training.resume_exp_name=1b_ttt_e2e_finetune_books_8k_1x_cc` instead, which is the flow the
vendor README describes under "Loading a model for extension" (README:86-93).

**Do not set `training.total_steps=0`** to force eval. `assert total_steps >= 1`
([ttt/train.py:219](../../vendor/ttt-e2e/ttt/train.py#L219)) fires before the eval branch is
reached. `eval_mode` already short-circuits, so the config's inherited `total_steps: 1250` is
harmless and should be left alone.

### 1.1 Which base experiment config — an unavoidable judgement call

The released checkpoint is `1b_ttt_e2e_finetune_books_8k_1x_cc` — Books, **8K**. There is no
`ext-1b-e2e-8K.yaml`; the only 1B extension config is 32K. So no stock config exactly
describes this checkpoint, and one override is unavoidable. Two defensible bases:

- **`1b/extension/ext-1b-e2e-32K` + `training.seq_length=8192`** ← recommended, used above.
  Inherits `dataset_name: books3` ([configs/training/1b/ext.yaml:4](../../vendor/ttt-e2e/configs/training/1b/ext.yaml#L4)),
  which is the corpus a Books checkpoint should be evaluated on.
- `1b/pretrain/pretrain-1b-e2e` + `training.dataset_name=books3`. Already 8K
  ([configs/training/1b/pretrain-8K.yaml:8](../../vendor/ttt-e2e/configs/training/1b/pretrain-8K.yaml#L8))
  but defaults to the DCLM corpus, so it needs the dataset override instead.

The choice is narrower than it looks, for two reasons:

1. **The `model:` blocks are identical.** `ext-1b-e2e-32K.yaml:19-28` and
   `pretrain-1b-e2e.yaml:17-26` specify the same `seq_modeling_block: SWA`,
   `sliding_window_size: 8192`, `rope_theta: 500000`, `prime: True`, `suffix_len: 6`,
   `intermediate_size: 4352`, `mini_batch_size: 1024`. Architecture is not in question.
2. **The inner-loop LR settings wash out at eval time.** The two configs differ in `ilr_init`
   (1 vs 0.1) and `ilr_warmup_steps` (0 vs 5000), which would otherwise change the forward
   pass. But [ttt/train.py:224](../../vendor/ttt-e2e/ttt/train.py#L224) forces `step_index` to
   `INT32_MAX - 100` before eval, and `get_ilr_multiplier`
   ([ttt/model/transformer.py:564-573](../../vendor/ttt-e2e/ttt/model/transformer.py#L564-L573))
   returns `1.0` when `ilr_warmup_steps == 0`, while for non-zero warmup `progress` saturates
   at `1.0`, giving `ilr_multiplier = ilr / optimizer_inner.lr = 1.0`. **Both paths yield
   exactly 1.0.** Both configs also set `optimizer_inner: sgd, lr: 1, clip_gradient: 1.0` and
   the same `spec_inner`.

So the effective difference between the two bases is only `dataset_name`, `seq_length`, and
`global_batch_size` — all three explicit in the command above.

---

## 2. Why this is a real eval path, not a vestigial flag

[ttt/train.py:222-227](../../vendor/ttt-e2e/ttt/train.py#L222-L227):

```python
with mesh:
    if cfg.training.eval_mode or start_step == total_steps:
        state = state.set(model.step_index, jnp.array(jnp.iinfo(jnp.int32).max - 100, dtype=jnp.int32))
        evaluator.eval_fn(model, state, start_step)
        jax.experimental.multihost_utils.sync_global_devices("eval finished")
        return
```

The `return` precedes the `for step in tqdm(range(start_step, total_steps), ...)` loop at
[ttt/train.py:230](../../vendor/ttt-e2e/ttt/train.py#L230). No `train_on_sequence` call, no
optimizer update, no checkpoint write. Three pieces of supporting machinery confirm the mode
was designed rather than incidental:

- [ttt/train.py:158-159](../../vendor/ttt-e2e/ttt/train.py#L158-L159) demotes `load_part` from
  `all` to `params` when `eval_mode` is set, with the comment
  `# prevent uncessary opt and loop state resumption` [sic].
- [ttt/train.py:224](../../vendor/ttt-e2e/ttt/train.py#L224) forces the inner-loop LR warmup to
  read as complete, so eval always runs at the fully-warmed inner LR.
- `Evaluator` is a dedicated class with its own loader and logging
  ([ttt/model/loop.py:50-129](../../vendor/ttt-e2e/ttt/model/loop.py#L50-L129)).

The same `evaluator.eval_fn` is also called at the end of a training run
([ttt/train.py:265](../../vendor/ttt-e2e/ttt/train.py#L265)), so this is the identical code
path the vendor's own published numbers came through — which is what experiment 000 needs.

### The complete set of eval knobs

A tree-wide grep for `eval|valid|perplex|ppl|test_loss` across all `.py`, `.yaml`, `.toml`
and `.md` files returned exactly three configurable fields:

| Field | Location | Default |
|---|---|---|
| `training.eval_mode` | [ttt/config.py:156](../../vendor/ttt-e2e/ttt/config.py#L156) | `False` |
| `training.eval_split` | [ttt/config.py:159](../../vendor/ttt-e2e/ttt/config.py#L159) | `"val"` |
| `training.eval_batch_size` | [ttt/config.py:170](../../vendor/ttt-e2e/ttt/config.py#L170) | `8` |

There is **no** `do_eval`, `val_every`, `eval_steps`, or `num_eval_batches`; no `eval` Hydra
config group; and no eval experiment config anywhere under `configs/`.

---

## 3. Which data the eval path reads → for D4

**Exactly one dataset, selected by config. Never both buckets.**

`Evaluator.__init__` builds a single loader
([ttt/model/loop.py:63-83](../../vendor/ttt-e2e/ttt/model/loop.py#L63-L83)), keyed
`train_holdout` ([ttt/model/loop.py:93](../../vendor/ttt-e2e/ttt/model/loop.py#L93)), from
`config.training.dataset_path` with `split=config.training.eval_split` (default `"val"`),
`repeat=False`, `shuffle=False`. And `dataset_path` resolves to one path
([configs/config.yaml:9](../../vendor/ttt-e2e/configs/config.yaml#L9)):

```yaml
dataset_path: ${deploy_paths.data[${training.dataset_name}]}
```

`dataset_name` is `books3` for extension configs
([configs/training/1b/ext.yaml:4](../../vendor/ttt-e2e/configs/training/1b/ext.yaml#L4)) and
`dclm_filter_8k` for pretrain
([configs/training/1b/pretrain-8K.yaml:4](../../vendor/ttt-e2e/configs/training/1b/pretrain-8K.yaml#L4)).

> **For the 1B Books @8K checkpoint, eval reads `gs://llama3-books3` only.**
> `gs://llama3-dclm-filter-8k` is **not** touched. It would only be needed to additionally
> evaluate the `1b_ttt_e2e_pretrain_dclm_8k_1x_cc` checkpoint, which is a separate decision.

Two facts the cost model must not get wrong:

**The dataset must be on local disk — `gs://` will not work.**
[ttt/dataloader/lm_dataset.py:14](../../vendor/ttt-e2e/ttt/dataloader/lm_dataset.py#L14) opens
`zarr.storage.LocalStore(path, read_only=True)`. There is no GCS store in the loader, which
is why the vendor README (lines 36-38) instructs a full `gcloud storage cp -r` download first.
Budget for the full transfer or a fuse mount. This is **asymmetric with checkpoints**, which
*do* accept a `gs://` path directly
([ttt/infra/checkpoint.py:83-84](../../vendor/ttt-e2e/ttt/infra/checkpoint.py#L83-L84)):

```python
if not checkpoint_path.startswith("gs://"):
    checkpoint_path = Path(checkpoint_path).resolve()
```

**Only the `val` sub-array is read for eval.**
[ttt/dataloader/lm_dataset.py:16](../../vendor/ttt-e2e/ttt/dataloader/lm_dataset.py#L16) opens
`zarr.open_array(store, path=f"/{split}")`, so `train` and `val` are sibling arrays inside one
store. `gsutil -u "$GCP_BILLING_PROJECT" du -s gs://llama3-books3/data.zarr/val` gives the eval-only byte
count, which should be far smaller than the whole bucket. **D4 should price `/val`, not the
full corpus** — provided a selective fetch is used rather than the README's `cp -r`.

**Eval duration is not config-boundable.** Because there is no `num_eval_batches` and
`repeat=False`, eval runs the *entire* val split for `len(ds)` batches
([ttt/model/loop.py:98-109](../../vendor/ttt-e2e/ttt/model/loop.py#L98-L109)). Wall-clock is set
by val-split size. D4 must price a full pass; D5 cannot offer a cheap partial eval except via
`training.dummy_dataset=true`, which evaluates on random tokens
([ttt/dataloader/lm_dataset.py:30-40](../../vendor/ttt-e2e/ttt/dataloader/lm_dataset.py#L30-L40))
and is therefore only a plumbing smoke test, not a measurement.

**⚠ `/val` alone will not start — resolved 2026-08-08 by D4.** `_make_train_iterator`
([ttt/train.py:125](../../vendor/ttt-e2e/ttt/train.py#L125)) runs *before* the eval branch, so
`zarr.open_array(store, path="/train")` executes even in eval mode — and it **raises if the
node is absent.** A store holding only `/val` therefore dies during setup, after billing has
started. This was previously logged here as an egress risk; it is a crash risk.

The fix costs nothing. The vendor pins `zarr>=3.0.4` (3.0.7 in `uv.lock`), and in zarr v3 each
array carries its own `zarr.json` while **absent chunks read as the fill value**. Copy the
group metadata, all of `/val`, and *only* `/train/zarr.json`: `/train` then opens at the right
shape with nothing behind it. `next()` is never called on that iterator in eval mode, and an
eager `prefetch_buffer_size=500`
([ttt/train.py:72-75](../../vendor/ttt-e2e/ttt/train.py#L72-L75)) would read zeros for free —
which retires the egress question too. Commands: [`COST_MODEL.md`](COST_MODEL.md) §1.1;
implemented in `scripts/bootstrap_gpu_box.sh` step 5.
**Still unverified against the live bucket** — no path has been listed.

---

## 4. W&B credentials: mandatory, and `log_wandb=false` does not avoid them

**Verdict: mandatory. Budget for a real W&B account before the first booking.**

This is the single most likely cause of a failed first launch, so it is worth being precise.
All three fields are `MISSING` in the schema
([ttt/config.py:133-135](../../vendor/ttt-e2e/ttt/config.py#L133-L135)), so Hydra raises on
access if they are unset. More importantly, setting `training.log_wandb=false` does **not**
make the run offline. In `WandbLogger.__init__`
([ttt/infra/wandb_utils.py:59-63](../../vendor/ttt-e2e/ttt/infra/wandb_utils.py#L59-L63)):

```python
if self.is_master:
    # Pass API key directly to Api()
    wandb.login(key=wandb_key)
    api = wandb.Api(api_key=wandb_key)
    runs = api.runs(f"{self.entity}/{self.project}", filters={"display_name": self.exp_name})
    num_existing = len(runs)
```

That block is guarded by `is_master` **only**. `self.enabled` — which is what
`training.log_wandb` sets ([ttt/train.py:110](../../vendor/ttt-e2e/ttt/train.py#L110)) — gates
only the *later* logging calls
([wandb_utils.py:71](../../vendor/ttt-e2e/ttt/infra/wandb_utils.py#L71),
[:88](../../vendor/ttt-e2e/ttt/infra/wandb_utils.py#L88),
[:95](../../vendor/ttt-e2e/ttt/infra/wandb_utils.py#L95),
[:102](../../vendor/ttt-e2e/ttt/infra/wandb_utils.py#L102)). So the login and the
authenticated `api.runs()` query execute regardless.

The query result is structurally load-bearing, not just telemetry: it sets `self.preexisting`
([wandb_utils.py:69](../../vendor/ttt-e2e/ttt/infra/wandb_utils.py#L69)), which drives the
resume decision at
[ttt/train.py:145-147](../../vendor/ttt-e2e/ttt/train.py#L145-L147). Removing it changes
control flow.

**⚠ Inference, not confirmed.** `WANDB_MODE=offline` is unlikely to rescue this, because
`wandb.Api()` is the public REST client and `api.runs()` is a live authenticated HTTP query
that offline mode is not documented to stub. I did not verify this — confirming requires
running the code.
*To confirm:* on any machine with the vendor env installed,
`WANDB_MODE=offline python -c "import wandb; wandb.Api(api_key='x').runs('e/p')"` and observe
whether it raises.

**Recommended posture for P1-1:** obtain a real W&B entity/project/key. A genuinely
air-gapped eval would require patching `wandb_utils.py`, which ADR-002 forbids. Use a fresh
`training.exp_name` so `num_existing == 0` and the run does not resume an unrelated one.

---

## 5. What to expect on first launch

Two things in the stock config will misbehave on real hardware.

**Gotcha 1 — the eval batch is 128, and `eval_batch_size` cannot lower it.**
[ttt/train.py:210-216](../../vendor/ttt-e2e/ttt/train.py#L210-L216):

```python
evaluator = Evaluator(
    global_batch_size=max(cfg.training.eval_batch_size, cfg.training.global_batch_size // cfg.training.accum_steps * 4),  # Larger bs to speed up eval
    ...
)
```

With the extension config's `global_batch_size: 32` and `accum_steps: 1`, this is
`max(8, 128) = 128` sequences at 8K context. Because it is a `max`, lowering
`training.eval_batch_size` has no effect — the second term dominates. **You must lower
`training.global_batch_size`.** To reach an eval batch of 8: `global_batch_size=2`
(`max(8, 2//1*4) = max(8, 8) = 8`). Also note `global_batch_size` must be divisible by the
process count ([ttt/dataloader/lm_dataset.py:76](../../vendor/ttt-e2e/ttt/dataloader/lm_dataset.py#L76)).
Tune this upward once the run is known to fit.

**Gotcha 2 — the default `exp_dir` writes into our own repo.**
`training.exp_dir` defaults to `./experiments`
([ttt/config.py:151](../../vendor/ttt-e2e/ttt/config.py#L151)) and `log_dir` is
`exp_dir/exp_folder/exp_name` ([ttt/train.py:98](../../vendor/ttt-e2e/ttt/train.py#L98)), with
`exp_folder` defaulting to `demo` ([ttt/config.py:152](../../vendor/ttt-e2e/ttt/config.py#L152)).
Launched from the repo root, that creates `experiments/demo/eval-1b-books-8k/` **inside our
tracked tree**. Our `.gitignore` ignores `experiments/*/results/` but not `experiments/demo/`
itself. Override `training.exp_dir` to a path outside the repo.

A third, benign note: a save-side `Checkpointer` is constructed unconditionally at
[ttt/train.py:116](../../vendor/ttt-e2e/ttt/train.py#L116) using `checkpoint.checkpoint_dir`,
so an empty checkpoint directory is created even in eval mode. Nothing is written to it — the
eval branch returns before any `save_checkpoint` call.

---

## 6. What the eval produces → for D2

Metrics come from the `MetricType` enum
([ttt/model/transformer.py:534-539](../../vendor/ttt-e2e/ttt/model/transformer.py#L534-L539)):
`loss`, `token_nll_loss`, `outer_grad_norm`. Eval emits the first two
([ttt/model/loop.py:115-129](../../vendor/ttt-e2e/ttt/model/loop.py#L115-L129)):

- **`train_holdout/loss`** — printed to stdout via `master_log` as
  `Eval -- train_holdout/loss: <value>`
  ([ttt/model/loop.py:120](../../vendor/ttt-e2e/ttt/model/loop.py#L120)) **and** logged to W&B.
  This is mean cross-entropy in **nats per token**: the sum of token-wise NLL divided by the
  valid-token count ([ttt/model/loss.py:26-27](../../vendor/ttt-e2e/ttt/model/loss.py#L26-L27)),
  where `valid` masks out BOS
  ([ttt/dataloader/lm_dataset.py:53](../../vendor/ttt-e2e/ttt/dataloader/lm_dataset.py#L53)).
- **`train_holdout_token_nll_loss.npy`** — the per-token NLL curve, saved into `log_dir`
  ([ttt/model/loop.py:125-128](../../vendor/ttt-e2e/ttt/model/loop.py#L125-L128)). This is the
  array a long-context claim is actually made from, and is likely what D2's tolerance should
  be stated against.

Two consequences for D2 and for the experiment record:

1. **No perplexity is computed anywhere in the vendor tree** — and none is needed. The grep
   found no `perplex` or `ppl` in any file. The paper's axis label "Loss (log perplexity)" is
   the *same* quantity as `train_holdout/loss`: mean CE in nats/token (paper §3.4.1).
   **Compare directly; no `exp()`.** Settled by D2 — [`TOLERANCE.md`](TOLERANCE.md) §2.
2. **`.npy` is gitignored** (`.gitignore:5`), and the `results/` directory is too
   (`.gitignore:9`). The per-token array must be deliberately copied into
   `experiments/000-repro-baseline/results/` and its numbers transcribed into a tracked file,
   or the record of the run will not survive.

Note the metric is logged under the key `train_holdout` even though it reads the `val` split —
the name refers to a held-out portion, not to the training split. Do not misread it as a
train-set number.

---

## Open items handed to other deliverables

- ✅ **D2** — done: [`TOLERANCE.md`](TOLERANCE.md). Stated against mean CE (nats/token), no
  conversion needed. **The paper reports no number for this checkpoint**, so the bar is a
  bracket plus structural checks, not a tolerance — see [`ADR-005`](../../docs/adr/ADR-005-baseline-comparison-basis.md).
- ✅ **D4** — done: [`COST_MODEL.md`](COST_MODEL.md). `/val` **plus both arrays' metadata**
  (see §3 above); cap $325 against $10–35 expected; egress under $1; the full val pass cannot
  be capped by config.
- ✅ **D5** — done: `scripts/bootstrap_gpu_box.sh`. Fetches dataset *and* checkpoint, refuses
  an `exp_dir` inside the repo. **Never executed.**
- **Before the first booking** — obtain a W&B entity/project/key. This is a procurement item,
  not a config item, and it is now the only Phase 0.4 task still open.
