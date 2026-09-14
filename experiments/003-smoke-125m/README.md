# Experiment 003 — 125M smoke run (NOT a measurement, NOT the kill-gate)

> **Read this before quoting any number from this directory.**
>
> This is **not** `001-attack-spike` and it is **not** `000-repro-baseline`. It
> renders no PROCEED/STOP verdict, it does not touch `PREREGISTERED.md`, and
> nothing here can discharge the kill-gate or the Phase 0.5 environment gate.
> Standing Rule 5 is untouched.
>
> Unlike `002-pilot-tiny-ttt`, this run produces **no effect sizes at all** —
> not weak ones, not suggestive ones. There is no attacker here.

## What this is

Instrument validation on free hardware. Specifically, the **first execution of
the vendor `MetaModel` and of `trustgate.eval.carry` against it** — the one
remaining structural risk named at the end of `docs/report.md`:

> The remaining risk is concentrated in one place: binding the carry overlay to
> the real vendor model. Everything downstream of that is written and tested.

Until now `carry.py` had only ever run against a CPU stand-in for
`MetaModel.inner_loop_step`, because the vendor is not importable in the CPU
test environment by design. That gap is exactly where a **false STOP** hides:
ADR-006 shows that a harness which resets fast weights between adaptation and
measurement produces a null indistinguishable from a real one — and you would
buy that null with a paid GPU session.

## Why it costs nothing

Three facts about the vendor, each verified by reading it:

| Blocker for Phase 0.5 | Why it does not apply here |
|---|---|
| ~5.9 GB requester-pays checkpoint | `train.py:198` — `else: # Create new model and optimizer state`. With `load_part=none` (the default) the vendor random-inits. |
| Books3 `/val` zarr + GCS egress | `training.dummy_dataset` (`config.py:149`) is wired into `train.py:55` **and** `loop.py:75`. |
| W&B entity/project/key | `log_wandb` gates the logger entirely (`train.py:103`). |

So: no gcloud auth, no bucket, no egress, **$0**, and no entry in
`docs/protocols/gpu-bookings.md` — because nothing is booked.

## What it deliberately does NOT claim

| | Pre-registered (001) | Phase 0.5 (000) | This run |
|---|---|---|---|
| Victim | TTT-E2E 1B, DCLM+Books @8K | released 1B checkpoint | **random-init 125M** |
| Weights | trained | trained | **random** |
| Data | Books3 | Books3 `/val` | **`DummyDataset`, ids in [0,20)** |
| Bar | Cohen's *d*, rel. degradation, fluency | `2.314 < loss < 2.805` | **ln(vocab) at init, and "did the carry carry"** |
| Verdict | PROCEED / STOP | proceed / debug env | **none** |

A random-init model has **no adaptation worth corrupting**. `002`'s central
finding already applies a fortiori: at *n* = 5 the `d ≥ 0.8` criterion fires on
**55% of draws with no attacker present**. Any effect size computed here would
be noise on noise. The runner therefore computes none, and nothing in `results/`
should be transcribed into the paper, the deck or a report as a measurement.

## Run it

```
bash experiments/003-smoke-125m/setup.sh
PYTHONPATH=src python experiments/003-smoke-125m/run_smoke.py --seq-length 8192
```

On a Colab **T4**, add `--compute-dtype fp32`: T4 is Turing and has no native
bf16, while the vendor default is `compute_dtype: bf16` (`config.py:107`). On
an 8 GB card, `--seq-length 4096` still gives 4 inner steps.

### Hardware

| | Local RTX 3070 Ti Laptop | Kaggle T4 x2 | Kaggle P100 | Colab |
|---|---|---|---|---|
| VRAM | 8 GB | 2 x 16 GB | 16 GB | 15 GB |
| Native bf16 | yes (Ampere) | **no** (Turing) | **no** (Pascal) | **no** (T4) |
| Devices JAX sees | 1 | **2 — see below** | 1 | 1 |
| JAX CUDA | **WSL2 only** — no Windows wheels | native | native | native |

**Colab was tried on 2026-09-14 and does not work.** It runs Python 3.13 with
JAX 0.11, against the vendor's `requires-python = ">=3.12"` and `jax[cuda12]<0.6`
— roughly six releases apart. Either no cp313 wheel exists for JAX 0.5.x or pip
drags `jaxlib` and the whole `nvidia-*` set backwards into a CUDA link that fails
later and less legibly. Bumping the pin is not a fix: the carry overlay depends
on JAX and equinox tree semantics and on `lax.scan` carry behaviour, so a pass
against a different JAX would not transfer to the 1B run.

**Kaggle works, via `uv`.** The vendor's README says it uses `uv` for package
management, so fetching a standalone 3.12 sidesteps the host Python entirely
rather than fighting it — which is more faithful to the vendor setup than
`pip install` against whatever the notebook ships:

```
pip install uv
uv venv --python 3.12 /kaggle/working/venv
uv pip install --python /kaggle/working/venv/bin/python -e vendor/ttt-e2e
```

Two Kaggle-specific notes. Its `sitecustomize` imports `wrapt`, which a clean
venv lacks; the resulting `ModuleNotFoundError` is printed and then ignored by
the interpreter, and is not a failure. And **prefer P100 over T4 x2**, or accept
that `CUDA_VISIBLE_DEVICES` is doing the work — see below.

#### More than one visible accelerator will abort model construction

`ModelSharding.__init__` (`sharding.py:33-35`) asserts
`n_data_parallel * n_state_parallel == jax.device_count()`. This run is
single-device by construction, so on a two-device box the assertion fires inside
`build_model`, after the vendor stack has imported and a minute of startup has
been spent.

The vendor's own knobs do not prevent it. `backend.local_device_ids` and
`backend.num_devices` are read only inside `if distributed_config.distributed:`
(`jax_utils.py:44-48`), and this runner sets `backend.distributed=false` — so
both are **dead config here**, despite being set. `run_smoke.py` therefore
sets `CUDA_VISIBLE_DEVICES=0` itself, before importing jax, via `setdefault`
so an explicit choice of card still wins.

Worth carrying forward: the same dead-knob behaviour applies to any multi-GPU
rental box, where the failure would land after billing had started.

### Config overrides, and why each one

`ext-125m-e2e-32K.yaml` will OOM immediately at `seq_length: 32768`,
`global_batch_size: 32`.

| Override | Reason |
|---|---|
| `seq_length=8192` | `transformer.py:681` asserts divisibility by `mini_batch_size` (1024). 8192/1024 = **8 inner steps** — the exact granularity the pre-registration is scoped to (ADR-004). |
| `global_batch_size=1`, `eval_batch_size=1` | Activation memory. `train.py:212` takes `max(eval_batch_size, …)`; `COST_MODEL.md` §4 lists the stock value as a known first-launch failure. |
| `backend.num_devices=1`, `distributed=false` | `configs/backend/gpu.yaml` hardcodes 8 device ids. |
| `+deploy_paths.data.books3=/dev/null` | `config.yaml` interpolates `dataset_path` from it; it is `???` in `interactive.yaml` and OmegaConf resolves eagerly even though `dummy_dataset` means nothing reads it. |
| `exp_dir=/tmp/…` | `bootstrap_gpu_box.sh` refuses an `exp_dir` inside the repo; same rule applies here. |
| `XLA_PYTHON_CLIENT_PREALLOCATE=false` | JAX otherwise takes 75% of the card at import. Set by the runner. |

## The checks, and what failing each one would mean

Every check reports rather than raises, so one session finds *all* the
first-launch failures instead of one per session.

| Check | Bar | What a failure would mean |
|---|---|---|
| `gpu-present` | any non-CPU device | A silent CPU fallback makes every later number meaningless while still passing. |
| `random-init-loss` | CE ≈ **ln(128256) = 11.7624** ± 0.05 | The loss, the masking or the dtype is wrong; nothing downstream is trustworthy. |
| `param-count` | non-zero inner params | Zero means the inner spec matched nothing — no fast weights, so any attack result is a structural null. |
| `determinism` | bit-identical across two builds | A non-deterministic forward pass means a poison/control difference is not attributable to the stream. |
| `inner-lr-saturated` | multiplier == 1.0 | Near-frozen inner loop → null for the wrong reason (ADR-006). **See the caveat below.** |
| **`carry-is-non-trivial`** | `‖adapted − initial‖ > 0` | **The reason this run exists.** Fast weights coming back unchanged means the vendor's scan-carry discard (`transformer.py:712`) is still in force. |
| **`adaptation-changes-benign-loss`** | `eval(carried) ≠ eval(fresh)` | The other half: weights can move and still be discarded before measurement — the two-call harness ADR-006 rules out. |

### Read the `inner-lr-saturated` pass narrowly

`training/125m/ext.yaml` leaves `ilr_warmup_steps` at 0, and
`get_ilr_multiplier` (`transformer.py:565-567`) then returns a hard `1.0`
regardless of `step_index`. **The check cannot fail at this config.** It is not
evidence that the guard works where the ramp is live, and the runner says so in
its own output. The 1B extension config must be re-checked on its own terms.

### Why `ln(128256)` and not `ln(20)`

`DummyDataset` (`lm_dataset.py:36`) emits ids in `[0, 20)`, so the *targets*
span 20 values. But a random-init model spreads its mass over all 128256 logits,
so CE starts at `ln(128256)`. A loss near `ln(20) ≈ 3.0` would mean the model
had *learned the dummy distribution* — which is the tell that this is not a
language model and that no number from it transfers.

## Outputs

`results/smoke.json` — every check with its bar, its observation, and why it
matters. Git-ignored. Its first key is a `disclaimer` field, so a number lifted
out of it carries its own caveat.

## What this unblocks, and what it does not

**Unblocks:** the paid 1B session stops being the first time any of this code
has touched a model. Every failure found here is one not found at $4/hour with
a booking clock running.

**Does not unblock:** Phase 0.5 still needs the checkpoint, the dataset and an
80 GB box; Phase 1 still needs Phase 0.5. A green run here says the instrument
is wired correctly. It says nothing about whether TTT-E2E fast weights can be
poisoned.
