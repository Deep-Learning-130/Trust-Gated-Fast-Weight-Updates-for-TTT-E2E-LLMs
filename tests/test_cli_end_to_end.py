"""C3 -> C4 -> C5 through the real CLI, against a trace-safe fake victim.

Every other test exercises one seam. This drives `cli.main` itself for the three
checkpoint runs of the session, in the order the runbook runs them, with only
two things faked: the victim (`_build_victim`, since the vendor is not
importable here) and the fluency reference (a network fetch). Everything
between -- conditions, the eval digest, the compiled inner step, the ordering
search, arms.pkl, the sequence arms, calibration, the gate, overhead and the
three reports -- is the code the box runs.

It exists because a one-token disagreement between `cli.py` and
`eval_benign_curve` about the eval digest passed every unit test and would have
refused the first benign eval of every run on the box. Only a test that crosses
all the seams at once sees that class of bug.
"""

import pickle
import sys
import types
from dataclasses import dataclass

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

MINI_BATCH = 64
SEQ = 256
SPAN = 16


@dataclass(frozen=True)
class _ModelCfg:
    bos_token_id: int = 128000


@dataclass(frozen=True)
class FakeCfg:
    model: _ModelCfg = _ModelCfg()


class FakeBatch(eqx.Module):
    input_ids: jnp.ndarray
    target_tokens: jnp.ndarray
    loss_masks: jnp.ndarray


class FakeStepResult(eqx.Module):
    new_model: object
    new_optimizer_state: object
    new_state: object
    metrics: dict


class _FakeLM:
    def __init__(self, m):
        self._m = m
        self.model = types.SimpleNamespace(h=types.SimpleNamespace(prefix_blocks=None))

    def wte_call(self, ids):
        return (ids.astype(jnp.float32) % 97.0) / 97.0

    def prefix_call(self, prefix_blocks, x, state, seq):
        return types.SimpleNamespace(last_hidden_state=2.0 * x - 1.0)

    def suffix_call(self, prefix_outputs, state, seq):
        logits = prefix_outputs[:, None] * self._m.inner[None, :] + self._m.outer
        return types.SimpleNamespace(logits=logits)


class _FakeOptimizer:
    def init(self, params):
        return jnp.zeros((), dtype=jnp.int32)


class FakeModel(eqx.Module):
    inner: jnp.ndarray
    outer: jnp.ndarray
    config: FakeCfg = eqx.field(static=True)

    def inner_parameters(self):
        return FakeModel(inner=self.inner, outer=None, config=self.config)

    def inner_optimizer(self, state):
        return _FakeOptimizer()

    @property
    def language_model(self):
        return _FakeLM(self)


_DIRECTION = jnp.array([1.0, 0.5, -0.3], dtype=jnp.float32)


class FakeMetaModel:
    @staticmethod
    def inner_loop_step(model, opt_state, state_tuple, seq, prefix_outputs):
        # Predict-then-adapt, like the vendor. The update is nonlinear in the
        # carry, so the ORDER of chunks changes where the weights end up -- the
        # attacker's only lever under SELECT has to exist in the fake too.
        x = prefix_outputs
        loss = jnp.mean((x[:, None] * model.inner[None, :] - 0.25) ** 2)
        step = 0.5 * jnp.tanh(jnp.mean(x) * 3.0 - jnp.sum(model.inner)) * _DIRECTION
        return FakeStepResult(
            new_model=FakeModel(inner=model.inner + step, outer=model.outer, config=model.config),
            new_optimizer_state=opt_state + 1,
            new_state=state_tuple,
            metrics={"loss": loss},
        )


class FakeScorer:
    """Mutates the streams exactly as `StreamFluencyScorer.__call__` does."""

    def __call__(self, poison, control):
        for s in (poison, control):
            s.perplexity = 10.0
            s.control_perplexity = 10.0
        return poison.fluency_ratio


@pytest.fixture
def session(monkeypatch, tmp_path):
    mods = {
        name: types.ModuleType(name)
        for name in (
            "ttt", "ttt.model", "ttt.model.data", "ttt.model.transformer",
            "ttt.utils", "ttt.utils.filter_utils", "ttt.utils.jax_utils",
        )
    }
    mods["ttt.model.data"].Batch = FakeBatch
    mods["ttt.model.data"].tree_slice = lambda tree, i: jax.tree.map(lambda x: x[i], tree)
    mods["ttt.model.transformer"].MetaModel = FakeMetaModel
    mods["ttt.model.transformer"].BlockCollectionSplit = object
    mods["ttt.utils.filter_utils"].get_filter_spec = lambda *a, **k: None
    mods["ttt.utils.jax_utils"].clone_pytree = lambda t: t
    for name, mod in mods.items():
        monkeypatch.setitem(sys.modules, name, mod)
    monkeypatch.delenv("TRUSTGATE_NO_JIT", raising=False)

    from trustgate.eval import cli, fluency, vendor_bind

    vendor_bind._STEP_CACHE.clear()
    cfg = FakeCfg()
    model = FakeModel(inner=jnp.zeros(3), outer=jnp.ones(3), config=cfg)
    binding = vendor_bind.VendorBinding(
        model_split=model,
        model_outer=model,
        spec_inner=FakeModel(inner=True, outer=False, config=cfg),
        state_all=None,
        state_prefix=None,
        state_suffix=None,
        mini_batch_size=MINI_BATCH,
        seq_length=SEQ,
        suffix_len=1,
        inner_lr_multiplier=1.0,
        state_dtype="fp32",
    )
    monkeypatch.setattr(
        cli, "_build_victim",
        lambda args, tag: (None, model, binding, MINI_BATCH, "fake-1b sha256:" + "0" * 64),
    )
    monkeypatch.setattr(fluency, "load_default_scorer", lambda *a, **k: FakeScorer())

    rng = np.random.default_rng(0)
    files = {}
    for name, n in (("train", 4096), ("val", SEQ + 1), ("probe", SEQ + 1)):
        files[name] = tmp_path / f"{name}.npy"
        np.save(files[name], rng.integers(10, 1000, n, dtype=np.int32))

    common = [
        "--objective", "degrade", "--strategy", "select",
        "--checkpoint", str(tmp_path / "ckpt"),
        "--corpus-file", str(files["train"]), "--eval-file", str(files["val"]),
        "--size", "1b", "--seq-length", str(SEQ), "--stream-tokens", str(SEQ),
        "--span-tokens", str(SPAN), "--seeds", "0", "1", "2",
    ]
    yield cli, common, files, tmp_path / "results"
    vendor_bind._STEP_CACHE.clear()


def test_c3_c4_c5_run_end_to_end_through_the_cli(session, capsys):
    cli, common, files, results = session

    # C3: the spike, with a real (small) ordering search.
    assert cli.main(common + ["--max-iters", "8", "--early-stop-patience", "0",
                              "--out", str(results)]) == 0
    out = capsys.readouterr().out
    assert "adapt-and-eval #2:" in out, "C2's per-evaluation timing line is missing"
    report = (results / "report.md").read_text(encoding="utf-8")
    assert "NOT A RESULT" not in report, "a checkpoint run must not carry the random-init banner"
    arms = pickle.loads((results / "arms.pkl").read_bytes())
    assert set(arms) == {0, 1, 2}

    # C4: the sequence arms on the spike's own streams.
    assert cli.main(common + ["--sequence-eval", "--arms-file", str(results / "arms.pkl"),
                              "--out", str(results / "sequence")]) == 0
    seq_report = (results / "sequence" / "sequence.md").read_text(encoding="utf-8")
    assert "NOT A RESULT" not in seq_report

    # C5: the gate, calibrated, at two thresholds, with overhead.
    assert cli.main(common + ["--gate-eval", "--arms-file", str(results / "arms.pkl"),
                              "--probe-file", str(files["probe"]),
                              "--overhead-repeats", "2", "--out", str(results / "gate")]) == 0
    gate = (results / "gate" / "gate.md").read_text(encoding="utf-8")
    assert "NOT A VERDICT" in gate
    assert "overhead:" in gate
    assert "crafted by the 001 search" in gate
    result = pickle.loads((results / "gate" / "gate-result.pkl").read_bytes())
    assert len(result.thresholds) == 2
    assert result.calibration.unscorable_steps == 0
    assert max(result.calibration.divergences) > 0.0
    assert all(len(t.arms.poisoned) == 3 for t in result.thresholds)


def test_c5_falls_back_to_uncrafted_arms_after_a_null_spike(session):
    cli, common, files, results = session

    assert cli.main(common + ["--gate-eval", "--uncrafted-arms",
                              "--probe-file", str(files["probe"]),
                              "--overhead-repeats", "0", "--out", str(results / "gate")]) == 0
    gate = (results / "gate" / "gate.md").read_text(encoding="utf-8")
    assert "UNCRAFTED" in gate
    assert "Not measured in this run." in gate


@pytest.mark.parametrize(
    "mode, report",
    [([], "report.md"), (["--sequence-eval", "--windows", "4"], "sequence.md"),
     (["--gate-eval", "--overhead-repeats", "2"], "gate.md")],
)
def test_random_init_runners_end_to_end(session, mode, report):
    """The three --random-init runners. The gate one is the smoke test
    `prepare_phase1.sh` runs on the box before any paid 1B work."""
    cli, _common, _files, results = session
    out = results / "ri"
    assert cli.main(["--objective", "degrade", "--strategy", "select", "--random-init",
                     "--seq-length", str(SEQ), "--stream-tokens", str(SEQ),
                     "--corpus-tokens", "4096", "--seeds", "0", "1", *mode,
                     "--out", str(out)]) == 0
    assert "NOT A RESULT" in (out / report).read_text(encoding="utf-8")


def test_c2_timing_run_finishes_or_reports_a_null(session, capsys):
    """C2 exactly as the runbook writes it. With one proposal per seed a null
    is likely, and that must surface as the documented NULL RESULT exit -- not
    as any other exception -- with the timing lines already printed."""
    cli, common, _files, results = session
    argv = [a for a in common]
    argv[argv.index("--seeds") + 1 : argv.index("--seeds") + 4] = ["0", "1"]
    try:
        code = cli.main(argv + ["--max-iters", "1", "--early-stop-patience", "0",
                                "--out", str(results / "c2")])
        assert code == 0
    except SystemExit as exc:
        assert "NULL RESULT" in str(exc)
    assert "adapt-and-eval #1:" in capsys.readouterr().out


# ------------------------------------------------------------ crash + resume --


class _Crash(RuntimeError):
    """Stands in for an OOM, a preemption or a Ctrl-C part-way through a run."""


def _crash_factory_after(monkeypatch, n):
    """Make every adapt-and-eval built from here on die on the n-th call overall."""
    from trustgate.eval import harness

    real = harness.make_adapt_and_eval
    calls = {"n": 0}

    def factory(*a, **k):
        fn = real(*a, **k)

        def dying(stream, condition):
            calls["n"] += 1
            if calls["n"] == n:
                raise _Crash(f"injected at evaluation {n}")
            return fn(stream, condition)

        return dying

    monkeypatch.setattr(harness, "make_adapt_and_eval", factory)
    return lambda: monkeypatch.setattr(harness, "make_adapt_and_eval", real)


def _real_evals(out: str) -> int:
    return out.count("adapt-and-eval #")


def test_c3_resumes_from_the_ledger_after_a_crash(session, capsys, monkeypatch):
    cli, common, _files, results = session
    argv = common + ["--max-iters", "6", "--early-stop-patience", "0"]

    assert cli.main(argv + ["--out", str(results / "ref")]) == 0
    total = _real_evals(capsys.readouterr().out)
    ref = pickle.loads((results / "ref" / "spike-result.pkl").read_bytes())
    assert (results / "ref" / "search-log.jsonl").read_text().count("\n") == 3

    restore = _crash_factory_after(monkeypatch, 9)
    with pytest.raises(_Crash):
        cli.main(argv + ["--out", str(results / "run")])
    restore()
    capsys.readouterr()
    assert (results / "run" / "ledger.jsonl").read_text().count("\n") == 8

    assert cli.main(argv + ["--out", str(results / "run")]) == 0
    out = capsys.readouterr().out
    assert "8 finished evaluation(s) on record" in out
    assert _real_evals(out) == total - 8, "a resumed run recomputed finished work"
    got = pickle.loads((results / "run" / "spike-result.pkl").read_bytes())
    assert got.per_seed_poisoned == ref.per_seed_poisoned
    assert got.per_seed_control == ref.per_seed_control
    assert (results / "run" / "arms.pkl").read_bytes() == (results / "ref" / "arms.pkl").read_bytes()


def test_c2_no_resume_times_every_evaluation_again(session, capsys):
    cli, common, _files, results = session
    argv = common + ["--max-iters", "4", "--early-stop-patience", "0", "--out", str(results / "c2")]
    assert cli.main(argv) == 0
    first = _real_evals(capsys.readouterr().out)
    assert cli.main(argv + ["--no-resume"]) == 0
    assert _real_evals(capsys.readouterr().out) == first


def test_c5_resumes_from_the_ledger_after_a_crash(session, monkeypatch):
    cli, common, files, results = session
    assert cli.main(common + ["--max-iters", "6", "--early-stop-patience", "0",
                              "--out", str(results)]) == 0
    gate = common + ["--gate-eval", "--arms-file", str(results / "arms.pkl"),
                     "--probe-file", str(files["probe"]), "--overhead-repeats", "0"]

    assert cli.main(gate + ["--out", str(results / "gref")]) == 0
    ref = pickle.loads((results / "gref" / "gate-result.pkl").read_bytes())

    # 6 ungated arm runs, then die inside the first gated threshold.
    restore = _crash_factory_after(monkeypatch, 8)
    with pytest.raises(_Crash):
        cli.main(gate + ["--out", str(results / "g")])
    restore()
    lines = (results / "g" / "ledger.jsonl").read_text().splitlines()
    assert sum('"gate/calibration"' in l for l in lines) == 1
    assert sum('"gate/arm"' in l for l in lines) == 7

    assert cli.main(gate + ["--out", str(results / "g")]) == 0
    got = pickle.loads((results / "g" / "gate-result.pkl").read_bytes())
    assert got.calibration == ref.calibration
    assert got.ungated == ref.ungated
    assert got.thresholds == ref.thresholds


def test_c4_resumes_from_the_ledger_after_a_crash(session, monkeypatch):
    cli, common, _files, results = session
    from trustgate.eval import harness

    assert cli.main(common + ["--max-iters", "6", "--early-stop-patience", "0",
                              "--out", str(results)]) == 0
    seq = common + ["--sequence-eval", "--arms-file", str(results / "arms.pkl")]
    assert cli.main(seq + ["--out", str(results / "sref")]) == 0
    ref = pickle.loads((results / "sref" / "sequence-result.pkl").read_bytes())

    real = harness.eval_benign
    calls = {"n": 0}

    def dying(*a, **k):
        calls["n"] += 1
        if calls["n"] == 20:  # floor + seed 0 (4 arms x 4 windows), then into seed 1
            raise _Crash("injected")
        return real(*a, **k)

    monkeypatch.setattr(harness, "eval_benign", dying)
    with pytest.raises(_Crash):
        cli.main(seq + ["--out", str(results / "s")])
    monkeypatch.setattr(harness, "eval_benign", real)
    assert (results / "s" / "ledger.jsonl").read_text().count('"sequence/arm"') == 4

    assert cli.main(seq + ["--out", str(results / "s")]) == 0
    got = pickle.loads((results / "s" / "sequence-result.pkl").read_bytes())
    assert got.floor_loss == ref.floor_loss
    assert got.per_seed == ref.per_seed


def test_a_null_spike_leaves_its_record_on_disk(session, monkeypatch):
    cli, common, _files, results = session
    from trustgate.eval import attacker

    def make_craft_fn(*, on_result=None, **_k):
        def craft_fn(seed):
            result = types.SimpleNamespace(
                n_evaluations=4, n_accepted=0, gain=0.0, baseline_score=1.0,
                best_score=1.0, stopped_early=False,
            )
            on_result(seed, result)
            raise attacker.NoOrderingFound(f"seed {seed} accepted nothing")
        return craft_fn

    monkeypatch.setattr(attacker, "make_craft_fn", make_craft_fn)
    with pytest.raises(SystemExit, match="NULL RESULT"):
        cli.main(common + ["--out", str(results)])
    assert "NULL RESULT" in (results / "NULL-RESULT.md").read_text(encoding="utf-8")
    assert '"n_accepted": 0' in (results / "search-log.jsonl").read_text()


def test_random_init_runs_keep_no_ledger(session):
    """A replayed smoke test would pass without touching the vendor forward."""
    cli, _common, _files, results = session
    out = results / "ri"
    args = ["--objective", "degrade", "--strategy", "select", "--random-init",
            "--seq-length", str(SEQ), "--stream-tokens", str(SEQ),
            "--corpus-tokens", "4096", "--seeds", "0", "1", "--gate-eval",
            "--overhead-repeats", "0", "--out", str(out)]
    assert cli.main(args) == 0
    assert not (out / "ledger.jsonl").exists()


def test_ledger_survives_a_torn_last_line(tmp_path):
    from trustgate.eval.ledger import Ledger

    path = tmp_path / "ledger.jsonl"
    led = Ledger(path)
    led.put("a", "k", {"loss": 1.5})
    with path.open("a", encoding="utf-8") as fh:
        fh.write('{"key": "b", "kind": "k", "val')  # died mid-write
    led = Ledger(path)
    assert led.get("a") == {"loss": 1.5} and led.get("b") is None
    led.put("c", "k", {"loss": float("nan")})
    led = Ledger(path)
    assert led.get("a") == {"loss": 1.5}
    assert np.isnan(led.get("c")["loss"])
