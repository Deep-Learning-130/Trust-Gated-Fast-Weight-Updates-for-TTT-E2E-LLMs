"""CLI routing: which invocations run, which are refused, and what they say.

CPU only, no vendor tree. Everything here is reachable before the first vendor
import, which is the point -- these are the guards that stop a
verdict-shaped artifact being produced by accident, and they must hold on any
machine.

There were no tests over this layer before. `cli.main` is where the
pre-registered thresholds are loaded and where `write_report` is called, and
`harness.run_attack_spike`'s own contract says the comparison is made "explicitly
by the caller, so the bar cannot be quietly relaxed inside the code that produces
the number". That makes this layer load-bearing for Standing Rule 5.
"""

import json
from pathlib import Path

import pytest

from trustgate.eval import cli


def run(args):
    """Invoke main, returning (exit_code_or_None, SystemExit message or None)."""
    try:
        return cli.main(args), None
    except SystemExit as exc:
        return None, str(exc.code if exc.code is not None else "")


BASE = ["--objective", "degrade", "--strategy", "select"]


# ---------------------------------------------------------------- refusals ---


def test_a_run_with_no_victim_is_refused(tmp_path):
    code, message = run([*BASE, "--out", str(tmp_path)])
    assert code is None
    assert "needs weights" in message


def test_the_refusal_names_both_escape_hatches(tmp_path):
    """A refusal that does not say what to do instead is a dead end."""
    _, message = run([*BASE, "--out", str(tmp_path)])
    assert "--random-init" in message
    assert "--dry-run" in message
    # and it must say what --random-init does NOT give you
    assert "NO verdict" in message


def test_random_init_and_checkpoint_are_mutually_exclusive(tmp_path):
    code, message = run(
        [*BASE, "--out", str(tmp_path), "--random-init", "--checkpoint", "gs://x"]
    )
    assert code is None
    assert "mutually exclusive" in message


@pytest.mark.parametrize(
    "missing,args",
    [
        ("--objective", ["--strategy", "select", "--out", "OUT"]),
        ("--strategy", ["--objective", "degrade", "--out", "OUT"]),
        ("--out", ["--objective", "degrade", "--strategy", "select"]),
    ],
)
def test_required_arguments_are_named_when_absent(missing, args, tmp_path):
    args = [str(tmp_path) if a == "OUT" else a for a in args]
    code, message = run(args)
    assert code is None
    assert missing in message


# ------------------------------------------------------ the dry-run banner ---


def test_dry_run_writes_a_report_marked_as_not_a_result(tmp_path, capsys):
    code, _ = run([*BASE, "--out", str(tmp_path), "--dry-run"])
    assert code == 0

    report = (tmp_path / "report.md").read_text(encoding="utf-8")
    assert report.startswith("> **")
    assert "NOT A RESULT" in report
    assert "do not cite it" in report.lower()


def test_dry_run_banner_precedes_the_verdict_line(tmp_path):
    """Order matters. A verdict read before its caveat is a verdict."""
    run([*BASE, "--out", str(tmp_path), "--dry-run"])
    report = (tmp_path / "report.md").read_text(encoding="utf-8")
    assert report.index("NOT A RESULT") < report.index("Verdict")


def test_fewer_than_five_seeds_warns(tmp_path, capsys):
    """The pre-registration names fewer than 5 usable seeds an invalidating
    condition, so this must be loud even though it is not fatal."""
    run([*BASE, "--out", str(tmp_path), "--dry-run", "--seeds", "0", "1"])
    stderr = capsys.readouterr().err
    assert "2 seeds given" in stderr
    assert "invalidating" in stderr


# ------------------------------------------------- the random-init contract ---


def test_random_init_banner_refuses_to_look_like_a_result():
    """The banner text itself is the guard. Assert its substance, not its
    wording drifting silently."""
    banner = cli.NOT_A_RESULT
    assert "NOT A RESULT" in banner
    assert "learned nothing" in banner
    assert "do not cite" in banner.lower()
    assert "verdict" in banner.lower()


def test_banner_report_prepends_without_losing_content(tmp_path):
    report = tmp_path / "report.md"
    report.write_text("# Original\n\nbody\n", encoding="utf-8")
    cli._banner_report(report, "CAVEAT HERE")

    text = report.read_text(encoding="utf-8")
    assert text.startswith("> **CAVEAT HERE**")
    assert "# Original" in text
    assert "body" in text


def test_random_init_without_a_vendor_tree_exits_legibly(tmp_path):
    """On a CPU box this is the expected path. It must not be a traceback, and
    it must distinguish 'expected here' from 'your GPU install is broken'."""
    code, message = run([*BASE, "--out", str(tmp_path), "--random-init"])
    assert code is None
    assert "EXPECTED to fail in the CPU test environment" in message
    assert "setup.sh" in message


def test_random_init_rejects_an_unknown_size_without_a_traceback(tmp_path):
    code, message = run([*BASE, "--out", str(tmp_path), "--random-init", "--size", "7b"])
    assert code is None
    assert "unknown size" in message


def test_random_init_rejects_a_ragged_seq_length_without_a_traceback(tmp_path):
    code, message = run(
        [*BASE, "--out", str(tmp_path), "--random-init", "--seq-length", "5000"]
    )
    assert code is None
    assert "1024" in message


# ------------------------------------------------------------------ parser ---


def test_defaults_match_the_pre_registered_granularity():
    args = cli.build_parser().parse_args([*BASE, "--out", "/tmp/x"])
    # 8192 / 1024 = 8 inner steps, the granularity PREREGISTERED.md is scoped to.
    assert args.seq_length == 8192
    assert args.seeds == [0, 1, 2, 3, 4]
    assert args.random_init is False
    assert args.dry_run is False


# ------------------------------------------------------ sequence-position ---


def test_sequence_eval_without_a_victim_is_refused(tmp_path):
    code, message = run([*BASE, "--out", str(tmp_path), "--sequence-eval"])
    assert code is None
    assert "needs a victim" in message


def test_the_no_victim_refusal_names_the_sequence_escape_hatch(tmp_path):
    """The refusal enumerates the ways to run something; keep it complete.

    A mode that exists but is absent from this message is a mode nobody finds.
    """
    _code, message = run([*BASE, "--out", str(tmp_path)])
    assert "--sequence-eval" in message


def test_sequence_defaults_are_the_fifty_windows_the_design_asks_for():
    args = cli.build_parser().parse_args([*BASE, "--out", "/tmp/x"])
    assert args.sequence_eval is False
    assert args.windows == 50
    assert args.eval_every == 1


# ------------------------------------------------------- checkpoint runs ---


def _ckpt_argv(tmp_path, **overrides):
    import numpy as np

    ckpt = tmp_path / "ckpt"
    (ckpt / "12000").mkdir(parents=True, exist_ok=True)
    for name in ("corpus.npy", "eval.npy"):
        if not (tmp_path / name).exists():
            np.save(tmp_path / name, np.arange(4096, dtype=np.int32) % 40_000)
    argv = {
        "--objective": "degrade",
        "--strategy": "select",
        "--checkpoint": str(ckpt),
        "--corpus-file": str(tmp_path / "corpus.npy"),
        "--eval-file": str(tmp_path / "eval.npy"),
        "--out": str(tmp_path / "out"),
    }
    argv.update(overrides)
    flat = []
    for flag, value in argv.items():
        if value is not None:
            flat += [flag, value]
    return flat


def test_a_checkpoint_run_refuses_to_fall_back_to_synthetic_tokens(tmp_path):
    """Against trained weights, `dummy_tokens` noise measures the reaction to
    gibberish. There is deliberately no synthetic fallback on this path."""
    argv = _ckpt_argv(tmp_path, **{"--corpus-file": None, "--eval-file": None})
    with pytest.raises(SystemExit) as excinfo:
        cli.main(argv)
    message = str(excinfo.value)
    assert "--corpus-file" in message
    assert "--eval-file" in message


def test_stream_and_eval_splits_must_differ(tmp_path):
    """Measuring degradation on the split the stream came from is named as an
    invalidating condition in PREREGISTERED.md."""
    argv = _ckpt_argv(tmp_path, **{"--corpus-split": "val", "--eval-split": "val"})
    with pytest.raises(SystemExit) as excinfo:
        cli.main(argv)
    assert "invalidating condition" in str(excinfo.value)


def test_a_nonexistent_checkpoint_fails_before_the_vendor_import(tmp_path):
    """On the box this failure should cost a second, not a launch."""
    argv = _ckpt_argv(tmp_path, **{"--checkpoint": str(tmp_path / "absent")})
    with pytest.raises(SystemExit) as excinfo:
        cli.main(argv)
    assert "absent" in str(excinfo.value)


def test_sequence_eval_now_accepts_a_checkpoint(tmp_path):
    """The guard used to say a checkpoint-backed run was 'not wired yet'."""
    argv = _ckpt_argv(tmp_path) + ["--sequence-eval"]
    with pytest.raises(SystemExit) as excinfo:
        cli.main(argv)
    assert "not wired yet" not in str(excinfo.value)


def test_the_weights_refusal_now_names_the_checkpoint_path(tmp_path):
    """The refusal lists every escape hatch; the real one had been missing."""
    with pytest.raises(SystemExit) as excinfo:
        cli.main(
            ["--objective", "degrade", "--strategy", "select", "--out", str(tmp_path)]
        )
    message = str(excinfo.value)
    assert "--checkpoint" in message
    assert "renders a verdict" in message


# ------------------------------------------------------ checkpoint label ---


def test_an_unfingerprinted_checkpoint_says_so(tmp_path):
    """A bare path names a directory that may have changed. If the run cannot
    quote a hash, the report must show that rather than imply provenance."""
    args = cli.build_parser().parse_args(
        ["--checkpoint", "/weights/1b", "--objective", "degrade"]
    )
    assert cli._checkpoint_label(args) == "/weights/1b (UNFINGERPRINTED)"


def test_the_manifest_hash_travels_into_the_label(tmp_path):
    digest = "a" * 64
    manifest = tmp_path / "manifest.txt"
    manifest.write_text(f"# files: 12\n# manifest_sha256: {digest}\n", encoding="utf-8")
    args = cli.build_parser().parse_args(
        [
            "--checkpoint", "/weights/1b",
            "--checkpoint-manifest", str(manifest),
            "--objective", "degrade",
        ]
    )
    assert cli._checkpoint_label(args) == f"/weights/1b sha256:{digest}"


def test_a_manifest_without_a_hash_is_refused(tmp_path):
    """Silently falling back to UNFINGERPRINTED would hide a broken manifest."""
    manifest = tmp_path / "manifest.txt"
    manifest.write_text("# files: 12\n", encoding="utf-8")
    args = cli.build_parser().parse_args(
        [
            "--checkpoint", "/weights/1b",
            "--checkpoint-manifest", str(manifest),
            "--objective", "degrade",
        ]
    )
    with pytest.raises(SystemExit) as excinfo:
        cli._checkpoint_label(args)
    assert "manifest_sha256" in str(excinfo.value)


# ---------------------------------------------------------- token files ---


def test_missing_token_file_names_how_to_make_one(tmp_path):
    with pytest.raises(SystemExit) as excinfo:
        cli._load_tokens(tmp_path / "absent.npy", "--corpus-file")
    assert ".npy" in str(excinfo.value)


def test_token_files_must_be_one_dimensional_integers(tmp_path):
    import numpy as np

    two_d = tmp_path / "two_d.npy"
    np.save(two_d, np.zeros((2, 2), dtype=np.int32))
    with pytest.raises(SystemExit) as excinfo:
        cli._load_tokens(two_d, "--corpus-file")
    assert "1-D" in str(excinfo.value)

    floats = tmp_path / "floats.npy"
    np.save(floats, np.zeros(4, dtype=np.float32))
    with pytest.raises(SystemExit) as excinfo:
        cli._load_tokens(floats, "--corpus-file")
    assert "integer" in str(excinfo.value)


def test_token_files_load_as_int32(tmp_path):
    import numpy as np

    path = tmp_path / "tokens.npy"
    np.save(path, np.arange(16, dtype=np.int64))
    loaded = cli._load_tokens(path, "--corpus-file")
    assert loaded.dtype.name == "int32"
    assert loaded.tolist() == list(range(16))
