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
