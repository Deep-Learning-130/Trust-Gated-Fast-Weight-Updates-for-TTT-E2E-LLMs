"""The acceptance checker scores what TOLERANCE.md says, and nothing looser.

CPU only. The collected session is fabricated in the layout run_gpu_session.sh
writes, with log text shaped like the vendor's own output (train.py:80-81 and
loop.py:120), so the regexes are tested against what they will actually read.
"""

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "check_baseline_acceptance.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("check_baseline_acceptance", _SCRIPT)
    module = importlib.util.module_from_spec(spec)
    sys.modules["check_baseline_acceptance"] = module
    spec.loader.exec_module(module)
    return module


acc = _load_module()


def vendor_log(loss, *, dataset_name="books3", extra=""):
    return (
        "[2026-09-20 10:00:00][__main__][INFO] - PATH=/usr/bin\nWANDB_KEY=<REDACTED>:WANDB_KEY\n"
        "[2026-09-20 10:00:00][__main__][INFO] - Launching with \n"
        f" {{'training': {{'dataset_name': '{dataset_name}', 'dataset_path': '/mnt/data/llama3-books3',"
        " 'wandb_key': '<REDACTED>', 'seq_length': 8192}}}}.\n"
        "Evaluating on sequence train_holdout: 100%|##########| 2288/2288\n"
        f"[2026-09-20 10:40:00][ttt.model.loop][INFO] - Eval -- train_holdout/loss: {loss}\n"
        + extra
    )


def falling_curve(n=8192):
    return (3.5 - np.log1p(np.arange(n)) * 0.1).astype(np.float32)


def make_session(root: Path, *, l1="2.6512", l2="2.6512", lc="11.7", curve=None, dataset="books3",
                 skip=(), extra=""):
    logs = root / "logs"
    logs.mkdir(parents=True)
    for name, loss in (("smoke", "2.7"), ("eval-1", l1), ("eval-2", l2), ("control", lc)):
        if name in skip:
            continue
        (logs / f"{name}.log").write_text(vendor_log(loss, dataset_name=dataset, extra=extra), encoding="utf-8")
    if "npy" not in skip:
        (root / "eval-1").mkdir()
        np.save(root / "eval-1" / "train_holdout_token_nll_loss.npy", falling_curve() if curve is None else curve)
    return root


def statuses(root):
    return {name: status for name, status, _ in acc.evaluate(root)}


def test_a_clean_session_passes(tmp_path):
    make_session(tmp_path)
    assert acc.main(["--collected", str(tmp_path)]) == 0
    s = statuses(tmp_path)
    assert all(v in (acc.PASS, acc.INFO) for v in s.values()), s


@pytest.mark.parametrize("loss", ["2.314", "2.805", "2.2", "3.1", "nan"])
def test_band_is_strict_and_open_at_both_ends(tmp_path, loss):
    make_session(tmp_path, l1=loss, l2=loss)
    assert statuses(tmp_path)["Band"] == acc.FAIL
    assert acc.main(["--collected", str(tmp_path)]) == 1


def test_s1_fails_on_a_flat_curve(tmp_path):
    """A checkpoint loaded but not using context: the curve does not fall."""
    make_session(tmp_path, curve=np.full(8192, 2.65, dtype=np.float32))
    assert statuses(tmp_path)["S1 monotonicity"] == acc.FAIL


def test_s1_averages_a_batched_curve(tmp_path):
    make_session(tmp_path, curve=np.stack([falling_curve(), falling_curve()]))
    assert statuses(tmp_path)["S1 monotonicity"] == acc.PASS


def test_s2_fails_past_four_decimals(tmp_path):
    make_session(tmp_path, l1="2.65120", l2="2.65130")
    assert statuses(tmp_path)["S2 determinism"] == acc.FAIL


def test_s2_passes_on_agreement_below_four_decimals(tmp_path):
    make_session(tmp_path, l1="2.651201", l2="2.651204")
    assert statuses(tmp_path)["S2 determinism"] == acc.PASS


def test_a_dropped_run_two_is_unverified_not_passed(tmp_path):
    """Runbook s4 drop order: S2 is recorded as unverified, never quietly passed."""
    make_session(tmp_path, skip=("eval-2",))
    assert statuses(tmp_path)["S2 determinism"] == acc.UNVERIFIED
    assert acc.main(["--collected", str(tmp_path)]) == 2


def test_s3_fails_when_the_control_is_inside_the_band(tmp_path):
    make_session(tmp_path, lc="2.7")
    assert statuses(tmp_path)["S3 negative control"] == acc.FAIL


def test_s4_catches_the_dclm_mixup_the_band_cannot(tmp_path):
    """TOLERANCE s4.2: a DCLM number would land inside the band."""
    make_session(tmp_path, dataset="dclm_filter_8k")
    assert statuses(tmp_path)["Band"] == acc.PASS
    assert statuses(tmp_path)["S4 resolved config"] == acc.FAIL


def test_s4_fails_if_trustgate_appears_in_any_log(tmp_path):
    make_session(tmp_path, extra='  File "/root/TTT/src/trustgate/interceptor.py", line 3\n')
    assert statuses(tmp_path)["S4 gate absent"] == acc.FAIL


def test_missing_logs_are_unverified_everywhere(tmp_path):
    (tmp_path / "logs").mkdir()
    assert acc.main(["--collected", str(tmp_path)]) == 2


def test_report_is_written_when_asked(tmp_path):
    make_session(tmp_path / "s")
    out = tmp_path / "ACCEPTANCE.txt"
    acc.main(["--collected", str(tmp_path / "s"), "--out", str(out)])
    assert "VERDICT: PASS" in out.read_text(encoding="utf-8")


# --- 125M: TOLERANCE.md s6 gives it no numeric bar (PREREGISTERED.md revision 2026-09-22) ---

C125 = "125m_ttt_e2e_finetune_books_8k_1x_cc"


def statuses_for(root, checkpoint):
    return {name: status for name, status, _ in acc.evaluate(root, checkpoint=checkpoint)}


def test_125m_is_not_scored_against_the_1b_band(tmp_path):
    """A plausible 125M loss sits above the 1B band. That must not FAIL a 125M session."""
    make_session(tmp_path, l1="3.05", l2="3.05")
    s = statuses_for(tmp_path, C125)
    assert s["Band"] == acc.INFO
    assert "s4.3 expectation" not in s
    assert acc.main(["--collected", str(tmp_path), "--checkpoint", C125]) == 0


def test_125m_pass_says_it_is_not_a_reproduction(tmp_path, capsys):
    make_session(tmp_path, l1="3.05", l2="3.05")
    acc.main(["--collected", str(tmp_path), "--checkpoint", C125])
    out = capsys.readouterr().out
    assert "not a reproduction" in out
    assert "s4.1" not in out.split("VERDICT:")[1]


def test_125m_still_fails_s1_to_s4(tmp_path):
    make_session(tmp_path, l1="3.05", l2="3.06", curve=np.full(8192, 3.0, dtype=np.float32))
    s = statuses_for(tmp_path, C125)
    assert s["S1 monotonicity"] == acc.FAIL
    assert s["S2 determinism"] == acc.FAIL
    assert acc.main(["--collected", str(tmp_path), "--checkpoint", C125]) == 1


def test_125m_s3_needs_the_control_above_the_real_loss(tmp_path):
    """No band to be above, so S3 reads 'above the model's own real-data loss'."""
    make_session(tmp_path, l1="3.05", l2="3.05", lc="2.9")
    assert statuses_for(tmp_path, C125)["S3 negative control"] == acc.FAIL


def test_125m_s3_passes_a_random_token_control(tmp_path):
    make_session(tmp_path, l1="3.05", l2="3.05", lc="11.7")
    assert statuses_for(tmp_path, C125)["S3 negative control"] == acc.PASS


def test_the_1b_checkpoint_keeps_the_band(tmp_path):
    make_session(tmp_path, l1="3.05", l2="3.05")
    assert statuses_for(tmp_path, "1b_ttt_e2e_finetune_books_8k_1x_cc")["Band"] == acc.FAIL


def test_an_unknown_checkpoint_is_refused(tmp_path):
    make_session(tmp_path)
    with pytest.raises(SystemExit):
        acc.main(["--collected", str(tmp_path), "--checkpoint", "350m_ttt_e2e_pretrain_dclm_8k_1x_cc"])
