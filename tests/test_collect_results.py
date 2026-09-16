"""Nothing secret leaves the box: the vendor logs its whole environment.

`ttt/train.py:80` logs every environment variable and `:81` the resolved config,
which carries `wandb_key`. This repository is public. These tests feed the
collector text shaped like that output and assert the key is gone from the copy.
"""

import importlib.util
import sys
from pathlib import Path

import pytest

_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "collect_results.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("collect_results", _SCRIPT)
    module = importlib.util.module_from_spec(spec)
    sys.modules["collect_results"] = module
    spec.loader.exec_module(module)
    return module


cr = _load_module()

KEY = "0123456789abcdef0123456789abcdef01234567"
ENV = {"WANDB_KEY": KEY, "GCP_BILLING_PROJECT": "project-a364e9db", "HOME": "/root", "SHLVL": "1"}

VENDOR_LOG = f"""[INFO] - HOME=/root
WANDB_KEY={KEY}
SOME_OTHER_API_TOKEN=unrelated-token-value-not-in-our-env
GCP_BILLING_PROJECT=project-a364e9db
[INFO] - Launching with
 {{'training': {{'dataset_name': 'books3', 'wandb_key': '{KEY}', 'overrides': ['training.wandb_key={KEY}']}}}}.
[INFO] - Eval -- train_holdout/loss: 2.6512
"""


def session(root: Path) -> Path:
    (root / "logs").mkdir(parents=True)
    (root / "logs" / "eval-1.log").write_text(VENDOR_LOG, encoding="utf-8")
    (root / "eval-1").mkdir()
    (root / "eval-1" / "train_holdout_token_nll_loss.npy").write_bytes(b"\x93NUMPY fake")
    return root


def test_the_key_is_gone_and_the_result_survives(tmp_path):
    src = session(tmp_path / "src")
    dest = tmp_path / "dest"
    assert cr.main(["--src", str(src), "--dest", str(dest)], environ=ENV) == 0

    text = (dest / "logs" / "eval-1.log").read_text(encoding="utf-8")
    assert KEY not in text
    assert "project-a364e9db" not in text
    assert "unrelated-token-value" not in text  # caught by name pattern, not by literal
    assert "Eval -- train_holdout/loss: 2.6512" in text
    assert "'dataset_name': 'books3'" in text  # S4 still readable
    assert "HOME=/root" in text
    assert (dest / "eval-1" / "train_holdout_token_nll_loss.npy").read_bytes() == b"\x93NUMPY fake"


def test_the_key_is_redacted_by_pattern_even_if_not_exported(tmp_path):
    """Collector run from a shell that forgot to export WANDB_KEY."""
    src = session(tmp_path / "src")
    dest = tmp_path / "dest"
    assert cr.main(["--src", str(src), "--dest", str(dest)], environ={"HOME": "/root"}) == 0
    assert KEY not in (dest / "logs" / "eval-1.log").read_text(encoding="utf-8")


def test_a_secret_in_a_binary_file_is_a_leak(tmp_path):
    """Binaries are copied verbatim, so the post-copy scan must read them too."""
    src = session(tmp_path / "src")
    (src / "blob.npy").write_bytes(b"xx" + KEY.encode() + b"yy")
    assert cr.main(["--src", str(src), "--dest", str(tmp_path / "dest")], environ=ENV) == 2


def test_short_values_are_not_treated_as_secrets():
    assert "SOME_KEY" not in cr.secret_values(dict(ENV, SOME_KEY="1"))


def test_refuses_to_mix_into_an_existing_session(tmp_path):
    src = session(tmp_path / "src")
    dest = tmp_path / "dest"
    dest.mkdir()
    (dest / "old.txt").write_text("x")
    with pytest.raises(SystemExit, match="not empty"):
        cr.main(["--src", str(src), "--dest", str(dest)], environ=ENV)
