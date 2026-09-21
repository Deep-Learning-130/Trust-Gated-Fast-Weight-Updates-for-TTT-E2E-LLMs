"""`scripts/dump_tokens.py` against a fabricated raw-bytes zarr store. No network."""

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest

_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "dump_tokens.py"


@pytest.fixture(scope="module")
def dump():
    sys.path.insert(0, str(_SCRIPT.parent))
    spec = importlib.util.spec_from_file_location("dump_tokens", _SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


CHUNK = 5000


def make_array(root: Path, name: str, data: np.ndarray, key: str = "c/0") -> Path:
    array_dir = root / name
    (array_dir / key).parent.mkdir(parents=True, exist_ok=True)
    meta = {
        "shape": [CHUNK],
        "data_type": "uint32",
        "chunk_grid": {"name": "regular", "configuration": {"chunk_shape": [CHUNK]}},
        "codecs": [{"name": "bytes", "configuration": {"endian": "little"}}],
        "fill_value": 0,
    }
    (array_dir / "zarr.json").write_text(json.dumps(meta), encoding="utf-8")
    data.astype("<u4").tofile(array_dir / key)
    return array_dir


def run(dump, tmp_path, val, train, **overrides):
    books3 = tmp_path / "books3"
    make_array(books3, "val", val)
    train_root = tmp_path / "train-root"
    make_array(train_root, "train", train, key="0")
    out = tmp_path / "out"
    args = {
        "--books3": books3, "--out": out, "--train-dir": train_root,
        "--seq-length": 64, "--mini-batch": 16, "--train-tokens": 1000,
        "--probe-offset": 2000,
    }
    args.update(overrides)
    argv = [str(x) for kv in args.items() for x in kv]
    assert dump.main(argv) == 0
    return out


def test_writes_disjoint_int32_slices_from_the_right_splits(dump, tmp_path):
    val = np.arange(10, 10 + CHUNK)
    train = np.arange(100_000, 100_000 + CHUNK)
    out = run(dump, tmp_path, val, train)

    v, p, t = (np.load(out / f"{n}.npy") for n in ("val", "probe", "train"))
    assert v.dtype == p.dtype == t.dtype == np.int32
    np.testing.assert_array_equal(v, val[:65])
    np.testing.assert_array_equal(p, val[2000:2017])
    np.testing.assert_array_equal(t, train[:1000])
    assert not set(v) & set(p), "probe overlaps the eval"

    manifest = json.loads((out / "tokens-manifest.json").read_text(encoding="utf-8"))
    assert set(manifest["files"]) == {"val.npy", "probe.npy", "train.npy"}


def test_refuses_the_fill_value_trap(dump, tmp_path):
    with pytest.raises(SystemExit, match="fill-value trap"):
        run(dump, tmp_path, np.zeros(CHUNK), np.arange(CHUNK))


def test_refuses_a_probe_that_overlaps_the_eval(dump, tmp_path):
    with pytest.raises(SystemExit, match="overlaps"):
        run(dump, tmp_path, np.arange(CHUNK), np.arange(CHUNK), **{"--probe-offset": 10})


def test_refuses_a_truncated_chunk(dump, tmp_path):
    books3 = tmp_path / "books3"
    array_dir = make_array(books3, "val", np.arange(1, CHUNK + 1))
    np.arange(1, 100).astype("<u4").tofile(array_dir / "c" / "0")
    meta = json.loads((array_dir / "zarr.json").read_text(encoding="utf-8"))
    with pytest.raises(SystemExit, match="Partial download"):
        dump.read_slice(array_dir, meta, array_dir / "c" / "0", 0, 10)
