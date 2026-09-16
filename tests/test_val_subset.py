"""Truncating the local /val store, and the guard that keeps it honest.

CPU only, no network, no `zarr` import -- the store is fabricated as plain JSON
and byte files, exactly as zarr v3 lays one out on disk. `zarr` is absent from
this project's dependencies and from CI, so the subset tool has to be correct
against the on-disk format by itself.

The property under test is the one the whole GPU session plan rests on:

    ttt/dataloader/lm_dataset.py:27
    def __len__(self):
        return (self.split.shape[0] - 1) // self.seq_len

If the declared shape does not control the eval length, the session plan does
not work. If a shape larger than the chunks on disk is allowed through, the run
evaluates on zarr fill values and returns a confident, meaningless number -- the
failure this module exists to make impossible.
"""

import importlib.util
import json
import sys
from pathlib import Path

import pytest

_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "make_val_subset.py"


def _load_module():
    """Import the script by path -- `scripts/` is not a package."""
    spec = importlib.util.spec_from_file_location("make_val_subset", _SCRIPT)
    module = importlib.util.module_from_spec(spec)
    sys.modules["make_val_subset"] = module
    spec.loader.exec_module(module)
    return module


vs = _load_module()


CHUNK = 1_000_000
ORIGINAL = 2_000_168_321

# Each sample is seq_len + 1 tokens, so N sequences needs N * 8192 + 1 tokens.
# 16 sequences is exactly two eval batches of 8 -- the smoke pass.
SMOKE_TOKENS = 16 * 8192 + 1


def build_store(root: Path, *, n_chunks: int, shape: int, nested: bool = True) -> Path:
    """Lay out a zarr v3 array on disk the way the real /val is laid out."""
    val = root / "val"
    val.mkdir(parents=True, exist_ok=True)
    meta = {
        "zarr_format": 3,
        "node_type": "array",
        "shape": [shape],
        "data_type": "uint32",
        "chunk_grid": {"name": "regular", "configuration": {"chunk_shape": [CHUNK]}},
        "chunk_key_encoding": {"name": "default", "configuration": {"separator": "/"}},
        "fill_value": 0,
        "codecs": [{"name": "bytes", "configuration": {"endian": "little"}}],
    }
    (val / "zarr.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    chunk_dir = val / "c" if nested else val
    chunk_dir.mkdir(parents=True, exist_ok=True)
    for i in range(n_chunks):
        # Content is irrelevant to the shape logic; size is not, so keep it small.
        (chunk_dir / str(i)).write_bytes(bytes([i % 256]) * 64)
    return root


def declared_shape(root: Path) -> int:
    return json.loads((root / "val" / "zarr.json").read_text(encoding="utf-8"))["shape"][0]


def vendor_len(tokens: int, seq_len: int = 8192) -> int:
    """Reproduce `lm_dataset.Dataset.__len__` exactly."""
    return (tokens - 1) // seq_len


# ------------------------------------------------------------------ reshape ---


def test_reshape_sets_the_shape_that_controls_eval_length(tmp_path):
    build_store(tmp_path, n_chunks=3, shape=ORIGINAL)

    vs.main(["--dest", str(tmp_path), "reshape", "--tokens", "2_000_000".replace("_", "")])

    assert declared_shape(tmp_path) == 2_000_000
    assert vs.sequences_for(2_000_000) == vendor_len(2_000_000) == 244


def test_reshape_is_reversible_without_refetching(tmp_path):
    """The smoke-pass move: tiny, then real, with no transfer in between."""
    build_store(tmp_path, n_chunks=3, shape=ORIGINAL)

    vs.main(["--dest", str(tmp_path), "reshape", "--tokens", str(SMOKE_TOKENS)])
    assert declared_shape(tmp_path) == SMOKE_TOKENS
    assert vendor_len(SMOKE_TOKENS) == 16  # exactly two eval batches of 8

    vs.main(["--dest", str(tmp_path), "reshape", "--tokens", "3000000"])
    assert declared_shape(tmp_path) == 3_000_000
    assert vendor_len(3_000_000) == 366


def test_reshape_never_exceeds_the_original(tmp_path):
    """Clamping happens before the fill guard, so --allow-fill isolates it."""
    build_store(tmp_path, n_chunks=3, shape=ORIGINAL)
    vs.main(["--dest", str(tmp_path), "reshape", "--tokens", str(ORIGINAL * 2), "--allow-fill"])
    assert declared_shape(tmp_path) == ORIGINAL


# --------------------------------------------------------- the fill-value guard --


def test_reshape_refuses_a_shape_larger_than_the_chunks_on_disk(tmp_path):
    """The failure mode nothing else in the stack would catch.

    Absent zarr chunks read as the fill value rather than raising, so an
    over-large shape yields a loss computed over token id 0 -- no error, no
    warning, and a number that looks like a result.
    """
    build_store(tmp_path, n_chunks=3, shape=ORIGINAL)  # 3M tokens on disk

    with pytest.raises(SystemExit) as excinfo:
        vs.main(["--dest", str(tmp_path), "reshape", "--tokens", "5000000"])

    message = str(excinfo.value)
    assert "exceeds" in message
    assert "fill value" in message
    # The shape must be untouched by a refused reshape.
    assert declared_shape(tmp_path) == ORIGINAL


def test_allow_fill_overrides_the_guard_deliberately(tmp_path):
    build_store(tmp_path, n_chunks=3, shape=ORIGINAL)
    vs.main(["--dest", str(tmp_path), "reshape", "--tokens", "5000000", "--allow-fill"])
    assert declared_shape(tmp_path) == 5_000_000


def test_exactly_the_tokens_on_disk_is_allowed(tmp_path):
    build_store(tmp_path, n_chunks=3, shape=ORIGINAL)
    vs.main(["--dest", str(tmp_path), "reshape", "--tokens", str(3 * CHUNK)])
    assert declared_shape(tmp_path) == 3 * CHUNK


# ------------------------------------------------------- chunk key discovery ---


@pytest.mark.parametrize("nested", [True, False])
def test_finds_chunks_under_either_key_encoding(tmp_path, nested):
    """zarr v3 writes `val/c/0`; the v2 style writes `val/0`. Discover, don't assume.

    Guessing wrong reports zero chunks, which would make the guard refuse a
    reshape that is in fact perfectly safe.
    """
    build_store(tmp_path, n_chunks=4, shape=ORIGINAL, nested=nested)
    assert len(vs.chunk_files(tmp_path)) == 4

    vs.main(["--dest", str(tmp_path), "reshape", "--tokens", str(4 * CHUNK)])
    assert declared_shape(tmp_path) == 4 * CHUNK


def test_chunk_files_are_ordered_numerically_not_lexically(tmp_path):
    """`10` sorts before `2` as a string; the count is right either way, but a
    lexical order would put the wrong bytes in the manifest's digest table."""
    build_store(tmp_path, n_chunks=12, shape=ORIGINAL)
    names = [p.name for p in vs.chunk_files(tmp_path)]
    assert names == [str(i) for i in range(12)]


# ------------------------------------------------------------------ manifest ---


def test_manifest_records_what_was_cut(tmp_path):
    build_store(tmp_path, n_chunks=3, shape=ORIGINAL)
    vs.main(["--dest", str(tmp_path), "reshape", "--tokens", "2000000"])

    manifest = json.loads((tmp_path / vs.MANIFEST_NAME).read_text(encoding="utf-8"))
    assert manifest["original_shape"] == ORIGINAL
    assert manifest["truncated_shape"] == 2_000_000
    assert manifest["chunks_on_disk"] == 3
    assert manifest["tokens_on_disk"] == 3 * CHUNK
    assert manifest["sequences"] == vendor_len(2_000_000)
    assert manifest["seq_length"] == 8192
    # Every chunk hashed, so the result can name the bytes it was computed over.
    assert set(manifest["chunk_sha256"]) == {"0", "1", "2"}


def test_manifest_original_shape_survives_repeated_reshaping(tmp_path):
    """The second reshape reads the truncated shape; it must not adopt it as the
    original, or the recorded fraction-of-/val drifts with every resize."""
    build_store(tmp_path, n_chunks=3, shape=ORIGINAL)

    vs.main(["--dest", str(tmp_path), "reshape", "--tokens", str(SMOKE_TOKENS)])
    vs.main(["--dest", str(tmp_path), "reshape", "--tokens", "2000000"])

    manifest = json.loads((tmp_path / vs.MANIFEST_NAME).read_text(encoding="utf-8"))
    assert manifest["original_shape"] == ORIGINAL
    assert manifest["fraction_of_val"] == pytest.approx(2_000_000 / ORIGINAL, rel=1e-6)


# -------------------------------------------------------------------- arithmetic --


@pytest.mark.parametrize(
    "tokens,expected_sequences",
    [
        (SMOKE_TOKENS, 16),    # the smoke pass: exactly two eval batches
        (8193, 1),
        (8192, 0),             # one token short of a full sequence
        (150_000_000, 18310),
        (ORIGINAL, 244161),    # the full split, for reference
    ],
)
def test_sequence_arithmetic_matches_the_vendor_loader(tokens, expected_sequences):
    assert vs.sequences_for(tokens) == vendor_len(tokens) == expected_sequences


def test_reshape_on_a_missing_store_fails_loudly(tmp_path):
    with pytest.raises(SystemExit):
        vs.main(["--dest", str(tmp_path), "reshape", "--tokens", "1000"])
