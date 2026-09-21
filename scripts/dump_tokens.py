#!/usr/bin/env python
"""Dump the token files the 001 spike and the Phase 2 gate measurement read.

`python -m trustgate.eval.cli --checkpoint ...` takes `.npy` files of int32 token
ids (`--corpus-file`, `--eval-file`, `--probe-file`), and the bootstrap stages
books3 as zarr. This is the bridge, run on the box after `bootstrap_gpu_box.sh`:

    python3 scripts/dump_tokens.py --books3 "$BOOKS3_LOCAL" \
        --billing "$GCP_BILLING_PROJECT" --out "$EXP_DIR/tokens"

Writes, into --out:

  train.npy   the attacker's span corpus, from chunk 0 of books3 /train
              (fetched here: the bootstrap stages only train/zarr.json)
  val.npy     the benign eval: seq_length + 1 tokens from the start of /val
  probe.npy   the gate's probe window: from /val at --probe-offset, disjoint
              from val.npy by construction
  tokens-manifest.json   offsets, lengths and sha256 of each file

Splits are kept apart on purpose. The corpus comes from /train and the eval
from /val because PREREGISTERED.md names measuring on the stream's own split as
an invalidating condition; the probe comes from a different region of /val than
the eval so the gate is not scored on the text it is judged by.

The fill-value trap
-------------------
In zarr v3 an absent chunk reads as the fill value (0) rather than raising
(see `make_val_subset.py`). A slice that is all one value is refused here: fed
to the spike, it would measure the model's reaction to a constant stream.
"""

from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import make_val_subset as mvs  # noqa: E402

DEFAULT_TRAIN_TOKENS = 16_000_000
DEFAULT_PROBE_OFFSET = 10_000_000


def _codecs(meta: dict) -> list[str]:
    return [c.get("name", "?") for c in meta.get("codecs", [])]


def _is_raw(meta: dict) -> bool:
    """Plain little-endian bytes, readable with `np.fromfile`."""
    codecs = meta.get("codecs", [])
    if len(codecs) != 1 or codecs[0].get("name") != "bytes":
        return False
    return codecs[0].get("configuration", {}).get("endian", "little") == "little"


def _np_dtype(meta: dict) -> np.dtype:
    name = meta.get("data_type") or meta.get("dtype")
    if name not in ("uint32", "int32", "<u4", "<i4"):
        sys.exit(f"unexpected token dtype {name!r}; expected uint32 or int32")
    return np.dtype("<u4") if "u" in name else np.dtype("<i4")


def read_slice(array_dir: Path, meta: dict, chunk_file: Path, start: int, length: int) -> np.ndarray:
    """Tokens [start, start + length) from chunk 0 of a 1-D array."""
    chunk = mvs.chunk_shape_of(meta)
    if start + length > chunk:
        sys.exit(
            f"slice [{start}, {start + length}) runs past chunk 0 "
            f"({chunk:,} tokens); only chunk 0 is read here"
        )
    if _is_raw(meta):
        data = np.fromfile(chunk_file, dtype=_np_dtype(meta))
        if data.shape[0] != chunk:
            sys.exit(
                f"{chunk_file} holds {data.shape[0]:,} tokens, the metadata "
                f"declares chunks of {chunk:,}. Partial download?"
            )
        out = data[start : start + length]
    else:
        # Compressed chunks: let zarr decode. The vendor env has it.
        try:
            import zarr
        except ImportError:
            sys.exit(
                f"{array_dir} uses codecs {_codecs(meta)}, which need the zarr "
                f"library. Run this under the vendor env: "
                f"(cd vendor/ttt-e2e && uv run python ../../scripts/dump_tokens.py ...)"
            )
        out = np.asarray(zarr.open_array(str(array_dir), mode="r")[start : start + length])

    out = np.asarray(out)
    if out.size and (out == out[0]).all():
        sys.exit(
            f"slice [{start}, {start + length}) of {array_dir} is the constant "
            f"{out[0]} -- the zarr fill-value trap (an absent chunk reads as "
            f"fill). Refusing to write it."
        )
    if int(out.max()) >= 2**31:
        sys.exit("token id does not fit int32")
    return out.astype(np.int32)


def _find_chunk0(array_dir: Path) -> Path | None:
    for candidate in (array_dir / "c" / "0", array_dir / "0"):
        if candidate.is_file():
            return candidate
    return None


def fetch_train_chunk0(billing: str, bucket: str, dest: Path) -> Path:
    """Copy /train's metadata and chunk 0 into `dest/train`. Returns the array dir."""
    array_dir = dest / "train"
    array_dir.mkdir(parents=True, exist_ok=True)
    meta_path = array_dir / "zarr.json"
    if not meta_path.is_file():
        for prefix in (f"{bucket}/train", f"{bucket}/data.zarr/train"):
            if mvs.gsutil(billing, "cp", f"{prefix}/zarr.json", str(meta_path), quiet=True).returncode == 0:
                break
        else:
            sys.exit(f"could not read train/zarr.json under {bucket}")
    else:
        prefix = None

    if _find_chunk0(array_dir) is None:
        prefixes = [prefix] if prefix else [f"{bucket}/train", f"{bucket}/data.zarr/train"]
        for p in prefixes:
            for key, local in (("c/0", array_dir / "c" / "0"), ("0", array_dir / "0")):
                local.parent.mkdir(parents=True, exist_ok=True)
                print(f"  fetching {p}/{key}")
                if mvs.gsutil(billing, "cp", f"{p}/{key}", str(local), quiet=True).returncode == 0:
                    return array_dir
        sys.exit(f"could not fetch chunk 0 of train under {bucket} (tried c/0 and 0)")
    return array_dir


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--books3", type=Path, default=mvs.default_dest(),
                    help="Local books3 store root (BOOKS3_LOCAL), holding val/.")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--billing", default=None,
                    help="GCP project for requester-pays reads of /train. Needed "
                         "unless train chunk 0 is already under --train-dir.")
    ap.add_argument("--bucket", default=mvs.DEFAULT_BUCKET)
    ap.add_argument("--train-dir", type=Path, default=None,
                    help="Where train chunk 0 is (or will be) kept. Default: "
                         "<out>/train-zarr, deliberately NOT inside --books3, "
                         "so the 000 baseline's store is never touched.")
    ap.add_argument("--seq-length", type=int, default=mvs.SEQ_LENGTH)
    ap.add_argument("--mini-batch", type=int, default=1024)
    ap.add_argument("--train-tokens", type=int, default=DEFAULT_TRAIN_TOKENS)
    ap.add_argument("--probe-offset", type=int, default=DEFAULT_PROBE_OFFSET)
    args = ap.parse_args(argv)

    args.out.mkdir(parents=True, exist_ok=True)
    manifest: dict = {
        "written_utc": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "files": {},
    }

    # ---- val + probe, from the local /val chunk 0
    val_dir = args.books3 / "val"
    val_meta = mvs.read_meta(val_dir / "zarr.json")
    files = mvs.chunk_files(args.books3)
    if not files or int(files[0].name) != 0:
        sys.exit(f"no chunk 0 under {val_dir}; run the bootstrap first")
    eval_len = args.seq_length + 1
    if args.probe_offset < eval_len:
        sys.exit("--probe-offset overlaps the eval slice")
    probe_len = args.mini_batch + 1

    for name, start, length in (("val", 0, eval_len), ("probe", args.probe_offset, probe_len)):
        tokens = read_slice(val_dir, val_meta, files[0], start, length)
        path = args.out / f"{name}.npy"
        np.save(path, tokens)
        manifest["files"][path.name] = {
            "source": f"{val_dir} chunk 0", "start": start, "length": length,
            "sha256": _sha256(path),
        }
        print(f"  {path}: {length:,} tokens from /val at {start:,}")

    # ---- train corpus, from /train chunk 0
    train_root = args.train_dir or (args.out / "train-zarr")
    train_dir = train_root / "train"
    if _find_chunk0(train_dir) is None:
        if not args.billing:
            sys.exit("train chunk 0 is not local; pass --billing to fetch it")
        train_dir = fetch_train_chunk0(args.billing, args.bucket, train_root)
    train_meta = mvs.read_meta(train_dir / "zarr.json")
    tokens = read_slice(train_dir, train_meta, _find_chunk0(train_dir), 0, args.train_tokens)
    path = args.out / "train.npy"
    np.save(path, tokens)
    manifest["files"][path.name] = {
        "source": f"{args.bucket}/train chunk 0", "start": 0, "length": args.train_tokens,
        "sha256": _sha256(path),
    }
    print(f"  {path}: {args.train_tokens:,} tokens from /train")

    (args.out / "tokens-manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"  manifest: {args.out / 'tokens-manifest.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
