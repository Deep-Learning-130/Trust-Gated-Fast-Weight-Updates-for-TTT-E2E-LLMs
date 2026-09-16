#!/usr/bin/env python
"""Build a TRUNCATED local copy of the books3 /val array, and record what was cut.

Why this exists
---------------
`COST_MODEL.md` §3 says the eval pass "cannot be shortened by configuration", and
that is true: there is no `num_eval_batches` and `repeat=False`. But the batch
count is not a config value at all -- it is read off the zarr array's declared
shape:

    ttt/dataloader/lm_dataset.py:27
    def __len__(self):
        return (self.split.shape[0] - 1) // self.seq_len

So copying K of `/val`'s chunks and rewriting `shape` in the **local**
`val/zarr.json` bounds the pass exactly. This touches our own data copy, never
the vendor tree, so ADR-002 is not in play.

The 2026-09-14 probe measured `/val` at 2,000,168,321 tokens (8.4 GB, uint32,
uncompressed). A full pass is ~7.5 GPU-hours at the 1B interpolated rate, and
`TOLERANCE.md` §5 bar S2 wants the identical command run **twice**. That does not
fit a 3-4 hour booking. This script is how it fits.

The manifest is not optional bookkeeping. A truncated baseline that cannot say
what it was truncated to is not a result.

Absent chunks are the trap
--------------------------
In zarr v3 an absent chunk reads as the **fill value** (0 for this array) rather
than raising. So a shape that exceeds the chunks actually on disk does not fail
loudly -- it silently evaluates the model on a stream of token id 0 and returns a
confident, meaningless loss. `reshape` refuses that by default; nothing else in
the stack would catch it.

Usage
-----
    # fetch + truncate (needs gsutil and a billing project)
    GCP_BILLING_PROJECT=proj python scripts/make_val_subset.py fetch --tokens 150000000

    # plan only, no transfer
    GCP_BILLING_PROJECT=proj python scripts/make_val_subset.py fetch --tokens 150000000 --probe-only

    # resize between runs, no network, no refetch -- the smoke-pass move
    python scripts/make_val_subset.py reshape --tokens 131073

    # what is on disk right now
    python scripts/make_val_subset.py status
"""

from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import math
import os
import shutil
import subprocess
import sys
from pathlib import Path

DEFAULT_BUCKET = "gs://llama3-books3"
DEFAULT_TOKENS = 150_000_000
SEQ_LENGTH = 8192  # ADR-004; the only seq_length this project evaluates at
EVAL_BATCH = 8  # train.py:211 floors the eval batch here regardless of gbs

MANIFEST_NAME = "val-subset-manifest.json"


def default_dest() -> Path:
    return Path(os.environ.get("DATA_ROOT", str(Path.home() / "ttt-data"))) / "llama3-books3"


# --------------------------------------------------------------------- zarr ---


def read_meta(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        sys.exit(f"{path} not found. Run `fetch` first.")


def chunk_shape_of(meta: dict) -> int:
    """Chunk length along the only axis, for zarr v3 or v2 metadata."""
    try:
        return int(meta["chunk_grid"]["configuration"]["chunk_shape"][0])
    except KeyError:
        pass
    if "chunks" in meta:  # zarr v2
        return int(meta["chunks"][0])
    sys.exit("could not find a chunk shape in the array metadata")


def shape_of(meta: dict) -> int:
    return int(meta["shape"][0])


def chunk_files(dest: Path) -> list[Path]:
    """Chunk files on disk, in index order, for either key encoding.

    zarr v3's default encoding puts 1-D chunks at `val/c/0`; the v2 style puts
    them at `val/0`. Discover rather than assume -- the wrong guess reports zero
    chunks and `reshape` would then refuse a shape that is in fact safe.
    """
    cdir = dest / "val" / "c"
    if cdir.is_dir():
        found = [p for p in cdir.iterdir() if p.name.isdigit()]
        if found:
            return sorted(found, key=lambda p: int(p.name))
    vdir = dest / "val"
    if vdir.is_dir():
        return sorted((p for p in vdir.iterdir() if p.name.isdigit()), key=lambda p: int(p.name))
    return []


def sequences_for(tokens: int) -> int:
    """What `lm_dataset.Dataset.__len__` will return for this many tokens."""
    return (tokens - 1) // SEQ_LENGTH


def describe(tokens: int, original: int | None = None) -> list[str]:
    seqs = sequences_for(tokens)
    lines = [
        f"    tokens      : {tokens:,}",
        f"    sequences   : {seqs:,} at seq_length={SEQ_LENGTH}",
        f"    eval batches: {math.ceil(seqs / EVAL_BATCH):,} at eval batch {EVAL_BATCH}",
    ]
    if original:
        lines.append(f"    fraction    : {tokens / original:.4%} of /val")
    return lines


# ------------------------------------------------------------------ manifest --


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def write_manifest(dest: Path, *, source: str, original: int, truncated: int,
                   chunk: int, meta: dict, digest: bool = True) -> Path:
    files = chunk_files(dest)
    manifest = {
        "written_utc": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "source": source,
        "original_shape": original,
        "truncated_shape": truncated,
        "chunk_shape": chunk,
        "chunks_on_disk": len(files),
        "tokens_on_disk": len(files) * chunk,
        # 10 places, not 8: a 0.007% subset rounds away at 8 and the recorded
        # fraction stops matching the shapes beside it.
        "fraction_of_val": round(truncated / original, 10) if original else None,
        "seq_length": SEQ_LENGTH,
        "sequences": sequences_for(truncated),
        "dtype": meta.get("data_type") or meta.get("dtype"),
        "fill_value": meta.get("fill_value"),
        "chunk_sha256": {p.name: sha256_of(p) for p in files} if digest else "not computed",
    }
    path = dest / MANIFEST_NAME
    path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return path


# --------------------------------------------------------------------- gsutil --


def gsutil(billing: str, *args: str, quiet: bool = False) -> subprocess.CompletedProcess:
    cmd = ["gsutil", "-u", billing, *args]
    return subprocess.run(
        cmd,
        capture_output=quiet,
        text=True,
        check=False,
    )


def find_val_prefix(billing: str, bucket: str, tmp: Path) -> str:
    """Locate the /val array, trying both layouts rather than assuming one.

    The 2026-09-14 probe recorded the array under `<bucket>/data.zarr/val`; the
    vendor README implies `<bucket>/val`. A wrong prefix here copies nothing and
    the failure would not surface until the eval read fill values.
    """
    for candidate in (f"{bucket}/val", f"{bucket}/data.zarr/val"):
        result = gsutil(billing, "cp", f"{candidate}/zarr.json", str(tmp), quiet=True)
        if result.returncode == 0:
            return candidate
    sys.exit(
        f"could not read val/zarr.json under {bucket}.\n"
        f"       Tried {bucket}/val and {bucket}/data.zarr/val.\n"
        f"       List the bucket and pass --bucket pointing at the store root."
    )


# ----------------------------------------------------------------- commands ---


def cmd_fetch(args: argparse.Namespace) -> int:
    billing = os.environ.get("GCP_BILLING_PROJECT")
    if not billing:
        sys.exit("Set GCP_BILLING_PROJECT -- the bucket is requester-pays")
    if not shutil.which("gsutil"):
        sys.exit("gsutil not found. Install the gcloud SDK first.")

    dest = Path(args.dest)
    (dest / "val").mkdir(parents=True, exist_ok=True)
    (dest / "train").mkdir(parents=True, exist_ok=True)

    tmp = dest / "val" / ".zarr.json.probe"
    print("\n==> Reading /val metadata (metadata only, no transfer)")
    prefix = find_val_prefix(billing, args.bucket, tmp)
    root = prefix.rsplit("/val", 1)[0]
    print(f"    val prefix  : {prefix}")

    meta = read_meta(tmp)
    original = shape_of(meta)
    chunk = chunk_shape_of(meta)
    want = min(args.tokens, original)
    n_chunks = math.ceil(want / chunk)
    total_chunks = math.ceil(original / chunk)

    print(f"    original    : {original:,} tokens in {total_chunks} chunks of {chunk:,}")
    print(f"    requested   : {args.tokens:,} -> truncating to {want:,}")
    print(f"    chunks      : {n_chunks} of {total_chunks}")
    print(f"    approx bytes: {n_chunks * chunk * 4 / 1e9:.2f} GB (uint32, uncompressed)")
    for line in describe(want, original):
        print(line)

    if args.probe_only:
        tmp.unlink(missing_ok=True)
        print("\n==> --probe-only: nothing transferred")
        return 0

    print(f"\n==> Discovering chunk key layout")
    listing = gsutil(billing, "ls", f"{prefix}/**", quiet=True)
    sample = next(
        (ln for ln in listing.stdout.splitlines() if ln.strip() and not ln.endswith("zarr.json")),
        None,
    )
    if sample is None:
        sys.exit(f"no chunk objects found under {prefix}")
    print(f"    sample key  : {sample}")
    nested = "/c/" in sample

    if nested:
        (dest / "val" / "c").mkdir(parents=True, exist_ok=True)

    def src_dst(i: int) -> tuple[str, Path]:
        if nested:
            return f"{prefix}/c/{i}", dest / "val" / "c" / str(i)
        return f"{prefix}/{i}", dest / "val" / str(i)

    print(f"\n==> Copying {n_chunks} chunk(s) -- this is the transfer")
    for i in range(n_chunks):
        src, dst = src_dst(i)
        if dst.exists():
            print(f"    [already done] chunk {i}")
            continue
        result = gsutil(billing, "cp", src, str(dst))
        if result.returncode != 0:
            sys.exit(f"failed to copy chunk {i} from {src}")

    print("\n==> Group and /train metadata")
    print("    COST_MODEL §1.1: without /train metadata the run dies after billing starts")
    for remote, local in (
        (f"{root}/zarr.json", dest / "zarr.json"),
        (f"{root}/train/zarr.json", dest / "train" / "zarr.json"),
    ):
        if gsutil(billing, "cp", remote, str(local), quiet=True).returncode != 0:
            print(f"    note: {remote} not found")

    shutil.move(str(tmp), str(dest / "val" / "zarr.json"))

    print("\n==> Rewriting shape and writing the manifest")
    meta["shape"] = [want]
    (dest / "val" / "zarr.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    manifest = write_manifest(
        dest, source=prefix, original=original, truncated=want, chunk=chunk, meta=meta
    )
    print(f"    shape       : {original:,} -> {want:,}")
    print(f"    manifest    : {manifest}")

    print(f"""
==> Done

  Local store : {dest}
  Manifest    : {manifest}

  To resize between runs WITHOUT refetching (the smoke-pass move):
    python scripts/make_val_subset.py reshape --tokens 131073
    python scripts/make_val_subset.py reshape --tokens {want}

  The manifest travels with the result. Copy it into
  experiments/000-repro-baseline/results/ alongside the loss.
""")
    return 0


def cmd_reshape(args: argparse.Namespace) -> int:
    """Rewrite `shape` against chunks already on disk. No network, no transfer.

    This is what makes the smoke pass cheap: run two batches to shake out the
    launch, then reshape and run for real without refetching a byte. Batch size
    and sequence length do not change, so a warmed XLA compilation cache still
    hits.
    """
    dest = Path(args.dest)
    meta_path = dest / "val" / "zarr.json"
    meta = read_meta(meta_path)
    chunk = chunk_shape_of(meta)

    manifest_path = dest / MANIFEST_NAME
    manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {}
    original = manifest.get("original_shape", shape_of(meta))

    on_disk = len(chunk_files(dest))
    if on_disk == 0:
        sys.exit(f"no chunk files found under {dest / 'val'}; run `fetch` first")
    covered = on_disk * chunk

    want = min(args.tokens, original)
    if want > covered and not args.allow_fill:
        sys.exit(
            f"--tokens {want:,} exceeds the {covered:,} tokens actually on disk\n"
            f"       ({on_disk} chunks x {chunk:,}).\n"
            f"       Absent zarr chunks read as fill value {meta.get('fill_value', 0)}, so the\n"
            f"       extra tokens would be padding -- a loss computed over zeros, returned\n"
            f"       without any error. Refetch with a larger --tokens, or pass --allow-fill\n"
            f"       if you genuinely intend to evaluate on fill values."
        )

    meta["shape"] = [want]
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    write_manifest(
        dest, source=manifest.get("source", "unknown"), original=original,
        truncated=want, chunk=chunk, meta=meta, digest=not args.no_digest,
    )

    print(f"==> Reshaped {meta_path}")
    print(f"    on disk     : {on_disk} chunks = {covered:,} tokens")
    for line in describe(want, original):
        print(line)
    return 0


def cmd_status(args: argparse.Namespace) -> int:
    dest = Path(args.dest)
    meta_path = dest / "val" / "zarr.json"
    if not meta_path.exists():
        print(f"no local store at {dest}")
        return 1
    meta = read_meta(meta_path)
    chunk = chunk_shape_of(meta)
    on_disk = len(chunk_files(dest))
    manifest_path = dest / MANIFEST_NAME
    manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {}

    print(f"store         : {dest}")
    print(f"declared shape: {shape_of(meta):,} tokens")
    print(f"original shape: {manifest.get('original_shape', 'unknown (no manifest)')}")
    print(f"chunks on disk: {on_disk} x {chunk:,} = {on_disk * chunk:,} tokens")
    for line in describe(shape_of(meta), manifest.get("original_shape")):
        print(line)
    if shape_of(meta) > on_disk * chunk:
        print("\nWARNING: declared shape exceeds chunks on disk.")
        print("         The tail reads as fill value and the loss would be meaningless.")
    return 0


def cmd_covers(args: argparse.Namespace) -> int:
    """Exit 0 iff a local store holds real chunks for `--tokens` (capped at /val).

    This is how the bootstrap decides whether it needs GCS at all. A side-loaded
    store that already covers the request must not trigger a fetch -- `fetch`
    reads metadata through gsutil, so it would demand a billing project for
    bytes already on disk. Silent: the caller prints the decision.
    """
    dest = Path(args.dest)
    meta_path = dest / "val" / "zarr.json"
    if not meta_path.exists():
        return 1
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    manifest_path = dest / MANIFEST_NAME
    manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {}
    original = manifest.get("original_shape", shape_of(meta))
    covered = len(chunk_files(dest)) * chunk_shape_of(meta)
    return 0 if covered >= min(args.tokens, original) > 0 else 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="make_val_subset.py",
        description="Build and resize a truncated local copy of the books3 /val array.",
    )
    parser.add_argument("--dest", default=str(default_dest()), help="local store root")
    sub = parser.add_subparsers(dest="command", required=True)

    f = sub.add_parser("fetch", help="copy chunks from GCS and truncate")
    f.add_argument("--tokens", type=int, default=DEFAULT_TOKENS)
    f.add_argument("--bucket", default=DEFAULT_BUCKET)
    f.add_argument("--probe-only", action="store_true", help="print the plan, transfer nothing")
    f.set_defaults(func=cmd_fetch)

    r = sub.add_parser("reshape", help="rewrite shape against chunks already on disk")
    r.add_argument("--tokens", type=int, required=True)
    r.add_argument("--allow-fill", action="store_true",
                   help="permit a shape larger than the chunks on disk (evaluates on zeros)")
    r.add_argument("--no-digest", action="store_true", help="skip re-hashing chunks")
    r.set_defaults(func=cmd_reshape)

    s = sub.add_parser("status", help="what is on disk and what shape is declared")
    s.set_defaults(func=cmd_status)

    c = sub.add_parser("covers", help="exit 0 iff chunks on disk cover --tokens (no output)")
    c.add_argument("--tokens", type=int, required=True)
    c.set_defaults(func=cmd_covers)

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
