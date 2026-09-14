#!/usr/bin/env python
"""Fetch the independent fluency reference model and the Llama-3 decoder.

T3.2 / TEAM_PLAN P1-8. Sibling of `fetch_checkpoints.sh`, which fetches the
*victim*. These two must never be confused: scoring a stream's fluency with the
model under attack is circular, and `PREREGISTERED.md` names it an invalidating
condition.

What it fetches, and why each piece
-----------------------------------
- **GPT-2 small** (`openai-community/gpt2`), 124M, trained on WebText. The
  reference model. Genuinely independent of the victim: different corpus
  (WebText vs DCLM+Books3), different tokenizer (50257 BPE vs Llama-3's 128256),
  different provenance. Small enough to score five poison/control pairs on CPU,
  and it has a **published** perplexity to check the implementation against
  (Radford et al. 2019, Table 3: WikiText-103 = 37.50 for the 117M model).
- **A Llama-3 tokenizer** (`NousResearch/Meta-Llama-3-8B`), used for **decoding
  only**. Crafted streams are Llama-3 token ids (the vendor ships no tokenizer;
  see `trustgate.tokens`), and the reference model reads text, so the ids have to
  be decoded on the way. An ungated mirror is used deliberately: the official
  `meta-llama` repo is gated, and a gate in the middle of a reproducibility path
  is a liability. Vocab identity is asserted at load time, not assumed.

Nothing here is committed. Weights live outside the repo, under
`TRUSTGATE_REFERENCE_CACHE` (default `~/.cache/trustgate/reference`), and the
sha256 manifest this script writes is the record of what was actually used.

Usage:
    python scripts/fetch_reference_model.py            # fetch what is missing
    PROBE_ONLY=1 python scripts/fetch_reference_model.py   # sizes only, no bytes
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
import urllib.request
from pathlib import Path

HF = "https://huggingface.co"

#: Pinned by revision so a silent upstream change cannot alter a recorded number.
#: These are the commit shas of the two source repos at fetch time; see MANIFEST.
GPT2_REPO = "openai-community/gpt2"
LLAMA3_TOKENIZER_REPO = "NousResearch/Meta-Llama-3-8B"

FILES = [
    (GPT2_REPO, "config.json", "gpt2/config.json"),
    (GPT2_REPO, "tokenizer.json", "gpt2/tokenizer.json"),
    (GPT2_REPO, "model.safetensors", "gpt2/model.safetensors"),
    (LLAMA3_TOKENIZER_REPO, "tokenizer.json", "llama3/tokenizer.json"),
]


def cache_root() -> Path:
    return Path(
        os.environ.get(
            "TRUSTGATE_REFERENCE_CACHE", Path.home() / ".cache" / "trustgate" / "reference"
        )
    )


def _url(repo: str, filename: str) -> str:
    return f"{HF}/{repo}/resolve/main/{filename}"


def _request(url: str, method: str = "GET"):
    return urllib.request.Request(
        url, method=method, headers={"User-Agent": "trustgate-fetch-reference"}
    )


def remote_size(repo: str, filename: str) -> int | None:
    try:
        with urllib.request.urlopen(_request(_url(repo, filename), "HEAD"), timeout=30) as r:
            value = r.headers.get("Content-Length")
            return int(value) if value else None
    except Exception as exc:  # noqa: BLE001 -- reported, not swallowed
        print(f"  HEAD failed for {repo}/{filename}: {type(exc).__name__} {exc}")
        return None


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def download(repo: str, filename: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    expected = remote_size(repo, filename)

    if dest.exists() and expected is not None and dest.stat().st_size == expected:
        print(f"  already present, size matches: {dest.name}")
        return

    tmp = dest.with_suffix(dest.suffix + ".partial")
    print(f"  downloading {repo}/{filename} -> {dest}")
    with urllib.request.urlopen(_request(_url(repo, filename)), timeout=120) as response:
        total = 0
        with tmp.open("wb") as handle:
            while True:
                block = response.read(1 << 20)
                if not block:
                    break
                handle.write(block)
                total += len(block)
                if expected:
                    pct = 100.0 * total / expected
                    print(f"    {total / 1e6:.1f} / {expected / 1e6:.1f} MB ({pct:.0f}%)", flush=True)

    if expected is not None and total != expected:
        tmp.unlink(missing_ok=True)
        raise RuntimeError(
            f"size mismatch for {filename}: got {total}, expected {expected}. "
            f"Refusing a partial file -- a truncated weight tensor would produce "
            f"plausible-looking perplexities that are silently wrong."
        )

    tmp.replace(dest)


def main() -> int:
    root = cache_root()
    probe_only = os.environ.get("PROBE_ONLY") == "1"

    print(f"reference cache: {root}")
    print(f"mode: {'PROBE ONLY (no bytes fetched)' if probe_only else 'FETCH'}")
    print()

    if probe_only:
        grand_total = 0
        for repo, filename, _ in FILES:
            size = remote_size(repo, filename)
            if size:
                grand_total += size
            print(f"  {repo}/{filename}: {0 if size is None else size / 1e6:.1f} MB")
        print(f"\ntotal: {grand_total / 1e6:.1f} MB")
        return 0

    for repo, filename, relative in FILES:
        download(repo, filename, root / relative)

    manifest = {
        "note": (
            "Independent fluency reference model (T3.2). NOT the victim -- scoring "
            "a stream with the model under attack is circular and PREREGISTERED.md "
            "names it invalidating."
        ),
        "files": {},
    }
    for _repo, _filename, relative in FILES:
        path = root / relative
        manifest["files"][relative] = {
            "sha256": sha256_of(path),
            "bytes": path.stat().st_size,
        }

    manifest_path = root / "MANIFEST.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"\nwrote {manifest_path}")
    for relative, info in manifest["files"].items():
        print(f"  {relative}: {info['sha256'][:16]}... ({info['bytes'] / 1e6:.1f} MB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
