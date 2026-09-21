"""Crash-safe record of finished evaluations, so a rerun resumes rather than restarts.

Every expensive number the session produces -- one adapt-and-eval, one gated
arm, one sequence arm -- is a pure function of what it read: the stream's
tokens, the `RunCondition` (which carries the checkpoint fingerprint and the
eval-token digest), and the gate, if any. So it is keyed by exactly that, and
appended to a JSONL file the moment it exists, flushed and fsynced.

Rerunning the same command with the same `--out` then replays every recorded
value instead of recomputing it. For the 001 search this is exact: proposals
come from `default_rng(seed)` and acceptance from the recorded objective values,
so the replay walks the same path to the point of the crash and continues from
there. A key that changes -- another checkpoint, another eval slice, another
gate -- simply misses, so a stale ledger can cost time but never a wrong number.

The file is also the raw record: if a run never finishes, every number it did
produce is on disk in plain text.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import os
from pathlib import Path

import numpy as np

LEDGER_NAME = "ledger.jsonl"


def tokens_digest(tokens) -> str:
    return hashlib.sha256(np.ascontiguousarray(tokens, dtype=np.int32).tobytes()).hexdigest()


def condition_key(condition) -> str:
    if dataclasses.is_dataclass(condition):
        return json.dumps(dataclasses.asdict(condition), sort_keys=True, default=str)
    return repr(condition)


def eval_key(kind: str, stream, condition, *extra: str) -> str:
    """The identity of one evaluation: what ran, on which tokens, under what."""
    parts = [kind, tokens_digest(stream.tokens), condition_key(condition), *extra]
    return hashlib.sha256("\x1f".join(parts).encode("utf-8")).hexdigest()


class Ledger:
    """Append-only JSONL of `{"key", "kind", "value"}` records.

    A torn last line (the process died mid-write) is skipped on load, never
    fatal: everything before it is intact because each record is fsynced.
    """

    def __init__(self, path: Path, *, reuse: bool = True, log=print):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._entries: dict[str, object] = {}
        self.hits = 0
        if self.path.exists() and self.path.stat().st_size:
            with self.path.open("rb") as fh:
                fh.seek(-1, os.SEEK_END)
                torn = fh.read(1) != b"\n"
            if torn:
                # Terminate the torn line, or the next record would be glued
                # onto it and be unparseable too.
                with self.path.open("a", encoding="utf-8") as fh:
                    fh.write("\n")
        if reuse and self.path.exists():
            with self.path.open("r", encoding="utf-8") as fh:
                for line in fh:
                    try:
                        rec = json.loads(line)
                        self._entries[rec["key"]] = rec["value"]
                    except (json.JSONDecodeError, KeyError, TypeError):
                        continue
            if self._entries:
                log(f"[ledger] {len(self._entries)} finished evaluation(s) on record in {self.path}; they will be reused")

    def __len__(self) -> int:
        return len(self._entries)

    def get(self, key: str):
        value = self._entries.get(key)
        if value is not None:
            self.hits += 1
        return value

    def put(self, key: str, kind: str, value) -> None:
        line = json.dumps({"key": key, "kind": kind, "value": value}, allow_nan=True)
        with self.path.open("a", encoding="utf-8") as fh:
            fh.write(line + "\n")
            fh.flush()
            os.fsync(fh.fileno())
        self._entries[key] = value


def memoize_adapt_and_eval(fn, ledger: Ledger | None, *, kind: str = "adapt_and_eval", extra: tuple[str, ...] = ()):
    """Wrap an `(stream, condition) -> loss` callable with the ledger."""
    if ledger is None:
        return fn

    def wrapped(stream, condition):
        key = eval_key(kind, stream, condition, *extra)
        cached = ledger.get(key)
        if cached is not None:
            return float(cached["loss"])
        loss = float(fn(stream, condition))
        ledger.put(key, kind, {"loss": loss, "seed": getattr(condition, "seed", None)})
        return loss

    return wrapped


def atomic_write_bytes(path: Path, data: bytes) -> None:
    """Write-then-rename, so a crash mid-write never leaves a torn file behind."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("wb") as fh:
        fh.write(data)
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, path)
