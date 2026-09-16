#!/usr/bin/env python
"""Copy a GPU session's artefacts into the repo, with every secret redacted.

Why this exists
---------------
The vendor's `_main` logs **every environment variable** before it does anything
else, and then the fully resolved config -- which includes `wandb_key`:

    ttt/train.py:80   logger.info("\\n".join([f"{k}={v}" for k, v in os.environ.items()]))
    ttt/train.py:81   logger.info(f"Launching with \\n {pformat(cfg_dict)}.")

Those logs are exactly what a baseline record needs (bar S4 reads the config
echo), and this repository is **public**. So logs never leave the box raw: they
go through here, and the copy is re-scanned afterwards. A secret still present
after redaction is exit 2 and nothing should be copied off the box.

Standard library only, so it runs on the box's system python3.

Usage
-----
    python3 scripts/collect_results.py --src $EXP_DIR/collected \\
        --dest experiments/000-repro-baseline/results/session-<utc>
"""

from __future__ import annotations

import argparse
import os
import re
import shutil
import sys
from pathlib import Path

REDACTED = "<REDACTED>"

# Environment variable names whose values are secret wherever they appear.
# Deliberately narrow: a literal is replaced everywhere it occurs, so treating
# e.g. SSH_AUTH_SOCK's path as a secret would shred every path in the logs.
SECRET_NAME = re.compile(r"(KEY|TOKEN|SECRET|PASSWORD|PASSWD|CREDENTIAL|BILLING_PROJECT)", re.I)

# `NAME=value` lines from the vendor's environment dump.
ENV_LINE = re.compile(
    r"^(?P<prefix>.*?\b)(?P<name>[A-Za-z_][A-Za-z0-9_]*(?:KEY|TOKEN|SECRET|PASSWORD|PASSWD|CREDENTIAL)[A-Za-z0-9_]*)=(?P<value>.*)$",
    re.I,
)
# `'wandb_key': '...'` in the pformat'd config, and `training.wandb_key=...` on a command line.
CONFIG_KEY = re.compile(r"""(?P<head>['"]?\w*(?:key|token|secret|password)['"]?\s*[:=]\s*)(?P<q>['"]?)(?P<value>[^'"\s,}]+)(?P=q)""", re.I)

# Values shorter than this are not treated as literal secrets: redacting "1" or
# "true" everywhere would destroy the logs without protecting anything.
MIN_SECRET_LEN = 8

TEXT_SUFFIXES = {".log", ".txt", ".json", ".sh", ".env", ".yaml", ".yml", ".md", ""}


def secret_values(environ: dict[str, str]) -> dict[str, str]:
    """Secret literals taken from this shell's environment, name -> value."""
    found = {}
    for name, value in environ.items():
        if SECRET_NAME.search(name) and len(value) >= MIN_SECRET_LEN:
            found[name] = value
    return found


def redact_text(text: str, secrets: dict[str, str]) -> str:
    # Longest first, so a secret containing another secret is replaced whole.
    for name, value in sorted(secrets.items(), key=lambda kv: -len(kv[1])):
        text = text.replace(value, f"{REDACTED}:{name}")

    lines = []
    for line in text.splitlines(keepends=True):
        body = line.rstrip("\r\n")
        ending = line[len(body):]
        m = ENV_LINE.match(body)
        if m and not m.group("value").startswith(REDACTED):
            body = f"{m.group('prefix')}{m.group('name')}={REDACTED}"
        body = CONFIG_KEY.sub(
            lambda k: k.group(0) if k.group("value").startswith(REDACTED) or len(k.group("value")) < MIN_SECRET_LEN
            else f"{k.group('head')}{k.group('q')}{REDACTED}{k.group('q')}",
            body,
        )
        lines.append(body + ending)
    return "".join(lines)


def is_text(path: Path) -> bool:
    return path.suffix.lower() in TEXT_SUFFIXES


def collect(src: Path, dest: Path, secrets: dict[str, str]) -> list[Path]:
    if not src.is_dir():
        sys.exit(f"{src} is not a directory")
    if dest.exists() and any(dest.iterdir()):
        sys.exit(f"{dest} already exists and is not empty; refusing to mix sessions")
    written = []
    for path in sorted(p for p in src.rglob("*") if p.is_file()):
        out = dest / path.relative_to(src)
        out.parent.mkdir(parents=True, exist_ok=True)
        if is_text(path):
            text = path.read_text(encoding="utf-8", errors="replace")
            out.write_text(redact_text(text, secrets), encoding="utf-8")
        else:
            shutil.copy2(path, out)
        written.append(out)
    return written


def leaks(dest: Path, secrets: dict[str, str]) -> list[tuple[Path, str]]:
    """Every (file, secret name) where a secret literal survives -- binaries included."""
    found = []
    for path in sorted(p for p in dest.rglob("*") if p.is_file()):
        data = path.read_bytes()
        for name, value in secrets.items():
            if value.encode("utf-8") in data:
                found.append((path, name))
    return found


def main(argv: list[str] | None = None, environ: dict[str, str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--src", type=Path, required=True)
    parser.add_argument("--dest", type=Path, required=True)
    args = parser.parse_args(argv)

    env = dict(os.environ if environ is None else environ)
    secrets = secret_values(env)
    if "WANDB_KEY" not in secrets:
        print("WARNING: WANDB_KEY is not set in this shell, so its literal value cannot be "
              "scrubbed. Pattern-based redaction still runs; export it and re-run to be sure.",
              file=sys.stderr)

    written = collect(args.src, args.dest, secrets)
    print(f"==> Collected {len(written)} files into {args.dest} (redacted {len(secrets)} secret value(s))")

    bad = leaks(args.dest, secrets)
    if bad:
        for path, name in bad:
            print(f"LEAK: {name} still present in {path}", file=sys.stderr)
        print("FATAL: redaction incomplete. Do not copy these results off the box.", file=sys.stderr)
        return 2
    print("==> Leak scan clean: no secret value from this shell appears in the copy")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
