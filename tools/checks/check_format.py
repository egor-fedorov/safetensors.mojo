#!/usr/bin/env python3
"""Check or rewrite Mojo source formatting with the pinned compiler."""

from __future__ import annotations

import argparse
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SOURCE_PATHS = (
    PROJECT_ROOT / "src",
    PROJECT_ROOT / "tests",
    PROJECT_ROOT / "tools",
)


def find_sources() -> list[Path]:
    sources: list[Path] = []
    for path in SOURCE_PATHS:
        if path.is_dir():
            sources.extend(path.rglob("*.mojo"))
        elif path.is_file():
            sources.append(path)
    return sorted(sources)


def run_formatter(mojo: str, sources: list[Path]) -> int:
    completed = subprocess.run(
        [mojo, "format", *(str(source) for source in sources)],
        cwd=PROJECT_ROOT,
        check=False,
    )
    return completed.returncode


def check_format(mojo: str, sources: list[Path]) -> int:
    with tempfile.TemporaryDirectory(prefix="safetensors-mojo-format-") as raw:
        temporary_root = Path(raw)
        copies: list[Path] = []
        for source in sources:
            relative = source.relative_to(PROJECT_ROOT)
            copied = temporary_root / relative
            copied.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, copied)
            copies.append(copied)

        if run_formatter(mojo, copies) != 0:
            return 1

        changed = [
            source.relative_to(PROJECT_ROOT)
            for source, copied in zip(sources, copies, strict=True)
            if source.read_bytes() != copied.read_bytes()
        ]

    if changed:
        print("The following Mojo files require formatting:", file=sys.stderr)
        for path in changed:
            print(f"  {path}", file=sys.stderr)
        print("Run `pixi run format` to rewrite them.", file=sys.stderr)
        return 1

    print(f"All {len(sources)} Mojo source file(s) are formatted.")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--write",
        action="store_true",
        help="rewrite source files instead of checking temporary copies",
    )
    arguments = parser.parse_args()

    mojo = shutil.which("mojo")
    if mojo is None:
        print("error: the Mojo compiler is not available on PATH", file=sys.stderr)
        return 1

    sources = find_sources()
    if not sources:
        print("error: no Mojo source files were found", file=sys.stderr)
        return 1

    if arguments.write:
        return run_formatter(mojo, sources)
    return check_format(mojo, sources)


if __name__ == "__main__":
    raise SystemExit(main())
