#!/usr/bin/env python3
"""Run each standalone Mojo test suite in a deterministic order."""

from __future__ import annotations

import os
from pathlib import Path
import shutil
import subprocess
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SOURCE_ROOT = PROJECT_ROOT / "src"
TEST_ROOTS = (
    PROJECT_ROOT / "tests" / "unit",
    PROJECT_ROOT / "tests" / "integration",
)


def main() -> int:
    mojo = shutil.which("mojo")
    if mojo is None:
        print("error: the Mojo compiler is not available on PATH", file=sys.stderr)
        return 1

    test_files: list[Path] = []
    for test_root in TEST_ROOTS:
        category_files = sorted(test_root.rglob("test_*.mojo"))
        if not category_files:
            relative_root = test_root.relative_to(PROJECT_ROOT)
            print(
                f"error: no Mojo tests were found under {relative_root}",
                file=sys.stderr,
            )
            return 1
        test_files.extend(category_files)

    environment = os.environ.copy()
    cache_root = PROJECT_ROOT / ".pixi" / "mojo-cache"
    cache_root.mkdir(parents=True, exist_ok=True)
    environment.setdefault("MODULAR_CACHE_DIR", str(cache_root))

    for test_file in test_files:
        relative_test = test_file.relative_to(PROJECT_ROOT)
        command = [
            mojo,
            "run",
            "-I",
            str(SOURCE_ROOT),
            str(test_file),
        ]
        print(f"Running {relative_test}", flush=True)
        completed = subprocess.run(
            command,
            cwd=PROJECT_ROOT,
            env=environment,
            check=False,
        )
        if completed.returncode != 0:
            print(f"error: {relative_test} failed", file=sys.stderr)
            return completed.returncode

    print(f"All {len(test_files)} Mojo test suite(s) passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
