#!/usr/bin/env python3
"""Run repository checks and packaging in a deterministic sequence."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile


PROJECT_ROOT = Path(__file__).resolve().parent.parent
SOURCE_ROOT = PROJECT_ROOT / "src" / "safetensors"
FIXTURE_ROOT = PROJECT_ROOT / "fixtures"


def run(command: list[str], label: str, environment: dict[str, str]) -> None:
    print(f"\n==> {label}", flush=True)
    completed = subprocess.run(
        command,
        cwd=PROJECT_ROOT,
        env=environment,
        check=False,
    )
    if completed.returncode != 0:
        raise SystemExit(completed.returncode)


def compare_directories(expected: Path, actual: Path) -> bool:
    expected_files = {
        path.relative_to(expected): path
        for path in expected.rglob("*")
        if path.is_file()
    }
    actual_files = {
        path.relative_to(actual): path
        for path in actual.rglob("*")
        if path.is_file()
    }
    if expected_files.keys() != actual_files.keys():
        return False
    return all(
        expected_files[relative].read_bytes() == actual_files[relative].read_bytes()
        for relative in expected_files
    )


def verify_fixtures(environment: dict[str, str]) -> None:
    temporary_parent = PROJECT_ROOT / ".pixi"
    temporary_parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix="safetensors-mojo-fixtures-",
        dir=temporary_parent,
    ) as raw_directory:
        generated = Path(raw_directory) / "fixtures"
        run(
            [
                sys.executable,
                "tools/generate_fixtures.py",
                "--output-root",
                str(generated),
            ],
            "Generate comparison fixtures",
            environment,
        )
        if not compare_directories(FIXTURE_ROOT, generated):
            print(
                "error: committed fixtures differ from deterministic output",
                file=sys.stderr,
            )
            raise SystemExit(1)
    print("Committed fixtures are reproducible.")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--package",
        action="store_true",
        help="build the Conda package after all checks pass",
    )
    arguments = parser.parse_args()

    mojo = shutil.which("mojo")
    pixi = shutil.which("pixi")
    if mojo is None:
        print("error: the Mojo compiler is not available on PATH", file=sys.stderr)
        return 1
    if arguments.package and pixi is None:
        print("error: Pixi is not available on PATH", file=sys.stderr)
        return 1

    environment = os.environ.copy()
    cache_root = PROJECT_ROOT / ".pixi" / "mojo-cache"
    cache_root.mkdir(parents=True, exist_ok=True)
    environment.setdefault("MODULAR_CACHE_DIR", str(cache_root))

    run(
        [sys.executable, "tools/check_format.py"],
        "Check Mojo formatting",
        environment,
    )
    run(
        [
            sys.executable,
            "-m",
            "unittest",
            "discover",
            "-s",
            "tests",
            "-p",
            "test_*.py",
        ],
        "Run Python tests",
        environment,
    )
    run(
        [
            mojo,
            "precompile",
            str(SOURCE_ROOT),
            "-o",
            str(PROJECT_ROOT / ".pixi" / "safetensors.mojoc"),
        ],
        "Compile the safetensors package",
        environment,
    )
    run(
        [
            mojo,
            "run",
            "-I",
            str(SOURCE_ROOT.parent),
            str(PROJECT_ROOT / "tools" / "package_smoke_format_core.mojo"),
        ],
        "Run the legacy format-core smoke consumer",
        environment,
    )
    run(
        [sys.executable, "tools/run_tests.py"],
        "Run Mojo tests",
        environment,
    )
    verify_fixtures(environment)

    if arguments.package:
        package_directory = PROJECT_ROOT / ".pixi" / "packages"
        if package_directory.exists():
            shutil.rmtree(package_directory)
        run(
            [
                pixi,
                "publish",
                "--clean",
                "--force",
                "--path",
                ".",
                "--target-dir",
                ".pixi/packages",
            ],
            "Build the safetensors-mojo Conda package",
            environment,
        )
        packages = sorted(package_directory.glob("*.conda"))
        if len(packages) != 1:
            print(
                "error: expected exactly one built .conda package, found "
                f"{len(packages)}",
                file=sys.stderr,
            )
            return 1
        run(
            [
                sys.executable,
                "tools/smoke_test_package.py",
                str(packages[0]),
            ],
            "Smoke-test a clean package installation",
            environment,
        )

    print("\nAll requested checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
