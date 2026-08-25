#!/usr/bin/env python3
"""Validate a release tag against version fields in a Pixi manifest."""

from __future__ import annotations

import argparse
from pathlib import Path
import re
from collections.abc import Sequence
import sys
import tomllib
from typing import Any


SEMVER_PATTERN = re.compile(
    r"(?:0|[1-9][0-9]*)\."
    r"(?:0|[1-9][0-9]*)\."
    r"(?:0|[1-9][0-9]*)"
    r"(?:-(?:0|[1-9][0-9]*|[0-9]*[A-Za-z-][0-9A-Za-z-]*)"
    r"(?:\.(?:0|[1-9][0-9]*|[0-9]*[A-Za-z-][0-9A-Za-z-]*))*)?"
    r"(?:\+[0-9A-Za-z-]+(?:\.[0-9A-Za-z-]+)*)?"
)


class ReleaseValidationError(ValueError):
    """A manifest or tag does not identify one valid release."""


def _manifest_version(manifest: dict[str, Any], section: str) -> str:
    try:
        version = manifest[section]["version"]
    except (KeyError, TypeError) as error:
        raise ReleaseValidationError(
            f"manifest must define {section}.version"
        ) from error
    if not isinstance(version, str):
        raise ReleaseValidationError(f"{section}.version must be a string")
    return version


def validate_manifest_tag(manifest_path: Path, tag: str) -> str:
    """Return the version when the manifest and release tag agree."""
    with manifest_path.open("rb") as manifest_file:
        manifest = tomllib.load(manifest_file)

    workspace_version = _manifest_version(manifest, "workspace")
    package_version = _manifest_version(manifest, "package")

    if workspace_version != package_version:
        raise ReleaseValidationError(
            "workspace and package versions differ: "
            f"{workspace_version!r} != {package_version!r}"
        )
    if SEMVER_PATTERN.fullmatch(package_version) is None:
        raise ReleaseValidationError(
            f"package version is not strict SemVer: {package_version!r}"
        )

    expected_tag = f"v{package_version}"
    if tag != expected_tag:
        raise ReleaseValidationError(
            f"release tag {tag!r} does not exactly match {expected_tag!r}"
        )
    return package_version


def parse_arguments(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        required=True,
        type=Path,
        help="path to the pixi.toml manifest",
    )
    parser.add_argument(
        "--tag",
        required=True,
        help="release tag expected to equal v<manifest-version>",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    arguments = parse_arguments(argv)
    try:
        version = validate_manifest_tag(arguments.manifest, arguments.tag)
    except (
        OSError,
        ReleaseValidationError,
        UnicodeError,
        tomllib.TOMLDecodeError,
    ) as error:
        print(f"error: release validation failed: {error}", file=sys.stderr)
        return 1

    print(f"Validated release tag {arguments.tag} for version {version}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
