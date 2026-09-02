#!/usr/bin/env python3
"""Validate a release manifest, native matrix, tag object, and checkout."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from dataclasses import dataclass
import json
from pathlib import Path
import re
import subprocess
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
GIT_OBJECT_ID_PATTERN = re.compile(r"(?:[0-9a-f]{40}|[0-9a-f]{64})")
PLATFORM_RUNNERS = {
    "linux-64": "ubuntu-latest",
    "linux-aarch64": "ubuntu-24.04-arm",
    "osx-arm64": "macos-15",
}
PLATFORM_BUILD_PREFIXES = {
    "linux-64": "linux64",
    "linux-aarch64": "linuxaarch64",
    "osx-arm64": "osxarm64",
}


class ReleaseValidationError(ValueError):
    """A manifest or tag does not identify one valid release."""


@dataclass(frozen=True)
class ReleaseTag:
    """The immutable tag object and peeled commit selected for a release."""

    tag_object: str
    commit: str


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


def release_matrix(manifest_path: Path) -> dict[str, list[dict[str, str]]]:
    """Return the native GitHub runner matrix declared by a manifest."""
    with manifest_path.open("rb") as manifest_file:
        manifest = tomllib.load(manifest_file)
    try:
        platforms = manifest["workspace"]["platforms"]
    except (KeyError, TypeError) as error:
        raise ReleaseValidationError(
            "manifest must define workspace.platforms"
        ) from error
    if not isinstance(platforms, list) or not platforms:
        raise ReleaseValidationError(
            "workspace.platforms must be a non-empty array"
        )

    include: list[dict[str, str]] = []
    seen: set[str] = set()
    for platform_name in platforms:
        if not isinstance(platform_name, str):
            raise ReleaseValidationError(
                "workspace.platforms entries must be strings"
            )
        if platform_name in seen:
            raise ReleaseValidationError(
                f"workspace.platforms contains duplicate {platform_name!r}"
            )
        seen.add(platform_name)
        try:
            runner = PLATFORM_RUNNERS[platform_name]
        except KeyError as error:
            raise ReleaseValidationError(
                f"unsupported release platform: {platform_name!r}"
            ) from error
        include.append({"platform": platform_name, "runner": runner})
    for entry in include:
        entry["build_prefix"] = (
            PLATFORM_BUILD_PREFIXES[entry["platform"]]
            if len(include) > 1
            else ""
        )
    return {"include": include}


def _run_git(repository: Path, arguments: Sequence[str]) -> str:
    completed = subprocess.run(
        ["git", "-C", str(repository), *arguments],
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        detail = completed.stderr.strip()
        suffix = f": {detail}" if detail else ""
        raise ReleaseValidationError(
            f"git {' '.join(arguments)} failed{suffix}"
        )
    return completed.stdout.strip()


def _git_object_id(repository: Path, revision: str) -> str:
    object_id = _run_git(repository, ["rev-parse", "--verify", revision])
    if GIT_OBJECT_ID_PATTERN.fullmatch(object_id) is None:
        raise ReleaseValidationError(
            f"git returned an invalid object ID for {revision!r}"
        )
    return object_id


def validate_repository_tag(repository: Path, tag: str) -> ReleaseTag:
    """Return the tag identity when its peeled commit is checked out."""
    reference = f"refs/tags/{tag}"
    _run_git(repository, ["show-ref", "--verify", "--quiet", reference])
    tag_object = _git_object_id(repository, reference)
    commit = _git_object_id(repository, f"{reference}^{{commit}}")
    head = _git_object_id(repository, "HEAD")
    if commit != head:
        raise ReleaseValidationError(
            f"checked-out commit does not match {tag}"
        )
    return ReleaseTag(tag_object=tag_object, commit=commit)


def write_github_output(
    path: Path,
    version: str,
    matrix: dict[str, list[dict[str, str]]],
    release_tag: ReleaseTag | None = None,
) -> None:
    """Append the validated version and compact matrix to GITHUB_OUTPUT."""
    serialized_matrix = json.dumps(matrix, separators=(",", ":"))
    serialized_platforms = json.dumps(
        [entry["platform"] for entry in matrix["include"]],
        separators=(",", ":"),
    )
    with path.open("a", encoding="utf-8") as output:
        output.write(f"version={version}\n")
        output.write(f"matrix={serialized_matrix}\n")
        output.write(f"platforms={serialized_platforms}\n")
        if release_tag is not None:
            output.write(f"commit={release_tag.commit}\n")
            output.write(f"tag_object={release_tag.tag_object}\n")


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
    parser.add_argument(
        "--github-output",
        type=Path,
        help="optional GitHub Actions output file for the validated plan",
    )
    parser.add_argument(
        "--repository",
        type=Path,
        help="optional Git checkout whose release tag must point to HEAD",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    arguments = parse_arguments(argv)
    try:
        version = validate_manifest_tag(arguments.manifest, arguments.tag)
        release_tag = (
            validate_repository_tag(arguments.repository, arguments.tag)
            if arguments.repository is not None
            else None
        )
        if arguments.github_output is not None:
            if release_tag is None:
                raise ReleaseValidationError(
                    "--repository is required with --github-output"
                )
            matrix = release_matrix(arguments.manifest)
            write_github_output(
                arguments.github_output,
                version,
                matrix,
                release_tag,
            )
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
