#!/usr/bin/env python3
"""Stage and collect the exact Conda artifacts for one release."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
import hashlib
import json
from pathlib import Path
import shutil
import sys
import zipfile


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tools.packaging.smoke_test import PACKAGE_NAME, artifact_identity  # noqa: E402
from tools.release.validate_release import PLATFORM_BUILD_PREFIXES  # noqa: E402


SUPPORTED_PLATFORMS = tuple(PLATFORM_BUILD_PREFIXES)
ARTIFACT_LIST_NAME = "release-artifacts.tsv"


def sha256_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def validate_artifact(path: Path, version: str) -> str:
    name, actual_version, build = artifact_identity(path)
    if name != PACKAGE_NAME or actual_version != version:
        raise ValueError(
            f"unexpected release artifact identity: {name} {actual_version}"
        )
    return build


def validate_platform_artifact(
    path: Path,
    version: str,
    platform: str,
    *,
    multi_platform: bool,
) -> None:
    build = validate_artifact(path, version)
    if multi_platform:
        expected_prefix = f"{PLATFORM_BUILD_PREFIXES[platform]}_"
        if not build.startswith(expected_prefix):
            raise ValueError(
                f"{platform} build string must start with {expected_prefix!r}: "
                f"{build!r}"
            )


def stage_artifact(
    input_directory: Path,
    output_directory: Path,
    version: str,
) -> Path:
    """Copy exactly one matching native build result into a job artifact."""
    candidates = sorted(input_directory.glob("*.conda"))
    if len(candidates) != 1:
        raise ValueError(
            "expected exactly one .conda package, found "
            f"{len(candidates)} in {input_directory}"
        )
    validate_artifact(candidates[0], version)
    output_directory.mkdir(parents=True, exist_ok=True)
    destination = output_directory / candidates[0].name
    shutil.copy2(candidates[0], destination)
    return destination.resolve()


def parse_platforms(raw_value: str) -> list[str]:
    value = json.loads(raw_value)
    if not isinstance(value, list) or any(
        not isinstance(item, str) for item in value
    ):
        raise ValueError("release platforms must be a JSON string array")
    if not value or len(set(value)) != len(value):
        raise ValueError("release platforms must be unique and non-empty")
    if any(item not in SUPPORTED_PLATFORMS for item in value):
        raise ValueError("release platforms contain an unsupported value")
    return value


def collect_artifacts(
    input_root: Path,
    output_directory: Path,
    version: str,
    platforms: Sequence[str],
    *,
    existing_root: Path | None = None,
) -> Path:
    """Select one canonical artifact per platform for aggregate publishing."""
    expected = tuple(platforms)
    parse_platforms(json.dumps(expected))
    rows: list[str] = []
    filenames: set[str] = set()
    for platform in expected:
        matches = sorted((input_root / f"conda-{platform}").rglob("*.conda"))
        if len(matches) != 1:
            raise ValueError(
                f"expected one {platform} artifact, found {len(matches)}"
            )
        candidate = matches[0]
        validate_platform_artifact(
            candidate,
            version,
            platform,
            multi_platform=len(expected) > 1,
        )
        if candidate.name in filenames:
            raise ValueError(
                "GitHub Release asset filenames must be unique across platforms: "
                f"{candidate.name}"
            )
        filenames.add(candidate.name)
        candidate_digest = sha256_digest(candidate)
        existing = (
            None
            if existing_root is None
            else existing_root / platform / candidate.name
        )
        if existing is not None and existing.is_file():
            selected = existing
            reused = True
            selected_digest = sha256_digest(selected)
        else:
            selected = candidate
            reused = False
            selected_digest = candidate_digest
        validate_platform_artifact(
            selected,
            version,
            platform,
            multi_platform=len(expected) > 1,
        )
        if selected_digest != candidate_digest:
            raise ValueError(
                f"existing {platform} Release asset differs from the "
                "native-verified artifact"
            )

        target_directory = output_directory / platform
        target_directory.mkdir(parents=True, exist_ok=True)
        target = (target_directory / candidate.name).resolve()
        shutil.copy2(selected, target)
        rows.append(
            f"{platform}\t{target}\t{candidate.name}\t"
            f"{str(reused).lower()}\t{selected_digest}\n"
        )

    artifact_list = (output_directory / ARTIFACT_LIST_NAME).resolve()
    artifact_list.write_text("".join(rows), encoding="utf-8")
    return artifact_list


def write_github_output(path: Path, **values: str) -> None:
    with path.open("a", encoding="utf-8") as output:
        for key, value in values.items():
            if "\n" in key or "\n" in value:
                raise ValueError("GitHub output keys and values must be one line")
            output.write(f"{key}={value}\n")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    stage = commands.add_parser("stage")
    stage.add_argument("--input-directory", required=True, type=Path)
    stage.add_argument("--output-directory", required=True, type=Path)
    stage.add_argument("--version", required=True)
    stage.add_argument("--github-output", required=True, type=Path)
    collect = commands.add_parser("collect")
    collect.add_argument("--input-root", required=True, type=Path)
    collect.add_argument("--output-directory", required=True, type=Path)
    collect.add_argument("--version", required=True)
    collect.add_argument("--platforms-json", required=True)
    collect.add_argument("--existing-root", type=Path)
    collect.add_argument("--github-output", required=True, type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    try:
        if arguments.command == "stage":
            artifact = stage_artifact(
                arguments.input_directory,
                arguments.output_directory,
                arguments.version,
            )
            write_github_output(
                arguments.github_output,
                path=str(artifact),
                name=artifact.name,
            )
            print(f"Staged {artifact.name}.")
        else:
            artifact_list = collect_artifacts(
                arguments.input_root,
                arguments.output_directory,
                arguments.version,
                parse_platforms(arguments.platforms_json),
                existing_root=arguments.existing_root,
            )
            write_github_output(
                arguments.github_output,
                artifact_list=str(artifact_list),
            )
            print(f"Collected release artifacts in {artifact_list}.")
    except (
        json.JSONDecodeError,
        OSError,
        ValueError,
        zipfile.BadZipFile,
    ) as error:
        print(f"error: release artifact validation failed: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
