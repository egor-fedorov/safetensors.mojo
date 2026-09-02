#!/usr/bin/env python3
"""Discover, reuse, and publish immutable GitHub Release artifacts."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from dataclasses import dataclass
import json
from pathlib import Path
import re
import subprocess
import sys
import zipfile


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tools.release.artifacts import (  # noqa: E402
    read_artifact_list,
    validate_platform_artifact,
    write_github_output,
)
from tools.release.validate_release import (  # noqa: E402
    GIT_OBJECT_ID_PATTERN,
    PLATFORM_BUILD_PREFIXES,
    SEMVER_PATTERN,
    validate_repository_tag,
)


REPOSITORY_PATTERN = re.compile(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+")


class GitHubReleaseError(ValueError):
    """A GitHub Release operation would violate the release contract."""


@dataclass(frozen=True)
class ReleaseSnapshot:
    """The existence and immutable asset names observed for one release."""

    exists: bool
    assets: frozenset[str]


def run_command(
    command: Sequence[str],
    *,
    cwd: Path | None = None,
    input_text: str | None = None,
) -> str:
    """Run one `gh` or `git` command and return its standard output."""
    completed = subprocess.run(
        list(command),
        cwd=cwd,
        input=input_text,
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        detail = completed.stderr.strip()
        suffix = f": {detail}" if detail else ""
        raise GitHubReleaseError(f"command failed: {' '.join(command)}{suffix}")
    return completed.stdout


def _validate_repository(repository: str) -> None:
    if REPOSITORY_PATTERN.fullmatch(repository) is None:
        raise GitHubReleaseError(f"invalid GitHub repository: {repository!r}")


def _validate_tag(tag: str) -> None:
    if not tag.startswith("v") or SEMVER_PATTERN.fullmatch(tag[1:]) is None:
        raise GitHubReleaseError(f"invalid release tag: {tag!r}")


def _validate_asset_name(name: str) -> None:
    if (
        name == ""
        or Path(name).name != name
        or any(character in name for character in "*?[]\\/")
        or any(ord(character) < 0x20 for character in name)
    ):
        raise GitHubReleaseError(f"unsafe GitHub Release asset name: {name!r}")


def release_snapshot(repository: str, tag: str) -> ReleaseSnapshot:
    """Return one tag's release and asset names from the paginated API."""
    _validate_repository(repository)
    _validate_tag(tag)
    query = (
        f".[] | select(.tag_name == {json.dumps(tag)}) | "
        "{tag: .tag_name, assets: [.assets[].name]} | @json"
    )
    output = run_command(
        [
            "gh",
            "api",
            "--paginate",
            f"repos/{repository}/releases?per_page=100",
            "--jq",
            query,
        ]
    )
    documents = [line for line in output.splitlines() if line]
    if not documents:
        return ReleaseSnapshot(False, frozenset())
    if len(documents) != 1:
        raise GitHubReleaseError(f"GitHub returned multiple releases for {tag}")
    try:
        document = json.loads(documents[0])
    except json.JSONDecodeError as error:
        raise GitHubReleaseError("GitHub returned invalid release JSON") from error
    if not isinstance(document, dict) or document.get("tag") != tag:
        raise GitHubReleaseError("GitHub returned an invalid release record")
    assets = document.get("assets")
    if not isinstance(assets, list) or any(
        not isinstance(asset, str) for asset in assets
    ):
        raise GitHubReleaseError("GitHub returned an invalid release asset list")
    if len(set(assets)) != len(assets):
        raise GitHubReleaseError("GitHub returned duplicate release asset names")
    return ReleaseSnapshot(True, frozenset(assets))


def _download_asset(
    repository: str,
    tag: str,
    name: str,
    output_directory: Path,
) -> Path:
    _validate_asset_name(name)
    destination = output_directory.resolve()
    destination.mkdir(parents=True, exist_ok=True)
    run_command(
        [
            "gh",
            "release",
            "download",
            tag,
            "--repo",
            repository,
            "--pattern",
            name,
            "--dir",
            str(destination),
        ]
    )
    downloaded = destination / name
    if not downloaded.is_file():
        raise GitHubReleaseError(f"GitHub did not download expected asset {name}")
    return downloaded


def download_package(
    repository: str,
    tag: str,
    version: str,
    platform: str,
    build_prefix: str,
    output_directory: Path,
) -> Path | None:
    """Download the one reusable native package for a matrix build, if any."""
    if SEMVER_PATTERN.fullmatch(version) is None or tag != f"v{version}":
        raise GitHubReleaseError("release tag and package version differ")
    try:
        expected_build_prefix = PLATFORM_BUILD_PREFIXES[platform]
    except KeyError as error:
        raise GitHubReleaseError(
            f"unsupported release platform: {platform!r}"
        ) from error
    if build_prefix not in ("", expected_build_prefix):
        raise GitHubReleaseError(
            f"unexpected build prefix for {platform}: {build_prefix!r}"
        )
    snapshot = release_snapshot(repository, tag)
    expected_stem = f"safetensors-mojo-{version}-"
    if build_prefix:
        expected_stem += f"{build_prefix}_"
    matches = sorted(
        name
        for name in snapshot.assets
        if name.startswith(expected_stem) and name.endswith(".conda")
    )
    if len(matches) > 1:
        raise GitHubReleaseError(
            f"release has multiple {platform} packages"
        )
    if not matches:
        return None
    package = _download_asset(repository, tag, matches[0], output_directory)
    validate_platform_artifact(
        package,
        version,
        platform,
        multi_platform=bool(build_prefix),
    )
    return package


def download_reusable_assets(
    repository: str,
    tag: str,
    artifact_list: Path,
    output_root: Path,
) -> ReleaseSnapshot:
    """Download exact existing assets that correspond to verified candidates."""
    artifacts = read_artifact_list(artifact_list)
    snapshot = release_snapshot(repository, tag)
    root = output_root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    if snapshot.exists:
        for artifact in artifacts:
            if artifact.filename in snapshot.assets:
                _download_asset(
                    repository,
                    tag,
                    artifact.filename,
                    root / artifact.platform,
                )
    return snapshot


def _remote_tag_object(repository: str, tag: str) -> str:
    object_id = run_command(
        [
            "gh",
            "api",
            f"repos/{repository}/git/ref/tags/{tag}",
            "--jq",
            ".object.sha",
        ]
    ).strip()
    if GIT_OBJECT_ID_PATTERN.fullmatch(object_id) is None:
        raise GitHubReleaseError("GitHub returned an invalid tag object ID")
    return object_id


def _git(repository: Path, *arguments: str) -> str:
    return run_command(["git", "-C", str(repository), *arguments]).strip()


def _release_notes(repository: Path, tag: str) -> str:
    reference = f"refs/tags/{tag}"
    if _git(repository, "cat-file", "-t", reference) != "tag":
        raise GitHubReleaseError(f"{tag} must be an annotated tag")
    notes = run_command(
        [
            "git",
            "-C",
            str(repository),
            "for-each-ref",
            "--format=%(contents)",
            reference,
        ]
    )
    first_content_line = next(
        (line for line in notes.splitlines() if line.strip()),
        "",
    )
    if first_content_line != "## Summary":
        raise GitHubReleaseError(
            "annotated tag notes must start with ## Summary"
        )
    return notes


def publish_release(
    repository: str,
    tag: str,
    expected_tag_object: str,
    expected_commit: str,
    source_directory: Path,
    artifact_list: Path,
    *,
    release_exists: bool,
) -> None:
    """Create or complete one release after revalidating its immutable tag."""
    _validate_repository(repository)
    _validate_tag(tag)
    if GIT_OBJECT_ID_PATTERN.fullmatch(expected_tag_object) is None:
        raise GitHubReleaseError("invalid expected tag object ID")
    if GIT_OBJECT_ID_PATTERN.fullmatch(expected_commit) is None:
        raise GitHubReleaseError("invalid expected tag commit ID")

    remote_tag_object = _remote_tag_object(repository, tag)
    local_tag = validate_repository_tag(source_directory, tag)
    if (
        remote_tag_object != expected_tag_object
        or local_tag.tag_object != expected_tag_object
        or local_tag.commit != expected_commit
    ):
        raise GitHubReleaseError(f"{tag} changed after release validation")

    artifacts = read_artifact_list(artifact_list)
    if release_exists:
        uploads = [
            str(artifact.path) for artifact in artifacts if not artifact.reused
        ]
        if uploads:
            run_command(
                [
                    "gh",
                    "release",
                    "upload",
                    tag,
                    *uploads,
                    "--repo",
                    repository,
                ]
            )
        else:
            print("GitHub Release already contains every canonical package.")
        return

    if any(artifact.reused for artifact in artifacts):
        raise GitHubReleaseError(
            "a new GitHub Release cannot reuse existing release assets"
        )
    notes = _release_notes(source_directory, tag)
    run_command(
        [
            "gh",
            "release",
            "create",
            tag,
            *(str(artifact.path) for artifact in artifacts),
            "--repo",
            repository,
            "--verify-tag",
            "--title",
            f"safetensors.mojo {tag}",
            "--notes-file",
            "-",
        ],
        input_text=notes,
    )


def parse_boolean(value: str) -> bool:
    if value == "true":
        return True
    if value == "false":
        return False
    raise argparse.ArgumentTypeError("expected true or false")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    package = commands.add_parser("download-package")
    package.add_argument("--repository", required=True)
    package.add_argument("--tag", required=True)
    package.add_argument("--version", required=True)
    package.add_argument("--platform", required=True)
    package.add_argument("--build-prefix", default="")
    package.add_argument("--output-directory", required=True, type=Path)
    package.add_argument("--github-output", required=True, type=Path)

    reusable = commands.add_parser("download-reusable")
    reusable.add_argument("--repository", required=True)
    reusable.add_argument("--tag", required=True)
    reusable.add_argument("--artifact-list", required=True, type=Path)
    reusable.add_argument("--output-root", required=True, type=Path)
    reusable.add_argument("--github-output", required=True, type=Path)

    publish = commands.add_parser("publish")
    publish.add_argument("--repository", required=True)
    publish.add_argument("--tag", required=True)
    publish.add_argument("--expected-tag-object", required=True)
    publish.add_argument("--expected-commit", required=True)
    publish.add_argument("--source-directory", required=True, type=Path)
    publish.add_argument("--artifact-list", required=True, type=Path)
    publish.add_argument("--release-exists", required=True, type=parse_boolean)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    try:
        if arguments.command == "download-package":
            package = download_package(
                arguments.repository,
                arguments.tag,
                arguments.version,
                arguments.platform,
                arguments.build_prefix,
                arguments.output_directory,
            )
            write_github_output(
                arguments.github_output,
                exists=str(package is not None).lower(),
                path="" if package is None else str(package),
            )
        elif arguments.command == "download-reusable":
            snapshot = download_reusable_assets(
                arguments.repository,
                arguments.tag,
                arguments.artifact_list,
                arguments.output_root,
            )
            write_github_output(
                arguments.github_output,
                release_exists=str(snapshot.exists).lower(),
                root=str(arguments.output_root.resolve()),
            )
        else:
            publish_release(
                arguments.repository,
                arguments.tag,
                arguments.expected_tag_object,
                arguments.expected_commit,
                arguments.source_directory,
                arguments.artifact_list,
                release_exists=arguments.release_exists,
            )
    except (OSError, ValueError, zipfile.BadZipFile) as error:
        print(f"error: GitHub release operation failed: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
