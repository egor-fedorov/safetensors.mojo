#!/usr/bin/env python3
"""Decide which release artifacts should be uploaded to Prefix.dev."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import string
import sys
import urllib.error
import urllib.request
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tools.release.artifacts import (  # noqa: E402
    ReleaseArtifact,
    read_artifact_list,
    write_github_output,
)


USER_AGENT = "safetensors-mojo-release-preflight/1"
REQUEST_TIMEOUT_SECONDS = 30


def validate_repodata(
    repodata: Any,
    expected_subdir: str,
) -> dict[str, Any]:
    """Validate the parts of a successful Conda repodata response we use."""
    if not isinstance(repodata, dict):
        raise ValueError("prefix.dev repodata must be a JSON object")

    info = repodata.get("info")
    if not isinstance(info, dict):
        raise ValueError("prefix.dev repodata must contain an info object")
    actual_subdir = info.get("subdir")
    if actual_subdir != expected_subdir:
        raise ValueError(
            "prefix.dev repodata subdir differs from the request: "
            f"{actual_subdir!r} != {expected_subdir!r}"
        )

    section_names = ("packages.conda", "packages")
    if not any(name in repodata for name in section_names):
        raise ValueError("prefix.dev repodata has no package index")
    for section_name in section_names:
        if section_name in repodata and not isinstance(
            repodata[section_name],
            dict,
        ):
            raise ValueError(
                f"prefix.dev repodata section {section_name!r} must be an object"
            )
    return repodata


def fetch_repodata(channel: str, subdir: str) -> dict[str, Any] | None:
    """Fetch one prefix.dev repodata document without using a cached response."""
    url = f"https://prefix.dev/{channel}/{subdir}/repodata.json"
    request = urllib.request.Request(
        url,
        headers={
            "User-Agent": USER_AGENT,
            "Accept": "application/json",
            "Cache-Control": "no-cache",
        },
    )
    try:
        with urllib.request.urlopen(
            request,
            timeout=REQUEST_TIMEOUT_SECONDS,
        ) as response:
            return validate_repodata(json.load(response), subdir)
    except urllib.error.HTTPError as error:
        if error.code == 404:
            return None
        raise


def find_package_record(
    repodata: Any,
    package_filename: str,
) -> dict[str, Any] | None:
    """Find one package record in current or legacy Conda repodata sections."""
    if not isinstance(repodata, dict):
        raise ValueError("prefix.dev repodata must be a JSON object")
    if not package_filename:
        raise ValueError("package filename must not be empty")

    section_names = ("packages.conda", "packages")
    if not any(name in repodata for name in section_names):
        raise ValueError("prefix.dev repodata has no package index")
    matches: list[dict[str, Any]] = []
    for section_name in section_names:
        section = repodata.get(section_name, {})
        if not isinstance(section, dict):
            raise ValueError(
                f"prefix.dev repodata section {section_name!r} must be an object"
            )
        record = section.get(package_filename)
        if record is None:
            continue
        if not isinstance(record, dict):
            raise ValueError(
                f"prefix.dev record for {package_filename!r} must be an object"
            )
        matches.append(record)

    if len(matches) > 1:
        raise ValueError(
            f"prefix.dev indexes {package_filename!r} more than once"
        )
    return matches[0] if matches else None


def decide_publish(
    record: dict[str, Any] | None,
    local_sha256: str,
) -> bool:
    """Return whether to publish, rejecting an immutable filename collision."""
    if not isinstance(local_sha256, str):
        raise ValueError("local package has an invalid SHA-256")
    normalized_local = local_sha256.lower()
    if len(normalized_local) != 64 or any(
        character not in string.hexdigits for character in normalized_local
    ):
        raise ValueError("local package has an invalid SHA-256")
    if record is None:
        return True
    if not isinstance(record, dict):
        raise ValueError("prefix.dev package record must be an object")
    remote_sha256 = record.get("sha256")
    if not isinstance(remote_sha256, str) or not remote_sha256:
        raise ValueError("prefix.dev package record has no SHA-256")
    normalized_remote = remote_sha256.lower()
    if len(normalized_remote) != 64 or any(
        character not in string.hexdigits for character in normalized_remote
    ):
        raise ValueError("prefix.dev package record has an invalid SHA-256")

    if normalized_remote != normalized_local:
        raise ValueError(
            "prefix.dev already contains the package filename with a different "
            "SHA-256; package filenames are immutable"
        )
    return False


def preflight_artifacts(
    channel: str,
    artifact_list: Path,
) -> list[ReleaseArtifact]:
    """Return artifacts absent from Prefix after every row validates."""
    pending: list[ReleaseArtifact] = []
    for artifact in read_artifact_list(artifact_list):
        repodata = fetch_repodata(channel, artifact.platform)
        record = (
            None
            if repodata is None
            else find_package_record(repodata, artifact.filename)
        )
        if decide_publish(record, artifact.sha256):
            pending.append(artifact)
    return pending


def write_publish_plan(
    publish_list: Path,
    github_output: Path,
    artifacts: list[ReleaseArtifact],
) -> None:
    """Write the complete Prefix upload plan after a successful preflight."""
    resolved = publish_list.resolve()
    resolved.write_text(
        "".join(
            f"{artifact.platform}\t{artifact.path}\n" for artifact in artifacts
        ),
        encoding="utf-8",
    )
    write_github_output(github_output, publish_list=str(resolved))


def parse_arguments(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--channel", required=True, help="prefix.dev channel name")
    parser.add_argument(
        "--artifact-list",
        required=True,
        type=Path,
        help="validated five-column release artifact manifest",
    )
    parser.add_argument(
        "--publish-list",
        required=True,
        type=Path,
        help="output TSV containing artifacts that require upload",
    )
    parser.add_argument(
        "--github-output",
        required=True,
        type=Path,
        help="path named by the GitHub Actions GITHUB_OUTPUT variable",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    arguments = parse_arguments(argv)
    try:
        pending = preflight_artifacts(
            arguments.channel,
            arguments.artifact_list,
        )
        write_publish_plan(
            arguments.publish_list,
            arguments.github_output,
            pending,
        )
    except (OSError, ValueError) as error:
        print(f"error: prefix.dev preflight failed: {error}", file=sys.stderr)
        return 1

    print(f"Prefix.dev preflight selected {len(pending)} artifacts for upload.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
