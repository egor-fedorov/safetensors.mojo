#!/usr/bin/env python3
"""Install one local Conda artifact with Pixi and smoke-test its Mojo API."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
import hashlib
import json
import os
from pathlib import Path
import platform
import shlex
import shutil
import subprocess
import sys
import tempfile
from typing import Any, Iterator
import zipfile


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONSUMER_SOURCE = (
    PROJECT_ROOT / "tools" / "packaging" / "consumers" / "package_smoke.mojo"
)
PACKAGE_NAME = "safetensors-mojo"
MODULAR_CHANNEL = "https://conda.modular.com/max"
CONDA_FORGE_CHANNEL = "conda-forge"


def parse_arguments(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Install a built safetensors-mojo Conda artifact in an isolated "
            "Pixi workspace and run a Mojo import smoke test."
        )
    )
    parser.add_argument(
        "artifact",
        type=Path,
        help="path to one safetensors-mojo .conda artifact",
    )
    parser.add_argument(
        "--consumer-source",
        type=Path,
        default=CONSUMER_SOURCE,
        help=(
            "Mojo smoke-test source to run against the installed package; "
            "defaults to tools/packaging/consumers/package_smoke.mojo"
        ),
    )
    parser.add_argument(
        "--platform",
        dest="target_platform",
        help="Conda target platform; defaults to the current machine",
    )
    return parser.parse_args(argv)


def resolve_consumer_source(source: Path) -> Path:
    """Return an absolute existing Mojo consumer source path."""
    try:
        resolved = source.expanduser().resolve(strict=True)
    except OSError as error:
        raise ValueError(f"consumer source does not exist: {source}") from error
    if not resolved.is_file():
        raise ValueError(f"consumer source is not a file: {source}")
    return resolved


def current_conda_platform() -> str:
    machine = platform.machine().lower()
    if sys.platform.startswith("linux") and machine in {"x86_64", "amd64"}:
        return "linux-64"
    if sys.platform == "darwin" and machine in {"arm64", "aarch64"}:
        return "osx-arm64"
    if sys.platform == "darwin" and machine in {"x86_64", "amd64"}:
        return "osx-64"
    if sys.platform.startswith("win") and machine in {"x86_64", "amd64"}:
        return "win-64"
    raise RuntimeError(
        f"cannot infer a supported Conda platform from {sys.platform!r}/{machine!r}"
    )


def artifact_identity(artifact: Path) -> tuple[str, str, str]:
    if artifact.suffix != ".conda":
        raise ValueError(f"expected a .conda artifact, received: {artifact}")

    prefix = f"{PACKAGE_NAME}-"
    if not artifact.name.startswith(prefix):
        raise ValueError(
            f"artifact filename must start with {prefix!r}: {artifact.name}"
        )

    version_and_build = artifact.name[len(prefix) : -len(".conda")]
    try:
        version, build = version_and_build.rsplit("-", 1)
    except ValueError as error:
        raise ValueError(
            f"artifact filename has no Conda build string: {artifact.name}"
        ) from error
    if not version or not build:
        raise ValueError(f"invalid Conda artifact filename: {artifact.name}")

    with zipfile.ZipFile(artifact) as archive:
        members = set(archive.namelist())
        if "metadata.json" not in members:
            raise ValueError(f"artifact has no metadata.json member: {artifact}")
        package_format = json.loads(archive.read("metadata.json"))
    if package_format.get("conda_pkg_format_version") != 2:
        raise ValueError(f"artifact is not a version 2 Conda package: {artifact}")
    if (
        len(
            [
                name
                for name in members
                if name.startswith("info-") and name.endswith(".tar.zst")
            ]
        )
        != 1
    ):
        raise ValueError(f"artifact must contain exactly one info archive: {artifact}")
    if (
        len(
            [
                name
                for name in members
                if name.startswith("pkg-") and name.endswith(".tar.zst")
            ]
        )
        != 1
    ):
        raise ValueError(f"artifact must contain exactly one payload archive: {artifact}")

    return PACKAGE_NAME, version, build


def sha256_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def run(
    command: list[str],
    *,
    cwd: Path,
    environment: dict[str, str],
    capture_output: bool = False,
) -> subprocess.CompletedProcess[str]:
    print(f"+ {shlex.join(command)}", flush=True)
    return subprocess.run(
        command,
        cwd=cwd,
        env=environment,
        check=True,
        text=True,
        capture_output=capture_output,
    )


def dictionaries(value: Any) -> Iterator[dict[str, Any]]:
    if isinstance(value, dict):
        yield value
        for child in value.values():
            yield from dictionaries(child)
    elif isinstance(value, list):
        for child in value:
            yield from dictionaries(child)


def package_record(listing: Any, name: str) -> dict[str, Any]:
    matches = [
        record
        for record in dictionaries(listing)
        if record.get("name") == name and "version" in record
    ]
    if len(matches) != 1:
        raise RuntimeError(
            f"expected one installed {name!r} record, found {len(matches)}"
        )
    return matches[0]


def initialize_workspace(
    pixi: str,
    workspace: Path,
    channels: list[str],
    target_platform: str,
    environment: dict[str, str],
) -> None:
    command = [
        pixi,
        "init",
        str(workspace),
        "--format",
        "pixi",
    ]
    for channel in channels:
        command.extend(["--channel", channel])
    command.extend(["--platform", target_platform])
    run(
        command,
        cwd=workspace.parent,
        environment=environment,
    )


def list_packages(
    pixi: str,
    workspace: Path,
    environment: dict[str, str],
    *,
    no_install: bool = False,
) -> Any:
    command = [
        pixi,
        "list",
        "--manifest-path",
        str(workspace),
        "--no-config",
        "--json",
    ]
    if no_install:
        command.append("--no-install")
    completed = run(
        command,
        cwd=workspace,
        environment=environment,
        capture_output=True,
    )
    return json.loads(completed.stdout)


def verify_identity(
    package: dict[str, Any],
    expected_version: str,
    expected_build: str,
) -> None:
    if package["version"] != expected_version:
        raise RuntimeError(
            "installed package version does not match the artifact: "
            f"{package['version']} != {expected_version}"
        )
    if package.get("build") != expected_build:
        raise RuntimeError(
            "installed package build does not match the artifact: "
            f"{package.get('build')} != {expected_build}"
        )


def write_repodata(
    channel: Path,
    target_platform: str,
    artifact: Path,
    package: dict[str, Any],
) -> None:
    package_subdir = package.get("subdir")
    if package_subdir != target_platform:
        raise RuntimeError(
            "artifact platform does not match the smoke-test target: "
            f"{package_subdir} != {target_platform}"
        )

    target_directory = channel / target_platform
    target_directory.mkdir(parents=True)
    shutil.copy2(artifact, target_directory / artifact.name)

    entry: dict[str, Any] = {
        "arch": package.get("arch"),
        "build": package["build"],
        "build_number": package["build_number"],
        "constrains": package.get("constrains", []),
        "depends": package.get("depends", []),
        "license": package.get("license"),
        "name": package["name"],
        "platform": package.get("platform"),
        "sha256": sha256_digest(artifact),
        "size": artifact.stat().st_size,
        "subdir": package_subdir,
        "timestamp": package.get("timestamp"),
        "version": package["version"],
    }
    entry = {key: value for key, value in entry.items() if value is not None}
    repodata = {
        "info": {"subdir": target_platform},
        "packages": {},
        "packages.conda": {artifact.name: entry},
        "removed": [],
        "repodata_version": 1,
    }
    (target_directory / "repodata.json").write_text(
        json.dumps(repodata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    noarch_directory = channel / "noarch"
    noarch_directory.mkdir()
    empty_repodata = {
        "info": {"subdir": "noarch"},
        "packages": {},
        "packages.conda": {},
        "removed": [],
        "repodata_version": 1,
    }
    (noarch_directory / "repodata.json").write_text(
        json.dumps(empty_repodata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def main() -> int:
    arguments = parse_arguments()
    pixi = shutil.which("pixi")
    if pixi is None:
        print("error: pixi is not available on PATH", file=sys.stderr)
        return 1

    try:
        artifact = arguments.artifact.expanduser().resolve(strict=True)
        consumer_source = resolve_consumer_source(arguments.consumer_source)
        package_name, expected_version, expected_build = artifact_identity(artifact)
        target_platform = arguments.target_platform or current_conda_platform()
        with tempfile.TemporaryDirectory(prefix="safetensors-mojo-smoke-") as root:
            temporary_root = Path(root)
            metadata_workspace = temporary_root / "metadata-workspace"
            channel = temporary_root / "channel"
            workspace = temporary_root / "workspace"
            consumer = workspace / "package_smoke.mojo"
            cache = temporary_root / "pixi-cache"
            modular_cache = temporary_root / "modular-cache"

            environment = os.environ.copy()
            environment["PIXI_CACHE_DIR"] = str(cache)
            environment["PIXI_NO_PROGRESS"] = "true"
            environment["MODULAR_CACHE_DIR"] = str(modular_cache)

            initialize_workspace(
                pixi,
                metadata_workspace,
                [MODULAR_CHANNEL, CONDA_FORGE_CHANNEL],
                target_platform,
                environment,
            )
            run(
                [
                    pixi,
                    "add",
                    "--manifest-path",
                    str(metadata_workspace),
                    "--no-config",
                    "--no-install",
                    str(artifact),
                ],
                cwd=metadata_workspace,
                environment=environment,
            )
            metadata_listing = list_packages(
                pixi,
                metadata_workspace,
                environment,
                no_install=True,
            )
            metadata_package = package_record(metadata_listing, package_name)
            verify_identity(metadata_package, expected_version, expected_build)
            write_repodata(
                channel,
                target_platform,
                artifact,
                metadata_package,
            )

            local_channel = channel.as_uri()
            initialize_workspace(
                pixi,
                workspace,
                [local_channel, MODULAR_CHANNEL, CONDA_FORGE_CHANNEL],
                target_platform,
                environment,
            )
            shutil.copy2(consumer_source, consumer)
            exact_spec = f"{package_name}=={expected_version}={expected_build}"
            run(
                [
                    pixi,
                    "add",
                    "--manifest-path",
                    str(workspace),
                    "--no-config",
                    exact_spec,
                ],
                cwd=workspace,
                environment=environment,
            )

            listing = list_packages(pixi, workspace, environment)
            package = package_record(listing, package_name)
            verify_identity(package, expected_version, expected_build)
            package_url = str(package.get("url", ""))
            expected_package_path = (
                channel / target_platform / artifact.name
            ).resolve()
            if package_url.startswith("file://"):
                selected_package = Path(package_url.removeprefix("file://"))
            else:
                selected_package = Path(package_url)
            if not selected_package.is_absolute():
                selected_package = workspace / selected_package
            if selected_package.resolve() != expected_package_path:
                raise RuntimeError(
                    "the installed package was not selected from the local channel: "
                    f"{package_url}"
                )

            compiler = package_record(listing, "mojo-compiler")
            if compiler["version"] != "1.0.0":
                raise RuntimeError(
                    "the artifact did not resolve the required Mojo compiler: "
                    f"{compiler['version']}"
                )

            run(
                [
                    pixi,
                    "run",
                    "--manifest-path",
                    str(workspace),
                    "--no-config",
                    "--clean-env",
                    "--executable",
                    "mojo",
                    "run",
                    str(consumer),
                ],
                cwd=workspace,
                environment=environment,
            )
    except (
        OSError,
        ValueError,
        RuntimeError,
        subprocess.CalledProcessError,
        zipfile.BadZipFile,
    ) as error:
        print(f"error: package smoke test failed: {error}", file=sys.stderr)
        return 1

    print(
        f"Verified {package_name} {expected_version} ({expected_build}) "
        f"from {artifact.name}."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
