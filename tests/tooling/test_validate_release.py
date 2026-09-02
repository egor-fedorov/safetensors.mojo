from __future__ import annotations

from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
import json
from pathlib import Path
import subprocess
import tempfile
import unittest

from tools.release.validate_release import (
    PLATFORM_BUILD_PREFIXES,
    PLATFORM_RUNNERS,
    ReleaseValidationError,
    ReleaseTag,
    main,
    release_matrix,
    validate_manifest_tag,
    validate_repository_tag,
    write_github_output,
)


class ValidateReleaseTests(unittest.TestCase):
    def git(self, repository: Path, *arguments: str) -> str:
        return subprocess.run(
            ["git", "-C", str(repository), *arguments],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()

    def initialize_repository(self, repository: Path) -> str:
        self.git(repository, "init", "--quiet")
        self.git(repository, "config", "user.name", "Release Test")
        self.git(repository, "config", "user.email", "release@example.com")
        self.git(repository, "add", ".")
        self.git(repository, "commit", "--quiet", "-m", "release")
        return self.git(repository, "rev-parse", "HEAD")

    def write_manifest(
        self,
        directory: str,
        workspace_version: str,
        package_version: str | None = None,
        platforms: list[str] | None = None,
    ) -> Path:
        manifest_path = Path(directory) / "pixi.toml"
        resolved_package_version = package_version or workspace_version
        contents = "[workspace]\n" f'version = "{workspace_version}"\n'
        if platforms is not None:
            contents += f"platforms = {json.dumps(platforms)}\n"
        contents += (
            "\n"
            "[package]\n"
            f'version = "{resolved_package_version}"\n'
        )
        manifest_path.write_text(contents, encoding="utf-8")
        return manifest_path

    def test_accepts_stable_and_prerelease_versions(self) -> None:
        versions = [
            "0.2.0",
            "1.2.3-alpha",
            "1.2.3-alpha.1",
            "1.2.3-0.3.7",
            "1.2.3-rc.1+build.5",
        ]
        with tempfile.TemporaryDirectory() as directory:
            for version in versions:
                with self.subTest(version=version):
                    manifest = self.write_manifest(directory, version)
                    self.assertEqual(
                        validate_manifest_tag(manifest, f"v{version}"),
                        version,
                    )

    def test_rejects_workspace_and_package_version_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            manifest = self.write_manifest(directory, "0.2.0", "0.2.1")
            with self.assertRaisesRegex(
                ReleaseValidationError,
                "workspace and package versions differ",
            ):
                validate_manifest_tag(manifest, "v0.2.1")

    def test_rejects_malformed_semver(self) -> None:
        versions = [
            "1.2",
            "01.2.3",
            "1.02.3",
            "1.2.03",
            "1.2.3-",
            "1.2.3-01",
            "1.2.3-alpha..1",
            "1.2.3-" + ("a" * 5_000) + "!",
            "v1.2.3",
        ]
        with tempfile.TemporaryDirectory() as directory:
            for version in versions:
                with self.subTest(version=version):
                    manifest = self.write_manifest(directory, version)
                    with self.assertRaisesRegex(
                        ReleaseValidationError,
                        "package version is not strict SemVer",
                    ):
                        validate_manifest_tag(manifest, f"v{version}")

    def test_rejects_tag_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            manifest = self.write_manifest(directory, "0.2.0")
            with self.assertRaisesRegex(
                ReleaseValidationError,
                "release tag 'v0.2.1' does not exactly match 'v0.2.0'",
            ):
                validate_manifest_tag(manifest, "v0.2.1")

    def test_release_matrix_maps_declared_platforms_to_native_runners(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            platforms = ["linux-64", "linux-aarch64", "osx-arm64"]
            manifest = self.write_manifest(
                directory,
                "0.7.0",
                platforms=platforms,
            )

            self.assertEqual(
                release_matrix(manifest),
                {
                    "include": [
                        {
                            "platform": platform_name,
                            "runner": PLATFORM_RUNNERS[platform_name],
                            "build_prefix": PLATFORM_BUILD_PREFIXES[
                                platform_name
                            ],
                        }
                        for platform_name in platforms
                    ]
                },
            )

    def test_release_matrix_preserves_legacy_linux_only_tags(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            manifest = self.write_manifest(
                directory,
                "0.5.0",
                platforms=["linux-64"],
            )

            self.assertEqual(
                release_matrix(manifest),
                {
                    "include": [
                        {
                            "platform": "linux-64",
                            "runner": "ubuntu-latest",
                            "build_prefix": "",
                        }
                    ]
                },
            )

    def test_release_matrix_rejects_missing_duplicate_and_unknown_platforms(
        self,
    ) -> None:
        cases = (
            (None, "must define workspace.platforms"),
            ([], "must be a non-empty array"),
            (["linux-64", "linux-64"], "contains duplicate"),
            (["win-64"], "unsupported release platform"),
        )
        with tempfile.TemporaryDirectory() as directory:
            for platforms, message in cases:
                with self.subTest(platforms=platforms):
                    manifest = self.write_manifest(
                        directory,
                        "0.7.0",
                        platforms=platforms,
                    )
                    with self.assertRaisesRegex(ReleaseValidationError, message):
                        release_matrix(manifest)

    def test_write_github_output_appends_version_and_compact_matrix(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "github-output"
            output.write_text("existing=value\n", encoding="utf-8")
            matrix = {
                "include": [
                    {
                        "platform": "osx-arm64",
                        "runner": "macos-15",
                        "build_prefix": "",
                    }
                ]
            }

            write_github_output(output, "0.7.0", matrix)

            self.assertEqual(
                output.read_text(encoding="utf-8"),
                "existing=value\n"
                "version=0.7.0\n"
                'matrix={"include":[{"platform":"osx-arm64",'
                '"runner":"macos-15","build_prefix":""}]}\n'
                'platforms=["osx-arm64"]\n',
            )

    def test_repository_tag_preserves_annotated_object_and_commit(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            repository = Path(directory)
            (repository / "README.md").write_text("release\n", encoding="utf-8")
            commit = self.initialize_repository(repository)
            self.git(repository, "tag", "--annotate", "v0.7.0", "-m", "notes")

            release_tag = validate_repository_tag(repository, "v0.7.0")

            self.assertEqual(release_tag.commit, commit)
            self.assertEqual(
                release_tag.tag_object,
                self.git(repository, "rev-parse", "refs/tags/v0.7.0"),
            )
            self.assertNotEqual(release_tag.tag_object, release_tag.commit)

    def test_repository_tag_preserves_lightweight_rerun_compatibility(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            repository = Path(directory)
            (repository / "README.md").write_text("release\n", encoding="utf-8")
            commit = self.initialize_repository(repository)
            self.git(repository, "tag", "v0.7.0")

            self.assertEqual(
                validate_repository_tag(repository, "v0.7.0"),
                ReleaseTag(tag_object=commit, commit=commit),
            )

    def test_repository_tag_rejects_missing_tag_and_different_head(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            repository = Path(directory)
            tracked = repository / "README.md"
            tracked.write_text("release\n", encoding="utf-8")
            self.initialize_repository(repository)
            with self.assertRaisesRegex(ReleaseValidationError, "show-ref"):
                validate_repository_tag(repository, "v0.7.0")

            self.git(repository, "tag", "v0.7.0")
            tracked.write_text("next\n", encoding="utf-8")
            self.git(repository, "add", ".")
            self.git(repository, "commit", "--quiet", "-m", "next")
            with self.assertRaisesRegex(
                ReleaseValidationError,
                "checked-out commit does not match",
            ):
                validate_repository_tag(repository, "v0.7.0")

    def test_cli_success(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            manifest = self.write_manifest(directory, "1.2.3-rc.1")
            stdout = StringIO()
            stderr = StringIO()
            with redirect_stdout(stdout), redirect_stderr(stderr):
                result = main(
                    [
                        "--manifest",
                        str(manifest),
                        "--tag",
                        "v1.2.3-rc.1",
                    ]
                )

            self.assertEqual(result, 0)
            self.assertEqual(stderr.getvalue(), "")
            self.assertEqual(
                stdout.getvalue(),
                "Validated release tag v1.2.3-rc.1 for version 1.2.3-rc.1.\n",
            )

    def test_cli_writes_release_plan_for_github_actions(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            repository = Path(directory) / "repository"
            repository.mkdir()
            manifest = self.write_manifest(
                str(repository),
                "0.7.0",
                platforms=["linux-64", "osx-arm64"],
            )
            commit = self.initialize_repository(repository)
            self.git(repository, "tag", "--annotate", "v0.7.0", "-m", "notes")
            tag_object = self.git(
                repository,
                "rev-parse",
                "refs/tags/v0.7.0",
            )
            output = Path(directory) / "github-output"

            result = main(
                [
                    "--manifest",
                    str(manifest),
                    "--tag",
                    "v0.7.0",
                    "--repository",
                    str(repository),
                    "--github-output",
                    str(output),
                ]
            )

            self.assertEqual(result, 0)
            self.assertEqual(
                output.read_text(encoding="utf-8"),
                "version=0.7.0\n"
                'matrix={"include":[{"platform":"linux-64",'
                '"runner":"ubuntu-latest","build_prefix":"linux64"},'
                '{"platform":"osx-arm64","runner":"macos-15",'
                '"build_prefix":"osxarm64"}]}\n'
                'platforms=["linux-64","osx-arm64"]\n'
                f"commit={commit}\n"
                f"tag_object={tag_object}\n",
            )

    def test_cli_requires_repository_before_writing_github_output(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            manifest = self.write_manifest(
                directory,
                "0.7.0",
                platforms=["linux-64"],
            )
            output = Path(directory) / "github-output"
            stderr = StringIO()

            with redirect_stderr(stderr):
                result = main(
                    [
                        "--manifest",
                        str(manifest),
                        "--tag",
                        "v0.7.0",
                        "--github-output",
                        str(output),
                    ]
                )

            self.assertEqual(result, 1)
            self.assertFalse(output.exists())
            self.assertIn("--repository is required", stderr.getvalue())

    def test_cli_reports_invalid_utf8_without_a_traceback(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            manifest = Path(directory) / "pixi.toml"
            manifest.write_bytes(b"\xff")
            stdout = StringIO()
            stderr = StringIO()

            with redirect_stdout(stdout), redirect_stderr(stderr):
                result = main(
                    [
                        "--manifest",
                        str(manifest),
                        "--tag",
                        "v0.2.0",
                    ]
                )

            self.assertEqual(result, 1)
            self.assertEqual(stdout.getvalue(), "")
            self.assertIn("error: release validation failed:", stderr.getvalue())
            self.assertNotIn("Traceback", stderr.getvalue())


if __name__ == "__main__":
    unittest.main()
