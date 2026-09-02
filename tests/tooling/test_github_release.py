from __future__ import annotations

from collections.abc import Callable
from contextlib import redirect_stderr
import hashlib
import io
import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import MagicMock, patch
import zipfile

from tools.release import github_release


class GitHubReleaseTests(unittest.TestCase):
    def git(self, repository: Path, *arguments: str) -> str:
        return subprocess.run(
            ["git", "-C", str(repository), *arguments],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()

    def initialize_repository(
        self,
        repository: Path,
        *,
        annotated: bool = True,
        notes: str = "## Summary\n\nRelease notes.",
    ) -> tuple[str, str]:
        repository.mkdir()
        (repository / "README.md").write_text("release\n", encoding="utf-8")
        self.git(repository, "init", "--quiet")
        self.git(repository, "config", "user.name", "Release Test")
        self.git(repository, "config", "user.email", "release@example.com")
        self.git(repository, "add", ".")
        self.git(repository, "commit", "--quiet", "-m", "release")
        if annotated:
            self.git(
                repository,
                "tag",
                "--annotate",
                "--cleanup=verbatim",
                "v0.7.0",
                "-m",
                notes,
            )
        else:
            self.git(repository, "tag", "v0.7.0")
        return (
            self.git(repository, "rev-parse", "refs/tags/v0.7.0"),
            self.git(repository, "rev-parse", "HEAD"),
        )

    def make_conda(self, path: Path, *, build: str) -> Path:
        artifact = path / f"safetensors-mojo-0.7.0-{build}.conda"
        artifact.parent.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(artifact, "w") as archive:
            archive.writestr(
                "metadata.json",
                json.dumps({"conda_pkg_format_version": 2}),
            )
            archive.writestr("info-test.tar.zst", b"info")
            archive.writestr("pkg-test.tar.zst", build.encode())
        return artifact

    def write_artifact_list(
        self,
        root: Path,
        artifacts: list[tuple[str, Path, bool]],
    ) -> Path:
        manifest = root / "release-artifacts.tsv"
        manifest.write_text(
            "".join(
                f"{platform}\t{artifact}\t{artifact.name}\t"
                f"{str(reused).lower()}\t"
                f"{hashlib.sha256(artifact.read_bytes()).hexdigest()}\n"
                for platform, artifact, reused in artifacts
            ),
            encoding="utf-8",
        )
        return manifest

    def real_git_fake_gh(
        self,
        remote_tag_object: str,
        calls: list[tuple[list[str], str | None]],
    ) -> Callable[..., str]:
        def run(
            command: list[str],
            *,
            cwd: Path | None = None,
            input_text: str | None = None,
        ) -> str:
            calls.append((command, input_text))
            if command[0] == "git":
                return subprocess.run(
                    command,
                    cwd=cwd,
                    input=input_text,
                    check=True,
                    capture_output=True,
                    text=True,
                ).stdout
            if command[:2] == ["gh", "api"]:
                return remote_tag_object + "\n"
            return ""

        return run

    @patch("tools.release.github_release.run_command")
    def test_release_snapshot_distinguishes_missing_and_existing(
        self,
        run_command: MagicMock,
    ) -> None:
        run_command.return_value = ""
        self.assertEqual(
            github_release.release_snapshot("owner/repository", "v0.7.0"),
            github_release.ReleaseSnapshot(False, frozenset()),
        )

        run_command.return_value = json.dumps(
            {
                "tag": "v0.7.0",
                "assets": ["first.conda", "second.conda"],
            }
        )
        self.assertEqual(
            github_release.release_snapshot("owner/repository", "v0.7.0"),
            github_release.ReleaseSnapshot(
                True,
                frozenset({"first.conda", "second.conda"}),
            ),
        )

        run_command.return_value = json.dumps(
            {
                "tag": "v0.7.0",
                "assets": ["checksums[legacy].txt"],
            }
        )
        self.assertEqual(
            github_release.release_snapshot("owner/repository", "v0.7.0"),
            github_release.ReleaseSnapshot(
                True,
                frozenset({"checksums[legacy].txt"}),
            ),
        )

    @patch("tools.release.github_release.run_command")
    def test_release_snapshot_rejects_malformed_or_duplicate_records(
        self,
        run_command: MagicMock,
    ) -> None:
        invalid_outputs = (
            "not-json\n",
            json.dumps({"tag": "other", "assets": []}),
            json.dumps({"tag": "v0.7.0", "assets": "bad"}),
            json.dumps({"tag": "v0.7.0", "assets": ["same", "same"]}),
            "{}\n{}\n",
        )
        for output in invalid_outputs:
            with self.subTest(output=output):
                run_command.return_value = output
                with self.assertRaises(github_release.GitHubReleaseError):
                    github_release.release_snapshot(
                        "owner/repository",
                        "v0.7.0",
                    )

    @patch("tools.release.github_release.run_command")
    def test_download_package_handles_missing_unique_and_duplicate_assets(
        self,
        run_command: MagicMock,
    ) -> None:
        with tempfile.TemporaryDirectory() as raw_directory:
            root = Path(raw_directory)
            output = root / "download"
            run_command.return_value = ""
            self.assertIsNone(
                github_release.download_package(
                    "owner/repository",
                    "v0.7.0",
                    "0.7.0",
                    "linux-64",
                    "linux64",
                    output,
                )
            )

            names = [
                "safetensors-mojo-0.7.0-linux64_first_0.conda",
                "safetensors-mojo-0.7.0-linux64_second_0.conda",
            ]
            run_command.return_value = json.dumps(
                {"tag": "v0.7.0", "assets": names}
            )
            with self.assertRaisesRegex(
                github_release.GitHubReleaseError,
                "multiple linux-64 packages",
            ):
                github_release.download_package(
                    "owner/repository",
                    "v0.7.0",
                    "0.7.0",
                    "linux-64",
                    "linux64",
                    output,
                )

            name = "safetensors-mojo-0.7.0-linux64_hash_0.conda"

            def download(command: list[str], **_: object) -> str:
                if command[:3] == ["gh", "api", "--paginate"]:
                    return json.dumps({"tag": "v0.7.0", "assets": [name]})
                destination = Path(command[command.index("--dir") + 1])
                self.make_conda(destination, build="linux64_hash_0")
                return ""

            run_command.side_effect = download
            package = github_release.download_package(
                "owner/repository",
                "v0.7.0",
                "0.7.0",
                "linux-64",
                "linux64",
                output,
            )
            self.assertIsNotNone(package)
            self.assertEqual(package.name, name)

    def test_download_package_rejects_invalid_platform_or_build_prefix(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as raw_directory:
            for platform, build_prefix, expected in (
                ("unknown", "", "unsupported release platform"),
                ("linux-64", "wrong", "unexpected build prefix"),
            ):
                with self.subTest(platform=platform, build_prefix=build_prefix):
                    with self.assertRaisesRegex(
                        github_release.GitHubReleaseError,
                        expected,
                    ):
                        github_release.download_package(
                            "owner/repository",
                            "v0.7.0",
                            "0.7.0",
                            platform,
                            build_prefix,
                            Path(raw_directory),
                        )

    @patch("tools.release.github_release.run_command")
    def test_download_reusable_assets_uses_exact_platform_layout(
        self,
        run_command: MagicMock,
    ) -> None:
        with tempfile.TemporaryDirectory() as raw_directory:
            root = Path(raw_directory)
            linux = self.make_conda(root / "candidates", build="linux64-a_0")
            arm = self.make_conda(root / "candidates", build="linuxaarch64-a_0")
            mac = self.make_conda(root / "candidates", build="osxarm64-a_0")
            manifest = self.write_artifact_list(
                root,
                [
                    ("linux-64", linux, False),
                    ("linux-aarch64", arm, False),
                    ("osx-arm64", mac, False),
                ],
            )

            def download(command: list[str], **_: object) -> str:
                if command[:3] == ["gh", "api", "--paginate"]:
                    return json.dumps(
                        {
                            "tag": "v0.7.0",
                            "assets": [linux.name, mac.name],
                        }
                    )
                name = command[command.index("--pattern") + 1]
                destination = Path(command[command.index("--dir") + 1])
                destination.mkdir(parents=True, exist_ok=True)
                (destination / name).write_bytes(b"downloaded")
                return ""

            run_command.side_effect = download
            snapshot = github_release.download_reusable_assets(
                "owner/repository",
                "v0.7.0",
                manifest,
                root / "existing",
            )

            self.assertTrue(snapshot.exists)
            self.assertTrue((root / "existing" / "linux-64" / linux.name).is_file())
            self.assertFalse(
                (root / "existing" / "linux-aarch64" / arm.name).exists()
            )
            self.assertTrue((root / "existing" / "osx-arm64" / mac.name).is_file())

    @patch("tools.release.github_release.run_command")
    def test_existing_release_uploads_only_missing_canonical_assets(
        self,
        run_command: MagicMock,
    ) -> None:
        with tempfile.TemporaryDirectory() as raw_directory:
            root = Path(raw_directory)
            source = root / "source"
            tag_object, commit = self.initialize_repository(source)
            existing = self.make_conda(root / "packages", build="linux64-a_0")
            missing = self.make_conda(root / "packages", build="osxarm64-a_0")
            manifest = self.write_artifact_list(
                root,
                [
                    ("linux-64", existing, True),
                    ("osx-arm64", missing, False),
                ],
            )
            calls: list[tuple[list[str], str | None]] = []
            run_command.side_effect = self.real_git_fake_gh(tag_object, calls)

            github_release.publish_release(
                "owner/repository",
                "v0.7.0",
                tag_object,
                commit,
                source,
                manifest,
                release_exists=True,
            )

            upload = next(
                command
                for command, _ in calls
                if command[:3] == ["gh", "release", "upload"]
            )
            self.assertIn(str(missing), upload)
            self.assertNotIn(str(existing), upload)
            self.assertNotIn("--clobber", upload)

    @patch("tools.release.github_release.run_command")
    def test_new_release_uses_exact_tag_notes_and_all_assets(
        self,
        run_command: MagicMock,
    ) -> None:
        with tempfile.TemporaryDirectory() as raw_directory:
            root = Path(raw_directory)
            source = root / "source"
            tag_object, commit = self.initialize_repository(source)
            artifact = self.make_conda(root / "packages", build="linux64-a_0")
            manifest = self.write_artifact_list(
                root,
                [("linux-64", artifact, False)],
            )
            calls: list[tuple[list[str], str | None]] = []
            run_command.side_effect = self.real_git_fake_gh(tag_object, calls)

            github_release.publish_release(
                "owner/repository",
                "v0.7.0",
                tag_object,
                commit,
                source,
                manifest,
                release_exists=False,
            )

            create, notes = next(
                (command, stdin)
                for command, stdin in calls
                if command[:3] == ["gh", "release", "create"]
            )
            self.assertIn(str(artifact), create)
            self.assertIn("--verify-tag", create)
            self.assertEqual(notes, "## Summary\n\nRelease notes.\n")

    @patch("tools.release.github_release.run_command")
    def test_publish_rejects_changed_or_invalid_tag_before_mutation(
        self,
        run_command: MagicMock,
    ) -> None:
        with tempfile.TemporaryDirectory() as raw_directory:
            root = Path(raw_directory)
            source = root / "source"
            tag_object, commit = self.initialize_repository(source)
            artifact = self.make_conda(root / "packages", build="linux64-a_0")
            manifest = self.write_artifact_list(
                root,
                [("linux-64", artifact, False)],
            )
            calls: list[tuple[list[str], str | None]] = []
            run_command.side_effect = self.real_git_fake_gh("0" * 40, calls)

            with self.assertRaisesRegex(
                github_release.GitHubReleaseError,
                "changed after release validation",
            ):
                github_release.publish_release(
                    "owner/repository",
                    "v0.7.0",
                    tag_object,
                    commit,
                    source,
                    manifest,
                    release_exists=True,
                )
            self.assertFalse(
                any(
                    command[:2] == ["gh", "release"]
                    for command, _ in calls
                )
            )

    @patch("tools.release.github_release.run_command")
    def test_new_release_rejects_lightweight_tag_and_bad_notes(
        self,
        run_command: MagicMock,
    ) -> None:
        for annotated, notes, expected in (
            (False, "", "annotated tag"),
            (True, "Release notes.", "must start with ## Summary"),
        ):
            with self.subTest(annotated=annotated):
                with tempfile.TemporaryDirectory() as raw_directory:
                    root = Path(raw_directory)
                    source = root / "source"
                    tag_object, commit = self.initialize_repository(
                        source,
                        annotated=annotated,
                        notes=notes,
                    )
                    artifact = self.make_conda(
                        root / "packages",
                        build="linux64-a_0",
                    )
                    manifest = self.write_artifact_list(
                        root,
                        [("linux-64", artifact, False)],
                    )
                    calls: list[tuple[list[str], str | None]] = []
                    run_command.side_effect = self.real_git_fake_gh(
                        tag_object,
                        calls,
                    )

                    with self.assertRaisesRegex(
                        github_release.GitHubReleaseError,
                        expected,
                    ):
                        github_release.publish_release(
                            "owner/repository",
                            "v0.7.0",
                            tag_object,
                            commit,
                            source,
                            manifest,
                            release_exists=False,
                        )
                    self.assertFalse(
                        any(
                            command[:3] == ["gh", "release", "create"]
                            for command, _ in calls
                        )
                    )

    @patch("tools.release.github_release.release_snapshot")
    def test_cli_failure_is_concise_and_does_not_write_outputs(
        self,
        release_snapshot: MagicMock,
    ) -> None:
        release_snapshot.side_effect = github_release.GitHubReleaseError("boom")
        with tempfile.TemporaryDirectory() as raw_directory:
            output = Path(raw_directory) / "github-output"
            stderr = io.StringIO()
            with redirect_stderr(stderr):
                result = github_release.main(
                    [
                        "download-package",
                        "--repository",
                        "owner/repository",
                        "--tag",
                        "v0.7.0",
                        "--version",
                        "0.7.0",
                        "--platform",
                        "linux-64",
                        "--build-prefix",
                        "linux64",
                        "--output-directory",
                        raw_directory,
                        "--github-output",
                        str(output),
                    ]
                )

            self.assertEqual(result, 1)
            self.assertFalse(output.exists())
            self.assertIn("boom", stderr.getvalue())
            self.assertNotIn("Traceback", stderr.getvalue())

    @patch("tools.release.github_release.run_command")
    def test_corrupt_download_is_reported_without_traceback(
        self,
        run_command: MagicMock,
    ) -> None:
        name = "safetensors-mojo-0.7.0-linux64_hash_0.conda"

        def download(command: list[str], **_: object) -> str:
            if command[:3] == ["gh", "api", "--paginate"]:
                return json.dumps({"tag": "v0.7.0", "assets": [name]})
            destination = Path(command[command.index("--dir") + 1])
            destination.mkdir(parents=True, exist_ok=True)
            (destination / name).write_bytes(b"not a conda package")
            return ""

        run_command.side_effect = download
        with tempfile.TemporaryDirectory() as raw_directory:
            output = Path(raw_directory) / "github-output"
            stderr = io.StringIO()
            with redirect_stderr(stderr):
                result = github_release.main(
                    [
                        "download-package",
                        "--repository",
                        "owner/repository",
                        "--tag",
                        "v0.7.0",
                        "--version",
                        "0.7.0",
                        "--platform",
                        "linux-64",
                        "--build-prefix",
                        "linux64",
                        "--output-directory",
                        raw_directory,
                        "--github-output",
                        str(output),
                    ]
                )

            self.assertEqual(result, 1)
            self.assertFalse(output.exists())
            self.assertIn("File is not a zip file", stderr.getvalue())
            self.assertNotIn("Traceback", stderr.getvalue())

    @patch("tools.release.github_release.download_package")
    def test_download_package_cli_writes_stable_outputs(
        self,
        download_package: MagicMock,
    ) -> None:
        download_package.return_value = None
        with tempfile.TemporaryDirectory() as raw_directory:
            output = Path(raw_directory) / "github-output"
            result = github_release.main(
                [
                    "download-package",
                    "--repository",
                    "owner/repository",
                    "--tag",
                    "v0.7.0",
                    "--version",
                    "0.7.0",
                    "--platform",
                    "linux-64",
                    "--build-prefix",
                    "linux64",
                    "--output-directory",
                    raw_directory,
                    "--github-output",
                    str(output),
                ]
            )

            self.assertEqual(result, 0)
            self.assertEqual(
                output.read_text(encoding="utf-8"),
                "exists=false\npath=\n",
            )

    @patch("tools.release.github_release.download_reusable_assets")
    def test_download_reusable_cli_writes_stable_outputs(
        self,
        download_reusable_assets: MagicMock,
    ) -> None:
        download_reusable_assets.return_value = github_release.ReleaseSnapshot(
            True,
            frozenset(),
        )
        with tempfile.TemporaryDirectory() as raw_directory:
            root = Path(raw_directory)
            output = root / "github-output"
            existing = root / "existing"
            result = github_release.main(
                [
                    "download-reusable",
                    "--repository",
                    "owner/repository",
                    "--tag",
                    "v0.7.0",
                    "--artifact-list",
                    str(root / "artifacts.tsv"),
                    "--output-root",
                    str(existing),
                    "--github-output",
                    str(output),
                ]
            )

            self.assertEqual(result, 0)
            self.assertEqual(
                output.read_text(encoding="utf-8"),
                f"release_exists=true\nroot={existing.resolve()}\n",
            )


if __name__ == "__main__":
    unittest.main()
