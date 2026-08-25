from __future__ import annotations

from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
from pathlib import Path
import tempfile
import unittest

from tools.release.validate_release import (
    ReleaseValidationError,
    main,
    validate_manifest_tag,
)


class ValidateReleaseTests(unittest.TestCase):
    def write_manifest(
        self,
        directory: str,
        workspace_version: str,
        package_version: str | None = None,
    ) -> Path:
        manifest_path = Path(directory) / "pixi.toml"
        resolved_package_version = package_version or workspace_version
        manifest_path.write_text(
            "[workspace]\n"
            f'version = "{workspace_version}"\n'
            "\n"
            "[package]\n"
            f'version = "{resolved_package_version}"\n',
            encoding="utf-8",
        )
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
