from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

from tools.packaging.smoke_test import (
    CONSUMER_SOURCE,
    MANIFEST,
    compiler_version_from_manifest,
    conda_platform,
    parse_arguments,
    require_native_platform,
    resolve_consumer_source,
    verify_compiler_dependency,
    verify_compiler_version,
)


class SmokeTestPackageArgumentsTests(unittest.TestCase):
    def test_default_consumer_source_is_preserved(self) -> None:
        arguments = parse_arguments(["package.conda"])

        self.assertEqual(arguments.artifact, Path("package.conda"))
        self.assertEqual(arguments.consumer_source, CONSUMER_SOURCE)
        self.assertEqual(arguments.manifest, MANIFEST)

    def test_manifest_can_select_a_historical_release(self) -> None:
        arguments = parse_arguments(
            ["package.conda", "--manifest", "tagged/pixi.toml"]
        )

        self.assertEqual(arguments.manifest, Path("tagged/pixi.toml"))

    def test_consumer_source_can_be_overridden(self) -> None:
        arguments = parse_arguments(
            [
                "package.conda",
                "--consumer-source",
                "tagged/tools/packaging/consumers/package_smoke.mojo",
            ]
        )

        self.assertEqual(
            arguments.consumer_source,
            Path("tagged/tools/packaging/consumers/package_smoke.mojo"),
        )

    def test_consumer_source_resolves_to_an_existing_file(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "package_smoke.mojo"
            source.write_text("def main():\n    pass\n", encoding="utf-8")

            self.assertEqual(resolve_consumer_source(source), source.resolve())

    def test_consumer_source_rejects_missing_paths_and_directories(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with self.assertRaisesRegex(ValueError, "does not exist"):
                resolve_consumer_source(root / "missing.mojo")
            with self.assertRaisesRegex(ValueError, "is not a file"):
                resolve_consumer_source(root)

    def test_native_platform_mapping_covers_release_targets(self) -> None:
        cases = {
            ("linux", "x86_64"): "linux-64",
            ("linux", "AMD64"): "linux-64",
            ("linux", "aarch64"): "linux-aarch64",
            ("linux", "ARM64"): "linux-aarch64",
            ("darwin", "arm64"): "osx-arm64",
            ("darwin", "x86_64"): "osx-64",
            ("win32", "AMD64"): "win-64",
        }
        for (sys_platform, machine), expected in cases.items():
            with self.subTest(sys_platform=sys_platform, machine=machine):
                self.assertEqual(
                    conda_platform(sys_platform, machine),
                    expected,
                )

    def test_native_platform_mapping_rejects_unknown_pairs(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "cannot infer"):
            conda_platform("linux", "riscv64")

    def test_smoke_target_must_match_the_native_runner(self) -> None:
        require_native_platform("osx-arm64", "osx-arm64")
        with self.assertRaisesRegex(RuntimeError, "must run natively"):
            require_native_platform("osx-arm64", "linux-64")


class SmokeTestPackageCompilerTests(unittest.TestCase):
    def test_compiler_version_comes_from_each_release_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            manifest = Path(directory) / "pixi.toml"
            for version in ("1.0.0", "1.1.0"):
                with self.subTest(version=version):
                    manifest.write_text(
                        "[package.run-dependencies]\n"
                        f'mojo-compiler = "=={version}"\n',
                        encoding="utf-8",
                    )
                    self.assertEqual(
                        compiler_version_from_manifest(manifest), version
                    )

    def test_manifest_requires_an_exact_compiler_pin(self) -> None:
        specifications = (
            '"1.1.*"',
            '">=1.0.0,<2.0a0"',
            '"==1.1.0,!=1.1.0"',
            '"==1.1"',
            '""',
            "42",
        )
        with tempfile.TemporaryDirectory() as directory:
            manifest = Path(directory) / "pixi.toml"
            for specification in specifications:
                with self.subTest(specification=specification):
                    manifest.write_text(
                        "[package.run-dependencies]\n"
                        f"mojo-compiler = {specification}\n",
                        encoding="utf-8",
                    )
                    with self.assertRaisesRegex(ValueError, "exact release"):
                        compiler_version_from_manifest(manifest)

    def test_manifest_requires_a_compiler_dependency(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            manifest = Path(directory) / "pixi.toml"
            for contents in ("", "package = 1\n", "[package.run-dependencies]\n"):
                with self.subTest(contents=contents):
                    manifest.write_text(contents, encoding="utf-8")
                    with self.assertRaisesRegex(ValueError, "must define"):
                        compiler_version_from_manifest(manifest)

    def test_malformed_and_missing_manifests_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            manifest = Path(directory) / "pixi.toml"
            with self.assertRaises(OSError):
                compiler_version_from_manifest(manifest)
            manifest.write_text("[broken", encoding="utf-8")
            with self.assertRaises(ValueError):
                compiler_version_from_manifest(manifest)

    def test_artifact_must_pin_the_manifest_compiler(self) -> None:
        for version in ("1.0.0", "1.1.0"):
            with self.subTest(version=version):
                verify_compiler_dependency(
                    {"depends": [f"mojo-compiler =={version}"]}, version
                )

        for dependencies in (
            [],
            ["mojo-compiler ==1.0.0"],
            ["mojo-compiler >=1.0.0,<2.0a0"],
            ["mojo-compiler 1.1.0.*"],
            ["mojo-compiler ==1.1.0", "mojo-compiler ==1.1.0"],
        ):
            with self.subTest(dependencies=dependencies):
                with self.assertRaisesRegex(RuntimeError, "exact compiler"):
                    verify_compiler_dependency({"depends": dependencies}, "1.1.0")

    def test_installed_compiler_must_match_the_selected_manifest(self) -> None:
        for version in ("1.0.0", "1.1.0"):
            with self.subTest(version=version):
                verify_compiler_version({"version": version}, version)
        with self.assertRaisesRegex(RuntimeError, "1.0.0 != 1.1.0"):
            verify_compiler_version({"version": "1.0.0"}, "1.1.0")
        with self.assertRaisesRegex(RuntimeError, "1.1.0 != 1.0.0"):
            verify_compiler_version({"version": "1.1.0"}, "1.0.0")


if __name__ == "__main__":
    unittest.main()
