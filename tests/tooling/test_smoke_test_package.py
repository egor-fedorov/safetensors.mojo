from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

from tools.packaging.smoke_test import (
    CONSUMER_SOURCE,
    conda_platform,
    parse_arguments,
    require_native_platform,
    resolve_consumer_source,
)


class SmokeTestPackageArgumentsTests(unittest.TestCase):
    def test_default_consumer_source_is_preserved(self) -> None:
        arguments = parse_arguments(["package.conda"])

        self.assertEqual(arguments.artifact, Path("package.conda"))
        self.assertEqual(arguments.consumer_source, CONSUMER_SOURCE)

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


if __name__ == "__main__":
    unittest.main()
