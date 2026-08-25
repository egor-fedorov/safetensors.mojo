from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

from tools.packaging.smoke_test import (
    CONSUMER_SOURCE,
    parse_arguments,
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


if __name__ == "__main__":
    unittest.main()
