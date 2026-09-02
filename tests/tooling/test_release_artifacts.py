from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest
import zipfile

from tools.release.artifacts import (
    collect_artifacts,
    parse_platforms,
    stage_artifact,
)


class ReleaseArtifactsTests(unittest.TestCase):
    def make_artifact(
        self,
        directory: Path,
        *,
        version: str = "0.7.0",
        build: str = "build_0",
        payload: bytes = b"payload",
    ) -> Path:
        directory.mkdir(parents=True, exist_ok=True)
        artifact = directory / f"safetensors-mojo-{version}-{build}.conda"
        with zipfile.ZipFile(artifact, "w") as archive:
            archive.writestr(
                "metadata.json",
                json.dumps({"conda_pkg_format_version": 2}),
            )
            archive.writestr("info-test.tar.zst", b"info")
            archive.writestr("pkg-test.tar.zst", payload)
        return artifact

    def add_platform(
        self,
        input_root: Path,
        platform: str,
        build: str,
    ) -> Path:
        return self.make_artifact(
            input_root / f"conda-{platform}",
            build=build,
        )

    def test_stage_requires_one_artifact_for_the_release_version(self) -> None:
        with tempfile.TemporaryDirectory() as raw_directory:
            root = Path(raw_directory)
            input_directory = root / "packages"
            artifact = self.make_artifact(input_directory)

            staged = stage_artifact(
                input_directory,
                root / "staged",
                "0.7.0",
            )
            self.assertEqual(staged.read_bytes(), artifact.read_bytes())

            self.make_artifact(input_directory, build="second_0")
            with self.assertRaisesRegex(ValueError, "exactly one"):
                stage_artifact(input_directory, root / "staged", "0.7.0")

    def test_stage_rejects_a_different_version(self) -> None:
        with tempfile.TemporaryDirectory() as raw_directory:
            root = Path(raw_directory)
            self.make_artifact(root / "packages", version="0.6.0")
            with self.assertRaisesRegex(ValueError, "unexpected release artifact"):
                stage_artifact(root / "packages", root / "staged", "0.7.0")

    def test_collect_requires_the_complete_platform_set(self) -> None:
        with tempfile.TemporaryDirectory() as raw_directory:
            root = Path(raw_directory)
            inputs = root / "inputs"
            platforms = ["linux-64", "linux-aarch64", "osx-arm64"]
            builds = ["linux64_hash_0", "linuxaarch64_hash_0", "osxarm64_hash_0"]
            for platform, build in zip(platforms, builds):
                self.add_platform(inputs, platform, build)

            artifact_list = collect_artifacts(
                inputs,
                root / "collected",
                "0.7.0",
                platforms,
            )
            rows = [
                line.split("\t")
                for line in artifact_list.read_text(encoding="utf-8").splitlines()
            ]
            self.assertEqual([row[0] for row in rows], platforms)
            self.assertEqual([row[3] for row in rows], ["false"] * 3)
            self.assertEqual(len({row[2] for row in rows}), 3)
            self.assertEqual(
                [row[2] for row in rows],
                [f"safetensors-mojo-0.7.0-{build}.conda" for build in builds],
            )
            self.assertTrue(all(Path(row[1]).name == row[2] for row in rows))
            self.assertTrue(all(len(row[4]) == 64 for row in rows))

            missing = root / "missing"
            self.add_platform(missing, "linux-64", "linux64_hash_0")
            with self.assertRaisesRegex(ValueError, "expected one osx-arm64"):
                collect_artifacts(
                    missing,
                    root / "missing-output",
                    "0.7.0",
                    ["linux-64", "osx-arm64"],
                )

    def test_collect_requires_the_platform_build_prefix(self) -> None:
        with tempfile.TemporaryDirectory() as raw_directory:
            root = Path(raw_directory)
            self.add_platform(root / "inputs", "linux-64", "linux64_hash_0")
            self.add_platform(root / "inputs", "osx-arm64", "hash_0")
            with self.assertRaisesRegex(ValueError, "osxarm64_"):
                collect_artifacts(
                    root / "inputs",
                    root / "output",
                    "0.7.0",
                    ["linux-64", "osx-arm64"],
                )

    def test_collect_prefers_an_existing_release_asset(self) -> None:
        with tempfile.TemporaryDirectory() as raw_directory:
            root = Path(raw_directory)
            candidate = self.add_platform(
                root / "inputs",
                "linux-64",
                "linux_0",
            )
            existing = root / "existing" / "linux-64" / candidate.name
            existing.parent.mkdir(parents=True)
            existing.write_bytes(candidate.read_bytes())

            artifact_list = collect_artifacts(
                root / "inputs",
                root / "output",
                "0.7.0",
                ["linux-64"],
                existing_root=root / "existing",
            )
            row = artifact_list.read_text(encoding="utf-8").strip().split("\t")
            selected = Path(row[1])
            self.assertEqual(row[2], candidate.name)
            self.assertEqual(row[3], "true")
            self.assertEqual(selected.read_bytes(), existing.read_bytes())

    def test_collect_rejects_a_changed_existing_release_asset(self) -> None:
        with tempfile.TemporaryDirectory() as raw_directory:
            root = Path(raw_directory)
            candidate = self.add_platform(
                root / "inputs",
                "linux-64",
                "linux_0",
            )
            self.make_artifact(
                root / "existing" / "linux-64",
                build="linux_0",
                payload=b"different bytes",
            )
            with self.assertRaisesRegex(ValueError, "differs"):
                collect_artifacts(
                    root / "inputs",
                    root / "output",
                    "0.7.0",
                    ["linux-64"],
                    existing_root=root / "existing",
                )
            self.assertTrue(candidate.is_file())

    def test_collect_reuses_multi_platform_canonical_assets(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as raw_directory:
            root = Path(raw_directory)
            inputs = root / "inputs"
            linux = self.add_platform(inputs, "linux-64", "linux64_hash_0")
            self.add_platform(inputs, "osx-arm64", "osxarm64_hash_0")
            existing_directory = root / "existing" / "linux-64"
            existing_directory.mkdir(parents=True)
            canonical_asset = existing_directory / linux.name
            canonical_asset.write_bytes(linux.read_bytes())

            artifact_list = collect_artifacts(
                inputs,
                root / "output",
                "0.7.0",
                ["linux-64", "osx-arm64"],
                existing_root=root / "existing",
            )
            rows = [
                line.split("\t")
                for line in artifact_list.read_text(encoding="utf-8").splitlines()
            ]
            linux_row = rows[0]
            selected = Path(linux_row[1])
            self.assertEqual(selected.name, linux.name)
            self.assertEqual(selected.read_bytes(), canonical_asset.read_bytes())
            self.assertEqual(linux_row[3], "true")

    def test_parse_platforms_rejects_invalid_release_plans(self) -> None:
        self.assertEqual(
            parse_platforms('["linux-64","osx-arm64"]'),
            ["linux-64", "osx-arm64"],
        )
        invalid_values = (
            "{}",
            "[]",
            '["linux-64", 1]',
            '["linux-64","linux-64"]',
            '["win-64"]',
        )
        for value in invalid_values:
            with self.subTest(value=value), self.assertRaises(ValueError):
                parse_platforms(value)


if __name__ == "__main__":
    unittest.main()
