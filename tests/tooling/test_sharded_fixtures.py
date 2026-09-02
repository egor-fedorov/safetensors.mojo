from __future__ import annotations

import json
from pathlib import Path
import unittest

import huggingface_hub
import numpy as np
from safetensors import safe_open

from tools.fixtures.generate import (
    reference_sharded_fixture_tree,
    sharded_fixture_tree,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SHARDED_ROOT = PROJECT_ROOT / "fixtures" / "sharded"


class ShardedFixtureTests(unittest.TestCase):
    def test_handcrafted_tree_is_reproducible(self) -> None:
        entries, failures, symlinks = sharded_fixture_tree()

        for relative, expected in entries.items():
            with self.subTest(path=relative):
                self.assertEqual((SHARDED_ROOT / relative).read_bytes(), expected)

        for relative, target in symlinks.items():
            with self.subTest(symlink=relative):
                path = SHARDED_ROOT / relative
                self.assertTrue(path.is_symlink())
                self.assertEqual(path.readlink(), Path(target))

        manifest = json.loads((PROJECT_ROOT / "fixtures/manifest.json").read_text())
        self.assertEqual(manifest["sharded"]["malformed"], failures)

    def test_reference_splitter_reproduces_exact_tree(self) -> None:
        self.assertEqual(huggingface_hub.__version__, "1.29.0")
        entries = reference_sharded_fixture_tree()
        committed = {
            str(path.relative_to(SHARDED_ROOT)): path.read_bytes()
            for path in (SHARDED_ROOT / "valid" / "reference").iterdir()
            if path.is_file()
        }
        self.assertEqual(committed, entries)

    def test_reference_index_routes_every_tensor(self) -> None:
        directory = SHARDED_ROOT / "valid" / "reference"
        index = json.loads(
            (directory / "model.safetensors.index.json").read_text(
                encoding="utf-8"
            )
        )
        weight_map = index["weight_map"]
        self.assertEqual(set(weight_map), {"alpha", "beta", "gamma", "empty"})
        self.assertEqual(len(set(weight_map.values())), 3)
        self.assertEqual(index["metadata"]["total_size"], 16)

        expected = {
            "alpha": np.array([0, 1, 2, 255], dtype=np.uint8),
            "beta": np.array([1.5, -2.25], dtype=np.float32),
            "gamma": np.array([0x1234, 0xABCD], dtype=np.uint16),
            "empty": np.array([], dtype=np.int8),
        }
        observed: set[str] = set()
        for filename in sorted(set(weight_map.values())):
            with safe_open(directory / filename, framework="numpy") as shard:
                for name in shard.keys():
                    self.assertEqual(weight_map[name], filename)
                    np.testing.assert_array_equal(
                        shard.get_tensor(name), expected[name]
                    )
                    observed.add(name)
        self.assertEqual(observed, set(expected))

    def test_manual_index_has_exact_global_coverage(self) -> None:
        directory = SHARDED_ROOT / "valid" / "multiple"
        index = json.loads(
            (directory / "model.safetensors.index.json").read_text(
                encoding="utf-8"
            )
        )
        weight_map = index["weight_map"]
        observed: set[str] = set()
        total_size = 0
        for filename in sorted(set(weight_map.values())):
            with safe_open(directory / filename, framework="numpy") as shard:
                for name in shard.keys():
                    self.assertEqual(weight_map[name], filename)
                    tensor = shard.get_tensor(name)
                    total_size += tensor.nbytes
                    observed.add(name)

        self.assertEqual(observed, set(weight_map))
        self.assertEqual(total_size, index["metadata"]["total_size"])
        self.assertTrue((directory / "unreferenced-invalid.safetensors").exists())

    def test_symlink_fixtures_encode_the_trust_boundary(self) -> None:
        trusted_index = (
            SHARDED_ROOT
            / "valid"
            / "index-symlink"
            / "model.safetensors.index.json"
        )
        self.assertTrue(trusted_index.is_symlink())
        self.assertEqual(
            trusted_index.readlink(),
            Path("../index-symlink-target/actual.json"),
        )
        self.assertFalse(
            (SHARDED_ROOT / "valid" / "index-symlink-target" / "shard.safetensors").exists()
        )

        cache_shard = (
            SHARDED_ROOT / "security" / "symlink-shard" / "shard.safetensors"
        )
        self.assertTrue(cache_shard.is_symlink())
        self.assertEqual(cache_shard.readlink(), Path("target/real.safetensors"))
        with safe_open(cache_shard, framework="numpy") as archive:
            np.testing.assert_array_equal(
                archive.get_tensor("alpha"),
                np.array([1, 2, 3], dtype=np.uint8),
            )


if __name__ == "__main__":
    unittest.main()
