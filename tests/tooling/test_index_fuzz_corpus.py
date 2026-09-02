from __future__ import annotations

import argparse
from pathlib import Path
import tempfile
import unittest

from tools.fuzz.generate_index_corpus import (
    CORPUS_MARKER,
    canonical_index,
    main,
    nonnegative_int,
    prepare_output_directory,
    structured_cases,
)


class IndexFuzzCorpusTests(unittest.TestCase):
    def test_small_seeded_corpora_are_reproducible(self) -> None:
        with tempfile.TemporaryDirectory() as raw_directory:
            root = Path(raw_directory)
            seeds = root / "seeds"
            seeds.mkdir()
            (seeds / "seed.safetensors.index.json").write_bytes(
                canonical_index()
            )
            first = root / "first"
            second = root / "second"
            common = [
                "--seeds",
                str(seeds),
                "--mutations",
                "8",
                "--seed",
                "20260825",
                "--no-structured",
            ]

            main([*common, "--output", str(first)])
            main([*common, "--output", str(second)])

            first_files = {
                path.name: path.read_bytes() for path in sorted(first.iterdir())
            }
            second_files = {
                path.name: path.read_bytes() for path in sorted(second.iterdir())
            }
            self.assertEqual(first_files, second_files)
            self.assertEqual((first / "count.txt").read_text(), "8\n")
            self.assertTrue((first / "shard-a.safetensors").is_file())
            self.assertTrue((first / "shard-b.safetensors").is_file())

    def test_structured_cases_cover_hostile_index_boundaries(self) -> None:
        cases = structured_cases()
        combined = b"\n".join(cases)

        self.assertIn(b"9007199254740993", combined)
        self.assertIn(b"18446744073709551615", combined)
        self.assertIn(b"18446744073709551616", combined)
        self.assertIn(b"\\u0077eight_map", combined)
        self.assertIn(b"../shard-a.safetensors", combined)
        self.assertIn(b"shard-a.safetensors", combined)
        self.assertGreaterEqual(max(case.count(b"[") for case in cases), 129)

    def test_unmarked_output_directory_is_never_replaced(self) -> None:
        with tempfile.TemporaryDirectory() as raw_directory:
            output = Path(raw_directory) / "output"
            output.mkdir()
            sentinel = output / "keep.txt"
            sentinel.write_text("keep", encoding="utf-8")

            with self.assertRaisesRegex(SystemExit, "unmarked output directory"):
                prepare_output_directory(output)

            self.assertEqual(sentinel.read_text(encoding="utf-8"), "keep")

    def test_marked_output_removes_only_known_generated_entries(self) -> None:
        with tempfile.TemporaryDirectory() as raw_directory:
            output = Path(raw_directory) / "output"
            prepare_output_directory(output)
            (output / "case000000.safetensors.index.json").write_bytes(b"old")
            (output / "shard-a.safetensors").write_bytes(b"old")
            (output / "shard-b.safetensors").write_bytes(b"old")
            (output / "count.txt").write_text("1\n", encoding="utf-8")

            prepare_output_directory(output)

            self.assertEqual(
                {path.name for path in output.iterdir()},
                {CORPUS_MARKER},
            )

    def test_marked_output_preserves_unexpected_entries(self) -> None:
        with tempfile.TemporaryDirectory() as raw_directory:
            output = Path(raw_directory) / "output"
            prepare_output_directory(output)
            sentinel = output / "keep.txt"
            sentinel.write_text("keep", encoding="utf-8")

            with self.assertRaisesRegex(SystemExit, "unexpected corpus entry"):
                prepare_output_directory(output)

            self.assertEqual(sentinel.read_text(encoding="utf-8"), "keep")

    def test_negative_mutation_count_is_rejected(self) -> None:
        with self.assertRaises(argparse.ArgumentTypeError):
            nonnegative_int("-1")


if __name__ == "__main__":
    unittest.main()
