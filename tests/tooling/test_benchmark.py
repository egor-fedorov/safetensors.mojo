from __future__ import annotations

from pathlib import Path
import sys
import unittest


BENCHMARK_ROOT = Path(__file__).resolve().parents[2] / "benchmarks"
sys.path.insert(0, str(BENCHMARK_ROOT))

import run as benchmark  # noqa: E402


class BenchmarkTests(unittest.TestCase):
    def test_native_output_requires_all_samples_and_checksum(self) -> None:
        output = "sample_ns 11\nsample_ns 17\nchecksum 5\n"
        self.assertEqual(
            benchmark._parse_native_samples(output, "test", 2, 5),
            [11, 17],
        )

        with self.assertRaisesRegex(RuntimeError, "returned 2 samples"):
            benchmark._parse_native_samples(output, "test", 3, 5)
        with self.assertRaisesRegex(RuntimeError, "checksum"):
            benchmark._parse_native_samples(output, "test", 2, 6)

    def test_native_output_rejects_non_positive_durations(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "non-positive duration"):
            benchmark._parse_native_samples(
                "sample_ns 0\nchecksum 1\n",
                "test",
                1,
                1,
            )

    def test_warm_backends_rotate_execution_order(self) -> None:
        calls: list[tuple[str, int, int]] = []

        def runner(name: str) -> benchmark.WarmRunner:
            def run(_path: Path, warmups: int, samples: int) -> list[int]:
                calls.append((name, warmups, samples))
                return [len(calls)] * samples

            return run

        samples = benchmark._warm_backend_samples(
            Path("archive.safetensors"),
            warmups=7,
            samples=4,
            batches=3,
            runners={
                "mojo": runner("mojo"),
                "rust": runner("rust"),
                "python": runner("python"),
            },
        )

        self.assertEqual(
            [name for name, _, _ in calls],
            [
                "mojo",
                "rust",
                "python",
                "rust",
                "python",
                "mojo",
                "python",
                "mojo",
                "rust",
            ],
        )
        self.assertTrue(all(warmups == 7 for _, warmups, _ in calls))
        self.assertEqual(
            {name: len(values) for name, values in samples.items()},
            {
                "mojo": 4,
                "rust": 4,
                "python": 4,
            },
        )

    def test_rust_reference_versions_are_exactly_pinned(self) -> None:
        self.assertEqual(benchmark._cargo_package_version("safetensors"), "0.8.0")
        self.assertEqual(benchmark._cargo_package_version("memmap2"), "0.9.11")


if __name__ == "__main__":
    unittest.main()
