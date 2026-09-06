from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path
import sys
import unittest


PROJECT_ROOT = Path(__file__).resolve().parents[2]
BENCHMARK_ROOT = PROJECT_ROOT / "benchmarks"
RESULTS_ROOT = BENCHMARK_ROOT / "results"
sys.path.insert(0, str(BENCHMARK_ROOT))

import run as benchmark  # noqa: E402


class BenchmarkTests(unittest.TestCase):
    def test_published_reports_match_raw_samples_and_are_linked(self) -> None:
        expected_labels = {
            1: {
                "mojo_warm",
                "python_warm",
                "mojo_fresh_process",
                "python_fresh_process",
            },
            2: {
                "mojo_warm",
                "rust_warm",
                "python_warm",
                "mojo_fresh_process",
                "rust_fresh_process",
                "python_fresh_process",
            },
        }
        readme = (PROJECT_ROOT / "README.md").read_text(encoding="utf-8")
        paths = sorted(RESULTS_ROOT.glob("*.json"))
        self.assertTrue(paths)

        for path in paths:
            report = json.loads(path.read_text(encoding="utf-8"))
            labels = expected_labels[report["schema_version"]]
            self.assertEqual(set(report["raw_nanoseconds"]), labels)
            self.assertEqual(set(report["summary"]), labels)
            self.assertIn(path.relative_to(PROJECT_ROOT).as_posix(), readme)
            for label, values in report["raw_nanoseconds"].items():
                sample_key = (
                    "fresh_samples"
                    if label.endswith("fresh_process")
                    else "samples"
                )
                self.assertEqual(
                    len(values),
                    report["configuration"][sample_key],
                )
                self.assertEqual(
                    asdict(benchmark._summarize(values)),
                    report["summary"][label],
                )

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
