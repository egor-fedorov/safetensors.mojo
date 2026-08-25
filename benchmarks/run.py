#!/usr/bin/env python3
"""Run comparable warm and fresh-process Safetensors open benchmarks."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import platform
import statistics
import subprocess
import sys
import time
import tomllib
from typing import Any

import numpy as np
import safetensors
from safetensors import safe_open

from generate_archive import (
    DEFAULT_OUTPUT,
    FIRST_TENSOR,
    PAYLOAD_BYTES,
    TENSOR_COUNT,
)
from python_worker import touch_first


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BINARY = PROJECT_ROOT / ".pixi" / "benchmarks" / "map-first"
DEFAULT_REPORT = PROJECT_ROOT / ".pixi" / "benchmarks" / "latest.json"


@dataclass(frozen=True)
class Summary:
    samples: int
    median_ms: float
    p95_ms: float
    minimum_ms: float
    maximum_ms: float


def _percentile(values: list[int], fraction: float) -> float:
    ordered = sorted(values)
    position = fraction * (len(ordered) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return float(ordered[lower])
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _summarize(values: list[int]) -> Summary:
    if not values:
        raise ValueError("at least one timing sample is required")
    return Summary(
        samples=len(values),
        median_ms=statistics.median(values) / 1_000_000,
        p95_ms=_percentile(values, 0.95) / 1_000_000,
        minimum_ms=min(values) / 1_000_000,
        maximum_ms=max(values) / 1_000_000,
    )


def _run_checked(command: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        command,
        cwd=PROJECT_ROOT,
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )


def _validate_archive(path: Path) -> None:
    expected_size_floor = PAYLOAD_BYTES
    if path.stat().st_size <= expected_size_floor:
        raise SystemExit("benchmark archive is missing its header or payload")

    with safe_open(path, framework="numpy") as archive:
        names = list(archive.keys())
        if len(names) != TENSOR_COUNT:
            raise SystemExit(
                f"benchmark archive contains {len(names)} tensors, expected "
                f"{TENSOR_COUNT}"
            )
        if FIRST_TENSOR not in names:
            raise SystemExit(f"benchmark archive is missing {FIRST_TENSOR}")
        element_count = 0
        for name in names:
            tensor_slice = archive.get_slice(name)
            shape = tensor_slice.get_shape()
            if tensor_slice.get_dtype() != "F32":
                raise SystemExit(f"benchmark tensor {name} is not F32")
            if len(shape) != 1 or shape[0] <= 0:
                raise SystemExit(
                    f"benchmark tensor {name} must be a non-empty vector"
                )
            element_count += shape[0]
        if element_count * 4 != PAYLOAD_BYTES:
            raise SystemExit("benchmark tensor shapes do not cover the payload")
        if archive.get_slice(FIRST_TENSOR).get_shape() != [1]:
            raise SystemExit("first benchmark tensor must have shape [1]")
        first = archive.get_tensor(FIRST_TENSOR)
        if first.dtype != np.dtype(np.float32):
            raise SystemExit("first benchmark tensor is not F32")
        if int(first.reshape(-1)[0] != 0.0) != 1:
            raise SystemExit("first benchmark tensor was not touched as expected")


def _python_warm_samples(path: Path, warmups: int, samples: int) -> list[int]:
    checksum = 0
    for _ in range(warmups):
        checksum += touch_first(str(path))

    durations: list[int] = []
    for _ in range(samples):
        started = time.perf_counter_ns()
        checksum += touch_first(str(path))
        durations.append(time.perf_counter_ns() - started)
    if checksum != warmups + samples:
        raise RuntimeError("Python benchmark checksum did not consume every value")
    return durations


def _mojo_warm_samples(
    binary: Path,
    path: Path,
    warmups: int,
    samples: int,
) -> list[int]:
    completed = _run_checked(
        [str(binary), str(path), str(warmups), str(samples)]
    )
    durations: list[int] = []
    checksum: int | None = None
    for line in completed.stdout.splitlines():
        fields = line.split()
        if len(fields) != 2:
            continue
        if fields[0] == "sample_ns":
            durations.append(int(fields[1]))
        elif fields[0] == "checksum":
            checksum = int(fields[1])
    if len(durations) != samples:
        raise RuntimeError(
            f"Mojo worker returned {len(durations)} samples, expected {samples}"
        )
    if checksum != warmups + samples:
        raise RuntimeError("Mojo benchmark checksum did not consume every value")
    return durations


def _warm_pair_samples(
    binary: Path,
    path: Path,
    warmups: int,
    samples: int,
    batches: int,
) -> tuple[list[int], list[int]]:
    base, remainder = divmod(samples, batches)
    mojo: list[int] = []
    python: list[int] = []
    for index in range(batches):
        batch_samples = base + int(index < remainder)
        if index % 2 == 0:
            mojo.extend(
                _mojo_warm_samples(binary, path, warmups, batch_samples)
            )
            python.extend(_python_warm_samples(path, warmups, batch_samples))
        else:
            python.extend(_python_warm_samples(path, warmups, batch_samples))
            mojo.extend(
                _mojo_warm_samples(binary, path, warmups, batch_samples)
            )
    return mojo, python


def _time_process(command: list[str]) -> int:
    started = time.perf_counter_ns()
    _run_checked(command)
    return time.perf_counter_ns() - started


def _fresh_pair_samples(
    mojo_command: list[str],
    python_command: list[str],
    warmups: int,
    samples: int,
) -> tuple[list[int], list[int]]:
    for _ in range(warmups):
        _run_checked(mojo_command)
        _run_checked(python_command)

    mojo: list[int] = []
    python: list[int] = []
    for index in range(samples):
        if index % 2 == 0:
            mojo.append(_time_process(mojo_command))
            python.append(_time_process(python_command))
        else:
            python.append(_time_process(python_command))
            mojo.append(_time_process(mojo_command))
    return mojo, python


def _tool_output(command: list[str]) -> str:
    return _run_checked(command).stdout.strip()


def _cpu_model() -> str:
    try:
        for line in Path("/proc/cpuinfo").read_text(encoding="utf-8").splitlines():
            if line.startswith("model name"):
                return line.partition(":")[2].strip()
    except OSError:
        pass
    return platform.processor() or "unknown"


def _environment() -> dict[str, Any]:
    try:
        commit = _tool_output(["git", "rev-parse", "--short", "HEAD"])
        git_dirty = bool(_tool_output(["git", "status", "--porcelain"]))
    except (OSError, subprocess.CalledProcessError):
        commit = "unknown"
        git_dirty = True
    with (PROJECT_ROOT / "pixi.toml").open("rb") as project_file:
        project = tomllib.load(project_file)
    return {
        "git_commit": commit,
        "git_dirty": git_dirty,
        "git_revision": f"{commit}-dirty" if git_dirty else commit,
        "safetensors_mojo": project["workspace"]["version"],
        "system": platform.system(),
        "kernel": platform.release(),
        "machine": platform.machine(),
        "cpu": _cpu_model(),
        "python": platform.python_version(),
        "python_executable": sys.executable,
        "numpy": np.__version__,
        "python_safetensors": safetensors.__version__,
        "mojo": _tool_output(["mojo", "--version"]),
    }


def _print_summary(label: str, summary: Summary) -> None:
    print(
        f"{label:<34} median {summary.median_ms:9.3f} ms  "
        f"p95 {summary.p95_ms:9.3f} ms"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--binary", type=Path, default=DEFAULT_BINARY)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--warmups", type=int, default=50)
    parser.add_argument("--samples", type=int, default=500)
    parser.add_argument("--warm-batches", type=int, default=4)
    parser.add_argument("--fresh-warmups", type=int, default=3)
    parser.add_argument("--fresh-samples", type=int, default=30)
    arguments = parser.parse_args()

    if arguments.warmups < 0 or arguments.samples <= 0:
        raise SystemExit("warmups must be non-negative and samples must be positive")
    if arguments.warm_batches <= 0 or arguments.warm_batches > arguments.samples:
        raise SystemExit("warm batches must be between one and the sample count")
    if arguments.fresh_warmups < 0 or arguments.fresh_samples <= 0:
        raise SystemExit(
            "fresh warmups must be non-negative and fresh samples must be positive"
        )

    archive = arguments.archive.resolve()
    binary = arguments.binary.resolve()
    if not archive.is_file():
        raise SystemExit(f"missing benchmark archive: {archive}")
    if not binary.is_file() or not os.access(binary, os.X_OK):
        raise SystemExit(f"missing benchmark executable: {binary}")
    _validate_archive(archive)

    mojo_warm, python_warm = _warm_pair_samples(
        binary,
        archive,
        arguments.warmups,
        arguments.samples,
        arguments.warm_batches,
    )
    mojo_fresh, python_fresh = _fresh_pair_samples(
        [str(binary), str(archive)],
        [
            sys.executable,
            str(Path(__file__).with_name("python_worker.py")),
            str(archive),
        ],
        arguments.fresh_warmups,
        arguments.fresh_samples,
    )

    summaries = {
        "mojo_warm": _summarize(mojo_warm),
        "python_warm": _summarize(python_warm),
        "mojo_fresh_process": _summarize(mojo_fresh),
        "python_fresh_process": _summarize(python_fresh),
    }
    archive_stat = archive.stat()
    report = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "operation": (
            "open/map and validate the complete header, obtain the one-element "
            "F32 tensor_000, touch its value, and close/unmap"
        ),
        "archive": {
            "path": str(archive),
            "tensors": TENSOR_COUNT,
            "dtype": "F32",
            "payload_bytes": PAYLOAD_BYTES,
            "logical_file_bytes": archive_stat.st_size,
            "allocated_file_bytes": archive_stat.st_blocks * 512,
            "sparse": archive_stat.st_blocks * 512 < archive_stat.st_size,
        },
        "configuration": {
            "warmups_per_batch": arguments.warmups,
            "warm_batches": arguments.warm_batches,
            "samples": arguments.samples,
            "fresh_warmups": arguments.fresh_warmups,
            "fresh_samples": arguments.fresh_samples,
            "page_cache": "warm for the header and first payload page",
        },
        "environment": _environment(),
        "summary": {key: asdict(value) for key, value in summaries.items()},
        "raw_nanoseconds": {
            "mojo_warm": mojo_warm,
            "python_warm": python_warm,
            "mojo_fresh_process": mojo_fresh,
            "python_fresh_process": python_fresh,
        },
    }
    report_path = arguments.report.resolve()
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    print("Safetensors open/map benchmark")
    print(f"archive: {archive_stat.st_size:,} logical bytes, {TENSOR_COUNT} tensors")
    print(f"CPU: {_cpu_model()}")
    _print_summary("Mojo warm map + typed touch", summaries["mojo_warm"])
    _print_summary("Python warm open + typed touch", summaries["python_warm"])
    _print_summary("Mojo fresh process", summaries["mojo_fresh_process"])
    _print_summary("Python fresh process", summaries["python_fresh_process"])
    print(f"report: {report_path}")


if __name__ == "__main__":
    main()
