#!/usr/bin/env python3
"""Generate the sparse archive used by the local open/map benchmark."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import struct
import tempfile


PROJECT_ROOT = Path(__file__).resolve().parents[1]
BENCHMARK_ROOT = PROJECT_ROOT / ".pixi" / "benchmarks"
DEFAULT_OUTPUT = BENCHMARK_ROOT / "f32-193-967mib.safetensors"
TENSOR_COUNT = 193
PAYLOAD_BYTES = 967 * 1024 * 1024
BYTES_PER_F32 = 4
FIRST_TENSOR = "tensor_000"


def _compact_json(value: object) -> bytes:
    return json.dumps(value, separators=(",", ":")).encode("utf-8")


def _header_and_data_length() -> tuple[bytes, int]:
    if PAYLOAD_BYTES % BYTES_PER_F32 != 0:
        raise AssertionError("the payload must contain complete F32 values")

    total_elements = PAYLOAD_BYTES // BYTES_PER_F32
    remaining_tensors = TENSOR_COUNT - 1
    elements_per_tensor, remainder = divmod(
        total_elements - 1, remaining_tensors
    )
    header: dict[str, object] = {}
    cursor = 0
    for index in range(TENSOR_COUNT):
        if index == 0:
            element_count = 1
        else:
            element_count = elements_per_tensor + int(index - 1 < remainder)
        byte_length = element_count * BYTES_PER_F32
        end = cursor + byte_length
        header[f"tensor_{index:03d}"] = {
            "dtype": "F32",
            "shape": [element_count],
            "data_offsets": [cursor, end],
        }
        cursor = end

    if cursor != PAYLOAD_BYTES:
        raise AssertionError("tensor ranges must cover the complete payload")

    encoded = _compact_json(header)
    padding = (-(8 + len(encoded))) % 8
    return encoded + (b" " * padding), cursor


def _require_safe_output(path: Path) -> Path:
    root = BENCHMARK_ROOT.resolve()
    output = path.resolve()
    if not output.is_relative_to(root):
        raise SystemExit(f"output must remain below {root}")
    if output == root:
        raise SystemExit("output must name a file below the benchmark directory")
    return output


def generate(output: Path = DEFAULT_OUTPUT) -> Path:
    """Atomically create a deterministic sparse archive at *output*."""
    destination = _require_safe_output(output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    header, data_length = _header_and_data_length()
    data_start = 8 + len(header)
    expected_size = data_start + data_length

    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w+b",
            dir=destination.parent,
            prefix=f".{destination.name}.",
            delete=False,
        ) as archive:
            temporary_path = Path(archive.name)
            archive.write(struct.pack("<Q", len(header)))
            archive.write(header)
            archive.write(struct.pack("<f", 1.0))
            archive.seek(expected_size - 1)
            archive.write(b"\0")
            archive.flush()

        if temporary_path.stat().st_size != expected_size:
            raise RuntimeError("generated archive has an unexpected file size")
        os.replace(temporary_path, destination)
        temporary_path = None
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)

    return destination


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    arguments = parser.parse_args()

    output = generate(arguments.output)
    allocated_bytes = output.stat().st_blocks * 512
    print(f"archive: {output}")
    print(f"tensors: {TENSOR_COUNT}")
    print(f"payload bytes: {PAYLOAD_BYTES}")
    print(f"logical file bytes: {output.stat().st_size}")
    print(f"allocated file bytes: {allocated_bytes}")


if __name__ == "__main__":
    main()
