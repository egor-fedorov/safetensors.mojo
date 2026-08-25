#!/usr/bin/env python3
"""Python safetensors side of the local open/map benchmark."""

from __future__ import annotations

import sys

from safetensors import safe_open


FIRST_TENSOR = "tensor_000"


def touch_first(path: str) -> int:
    """Open, validate, obtain an F32 tensor, and touch its first element."""
    with safe_open(path, framework="numpy") as archive:
        tensor = archive.get_tensor(FIRST_TENSOR)
        if tensor.dtype.name != "float32" or tensor.shape != (1,):
            raise RuntimeError("the first benchmark tensor must have shape [1]")
        return int(tensor.reshape(-1)[0] != 0.0)


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit("usage: python_worker.py <archive>")
    print(touch_first(sys.argv[1]))


if __name__ == "__main__":
    main()
