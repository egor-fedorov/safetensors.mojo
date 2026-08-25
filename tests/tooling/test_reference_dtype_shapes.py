from __future__ import annotations

from pathlib import Path
import unittest

import safetensors
from safetensors._safetensors_rust import deserialize

from tools.fixtures.generate import (
    SAFE_DTYPE_BITS,
    reference_dtype_shape_cases,
    reference_dtype_shapes_fixture,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
MATRIX_PATH = (
    PROJECT_ROOT / "fixtures" / "valid" / "reference_dtype_shapes.safetensors"
)
F6_PATH = PROJECT_ROOT / "fixtures" / "valid" / "reference_f6_shapes.safetensors"


class ReferenceDTypeShapeTests(unittest.TestCase):
    def test_reference_serializer_reproduces_exact_matrix_bytes(self) -> None:
        self.assertEqual(safetensors.__version__, "0.8.0")
        self.assertEqual(MATRIX_PATH.read_bytes(), reference_dtype_shapes_fixture())

    def test_reference_deserializer_observes_every_dtype_and_shape(self) -> None:
        matrix = dict(deserialize(MATRIX_PATH.read_bytes()))
        cases = reference_dtype_shape_cases()
        self.assertEqual(len(matrix), 79)
        self.assertEqual(set(matrix), {case.name for case in cases})

        observed_dtypes: set[str] = set()
        for case in cases:
            with self.subTest(name=case.name):
                tensor = matrix[case.name]
                self.assertEqual(tensor["dtype"], case.wire_dtype)
                self.assertEqual(tensor["shape"], case.logical_shape)
                self.assertEqual(len(tensor["data"]), case.byte_length)
                self.assertEqual(tensor["data"], bytearray(case.byte_length))
                observed_dtypes.add(tensor["dtype"])

        f6 = dict(deserialize(F6_PATH.read_bytes()))
        expected_f6: dict[str, tuple[str, list[int], int]] = {}
        for dtype in ["F6_E2M3", "F6_E3M2"]:
            for label, shape, byte_length in [
                ("vector", [4], 3),
                ("matrix", [2, 2], 3),
                ("zero", [2, 3, 0], 0),
            ]:
                expected_f6[f"{dtype.lower()}_{label}"] = (
                    dtype,
                    shape,
                    byte_length,
                )

        self.assertEqual(set(f6), set(expected_f6))
        for name, (dtype, shape, byte_length) in expected_f6.items():
            with self.subTest(name=name):
                tensor = f6[name]
                self.assertEqual(tensor["dtype"], dtype)
                self.assertEqual(tensor["shape"], shape)
                self.assertEqual(len(tensor["data"]), byte_length)
                self.assertEqual(tensor["data"], bytearray(byte_length))
                observed_dtypes.add(dtype)

        self.assertEqual(observed_dtypes, set(SAFE_DTYPE_BITS))


if __name__ == "__main__":
    unittest.main()
