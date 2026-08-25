from __future__ import annotations

import hashlib
import json
from pathlib import Path
import struct
import unittest

import numpy as np
import safetensors
from safetensors import safe_open
from safetensors.numpy import load_file


PROJECT_ROOT = Path(__file__).resolve().parents[2]
FIXTURE_PATH = PROJECT_ROOT / "fixtures" / "valid" / "canonical_writer.safetensors"

EXPECTED_SHA256 = "507b1e9502bdb1e1f6b6fc39c40e86670453d660109a21a68c4226c2de246b14"
EXPECTED_HEADER_LENGTH = 480
EXPECTED_PADDING = b" " * 4
EXPECTED_HEADER_CORE = (
    '{"__metadata__":{"author":"safetensors.mojo",'
    '"quote\\"and\\\\slash":"line one\\nline two",'
    '"unicode":"Zoë 😊"},'
    '"scalar_i64":{"dtype":"I64","shape":[],"data_offsets":[0,8]},'
    '"beta_f32":{"dtype":"F32","shape":[2],"data_offsets":[8,16]},'
    '"omega_u16":{"dtype":"U16","shape":[2],"data_offsets":[16,20]},'
    '"empty_i8":{"dtype":"I8","shape":[0],"data_offsets":[20,20]},'
    '"alpha_u8":{"dtype":"U8","shape":[3],"data_offsets":[20,23]},'
    '"zeta_u8":{"dtype":"U8","shape":[2],"data_offsets":[23,25]}}'
).encode("utf-8")
EXPECTED_DATA = bytes.fromhex(
    "d6ffffffffffffff"
    "0000c03f000010c0"
    "3412cdab"
    "0001ff"
    "fa07"
)
EXPECTED_TENSOR_ORDER = [
    "scalar_i64",
    "beta_f32",
    "omega_u16",
    "empty_i8",
    "alpha_u8",
    "zeta_u8",
]


class CanonicalWriterFixtureTests(unittest.TestCase):
    def test_exact_canonical_file_layout(self) -> None:
        contents = FIXTURE_PATH.read_bytes()
        prefix = contents[:8]
        header_length = struct.unpack("<Q", prefix)[0]
        header = contents[8 : 8 + header_length]
        data = contents[8 + header_length :]

        self.assertEqual(prefix, struct.pack("<Q", EXPECTED_HEADER_LENGTH))
        self.assertEqual(header_length, EXPECTED_HEADER_LENGTH)
        self.assertEqual((8 + header_length) % 8, 0)
        self.assertEqual(header, EXPECTED_HEADER_CORE + EXPECTED_PADDING)
        self.assertEqual(data, EXPECTED_DATA)
        self.assertEqual(hashlib.sha256(contents).hexdigest(), EXPECTED_SHA256)

        decoded = json.loads(EXPECTED_HEADER_CORE)
        self.assertEqual(
            list(decoded),
            ["__metadata__", *EXPECTED_TENSOR_ORDER],
        )
        self.assertEqual(
            list(decoded["__metadata__"]),
            ["author", 'quote"and\\slash', "unicode"],
        )

        cursor = 0
        for name in EXPECTED_TENSOR_ORDER:
            tensor = decoded[name]
            self.assertEqual(list(tensor), ["dtype", "shape", "data_offsets"])
            begin, end = tensor["data_offsets"]
            self.assertEqual(begin, cursor)
            cursor = end
        self.assertEqual(cursor, len(data))

    def test_python_reference_loads_canonical_fixture(self) -> None:
        self.assertEqual(safetensors.__version__, "0.8.0")

        expected_metadata = {
            "author": "safetensors.mojo",
            'quote"and\\slash': "line one\nline two",
            "unicode": "Zoë 😊",
        }
        with safe_open(FIXTURE_PATH, framework="numpy") as archive:
            self.assertEqual(set(archive.keys()), set(EXPECTED_TENSOR_ORDER))
            self.assertEqual(archive.metadata(), expected_metadata)

        tensors = load_file(FIXTURE_PATH)
        expected = {
            "scalar_i64": np.array(-42, dtype=np.int64),
            "beta_f32": np.array([1.5, -2.25], dtype=np.float32),
            "omega_u16": np.array([0x1234, 0xABCD], dtype=np.uint16),
            "empty_i8": np.array([], dtype=np.int8),
            "alpha_u8": np.array([0, 1, 255], dtype=np.uint8),
            "zeta_u8": np.array([250, 7], dtype=np.uint8),
        }
        self.assertEqual(set(tensors), set(expected))
        for name, expected_tensor in expected.items():
            self.assertEqual(tensors[name].dtype, expected_tensor.dtype)
            self.assertEqual(tensors[name].shape, expected_tensor.shape)
            np.testing.assert_array_equal(tensors[name], expected_tensor)


if __name__ == "__main__":
    unittest.main()
