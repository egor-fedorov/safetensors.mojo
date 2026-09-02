#!/usr/bin/env python3
"""Generate the deterministic Safetensors conformance fixture corpus."""

from __future__ import annotations

import argparse
import ctypes
import json
import math
import struct
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path


MAX_U64 = (1 << 64) - 1

SAFE_DTYPE_REFERENCE_ORDINAL = {
    "BOOL": 0,
    "F4": 1,
    "F6_E2M3": 2,
    "F6_E3M2": 3,
    "U8": 4,
    "I8": 5,
    "F8_E5M2": 6,
    "F8_E4M3": 7,
    "F8_E8M0": 8,
    "F8_E4M3FNUZ": 9,
    "F8_E5M2FNUZ": 10,
    "I16": 11,
    "U16": 12,
    "F16": 13,
    "BF16": 14,
    "I32": 15,
    "U32": 16,
    "F32": 17,
    "C64": 18,
    "F64": 19,
    "I64": 20,
    "U64": 21,
}

SAFE_DTYPE_BITS = {
    "BOOL": 8,
    "F4": 4,
    "F6_E2M3": 6,
    "F6_E3M2": 6,
    "U8": 8,
    "I8": 8,
    "F8_E5M2": 8,
    "F8_E4M3": 8,
    "F8_E8M0": 8,
    "F8_E4M3FNUZ": 8,
    "F8_E5M2FNUZ": 8,
    "I16": 16,
    "U16": 16,
    "F16": 16,
    "BF16": 16,
    "I32": 32,
    "U32": 32,
    "F32": 32,
    "C64": 64,
    "F64": 64,
    "I64": 64,
    "U64": 64,
}

REFERENCE_SERIALIZER_DTYPES = [
    ("BOOL", "bool"),
    ("F4", "float4_e2m1fn_x2"),
    ("U8", "uint8"),
    ("I8", "int8"),
    ("F8_E5M2", "float8_e5m2"),
    ("F8_E4M3", "float8_e4m3fn"),
    ("F8_E8M0", "float8_e8m0fnu"),
    ("F8_E4M3FNUZ", "float8_e4m3fnuz"),
    ("F8_E5M2FNUZ", "float8_e5m2fnuz"),
    ("I16", "int16"),
    ("U16", "uint16"),
    ("F16", "float16"),
    ("BF16", "bfloat16"),
    ("I32", "int32"),
    ("U32", "uint32"),
    ("F32", "float32"),
    ("C64", "complex64"),
    ("F64", "float64"),
    ("I64", "int64"),
    ("U64", "uint64"),
]


@dataclass(frozen=True)
class ReferenceDTypeShapeCase:
    name: str
    wire_dtype: str
    constructor_dtype: str
    logical_shape: list[int]
    storage_shape: list[int]
    byte_length: int


def reference_dtype_shape_cases() -> list[ReferenceDTypeShapeCase]:
    cases: list[ReferenceDTypeShapeCase] = []
    for wire_dtype, constructor_dtype in REFERENCE_SERIALIZER_DTYPES:
        shapes = [
            ("vector", [4], [4]),
            ("matrix", [2, 2], [2, 2]),
            ("zero", [2, 3, 0], [2, 3, 0]),
        ]
        if wire_dtype == "F4":
            shapes = [
                ("vector", [4], [2]),
                ("matrix", [2, 2], [2, 1]),
                ("zero", [2, 3, 0], [2, 3, 0]),
            ]
        else:
            shapes.append(("scalar", [], []))

        for label, logical_shape, storage_shape in shapes:
            bit_length = math.prod(logical_shape) * SAFE_DTYPE_BITS[wire_dtype]
            if bit_length % 8 != 0:
                raise AssertionError("reference case is not byte-addressable")
            cases.append(
                ReferenceDTypeShapeCase(
                    name=f"{wire_dtype.lower()}_{label}",
                    wire_dtype=wire_dtype,
                    constructor_dtype=constructor_dtype,
                    logical_shape=logical_shape,
                    storage_shape=storage_shape,
                    byte_length=bit_length // 8,
                )
            )
    return cases


def reference_dtype_shapes_fixture() -> bytes:
    from safetensors._safetensors_rust import TensorSpec, serialize

    buffers: list[bytearray] = []
    tensors: dict[str, TensorSpec] = {}
    for case in reference_dtype_shape_cases():
        buffer = bytearray(max(case.byte_length, 1))
        buffers.append(buffer)
        pointer = ctypes.addressof(ctypes.c_char.from_buffer(buffer))
        tensors[case.name] = TensorSpec(
            dtype=case.constructor_dtype,
            shape=case.storage_shape,
            data_ptr=pointer,
            data_len=case.byte_length,
        )

    return serialize(tensors)


def encode_file(header: bytes, data: bytes = b"") -> bytes:
    return struct.pack("<Q", len(header)) + header + data


def compact_json(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")


def encoded_json_file(value: object, data: bytes = b"") -> bytes:
    return encode_file(compact_json(value), data)


def aligned_json_file(
    value: object,
    data: bytes = b"",
    alignment: int = 8,
) -> bytes:
    header = compact_json(value)
    padding = (-(8 + len(header))) % alignment
    return encode_file(header + (b" " * padding), data)


def descriptor(
    dtype: str,
    shape: list[int],
    begin: int,
    end: int,
) -> dict[str, object]:
    return {
        "dtype": dtype,
        "shape": shape,
        "data_offsets": [begin, end],
    }


def canonical_writer_fixture() -> bytes:
    """Build an independent golden for the planned canonical writer layout."""
    metadata_entries = [
        ("unicode", "Zoë 😊"),
        ('quote"and\\slash', "line one\nline two"),
        ("author", "safetensors.mojo"),
    ]
    metadata = {
        key: value
        for key, value in sorted(metadata_entries, key=lambda entry: entry[0])
    }

    # This declaration order is intentionally unrelated to canonical wire order.
    tensors = [
        ("zeta_u8", "U8", [2], bytes([250, 7])),
        ("omega_u16", "U16", [2], struct.pack("<HH", 0x1234, 0xABCD)),
        ("scalar_i64", "I64", [], struct.pack("<q", -42)),
        ("empty_i8", "I8", [0], b""),
        ("alpha_u8", "U8", [3], bytes([0, 1, 255])),
        ("beta_f32", "F32", [2], struct.pack("<ff", 1.5, -2.25)),
    ]
    tensors.sort(
        key=lambda tensor: (
            -SAFE_DTYPE_REFERENCE_ORDINAL[tensor[1]],
            tensor[0],
        )
    )

    header: dict[str, object] = {"__metadata__": metadata}
    data = bytearray()
    for name, dtype, shape, payload in tensors:
        begin = len(data)
        data.extend(payload)
        header[name] = descriptor(dtype, shape, begin, len(data))

    return aligned_json_file(header, bytes(data))


def write_entries(directory: Path, entries: Mapping[str, bytes]) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    expected = {f"{name}.safetensors" for name in entries}
    for old_path in directory.glob("*.safetensors"):
        if old_path.name not in expected:
            old_path.unlink()
    for name, payload in sorted(entries.items()):
        (directory / f"{name}.safetensors").write_bytes(payload)


def write_tree(
    directory: Path,
    entries: Mapping[str, bytes],
    symlinks: Mapping[str, str] | None = None,
) -> None:
    """Write one deterministic nested fixture tree.

    Only files and symbolic links owned by this generator are allowed below the
    sharded fixture root, so stale entries can be removed deterministically.
    """
    links = symlinks or {}
    expected = {Path(name) for name in entries} | {Path(name) for name in links}
    directory.mkdir(parents=True, exist_ok=True)

    for old_path in sorted(directory.rglob("*"), reverse=True):
        relative = old_path.relative_to(directory)
        if old_path.is_symlink() or old_path.is_file():
            if relative not in expected:
                old_path.unlink()
        elif old_path.is_dir() and not any(old_path.iterdir()):
            old_path.rmdir()

    for name, payload in sorted(entries.items()):
        output = directory / name
        output.parent.mkdir(parents=True, exist_ok=True)
        if output.is_symlink():
            output.unlink()
        output.write_bytes(payload)

    for name, target in sorted(links.items()):
        output = directory / name
        output.parent.mkdir(parents=True, exist_ok=True)
        if output.is_symlink() or output.exists():
            output.unlink()
        output.symlink_to(target)


def pretty_json(value: object) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, indent=2).encode("utf-8") + b"\n"
    )


def sharded_tensor_files() -> tuple[bytes, bytes]:
    shard_a = encoded_json_file(
        {"alpha": descriptor("U8", [3], 0, 3)},
        b"\x01\x02\x03",
    )
    shard_b = encoded_json_file(
        {
            "beta": descriptor("I16", [2], 0, 4),
            "empty": descriptor("F32", [0], 4, 4),
        },
        struct.pack("<hh", -2, 300),
    )
    return shard_a, shard_b


def sharded_fixture_tree() -> tuple[
    dict[str, bytes],
    dict[str, str],
    dict[str, str],
]:
    """Return handcrafted sharded files, expected failures, and symlinks."""
    entries: dict[str, bytes] = {}
    malformed: dict[str, str] = {}
    symlinks: dict[str, str] = {}
    shard_a, shard_b = sharded_tensor_files()

    def add_index(
        group: str,
        name: str,
        index: object | bytes,
        shards: Mapping[str, bytes] | None = None,
        expected: str | None = None,
    ) -> None:
        base = f"{group}/{name}"
        payload = index if isinstance(index, bytes) else pretty_json(index)
        entries[f"{base}/model.safetensors.index.json"] = payload
        for filename, contents in (shards or {}).items():
            entries[f"{base}/{filename}"] = contents
        if expected is not None:
            malformed[f"{base}/model.safetensors.index.json"] = expected

    multiple_index = {
        "metadata": {
            "total_size": 7,
            "total_parameters": 5,
            "producer": "safetensors.mojo fixtures",
            "future": {"nested": [True, None, -1.25e2]},
        },
        "weight_map": {
            "empty": "shard-b.safetensors",
            "alpha": "shard-a.safetensors",
            "beta": "shard-b.safetensors",
        },
        "future_root": [False, {"value": 18446744073709551616}],
    }
    add_index(
        "valid",
        "multiple",
        multiple_index,
        {
            "shard-a.safetensors": shard_a,
            "shard-b.safetensors": shard_b,
            # A directory scan would trip over this. Index readers must ignore it.
            "unreferenced-invalid.safetensors": b"not a safetensors file",
        },
    )
    add_index(
        "valid",
        "single",
        {
            "weight_map": {"alpha": "model.safetensors"},
        },
        {"model.safetensors": shard_a},
    )

    # The index path is caller-trusted, so its final symlink may be followed.
    index_symlink_base = "valid/index-symlink"
    index_target_base = "valid/index-symlink-target"
    entries[f"{index_target_base}/actual.json"] = (
        b"\x09\x0A "
        + pretty_json(
            {
                "metadata": {"total_size": 3},
                "weight_map": {"alpha": "shard.safetensors"},
            }
        )
    )
    entries[f"{index_symlink_base}/shard.safetensors"] = shard_a
    symlinks[f"{index_symlink_base}/model.safetensors.index.json"] = (
        "../index-symlink-target/actual.json"
    )

    parser_failures: list[tuple[str, bytes, str]] = [
        ("invalid-json", b'{"weight_map":', "InvalidJson"),
        ("invalid-utf8", b'{"weight_map":{"\xff":"x.safetensors"}}', "InvalidUtf8"),
        (
            "duplicate-root-decoded",
            b'{"weight_map":{"a":"a.safetensors"},'
            b'"\\u0077eight_map":{"a":"a.safetensors"}}',
            "DuplicateKey",
        ),
        (
            "duplicate-metadata-decoded",
            b'{"metadata":{"total_size":3,"total_\\u0073ize":3},'
            b'"weight_map":{"alpha":"shard.safetensors"}}',
            "DuplicateKey",
        ),
        (
            "duplicate-weight-map-decoded",
            b'{"weight_map":{"alpha":"shard.safetensors",'
            b'"\\u0061lpha":"shard.safetensors"}}',
            "DuplicateKey",
        ),
    ]
    for name, index, expected in parser_failures:
        add_index("malformed", name, index, expected=expected)

    schema_failures: list[tuple[str, object, str]] = [
        ("missing-weight-map", {"metadata": {"total_size": 0}}, "MissingField"),
        ("empty-weight-map", {"weight_map": {}}, "InvalidIndex"),
        ("weight-map-not-object", {"weight_map": []}, "InvalidFieldType"),
        (
            "weight-map-value-not-string",
            {"weight_map": {"alpha": 1}},
            "InvalidFieldType",
        ),
        (
            "metadata-not-object",
            {"metadata": [], "weight_map": {"alpha": "shard.safetensors"}},
            "InvalidFieldType",
        ),
        (
            "total-size-negative",
            {
                "metadata": {"total_size": -1},
                "weight_map": {"alpha": "shard.safetensors"},
            },
            "InvalidFieldType",
        ),
        (
            "total-size-fractional",
            {
                "metadata": {"total_size": 1.5},
                "weight_map": {"alpha": "shard.safetensors"},
            },
            "InvalidFieldType",
        ),
        (
            "total-size-string",
            {
                "metadata": {"total_size": "3"},
                "weight_map": {"alpha": "shard.safetensors"},
            },
            "InvalidFieldType",
        ),
        (
            "total-size-boolean",
            {
                "metadata": {"total_size": True},
                "weight_map": {"alpha": "shard.safetensors"},
            },
            "InvalidFieldType",
        ),
        (
            "total-size-null",
            {
                "metadata": {"total_size": None},
                "weight_map": {"alpha": "shard.safetensors"},
            },
            "InvalidFieldType",
        ),
        (
            "total-size-exponent",
            b'{"metadata":{"total_size":1e0},'
            b'"weight_map":{"alpha":"shard.safetensors"}}',
            "InvalidFieldType",
        ),
        (
            "total-size-overflow",
            b'{"metadata":{"total_size":18446744073709551616},'
            b'"weight_map":{"alpha":"shard.safetensors"}}',
            "ValidationOverflow",
        ),
    ]
    for name, index, expected in schema_failures:
        add_index("malformed", name, index, expected=expected)

    deep_value: object = None
    for _ in range(130):
        deep_value = [deep_value]
    add_index(
        "malformed",
        "excessive-nesting",
        {
            "weight_map": {"alpha": "shard.safetensors"},
            "future": deep_value,
        },
        expected="InvalidJson",
    )

    unsafe_filenames = {
        "empty-filename": "",
        "dot": ".",
        "dot-dot": "..",
        "parent-traversal": "../shard.safetensors",
        "nested-path": "nested/shard.safetensors",
        "backslash-path": "nested\\shard.safetensors",
        "absolute-posix": "/tmp/shard.safetensors",
        "windows-drive": "C:\\shard.safetensors",
        "windows-unc": "\\\\server\\share.safetensors",
        "url": "https://example.com/shard.safetensors",
        "colon": "shard:one.safetensors",
        "nul": "shard\0.safetensors",
        "control": "shard\n.safetensors",
        "wrong-suffix": "shard.bin",
    }
    for name, filename in unsafe_filenames.items():
        add_index(
            "security",
            name,
            {"weight_map": {"alpha": filename}},
            expected="PathTraversal",
        )

    add_index(
        "malformed",
        "total-size-mismatch",
        {
            "metadata": {"total_size": 4},
            "weight_map": {"alpha": "shard.safetensors"},
        },
        {"shard.safetensors": shard_a},
        "TotalSizeMismatch",
    )
    add_index(
        "malformed",
        "missing-shard",
        {"weight_map": {"alpha": "missing.safetensors"}},
        expected="IoError",
    )
    add_index(
        "malformed",
        "wrong-route",
        {
            "weight_map": {
                "alpha": "shard-b.safetensors",
                "beta": "shard-a.safetensors",
                "empty": "shard-b.safetensors",
            }
        },
        {"shard-a.safetensors": shard_a, "shard-b.safetensors": shard_b},
        "ShardMismatch",
    )
    add_index(
        "malformed",
        "omitted-tensor",
        {"weight_map": {"beta": "shard.safetensors"}},
        {"shard.safetensors": shard_b},
        "ShardMismatch",
    )
    add_index(
        "malformed",
        "ghost-tensor",
        {
            "weight_map": {
                "alpha": "shard.safetensors",
                "ghost": "shard.safetensors",
            }
        },
        {"shard.safetensors": shard_a},
        "ShardMismatch",
    )
    duplicate_alpha = encoded_json_file(
        {
            "alpha": descriptor("U8", [1], 0, 1),
            "alias": descriptor("U8", [1], 1, 2),
        },
        b"\xff\x00",
    )
    add_index(
        "malformed",
        "duplicate-tensor-across-shards",
        {"weight_map": {"alpha": "shard-a.safetensors"}},
        {
            "shard-a.safetensors": shard_a,
            "shard-b.safetensors": duplicate_alpha,
        },
        "ShardMismatch",
    )
    # Reference both physical files so a global scan must see the duplicate.
    entries[
        "malformed/duplicate-tensor-across-shards/model.safetensors.index.json"
    ] = pretty_json(
        {
            "weight_map": {
                "alpha": "shard-a.safetensors",
                "alias": "shard-b.safetensors",
            }
        }
    )
    add_index(
        "malformed",
        "malformed-shard",
        {"weight_map": {"alpha": "shard.safetensors"}},
        {"shard.safetensors": b"\0" * 7},
        "HeaderTooSmall",
    )

    # A weight-map filename is untrusted and must not be allowed to cross a
    # symlink, while callers may pass the same symlink through the trusted path
    # list API (matching a Hugging Face cache snapshot).
    symlink_base = "security/symlink-shard"
    entries[f"{symlink_base}/target/real.safetensors"] = shard_a
    entries[f"{symlink_base}/model.safetensors.index.json"] = pretty_json(
        {"weight_map": {"alpha": "shard.safetensors"}}
    )
    symlinks[f"{symlink_base}/shard.safetensors"] = "target/real.safetensors"
    malformed[
        f"{symlink_base}/model.safetensors.index.json"
    ] = "PathTraversal"

    return entries, malformed, symlinks


def valid_fixtures() -> dict[str, bytes]:
    valid: dict[str, bytes] = {}
    valid["canonical_writer"] = canonical_writer_fixture()
    valid["empty_archive"] = encoded_json_file({})
    valid["leading_json_whitespace"] = encode_file(b"\x09\x0A{}\x0D\x20")
    valid["json_whitespace_padding"] = encode_file(b"{}\x0D\x20\x09\x0A")
    valid["metadata_only"] = encoded_json_file(
        {"__metadata__": {"producer": "safetensors.mojo", "format": "mojo"}}
    )
    valid["scalar_i64"] = encoded_json_file(
        {"scalar": descriptor("I64", [], 0, 8)},
        struct.pack("<q", -42),
    )
    valid["aligned_scalar_i64"] = aligned_json_file(
        {"scalar": descriptor("I64", [], 0, 8)},
        struct.pack("<q", -42),
    )
    valid["zero_dimension"] = encoded_json_file(
        {"empty": descriptor("F32", [MAX_U64, 0, MAX_U64], 0, 0)}
    )
    valid["multiple_empty_boundaries"] = encoded_json_file(
        {
            "empty_before": descriptor("F32", [0, 9], 0, 0),
            "payload": descriptor("U8", [3], 0, 3),
            "empty_after": descriptor("I16", [4, 0], 3, 3),
        },
        b"\x01\x02\x03",
    )
    valid["reordered_offsets"] = encoded_json_file(
        {
            "second": descriptor("U16", [1], 4, 6),
            "first": descriptor("I8", [4], 0, 4),
        },
        b"\x80\x00\x01\x7f\x34\x12",
    )
    valid["float8_scalars"] = encoded_json_file(
        {
            "e5m2": descriptor("F8_E5M2", [], 0, 1),
            "e4m3": descriptor("F8_E4M3", [], 1, 2),
            "e8m0": descriptor("F8_E8M0", [], 2, 3),
            "e4m3fnuz": descriptor("F8_E4M3FNUZ", [], 3, 4),
            "e5m2fnuz": descriptor("F8_E5M2FNUZ", [], 4, 5),
        },
        bytes([0x3C, 0x38, 0x7F, 0x40, 0x40]),
    )
    valid["unicode"] = encoded_json_file(
        {
            "__metadata__": {"author": "Zoë", "note": "tensor 😊"},
            "wëight😊": descriptor("F16", [2], 0, 4),
        },
        b"\x00\x3c\x00\xc0",
    )
    valid["high_rank"] = encoded_json_file(
        {"rank16": descriptor("F32", [1] * 16, 0, 4)},
        struct.pack("<f", 1.25),
    )
    valid["subbyte"] = encoded_json_file(
        {
            "f4": descriptor("F4", [2], 0, 1),
            "f6": descriptor("F6_E2M3", [4], 1, 4),
        },
        b"\x21\x01\x02\x03",
    )
    f6_header: dict[str, object] = {}
    f6_data = bytearray()
    for dtype in ["F6_E2M3", "F6_E3M2"]:
        for label, shape in [
            ("vector", [4]),
            ("matrix", [2, 2]),
            ("zero", [2, 3, 0]),
        ]:
            byte_length = math.prod(shape) * SAFE_DTYPE_BITS[dtype] // 8
            begin = len(f6_data)
            f6_data.extend(b"\0" * byte_length)
            f6_header[f"{dtype.lower()}_{label}"] = descriptor(
                dtype, shape, begin, len(f6_data)
            )
    valid["reference_f6_shapes"] = aligned_json_file(
        f6_header, bytes(f6_data)
    )
    padded_header = compact_json(
        {"padded": descriptor("BOOL", [1], 0, 1)}
    ) + b" " * 13
    valid["space_padding"] = encode_file(padded_header, b"\x01")
    extended = descriptor("U8", [1], 0, 1)
    extended["future_null"] = None
    extended["future_bool"] = True
    extended["future_number"] = -125.0
    extended["future_array"] = [False, {"nested": "value"}]
    extended["future_object"] = {"key": 0}
    valid["unknown_descriptor_fields"] = encoded_json_file(
        {"extended": extended}, b"\x2a"
    )
    return valid


def malformed_fixtures() -> tuple[dict[str, bytes], dict[str, str]]:
    malformed: dict[str, bytes] = {}
    expected: dict[str, str] = {}

    def add(name: str, payload: bytes, kind: str) -> None:
        malformed[name] = payload
        expected[name] = kind

    desc = b'{"dtype":"U8","shape":[1],"data_offsets":[0,1]}'
    add("header_too_small", b"\x00" * 7, "HeaderTooSmall")
    add(
        "header_too_large",
        struct.pack("<Q", 100_000_001),
        "HeaderTooLarge",
    )
    add("truncated_header", struct.pack("<Q", 20) + b"{}", "InvalidHeaderLength")
    add("invalid_header_start", encode_file(b"[]"), "InvalidHeaderStart")
    add("invalid_utf8", encode_file(b'{"\xff":{}}'), "InvalidUtf8")
    add("invalid_json", encode_file(b'{"tensor":'), "InvalidJson")
    add(
        "trailing_non_json_whitespace",
        encode_file(b"{}\x0B"),
        "InvalidHeaderPadding",
    )
    add("trailing_second_value", encode_file(b"{} {}"), "InvalidHeaderPadding")
    add(
        "duplicate_top_level",
        encode_file(b'{"a":' + desc + b',"a":' + desc + b"}"),
        "DuplicateKey",
    )
    add(
        "duplicate_top_level_decoded",
        encode_file(b'{"a":' + desc + b',"\\u0061":' + desc + b"}"),
        "DuplicateKey",
    )
    add(
        "duplicate_metadata_decoded",
        encode_file(b'{"__metadata__":{"a":"x","\\u0061":"y"}}'),
        "DuplicateKey",
    )
    add(
        "duplicate_descriptor_decoded",
        encode_file(
            b'{"a":{"dtype":"U8","\\u0064type":"U8",'
            b'"shape":[1],"data_offsets":[0,1]}}'
        ),
        "DuplicateKey",
    )
    add(
        "metadata_value_not_string",
        encode_file(b'{"__metadata__":{"version":1}}'),
        "InvalidFieldType",
    )
    add(
        "metadata_is_not_object",
        encode_file(b'{"__metadata__":"value"}'),
        "InvalidMetadata",
    )
    add(
        "missing_dtype",
        encoded_json_file({"a": {"shape": [1], "data_offsets": [0, 1]}}, b"\0"),
        "MissingField",
    )
    add(
        "missing_shape",
        encoded_json_file({"a": {"dtype": "U8", "data_offsets": [0, 1]}}, b"\0"),
        "MissingField",
    )
    add(
        "missing_offsets",
        encoded_json_file({"a": {"dtype": "U8", "shape": [1]}}, b"\0"),
        "MissingField",
    )
    add(
        "wrong_dtype_type",
        encode_file(
            b'{"a":{"dtype":1,"shape":[1],"data_offsets":[0,1]}}',
            b"\0",
        ),
        "InvalidFieldType",
    )
    for name, number in (
        ("negative_integer", b"-1"),
        ("fractional_integer", b"1.5"),
        ("exponent_integer", b"1e0"),
        ("leading_zero_integer", b"01"),
    ):
        add(
            name,
            encode_file(
                b'{"a":{"dtype":"U8","shape":['
                + number
                + b'],"data_offsets":[0,0]}}'
            ),
            "InvalidShape",
        )
    add(
        "overflowing_integer",
        encode_file(
            b'{"a":{"dtype":"U8","shape":['
            b"18446744073709551616"
            b'],"data_offsets":[0,0]}}'
        ),
        "ValidationOverflow",
    )
    add(
        "unknown_dtype",
        encoded_json_file({"a": descriptor("FLOAT32", [1], 0, 4)}, b"\0" * 4),
        "UnsupportedDType",
    )
    add(
        "shape_product_overflow",
        encoded_json_file({"a": descriptor("U8", [MAX_U64, 2], 0, 0)}),
        "ValidationOverflow",
    )
    add(
        "bit_length_overflow",
        encoded_json_file({"a": descriptor("U64", [MAX_U64], 0, 0)}),
        "ValidationOverflow",
    )
    add(
        "subbyte_not_byte_aligned",
        encoded_json_file({"a": descriptor("F4", [1], 0, 0)}),
        "MisalignedSlice",
    )
    add(
        "begin_after_end",
        encoded_json_file({"a": descriptor("U8", [1], 1, 0)}, b"\0"),
        "InvalidOffsets",
    )
    add(
        "offset_outside_data",
        encoded_json_file({"a": descriptor("U8", [2], 0, 2)}, b"\0"),
        "InvalidOffsets",
    )
    add(
        "wrong_tensor_size",
        encoded_json_file({"a": descriptor("U16", [1], 0, 1)}, b"\0"),
        "InvalidTensorSize",
    )
    add(
        "overlapping_ranges",
        encoded_json_file(
            {
                "a": descriptor("U8", [2], 0, 2),
                "b": descriptor("U8", [2], 1, 3),
            },
            b"\0" * 3,
        ),
        "IncompleteDataCoverage",
    )
    add(
        "gap_between_ranges",
        encoded_json_file(
            {
                "a": descriptor("U8", [1], 0, 1),
                "b": descriptor("U8", [1], 2, 3),
            },
            b"\0" * 3,
        ),
        "IncompleteDataCoverage",
    )
    add(
        "trailing_unindexed_data",
        encoded_json_file({"a": descriptor("U8", [1], 0, 1)}, b"\0\0"),
        "IncompleteDataCoverage",
    )
    add("data_without_tensors", encoded_json_file({}, b"\0"), "IncompleteDataCoverage")
    add(
        "reserved_name_conflict",
        encoded_json_file({"__metadata__": descriptor("U8", [1], 0, 1)}, b"\0"),
        "InvalidFieldType",
    )
    return malformed, expected


def generate_reference_fixtures(valid_dir: Path) -> None:
    try:
        import numpy as np
        from safetensors.numpy import save_file
    except ImportError as error:
        raise SystemExit(
            "Fixture generation requires the locked development dependencies "
            "numpy and safetensors."
        ) from error

    output = valid_dir / "reference_f32.safetensors"
    save_file(
        {"weights": np.array([[1.0, -2.0], [3.5, 4.25]], dtype=np.float32)},
        output,
        metadata={"producer": "Python safetensors reference"},
    )
    (valid_dir / "reference_dtype_shapes.safetensors").write_bytes(
        reference_dtype_shapes_fixture()
    )


def reference_sharded_fixture_tree() -> dict[str, bytes]:
    """Generate a sharded index through Hugging Face's public split helper."""
    try:
        import numpy as np
        from huggingface_hub import split_state_dict_into_shards_factory
        from safetensors.numpy import save
    except ImportError as error:
        raise SystemExit(
            "Sharded fixture generation requires the locked development "
            "dependencies huggingface_hub, numpy, and safetensors."
        ) from error

    tensors = {
        "alpha": np.array([0, 1, 2, 255], dtype=np.uint8),
        "beta": np.array([1.5, -2.25], dtype=np.float32),
        "gamma": np.array([0x1234, 0xABCD], dtype=np.uint16),
        "empty": np.array([], dtype=np.int8),
    }
    split = split_state_dict_into_shards_factory(
        tensors,
        get_storage_size=lambda tensor: int(tensor.nbytes),
        filename_pattern="model{suffix}.safetensors",
        max_shard_size=8,
    )
    if not split.is_sharded:
        raise AssertionError("reference fixture was expected to use multiple shards")

    entries: dict[str, bytes] = {}
    for filename, names in sorted(split.filename_to_tensors.items()):
        shard = {name: tensors[name] for name in names}
        entries[f"valid/reference/{filename}"] = save(
            shard,
            metadata={"format": "numpy"},
        )

    index = {
        "metadata": split.metadata,
        "weight_map": split.tensor_to_filename,
    }
    entries["valid/reference/model.safetensors.index.json"] = pretty_json(index)
    return entries


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path(__file__).resolve().parents[2] / "fixtures",
    )
    args = parser.parse_args()

    root = args.output_root.resolve()
    valid_dir = root / "valid"
    malformed_dir = root / "malformed"
    write_entries(valid_dir, valid_fixtures())
    malformed, expected = malformed_fixtures()
    write_entries(malformed_dir, malformed)
    generate_reference_fixtures(valid_dir)

    sharded_entries, sharded_failures, sharded_symlinks = sharded_fixture_tree()
    sharded_entries.update(reference_sharded_fixture_tree())
    sharded_dir = root / "sharded"
    write_tree(sharded_dir, sharded_entries, sharded_symlinks)

    manifest = {
        "valid": sorted(path.name for path in valid_dir.glob("*.safetensors")),
        "malformed": {
            f"{name}.safetensors": expected[name] for name in sorted(expected)
        },
        "sharded": {
            "valid": sorted(
                str(path.relative_to(sharded_dir))
                for path in (sharded_dir / "valid").rglob("*.index.json")
            ),
            "malformed": {
                name: sharded_failures[name] for name in sorted(sharded_failures)
            },
        },
    }
    (root / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
