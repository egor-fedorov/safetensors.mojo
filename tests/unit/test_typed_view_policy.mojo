"""Native typed-view dtype and layout policy tests."""

from std.sys import is_little_endian
from std.testing import TestSuite, assert_equal, assert_true

from safetensors import SafeDType, SafeTensorErrorKind
from safetensors.io._typed_view import (
    _requested_safe_dtype,
    _require_exact_typed_dtype,
    _supports_native_typed_view,
    _validate_typed_layout,
)


def _assert_exact_mapping[
    dtype: DType,
](expected: SafeDType) raises:
    assert_equal(_requested_safe_dtype[dtype](), expected)
    assert_true(_supports_native_typed_view(expected))
    _require_exact_typed_dtype[dtype](expected)


def _assert_requested_dtype_error[
    dtype: DType,
](expected: SafeTensorErrorKind) raises:
    var raised = False
    try:
        _ = _requested_safe_dtype[dtype]()
    except error:
        raised = True
        assert_equal(error.kind, expected)
    assert_true(raised)


def _assert_exact_dtype_error[
    dtype: DType,
](actual: SafeDType, expected: SafeTensorErrorKind) raises:
    var raised = False
    try:
        _require_exact_typed_dtype[dtype](actual)
    except error:
        raised = True
        assert_equal(error.kind, expected)
    assert_true(raised)


def _assert_layout_error[
    dtype: DType,
](
    element_count: UInt64,
    validated_byte_length: UInt64,
    raw_byte_length: Int,
    address: Int,
    host_is_little_endian: Bool,
    expected: SafeTensorErrorKind,
) raises:
    var raised = False
    try:
        _ = _validate_typed_layout[dtype](
            element_count,
            validated_byte_length,
            raw_byte_length,
            address,
            host_is_little_endian,
        )
    except error:
        raised = True
        assert_equal(error.kind, expected)
    assert_true(raised)


def test_exact_native_dtype_mapping() raises:
    _assert_exact_mapping[DType.uint8](SafeDType.U8)
    _assert_exact_mapping[DType.int8](SafeDType.I8)
    _assert_exact_mapping[DType.float8_e5m2](SafeDType.F8_E5M2)
    _assert_exact_mapping[DType.float8_e4m3fn](SafeDType.F8_E4M3)
    _assert_exact_mapping[DType.float8_e8m0fnu](SafeDType.F8_E8M0)
    _assert_exact_mapping[DType.float8_e4m3fnuz](SafeDType.F8_E4M3FNUZ)
    _assert_exact_mapping[DType.float8_e5m2fnuz](SafeDType.F8_E5M2FNUZ)
    _assert_exact_mapping[DType.int16](SafeDType.I16)
    _assert_exact_mapping[DType.uint16](SafeDType.U16)
    _assert_exact_mapping[DType.float16](SafeDType.F16)
    _assert_exact_mapping[DType.bfloat16](SafeDType.BF16)
    _assert_exact_mapping[DType.int32](SafeDType.I32)
    _assert_exact_mapping[DType.uint32](SafeDType.U32)
    _assert_exact_mapping[DType.float32](SafeDType.F32)
    _assert_exact_mapping[DType.float64](SafeDType.F64)
    _assert_exact_mapping[DType.int64](SafeDType.I64)
    _assert_exact_mapping[DType.uint64](SafeDType.U64)


def test_unsupported_and_mismatched_dtypes_are_distinct() raises:
    assert_equal(SafeTensorErrorKind.DTYPE_MISMATCH.code(), "DTypeMismatch")
    _assert_requested_dtype_error[DType.bool](
        SafeTensorErrorKind.UNSUPPORTED_DTYPE
    )
    _assert_requested_dtype_error[DType.float4_e2m1fn](
        SafeTensorErrorKind.UNSUPPORTED_DTYPE
    )

    assert_true(not _supports_native_typed_view(SafeDType.BOOL))
    assert_true(not _supports_native_typed_view(SafeDType.F4))
    assert_true(not _supports_native_typed_view(SafeDType.F6_E2M3))
    assert_true(not _supports_native_typed_view(SafeDType.F6_E3M2))
    assert_true(not _supports_native_typed_view(SafeDType.C64))
    _assert_exact_dtype_error[DType.uint8](
        SafeDType.BOOL, SafeTensorErrorKind.UNSUPPORTED_DTYPE
    )
    _assert_exact_dtype_error[DType.uint32](
        SafeDType.F32, SafeTensorErrorKind.DTYPE_MISMATCH
    )


def test_endianness_and_empty_layout_policy() raises:
    assert_true(is_little_endian())
    assert_equal(
        _validate_typed_layout[DType.uint8](1, 1, 1, 1, False),
        1,
    )
    _assert_layout_error[DType.float32](
        1,
        4,
        4,
        4,
        False,
        SafeTensorErrorKind.UNSUPPORTED_ENDIANNESS,
    )
    assert_equal(
        _validate_typed_layout[DType.float32](0, 0, 0, 1, False),
        0,
    )


def test_checked_native_layout_boundaries() raises:
    assert_equal(
        _validate_typed_layout[DType.float32](4, 16, 16, 8, True),
        4,
    )
    _assert_layout_error[DType.float32](
        4,
        15,
        15,
        8,
        True,
        SafeTensorErrorKind.INVALID_TENSOR_SIZE,
    )
    _assert_layout_error[DType.float32](
        4,
        16,
        15,
        8,
        True,
        SafeTensorErrorKind.INVALID_TENSOR_SIZE,
    )
    _assert_layout_error[DType.float32](
        1,
        4,
        4,
        2,
        True,
        SafeTensorErrorKind.MISALIGNED_TENSOR,
    )
    _assert_layout_error[DType.float64](
        UInt64.MAX,
        0,
        0,
        8,
        True,
        SafeTensorErrorKind.VALIDATION_OVERFLOW,
    )
    _assert_layout_error[DType.uint8](
        UInt64(Int.MAX) + 1,
        UInt64(Int.MAX) + 1,
        0,
        1,
        True,
        SafeTensorErrorKind.VALIDATION_OVERFLOW,
    )


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
