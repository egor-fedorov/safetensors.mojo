"""Exact native-dtype policy and checked layout validation for typed views."""

from std.sys import align_of, size_of

from safetensors.errors import SafeTensorError, SafeTensorErrorKind, make_error
from safetensors.format.checked import (
    checked_mul_u64,
    checked_u64_to_int,
)
from safetensors.format.dtype import SafeDType


def _requested_safe_dtype[
    dtype: DType,
]() raises SafeTensorError -> SafeDType:
    """Returns the exact whitelisted wire dtype for one native Mojo scalar."""
    comptime if dtype == DType.uint8:
        return SafeDType.U8
    elif dtype == DType.int8:
        return SafeDType.I8
    elif dtype == DType.float8_e5m2:
        return SafeDType.F8_E5M2
    elif dtype == DType.float8_e4m3fn:
        return SafeDType.F8_E4M3
    elif dtype == DType.float8_e8m0fnu:
        return SafeDType.F8_E8M0
    elif dtype == DType.float8_e4m3fnuz:
        return SafeDType.F8_E4M3FNUZ
    elif dtype == DType.float8_e5m2fnuz:
        return SafeDType.F8_E5M2FNUZ
    elif dtype == DType.int16:
        return SafeDType.I16
    elif dtype == DType.uint16:
        return SafeDType.U16
    elif dtype == DType.float16:
        return SafeDType.F16
    elif dtype == DType.bfloat16:
        return SafeDType.BF16
    elif dtype == DType.int32:
        return SafeDType.I32
    elif dtype == DType.uint32:
        return SafeDType.U32
    elif dtype == DType.float32:
        return SafeDType.F32
    elif dtype == DType.float64:
        return SafeDType.F64
    elif dtype == DType.int64:
        return SafeDType.I64
    elif dtype == DType.uint64:
        return SafeDType.U64
    else:
        raise make_error(
            SafeTensorErrorKind.UNSUPPORTED_DTYPE,
            "requested Mojo dtype has no safe native Safetensors view",
        )


def _supports_native_typed_view(dtype: SafeDType) -> Bool:
    """Returns whether a wire dtype has an exact whitelisted Mojo scalar."""
    return (
        dtype == SafeDType.U8
        or dtype == SafeDType.I8
        or dtype == SafeDType.F8_E5M2
        or dtype == SafeDType.F8_E4M3
        or dtype == SafeDType.F8_E8M0
        or dtype == SafeDType.F8_E4M3FNUZ
        or dtype == SafeDType.F8_E5M2FNUZ
        or dtype == SafeDType.I16
        or dtype == SafeDType.U16
        or dtype == SafeDType.F16
        or dtype == SafeDType.BF16
        or dtype == SafeDType.I32
        or dtype == SafeDType.U32
        or dtype == SafeDType.F32
        or dtype == SafeDType.F64
        or dtype == SafeDType.I64
        or dtype == SafeDType.U64
    )


def _require_exact_typed_dtype[
    dtype: DType,
](actual: SafeDType) raises SafeTensorError:
    """Requires a supported wire dtype matching the requested native scalar."""
    var requested = _requested_safe_dtype[dtype]()
    if not _supports_native_typed_view(actual):
        raise make_error(
            SafeTensorErrorKind.UNSUPPORTED_DTYPE,
            "tensor dtype has no safe native Mojo scalar view",
        )
    if actual != requested:
        raise make_error(
            SafeTensorErrorKind.DTYPE_MISMATCH,
            "requested Mojo dtype does not match the tensor dtype",
        )


def _validate_typed_layout[
    dtype: DType,
](
    element_count: UInt64,
    validated_byte_length: UInt64,
    raw_byte_length: Int,
    address: Int,
    host_is_little_endian: Bool,
) raises SafeTensorError -> Int:
    """Checks native size, count, byte order, and actual pointer alignment."""
    _ = _requested_safe_dtype[dtype]()

    var element_size = size_of[Scalar[dtype]]()
    var expected_byte_length = checked_mul_u64(
        element_count, UInt64(element_size)
    )
    if expected_byte_length != validated_byte_length:
        raise make_error(
            SafeTensorErrorKind.INVALID_TENSOR_SIZE,
            "native scalar size disagrees with validated tensor byte length",
        )

    var checked_raw_byte_length = checked_u64_to_int(validated_byte_length)
    if raw_byte_length != checked_raw_byte_length:
        raise make_error(
            SafeTensorErrorKind.INVALID_TENSOR_SIZE,
            "raw view length disagrees with validated tensor byte length",
        )

    var native_element_count = checked_u64_to_int(element_count)
    if native_element_count == 0:
        return 0

    if element_size > 1 and not host_is_little_endian:
        raise make_error(
            SafeTensorErrorKind.UNSUPPORTED_ENDIANNESS,
            "multi-byte typed views require a little-endian host",
        )

    var alignment = align_of[Scalar[dtype]]()
    if address % alignment != 0:
        raise make_error(
            SafeTensorErrorKind.MISALIGNED_TENSOR,
            "tensor address does not satisfy the native scalar alignment",
        )
    return native_element_count
