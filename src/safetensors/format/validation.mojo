"""Semantic validation for decoded Safetensors metadata."""

from std.builtin.sort import sort

from safetensors.errors import SafeTensorError, SafeTensorErrorKind, make_error
from safetensors.format.checked import (
    checked_mul_u64,
    checked_product_u64,
    checked_sub_u64,
)
from safetensors.format.dtype import SafeDType
from safetensors.format.model import (
    RawSafeTensorMetadata,
    SafeTensorMetadata,
    TensorInfo,
)


def _tensor_offset_less(left: TensorInfo, right: TensorInfo) capturing -> Bool:
    if left.begin != right.begin:
        return left.begin < right.begin
    if left.end != right.end:
        return left.end < right.end
    return left.name < right.name


def validate_metadata(
    raw: RawSafeTensorMetadata,
    data_length: UInt64,
    data_start: UInt64 = 0,
) raises SafeTensorError -> SafeTensorMetadata:
    """Validates dtype, shape, byte-size, offset, and coverage invariants."""
    var seen_names = Dict[String, Bool]()
    var tensors = List[TensorInfo]()

    for index in range(len(raw.tensors)):
        var source = raw.tensors[index].copy()
        if source.name == "__metadata__":
            raise make_error(
                SafeTensorErrorKind.INVALID_METADATA,
                "the reserved __metadata__ key cannot name a tensor",
            )
        if source.name in seen_names:
            raise make_error(
                SafeTensorErrorKind.DUPLICATE_KEY,
                "duplicate decoded tensor name",
            )
        seen_names[source.name.copy()] = True

        var dtype = SafeDType.from_wire_name(source.dtype_name)
        var element_count = checked_product_u64(source.shape)
        var bit_length = checked_mul_u64(
            element_count, dtype.bits_per_element()
        )
        if bit_length % 8 != 0:
            raise make_error(
                SafeTensorErrorKind.MISALIGNED_SLICE,
                "tensor bit length is not byte-addressable",
            )
        var byte_length = bit_length // 8

        if source.begin > source.end:
            raise make_error(
                SafeTensorErrorKind.INVALID_OFFSETS,
                "tensor begin offset is greater than its end offset",
            )
        if source.end > data_length:
            raise make_error(
                SafeTensorErrorKind.INVALID_OFFSETS,
                "tensor offsets extend beyond the data buffer",
            )
        var range_length = checked_sub_u64(source.end, source.begin)
        if range_length != byte_length:
            raise make_error(
                SafeTensorErrorKind.INVALID_TENSOR_SIZE,
                "tensor offsets do not match its validated byte length",
            )

        tensors.append(
            TensorInfo(
                source.name.copy(),
                dtype,
                source.shape.copy(),
                source.begin,
                source.end,
                element_count,
                bit_length,
                byte_length,
            )
        )

    sort[T=TensorInfo, cmp_fn=_tensor_offset_less](tensors)

    var cursor: UInt64 = 0
    for index in range(len(tensors)):
        if tensors[index].begin != cursor:
            raise make_error(
                SafeTensorErrorKind.INCOMPLETE_DATA_COVERAGE,
                "tensor offsets contain a gap or overlap",
            )
        cursor = tensors[index].end
    if cursor != data_length:
        raise make_error(
            SafeTensorErrorKind.INCOMPLETE_DATA_COVERAGE,
            "tensor offsets do not cover the complete data buffer",
        )

    return SafeTensorMetadata(
        raw.user_metadata.copy(), tensors^, data_start, data_length
    )
