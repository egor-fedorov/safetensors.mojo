"""Deterministic checked planning for Safetensors serialization."""

from std.builtin.sort import sort

from safetensors.errors import SafeTensorError, SafeTensorErrorKind, make_error
from safetensors.format.checked import (
    checked_add_u64,
    checked_mul_u64,
    checked_product_u64,
)
from safetensors.format.dtype import SafeDType
from safetensors.format.model import SafeTensorData, TensorInfo
from safetensors.format.parser import DEFAULT_MAX_HEADER_BYTES


@fieldwise_init
struct _ValidatedTensor(Copyable, Movable):
    """One fully checked input tensor before canonical offset assignment."""

    var original_index: Int
    var dtype_ordinal: Int
    var name: String
    var dtype: SafeDType
    var shape: List[UInt64]
    var element_count: UInt64
    var bit_length: UInt64
    var byte_length: UInt64


@fieldwise_init
struct _SerializationPlan(Movable):
    """A bounded canonical header and validated payload traversal plan."""

    var header: List[UInt8]
    var header_length: UInt64
    var tensor_indices: List[Int]
    var tensors: List[TensorInfo]
    var data_length: UInt64


@fieldwise_init
struct _AssignedOffsets(Movable):
    """Canonical input indices paired with validation-derived tensor offsets."""

    var tensor_indices: List[Int]
    var tensors: List[TensorInfo]
    var data_length: UInt64


def _dtype_ordinal(dtype: SafeDType) raises SafeTensorError -> Int:
    """Returns the reference enum ordinal or rejects a constructed invalid value.
    """
    var ordinal = Int(dtype._value)
    if ordinal <= Int(SafeDType.U64._value):
        return ordinal
    raise make_error(
        SafeTensorErrorKind.UNSUPPORTED_DTYPE,
        "constructed SafeDType is not a recognized wire dtype",
    )


def _validated_tensor_less(
    left: _ValidatedTensor, right: _ValidatedTensor
) capturing -> Bool:
    if left.dtype_ordinal != right.dtype_ordinal:
        return left.dtype_ordinal > right.dtype_ordinal
    return left.name < right.name


def _string_less(left: String, right: String) capturing -> Bool:
    return left < right


def _header_too_large() -> SafeTensorError:
    return make_error(
        SafeTensorErrorKind.HEADER_TOO_LARGE,
        "serialized header exceeds the fixed header limit",
    )


def _append_header_byte(
    mut output: List[UInt8], value: UInt8
) raises SafeTensorError:
    if UInt64(len(output)) >= DEFAULT_MAX_HEADER_BYTES:
        raise _header_too_large()
    output.append(value)


def _append_utf8(mut output: List[UInt8], value: String) raises SafeTensorError:
    for byte in value.as_bytes():
        _append_header_byte(output, byte)


def _append_hex_digit(
    mut output: List[UInt8], value: UInt8
) raises SafeTensorError:
    if value < 10:
        _append_header_byte(output, UInt8(0x30) + value)
    else:
        _append_header_byte(output, UInt8(0x61) + value - UInt8(10))


def _append_json_string(
    mut output: List[UInt8], value: String
) raises SafeTensorError:
    _append_header_byte(output, 0x22)
    for byte in value.as_bytes():
        if byte == 0x22:
            _append_utf8(output, '\\"')
        elif byte == 0x5C:
            _append_utf8(output, "\\\\")
        elif byte == 0x08:
            _append_utf8(output, "\\b")
        elif byte == 0x09:
            _append_utf8(output, "\\t")
        elif byte == 0x0A:
            _append_utf8(output, "\\n")
        elif byte == 0x0C:
            _append_utf8(output, "\\f")
        elif byte == 0x0D:
            _append_utf8(output, "\\r")
        elif byte < 0x20:
            _append_utf8(output, "\\u00")
            _append_hex_digit(output, byte >> 4)
            _append_hex_digit(output, byte & 0x0F)
        else:
            _append_header_byte(output, byte)
    _append_header_byte(output, 0x22)


def _append_u64(mut output: List[UInt8], value: UInt64) raises SafeTensorError:
    _append_utf8(output, String(value))


def _append_shape(
    mut output: List[UInt8], shape: List[UInt64]
) raises SafeTensorError:
    _append_header_byte(output, 0x5B)
    for index in range(len(shape)):
        if index != 0:
            _append_header_byte(output, 0x2C)
        _append_u64(output, shape[index])
    _append_header_byte(output, 0x5D)


def _validate_inputs(
    tensors: List[SafeTensorData],
) raises SafeTensorError -> List[_ValidatedTensor]:
    var seen_names = Dict[String, Bool]()
    var validated = List[_ValidatedTensor]()

    for index in range(len(tensors)):
        var name = tensors[index].name.copy()
        if name == "__metadata__":
            raise make_error(
                SafeTensorErrorKind.INVALID_METADATA,
                "the reserved __metadata__ key cannot name a tensor",
            )
        if name in seen_names:
            raise make_error(
                SafeTensorErrorKind.DUPLICATE_KEY,
                "duplicate tensor name in serialization input",
            )
        seen_names[name.copy()] = True

        var dtype = tensors[index].dtype
        var dtype_ordinal = _dtype_ordinal(dtype)
        var element_count = checked_product_u64(tensors[index].shape)
        var bit_length = checked_mul_u64(
            element_count, dtype.bits_per_element()
        )
        if bit_length % 8 != 0:
            raise make_error(
                SafeTensorErrorKind.INVALID_TENSOR_SIZE,
                "tensor bit length is not byte-addressable",
            )
        var byte_length = bit_length // 8
        if UInt64(len(tensors[index].data)) != byte_length:
            raise make_error(
                SafeTensorErrorKind.INVALID_TENSOR_SIZE,
                "tensor payload length does not match dtype and shape",
            )

        validated.append(
            _ValidatedTensor(
                index,
                dtype_ordinal,
                name^,
                dtype,
                tensors[index].shape.copy(),
                element_count,
                bit_length,
                byte_length,
            )
        )

    return validated^


def _assign_offsets(
    mut validated: List[_ValidatedTensor],
) raises SafeTensorError -> _AssignedOffsets:
    sort[T=_ValidatedTensor, cmp_fn=_validated_tensor_less](validated)

    var tensor_indices = List[Int]()
    var tensors = List[TensorInfo]()
    var cursor: UInt64 = 0
    for index in range(len(validated)):
        var source = validated[index].copy()
        var begin = cursor
        var end = checked_add_u64(begin, source.byte_length)
        cursor = end
        tensor_indices.append(source.original_index)
        tensors.append(
            TensorInfo(
                source.name.copy(),
                source.dtype,
                source.shape.copy(),
                begin,
                end,
                source.element_count,
                source.bit_length,
                source.byte_length,
            )
        )
    return _AssignedOffsets(tensor_indices^, tensors^, cursor)


def _sorted_metadata_keys(metadata: Dict[String, String]) -> List[String]:
    var keys = List[String]()
    for key in metadata:
        keys.append(key.copy())
    sort[T=String, cmp_fn=_string_less](keys)
    return keys^


def _serialize_header(
    tensors: List[TensorInfo], metadata: Dict[String, String]
) raises SafeTensorError -> List[UInt8]:
    var output = List[UInt8]()
    var metadata_keys = _sorted_metadata_keys(metadata)
    var needs_comma = False

    _append_header_byte(output, 0x7B)
    if len(metadata_keys) != 0:
        _append_json_string(output, "__metadata__")
        _append_header_byte(output, 0x3A)
        _append_header_byte(output, 0x7B)
        for index in range(len(metadata_keys)):
            if index != 0:
                _append_header_byte(output, 0x2C)
            var key = metadata_keys[index].copy()
            _append_json_string(output, key)
            _append_header_byte(output, 0x3A)
            var maybe_value = metadata.get(key)
            if not maybe_value:
                raise make_error(
                    SafeTensorErrorKind.INVALID_METADATA,
                    "metadata key disappeared during serialization planning",
                )
            _append_json_string(output, maybe_value.value())
        _append_header_byte(output, 0x7D)
        needs_comma = True

    for index in range(len(tensors)):
        if needs_comma:
            _append_header_byte(output, 0x2C)
        var tensor = tensors[index].copy()
        _append_json_string(output, tensor.name)
        _append_utf8(output, ':{"dtype":')
        _append_json_string(output, tensor.dtype.wire_name())
        _append_utf8(output, ',"shape":')
        _append_shape(output, tensor.shape)
        _append_utf8(output, ',"data_offsets":[')
        _append_u64(output, tensor.begin)
        _append_header_byte(output, 0x2C)
        _append_u64(output, tensor.end)
        _append_utf8(output, "]}")
        needs_comma = True

    _append_header_byte(output, 0x7D)
    var remainder = UInt64(len(output)) % 8
    var padding = (UInt64(8) - remainder) % 8
    for _ in range(Int(padding)):
        _append_header_byte(output, 0x20)
    return output^


def _plan_serialization(
    tensors: List[SafeTensorData],
    user_metadata: Dict[String, String],
) raises SafeTensorError -> _SerializationPlan:
    """Validates inputs and returns a bounded deterministic serialization plan.
    """
    var validated = _validate_inputs(tensors)
    var assigned = _assign_offsets(validated)
    var header = _serialize_header(assigned.tensors, user_metadata)
    var header_length = UInt64(len(header))
    return _SerializationPlan(
        header^,
        header_length,
        assigned.tensor_indices.copy(),
        assigned.tensors.copy(),
        assigned.data_length,
    )
