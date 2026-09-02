"""Schema-directed parser for validated Safetensors JSON headers."""

from std.collections import Dict, List

from safetensors.errors import SafeTensorError, SafeTensorErrorKind
from safetensors.format.json_lexical import (
    _expect,
    _fail,
    _fail_duplicate_key,
    _is_end,
    _parse_string,
    _parse_uint64,
    _peek,
    _record_key,
    _skip_json_value,
    _skip_json_whitespace,
    _take,
    _validate_utf8,
)
from safetensors.format.model import RawSafeTensorMetadata, RawTensorInfo


def _parse_shape[
    origin: Origin
](header: Span[Byte, origin], mut index: Int) raises SafeTensorError -> List[
    UInt64
]:
    _expect(header, index, 0x5B, "shape must be a JSON array")
    _skip_json_whitespace(header, index)
    var shape = List[UInt64]()
    if _peek(header, index) == 0x5D:
        _ = _take(header, index)
        return shape^

    while True:
        shape.append(
            _parse_uint64(
                header,
                index,
                SafeTensorErrorKind.INVALID_SHAPE,
            )
        )
        _skip_json_whitespace(header, index)
        var delimiter = _take(header, index)
        if delimiter == 0x5D:
            return shape^
        if delimiter != 0x2C:
            _fail(
                SafeTensorErrorKind.INVALID_SHAPE,
                "expected comma or closing bracket in shape",
            )
        _skip_json_whitespace(header, index)
        if _peek(header, index) == 0x5D:
            _fail(
                SafeTensorErrorKind.INVALID_SHAPE,
                "trailing comma in shape",
            )


def _parse_offsets[
    origin: Origin
](header: Span[Byte, origin], mut index: Int) raises SafeTensorError -> List[
    UInt64
]:
    _expect(
        header,
        index,
        0x5B,
        "data_offsets must be a JSON array",
    )
    _skip_json_whitespace(header, index)
    var offsets = List[UInt64]()
    offsets.append(
        _parse_uint64(
            header,
            index,
            SafeTensorErrorKind.INVALID_OFFSETS,
        )
    )
    _skip_json_whitespace(header, index)
    if _take(header, index) != 0x2C:
        _fail(
            SafeTensorErrorKind.INVALID_OFFSETS,
            "data_offsets must contain exactly two integers",
        )
    _skip_json_whitespace(header, index)
    offsets.append(
        _parse_uint64(
            header,
            index,
            SafeTensorErrorKind.INVALID_OFFSETS,
        )
    )
    _skip_json_whitespace(header, index)
    if _take(header, index) != 0x5D:
        _fail(
            SafeTensorErrorKind.INVALID_OFFSETS,
            "data_offsets must contain exactly two integers",
        )
    return offsets^


def _parse_metadata[
    origin: Origin
](header: Span[Byte, origin], mut index: Int) raises SafeTensorError -> Dict[
    String, String
]:
    _expect(
        header,
        index,
        0x7B,
        "__metadata__ must be a JSON object",
    )
    _skip_json_whitespace(header, index)
    var metadata = Dict[String, String]()
    if _peek(header, index) == 0x7D:
        _ = _take(header, index)
        return metadata^

    while True:
        var key = _parse_string(header, index)
        if key in metadata:
            _fail_duplicate_key()
        _skip_json_whitespace(header, index)
        _expect(
            header,
            index,
            0x3A,
            "expected colon after metadata key",
        )
        _skip_json_whitespace(header, index)
        if _peek(header, index) != 0x22:
            _fail(
                SafeTensorErrorKind.INVALID_FIELD_TYPE,
                "metadata values must be JSON strings",
            )
        var value = _parse_string(header, index)
        metadata[key.copy()] = value.copy()
        _skip_json_whitespace(header, index)
        var delimiter = _take(header, index)
        if delimiter == 0x7D:
            return metadata^
        if delimiter != 0x2C:
            _fail(
                SafeTensorErrorKind.INVALID_JSON,
                "expected comma or closing brace in metadata",
            )
        _skip_json_whitespace(header, index)
        if _peek(header, index) == 0x7D:
            _fail(
                SafeTensorErrorKind.INVALID_JSON,
                "trailing comma in metadata",
            )


def _parse_tensor[
    origin: Origin
](
    header: Span[Byte, origin],
    mut index: Int,
    name: String,
    strict: Bool,
) raises SafeTensorError -> RawTensorInfo:
    _expect(
        header,
        index,
        0x7B,
        "tensor descriptor must be a JSON object",
    )
    _skip_json_whitespace(header, index)

    var dtype_name = String()
    var shape = List[UInt64]()
    var begin: UInt64 = 0
    var end: UInt64 = 0
    var has_dtype = False
    var has_shape = False
    var has_offsets = False
    var seen_fields = Dict[String, Bool]()

    if _peek(header, index) == 0x7D:
        _ = _take(header, index)
    else:
        while True:
            var key = _parse_string(header, index)
            _record_key(seen_fields, key)
            _skip_json_whitespace(header, index)
            _expect(
                header,
                index,
                0x3A,
                "expected colon after tensor field",
            )
            _skip_json_whitespace(header, index)

            if key == "dtype":
                if _peek(header, index) != 0x22:
                    _fail(
                        SafeTensorErrorKind.INVALID_FIELD_TYPE,
                        "dtype must be a JSON string",
                    )
                dtype_name = _parse_string(header, index)
                has_dtype = True
            elif key == "shape":
                if _peek(header, index) != 0x5B:
                    _fail(
                        SafeTensorErrorKind.INVALID_FIELD_TYPE,
                        "shape must be a JSON array",
                    )
                shape = _parse_shape(header, index)
                has_shape = True
            elif key == "data_offsets":
                if _peek(header, index) != 0x5B:
                    _fail(
                        SafeTensorErrorKind.INVALID_FIELD_TYPE,
                        "data_offsets must be a JSON array",
                    )
                var offsets = _parse_offsets(header, index)
                begin = offsets[0]
                end = offsets[1]
                has_offsets = True
            else:
                if strict:
                    _fail(
                        SafeTensorErrorKind.UNKNOWN_FIELD,
                        "unknown tensor descriptor field",
                    )
                _skip_json_value(header, index)

            _skip_json_whitespace(header, index)
            var delimiter = _take(header, index)
            if delimiter == 0x7D:
                break
            if delimiter != 0x2C:
                _fail(
                    SafeTensorErrorKind.INVALID_JSON,
                    "expected comma or closing brace in tensor descriptor",
                )
            _skip_json_whitespace(header, index)
            if _peek(header, index) == 0x7D:
                _fail(
                    SafeTensorErrorKind.INVALID_JSON,
                    "trailing comma in tensor descriptor",
                )

    if not has_dtype:
        _fail(SafeTensorErrorKind.MISSING_FIELD, "missing dtype field")
    if not has_shape:
        _fail(SafeTensorErrorKind.MISSING_FIELD, "missing shape field")
    if not has_offsets:
        _fail(
            SafeTensorErrorKind.MISSING_FIELD,
            "missing data_offsets field",
        )
    return RawTensorInfo(name.copy(), dtype_name^, shape^, begin, end)


def parse_raw_header[
    origin: Origin
](
    header: Span[Byte, origin], strict: Bool = False
) raises SafeTensorError -> RawSafeTensorMetadata:
    """Parses one header with compatible or canonical-schema policy.

    Compatible mode still validates complete JSON and every known descriptor
    field. Strict mode additionally requires canonical boundary whitespace and
    rejects unknown descriptor fields.
    """
    _validate_utf8(header)
    var index = 0
    if not strict:
        _skip_json_whitespace(header, index)
    if _is_end(header, index) or _peek(header, index) != 0x7B:
        _fail(
            SafeTensorErrorKind.INVALID_HEADER_START,
            "Safetensors JSON header must start with an object",
        )

    _expect(header, index, 0x7B, "expected root JSON object")
    _skip_json_whitespace(header, index)

    var user_metadata = Dict[String, String]()
    var tensors = List[RawTensorInfo]()
    var seen = Dict[String, Bool]()

    if _peek(header, index) == 0x7D:
        _ = _take(header, index)
    else:
        while True:
            var key = _parse_string(header, index)
            _record_key(seen, key)
            _skip_json_whitespace(header, index)
            _expect(
                header,
                index,
                0x3A,
                "expected colon after root object key",
            )
            _skip_json_whitespace(header, index)

            if key == "__metadata__":
                if _peek(header, index) != 0x7B:
                    _fail(
                        SafeTensorErrorKind.INVALID_METADATA,
                        "__metadata__ must be a JSON object",
                    )
                user_metadata = _parse_metadata(header, index)
            else:
                if _peek(header, index) != 0x7B:
                    _fail(
                        SafeTensorErrorKind.INVALID_FIELD_TYPE,
                        "tensor descriptor must be a JSON object",
                    )
                tensors.append(_parse_tensor(header, index, key, strict))

            _skip_json_whitespace(header, index)
            var delimiter = _take(header, index)
            if delimiter == 0x7D:
                break
            if delimiter != 0x2C:
                _fail(
                    SafeTensorErrorKind.INVALID_JSON,
                    "expected comma or closing brace in root object",
                )
            _skip_json_whitespace(header, index)
            if _peek(header, index) == 0x7D:
                _fail(
                    SafeTensorErrorKind.INVALID_JSON,
                    "trailing comma in root object",
                )

    if strict:
        while not _is_end(header, index) and _peek(header, index) == 0x20:
            _ = _take(header, index)
    else:
        _skip_json_whitespace(header, index)
    if not _is_end(header, index):
        _fail(
            SafeTensorErrorKind.INVALID_HEADER_PADDING,
            "invalid content after the root object",
        )
    return RawSafeTensorMetadata(user_metadata^, tensors^)
