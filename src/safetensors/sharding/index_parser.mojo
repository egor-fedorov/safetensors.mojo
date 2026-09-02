"""Schema-directed parser for Safetensors shard index JSON documents."""

from std.collections import Dict

from safetensors.errors import SafeTensorError, SafeTensorErrorKind, make_error
from safetensors.format.checked import checked_add_u64
from safetensors.format.json_lexical import (
    _expect,
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


comptime DEFAULT_MAX_INDEX_BYTES = UInt64(100_000_000)
comptime DEFAULT_MAX_INDEX_ENTRIES = UInt64(1_000_000)
comptime DEFAULT_MAX_SHARDS = UInt64(256)


@fieldwise_init
struct _ParsedIndex(Copyable, Movable):
    """Validated retained fields from one shard index document."""

    var weight_map: Dict[String, String]
    var declared_total_size: Optional[UInt64]


def _parse_weight_map[
    origin: Origin
](
    document: Span[UInt8, origin],
    mut index: Int,
    max_index_entries: UInt64,
    max_shards: UInt64,
) raises SafeTensorError -> Dict[String, String]:
    _expect(document, index, 0x7B, "weight_map must be a JSON object")
    _skip_json_whitespace(document, index)
    var weight_map = Dict[String, String]()
    var seen_shards = Dict[String, Bool]()
    var entry_count: UInt64 = 0
    var shard_count: UInt64 = 0

    if _peek(document, index) == 0x7D:
        _ = _take(document, index)
        return weight_map^

    while True:
        if entry_count >= max_index_entries:
            raise make_error(
                SafeTensorErrorKind.INDEX_ENTRY_LIMIT_EXCEEDED,
                "weight_map entry count exceeds the configured limit",
            )
        entry_count = checked_add_u64(entry_count, 1)
        var tensor_name = _parse_string(document, index)
        if tensor_name in weight_map:
            _fail_duplicate_key()
        _skip_json_whitespace(document, index)
        _expect(document, index, 0x3A, "expected colon after tensor name")
        _skip_json_whitespace(document, index)
        if _peek(document, index) != 0x22:
            raise make_error(
                SafeTensorErrorKind.INVALID_FIELD_TYPE,
                "weight_map values must be JSON strings",
            )
        var shard_name = _parse_string(document, index)
        if shard_name not in seen_shards:
            if shard_count >= max_shards:
                raise make_error(
                    SafeTensorErrorKind.SHARD_LIMIT_EXCEEDED,
                    "unique index shard count exceeds the configured limit",
                )
            shard_count = checked_add_u64(shard_count, 1)
            seen_shards[shard_name.copy()] = True
        weight_map[tensor_name.copy()] = shard_name.copy()
        _skip_json_whitespace(document, index)
        var delimiter = _take(document, index)
        if delimiter == 0x7D:
            return weight_map^
        if delimiter != 0x2C:
            raise make_error(
                SafeTensorErrorKind.INVALID_JSON,
                "expected comma or closing brace in weight_map",
            )
        _skip_json_whitespace(document, index)
        if _peek(document, index) == 0x7D:
            raise make_error(
                SafeTensorErrorKind.INVALID_JSON,
                "trailing comma in weight_map",
            )


def _parse_index_metadata[
    origin: Origin
](
    document: Span[UInt8, origin], mut index: Int
) raises SafeTensorError -> Optional[UInt64]:
    _expect(document, index, 0x7B, "metadata must be a JSON object")
    _skip_json_whitespace(document, index)
    var seen = Dict[String, Bool]()
    var total_size = Optional[UInt64](None)

    if _peek(document, index) == 0x7D:
        _ = _take(document, index)
        return total_size

    while True:
        var key = _parse_string(document, index)
        _record_key(seen, key)
        _skip_json_whitespace(document, index)
        _expect(document, index, 0x3A, "expected colon after metadata key")
        _skip_json_whitespace(document, index)
        if key == "total_size":
            total_size = Optional(
                _parse_uint64(
                    document,
                    index,
                    SafeTensorErrorKind.INVALID_FIELD_TYPE,
                )
            )
        else:
            _skip_json_value(document, index)

        _skip_json_whitespace(document, index)
        var delimiter = _take(document, index)
        if delimiter == 0x7D:
            return total_size
        if delimiter != 0x2C:
            raise make_error(
                SafeTensorErrorKind.INVALID_JSON,
                "expected comma or closing brace in index metadata",
            )
        _skip_json_whitespace(document, index)
        if _peek(document, index) == 0x7D:
            raise make_error(
                SafeTensorErrorKind.INVALID_JSON,
                "trailing comma in index metadata",
            )


def _parse_index[
    origin: Origin
](
    document: Span[UInt8, origin],
    max_index_entries: UInt64 = DEFAULT_MAX_INDEX_ENTRIES,
    max_shards: UInt64 = DEFAULT_MAX_SHARDS,
) raises SafeTensorError -> _ParsedIndex:
    """Parses and retains the exact fields needed to validate a shard set."""
    _validate_utf8(document)
    var index = 0
    _skip_json_whitespace(document, index)
    if _is_end(document, index) or _peek(document, index) != 0x7B:
        raise make_error(
            SafeTensorErrorKind.INVALID_INDEX,
            "Safetensors index must be a JSON object",
        )

    _expect(document, index, 0x7B, "expected index root object")
    _skip_json_whitespace(document, index)
    var seen = Dict[String, Bool]()
    var weight_map = Dict[String, String]()
    var declared_total_size = Optional[UInt64](None)
    var has_weight_map = False

    if _peek(document, index) == 0x7D:
        _ = _take(document, index)
    else:
        while True:
            var key = _parse_string(document, index)
            _record_key(seen, key)
            _skip_json_whitespace(document, index)
            _expect(document, index, 0x3A, "expected colon after index key")
            _skip_json_whitespace(document, index)

            if key == "weight_map":
                if _peek(document, index) != 0x7B:
                    raise make_error(
                        SafeTensorErrorKind.INVALID_FIELD_TYPE,
                        "weight_map must be a JSON object",
                    )
                weight_map = _parse_weight_map(
                    document, index, max_index_entries, max_shards
                )
                has_weight_map = True
            elif key == "metadata":
                if _peek(document, index) != 0x7B:
                    raise make_error(
                        SafeTensorErrorKind.INVALID_FIELD_TYPE,
                        "metadata must be a JSON object",
                    )
                declared_total_size = _parse_index_metadata(document, index)
            else:
                _skip_json_value(document, index)

            _skip_json_whitespace(document, index)
            var delimiter = _take(document, index)
            if delimiter == 0x7D:
                break
            if delimiter != 0x2C:
                raise make_error(
                    SafeTensorErrorKind.INVALID_JSON,
                    "expected comma or closing brace in index root",
                )
            _skip_json_whitespace(document, index)
            if _peek(document, index) == 0x7D:
                raise make_error(
                    SafeTensorErrorKind.INVALID_JSON,
                    "trailing comma in index root",
                )

    _skip_json_whitespace(document, index)
    if not _is_end(document, index):
        raise make_error(
            SafeTensorErrorKind.INVALID_JSON,
            "content follows the index root object",
        )
    if not has_weight_map:
        raise make_error(
            SafeTensorErrorKind.MISSING_FIELD,
            "missing weight_map field",
        )
    if len(weight_map) == 0:
        raise make_error(
            SafeTensorErrorKind.INVALID_INDEX,
            "weight_map must contain at least one tensor",
        )
    return _ParsedIndex(weight_map^, declared_total_size)
