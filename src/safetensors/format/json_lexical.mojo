"""Duplicate-aware JSON lexical helpers with exact integer parsing."""

from std.collections import Dict, List

from safetensors.errors import SafeTensorError, SafeTensorErrorKind, make_error
from safetensors.format.checked import checked_decimal_append


comptime _MAX_SKIPPED_JSON_DEPTH = 128


def _fail(kind: SafeTensorErrorKind, message: String) raises SafeTensorError:
    raise make_error(kind, message)


def _is_end[origin: Origin](document: Span[Byte, origin], index: Int) -> Bool:
    return index >= len(document)


def _peek[
    origin: Origin
](document: Span[Byte, origin], index: Int) raises SafeTensorError -> Byte:
    if index < 0 or index >= len(document):
        _fail(
            SafeTensorErrorKind.INVALID_JSON,
            "unexpected end of JSON header",
        )
    return document[index]


def _take[
    origin: Origin
](document: Span[Byte, origin], mut index: Int) raises SafeTensorError -> Byte:
    var value = _peek(document, index)
    index += 1
    return value


def _is_json_whitespace(value: Byte) -> Bool:
    return value == 0x20 or value == 0x09 or value == 0x0A or value == 0x0D


def _skip_json_whitespace[
    origin: Origin
](document: Span[Byte, origin], mut index: Int) raises SafeTensorError:
    while not _is_end(document, index):
        if not _is_json_whitespace(_peek(document, index)):
            return
        _ = _take(document, index)


def _expect[
    origin: Origin
](
    document: Span[Byte, origin],
    mut index: Int,
    expected: Byte,
    message: String,
) raises SafeTensorError:
    if _take(document, index) != expected:
        _fail(SafeTensorErrorKind.INVALID_JSON, message)


def _hex_value(value: Byte) raises SafeTensorError -> UInt32:
    if value >= 0x30 and value <= 0x39:
        return UInt32(value - 0x30)
    if value >= 0x41 and value <= 0x46:
        return UInt32(value - 0x41) + 10
    if value >= 0x61 and value <= 0x66:
        return UInt32(value - 0x61) + 10
    _fail(
        SafeTensorErrorKind.INVALID_JSON,
        "invalid hexadecimal digit in JSON Unicode escape",
    )
    return 0


def _parse_hex_quad[
    origin: Origin
](
    document: Span[Byte, origin], mut index: Int
) raises SafeTensorError -> UInt32:
    var value: UInt32 = 0
    for _ in range(4):
        value = (value << 4) | _hex_value(_take(document, index))
    return value


def _append_utf8_scalar(
    mut output: List[Byte], scalar: UInt32
) raises SafeTensorError:
    if scalar <= 0x7F:
        output.append(Byte(scalar))
        return
    if scalar <= 0x7FF:
        output.append(Byte(0xC0 | (scalar >> 6)))
        output.append(Byte(0x80 | (scalar & 0x3F)))
        return
    if scalar >= 0xD800 and scalar <= 0xDFFF:
        _fail(
            SafeTensorErrorKind.INVALID_JSON,
            "unpaired surrogate in JSON string",
        )
    if scalar <= 0xFFFF:
        output.append(Byte(0xE0 | (scalar >> 12)))
        output.append(Byte(0x80 | ((scalar >> 6) & 0x3F)))
        output.append(Byte(0x80 | (scalar & 0x3F)))
        return
    if scalar <= 0x10FFFF:
        output.append(Byte(0xF0 | (scalar >> 18)))
        output.append(Byte(0x80 | ((scalar >> 12) & 0x3F)))
        output.append(Byte(0x80 | ((scalar >> 6) & 0x3F)))
        output.append(Byte(0x80 | (scalar & 0x3F)))
        return
    _fail(
        SafeTensorErrorKind.INVALID_JSON,
        "Unicode scalar is outside the valid range",
    )


def _parse_unicode_escape[
    origin: Origin
](
    document: Span[Byte, origin],
    mut index: Int,
    mut output: List[Byte],
) raises SafeTensorError:
    var first = _parse_hex_quad(document, index)
    if first >= 0xD800 and first <= 0xDBFF:
        if _take(document, index) != 0x5C or _take(document, index) != 0x75:
            _fail(
                SafeTensorErrorKind.INVALID_JSON,
                "high surrogate is not followed by a low surrogate",
            )
        var second = _parse_hex_quad(document, index)
        if second < 0xDC00 or second > 0xDFFF:
            _fail(
                SafeTensorErrorKind.INVALID_JSON,
                "high surrogate is not followed by a low surrogate",
            )
        var scalar = (
            UInt32(0x10000)
            + ((first - UInt32(0xD800)) << 10)
            + (second - UInt32(0xDC00))
        )
        _append_utf8_scalar(output, scalar)
        return
    if first >= 0xDC00 and first <= 0xDFFF:
        _fail(
            SafeTensorErrorKind.INVALID_JSON,
            "unpaired low surrogate in JSON string",
        )
    _append_utf8_scalar(output, first)


def _parse_escape[
    origin: Origin
](
    document: Span[Byte, origin],
    mut index: Int,
    mut output: List[Byte],
) raises SafeTensorError:
    var escaped = _take(document, index)
    if escaped == 0x22:
        output.append(0x22)
        return
    if escaped == 0x5C:
        output.append(0x5C)
        return
    if escaped == 0x2F:
        output.append(0x2F)
        return
    if escaped == 0x62:
        output.append(0x08)
        return
    if escaped == 0x66:
        output.append(0x0C)
        return
    if escaped == 0x6E:
        output.append(0x0A)
        return
    if escaped == 0x72:
        output.append(0x0D)
        return
    if escaped == 0x74:
        output.append(0x09)
        return
    if escaped == 0x75:
        _parse_unicode_escape(document, index, output)
        return
    _fail(SafeTensorErrorKind.INVALID_JSON, "invalid JSON string escape")


def _parse_string[
    origin: Origin
](
    document: Span[Byte, origin], mut index: Int
) raises SafeTensorError -> String:
    _expect(document, index, 0x22, "expected a JSON string")
    var output = List[Byte]()
    while True:
        var value = _take(document, index)
        if value == 0x22:
            try:
                return String(from_utf8=Span(output))
            except:
                _fail(
                    SafeTensorErrorKind.INVALID_UTF8,
                    "decoded JSON string is not valid UTF-8",
                )
        if value == 0x5C:
            _parse_escape(document, index, output)
            continue
        if value < 0x20:
            _fail(
                SafeTensorErrorKind.INVALID_JSON,
                "unescaped control character in JSON string",
            )
        output.append(value)


def _parse_uint64[
    origin: Origin
](
    document: Span[Byte, origin],
    mut index: Int,
    invalid_kind: SafeTensorErrorKind,
) raises SafeTensorError -> UInt64:
    var first = _peek(document, index)
    if first < 0x30 or first > 0x39:
        _fail(
            invalid_kind,
            "expected a non-negative integer",
        )
    if first == 0x30:
        _ = _take(document, index)
        if not _is_end(document, index):
            var following = _peek(document, index)
            if following >= 0x30 and following <= 0x39:
                _fail(
                    invalid_kind,
                    "leading zero in JSON integer",
                )
            if following == 0x2E or following == 0x65 or following == 0x45:
                _fail(
                    invalid_kind,
                    "integer must not use a fraction or exponent",
                )
        return 0

    var value: UInt64 = 0
    while not _is_end(document, index):
        var current = _peek(document, index)
        if current < 0x30 or current > 0x39:
            break
        value = checked_decimal_append(value, UInt8(current - 0x30))
        _ = _take(document, index)
    if not _is_end(document, index):
        var following = _peek(document, index)
        if following == 0x2E or following == 0x65 or following == 0x45:
            _fail(
                invalid_kind,
                "integer must not use a fraction or exponent",
            )
    return value


def _fail_duplicate_key() raises SafeTensorError:
    _fail(
        SafeTensorErrorKind.DUPLICATE_KEY,
        "duplicate decoded JSON object key",
    )


def _record_key(
    mut seen: Dict[String, Bool], key: String
) raises SafeTensorError:
    if key in seen:
        _fail_duplicate_key()
    seen[key.copy()] = True


def _skip_json_string[
    origin: Origin
](document: Span[Byte, origin], mut index: Int) raises SafeTensorError:
    """Validates and skips a JSON string without allocating its decoded form."""
    _expect(document, index, 0x22, "expected a JSON string")
    while True:
        var value = _take(document, index)
        if value == 0x22:
            return
        if value < 0x20:
            _fail(
                SafeTensorErrorKind.INVALID_JSON,
                "unescaped control character in JSON string",
            )
        if value != 0x5C:
            continue

        var escaped = _take(document, index)
        if (
            escaped == 0x22
            or escaped == 0x5C
            or escaped == 0x2F
            or escaped == 0x62
            or escaped == 0x66
            or escaped == 0x6E
            or escaped == 0x72
            or escaped == 0x74
        ):
            continue
        if escaped != 0x75:
            _fail(
                SafeTensorErrorKind.INVALID_JSON, "invalid JSON string escape"
            )

        var first = _parse_hex_quad(document, index)
        if first >= 0xD800 and first <= 0xDBFF:
            if _take(document, index) != 0x5C or _take(document, index) != 0x75:
                _fail(
                    SafeTensorErrorKind.INVALID_JSON,
                    "high surrogate is not followed by a low surrogate",
                )
            var second = _parse_hex_quad(document, index)
            if second < 0xDC00 or second > 0xDFFF:
                _fail(
                    SafeTensorErrorKind.INVALID_JSON,
                    "high surrogate is not followed by a low surrogate",
                )
        elif first >= 0xDC00 and first <= 0xDFFF:
            _fail(
                SafeTensorErrorKind.INVALID_JSON,
                "unpaired low surrogate in JSON string",
            )


def _skip_json_number[
    origin: Origin
](document: Span[Byte, origin], mut index: Int) raises SafeTensorError:
    if _peek(document, index) == 0x2D:
        _ = _take(document, index)

    var first = _peek(document, index)
    if first == 0x30:
        _ = _take(document, index)
        if not _is_end(document, index):
            var following = _peek(document, index)
            if following >= 0x30 and following <= 0x39:
                _fail(
                    SafeTensorErrorKind.INVALID_JSON,
                    "leading zero in skipped JSON number",
                )
    elif first >= 0x31 and first <= 0x39:
        while not _is_end(document, index):
            var digit = _peek(document, index)
            if digit < 0x30 or digit > 0x39:
                break
            _ = _take(document, index)
    else:
        _fail(SafeTensorErrorKind.INVALID_JSON, "invalid JSON number")

    if not _is_end(document, index) and _peek(document, index) == 0x2E:
        _ = _take(document, index)
        var fraction_start = index
        while not _is_end(document, index):
            var digit = _peek(document, index)
            if digit < 0x30 or digit > 0x39:
                break
            _ = _take(document, index)
        if index == fraction_start:
            _fail(
                SafeTensorErrorKind.INVALID_JSON,
                "JSON fraction requires at least one digit",
            )

    if not _is_end(document, index):
        var exponent = _peek(document, index)
        if exponent == 0x65 or exponent == 0x45:
            _ = _take(document, index)
            if not _is_end(document, index):
                var sign = _peek(document, index)
                if sign == 0x2B or sign == 0x2D:
                    _ = _take(document, index)
            var exponent_start = index
            while not _is_end(document, index):
                var digit = _peek(document, index)
                if digit < 0x30 or digit > 0x39:
                    break
                _ = _take(document, index)
            if index == exponent_start:
                _fail(
                    SafeTensorErrorKind.INVALID_JSON,
                    "JSON exponent requires at least one digit",
                )


def _skip_json_value[
    origin: Origin
](
    document: Span[Byte, origin],
    mut index: Int,
    depth: Int = 0,
) raises SafeTensorError:
    var first = _peek(document, index)
    if first == 0x22:
        _skip_json_string(document, index)
        return
    if first == 0x2D or (first >= 0x30 and first <= 0x39):
        _skip_json_number(document, index)
        return
    if first == 0x74:
        _expect(document, index, 0x74, "expected true")
        _expect(document, index, 0x72, "expected true")
        _expect(document, index, 0x75, "expected true")
        _expect(document, index, 0x65, "expected true")
        return
    if first == 0x66:
        _expect(document, index, 0x66, "expected false")
        _expect(document, index, 0x61, "expected false")
        _expect(document, index, 0x6C, "expected false")
        _expect(document, index, 0x73, "expected false")
        _expect(document, index, 0x65, "expected false")
        return
    if first == 0x6E:
        _expect(document, index, 0x6E, "expected null")
        _expect(document, index, 0x75, "expected null")
        _expect(document, index, 0x6C, "expected null")
        _expect(document, index, 0x6C, "expected null")
        return
    if first != 0x5B and first != 0x7B:
        _fail(SafeTensorErrorKind.INVALID_JSON, "invalid skipped JSON value")
    if depth >= _MAX_SKIPPED_JSON_DEPTH:
        _fail(
            SafeTensorErrorKind.INVALID_JSON,
            "skipped JSON value exceeds the nesting limit",
        )

    if first == 0x5B:
        _ = _take(document, index)
        _skip_json_whitespace(document, index)
        if _peek(document, index) == 0x5D:
            _ = _take(document, index)
            return
        while True:
            _skip_json_value(document, index, depth + 1)
            _skip_json_whitespace(document, index)
            var delimiter = _take(document, index)
            if delimiter == 0x5D:
                return
            if delimiter != 0x2C:
                _fail(
                    SafeTensorErrorKind.INVALID_JSON,
                    "expected comma or closing bracket in skipped array",
                )
            _skip_json_whitespace(document, index)
            if _peek(document, index) == 0x5D:
                _fail(
                    SafeTensorErrorKind.INVALID_JSON,
                    "trailing comma in skipped array",
                )

    _ = _take(document, index)
    _skip_json_whitespace(document, index)
    if _peek(document, index) == 0x7D:
        _ = _take(document, index)
        return
    while True:
        _skip_json_string(document, index)
        _skip_json_whitespace(document, index)
        _expect(document, index, 0x3A, "expected colon in skipped object")
        _skip_json_whitespace(document, index)
        _skip_json_value(document, index, depth + 1)
        _skip_json_whitespace(document, index)
        var delimiter = _take(document, index)
        if delimiter == 0x7D:
            return
        if delimiter != 0x2C:
            _fail(
                SafeTensorErrorKind.INVALID_JSON,
                "expected comma or closing brace in skipped object",
            )
        _skip_json_whitespace(document, index)
        if _peek(document, index) == 0x7D:
            _fail(
                SafeTensorErrorKind.INVALID_JSON,
                "trailing comma in skipped object",
            )


def _validate_utf8[
    origin: Origin
](document: Span[Byte, origin]) raises SafeTensorError:
    try:
        _ = String(from_utf8=document)
    except:
        _fail(
            SafeTensorErrorKind.INVALID_UTF8,
            "Safetensors header is not valid UTF-8",
        )
