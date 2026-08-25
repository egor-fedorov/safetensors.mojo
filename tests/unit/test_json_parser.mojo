"""Strict JSON header parser tests."""

from std.testing import TestSuite, assert_equal, assert_true

from safetensors import SafeTensorErrorKind, parse_raw_header


def _descriptor() -> String:
    return '{"dtype":"U8","shape":[1],"data_offsets":[0,1]}'


def _assert_header_error(header: String, expected: SafeTensorErrorKind) raises:
    var raised = False
    try:
        _ = parse_raw_header(header.as_bytes())
    except error:
        raised = True
        assert_equal(error.kind, expected)
    assert_true(raised)


def test_strict_header_and_metadata() raises:
    var header = (
        '{"__metadata__":{"producer":"Mojo","escaped":"line\\n"},'
        '"tensor":{"data_offsets":[0,1],"shape":[1],"dtype":"U8"}}   '
    )
    var raw = parse_raw_header(header.as_bytes())
    assert_equal(len(raw.tensors), 1)
    assert_equal(raw.tensors[0].name, "tensor")
    assert_equal(raw.tensors[0].dtype_name, "U8")
    assert_equal(raw.tensors[0].shape[0], UInt64(1))
    assert_equal(raw.tensors[0].begin, UInt64(0))
    assert_equal(raw.tensors[0].end, UInt64(1))
    assert_equal(raw.user_metadata["producer"], "Mojo")
    assert_equal(raw.user_metadata["escaped"], "line\n")


def test_unicode_strings_and_surrogate_pairs() raises:
    var header = '{"w\\u00ebight\\ud83d\\ude0a":' + _descriptor() + "}"
    var raw = parse_raw_header(header.as_bytes())
    assert_equal(raw.tensors[0].name, "wëight😊")


def test_every_json_string_escape() raises:
    var raw = parse_raw_header(
        '{"__metadata__":{"escapes":"\\"\\\\\\/\\b\\f\\n\\r\\t"}}'.as_bytes()
    )
    assert_equal(
        raw.user_metadata["escapes"],
        '"\\/\b\f\n\r\t',
    )


def test_decoded_duplicate_keys_are_rejected() raises:
    var descriptor = _descriptor()
    _assert_header_error(
        '{"a":' + descriptor + ',"a":' + descriptor + "}",
        SafeTensorErrorKind.DUPLICATE_KEY,
    )
    _assert_header_error(
        '{"a":' + descriptor + ',"\\u0061":' + descriptor + "}",
        SafeTensorErrorKind.DUPLICATE_KEY,
    )
    _assert_header_error(
        '{"😊":' + descriptor + ',"\\ud83d\\ude0a":' + descriptor + "}",
        SafeTensorErrorKind.DUPLICATE_KEY,
    )
    _assert_header_error(
        '{"__metadata__":{"a":"x","\\u0061":"y"}}',
        SafeTensorErrorKind.DUPLICATE_KEY,
    )
    _assert_header_error(
        (
            '{"a":{"dtype":"U8","\\u0064type":"U8",'
            '"shape":[1],"data_offsets":[0,1]}}'
        ),
        SafeTensorErrorKind.DUPLICATE_KEY,
    )
    _assert_header_error(
        (
            '{"a":{"dtype":"U8","shape":[1],"\\u0073hape":[1],'
            '"data_offsets":[0,1]}}'
        ),
        SafeTensorErrorKind.DUPLICATE_KEY,
    )
    _assert_header_error(
        (
            '{"a":{"dtype":"U8","shape":[1],"data_offsets":[0,1],'
            '"data_\\u006fffsets":[0,1]}}'
        ),
        SafeTensorErrorKind.DUPLICATE_KEY,
    )


def test_duplicate_descriptor_field_precedes_value_validation() raises:
    _assert_header_error(
        '{"a":{"dtype":"U8","dtype":1,"shape":[1],"data_offsets":[0,1]}}',
        SafeTensorErrorKind.DUPLICATE_KEY,
    )
    _assert_header_error(
        '{"a":{"dtype":"U8","shape":[1],"shape":"bad","data_offsets":[0,1]}}',
        SafeTensorErrorKind.DUPLICATE_KEY,
    )
    _assert_header_error(
        (
            '{"a":{"dtype":"U8","shape":[1],"data_offsets":[0,1],'
            '"data_offsets":{}}}'
        ),
        SafeTensorErrorKind.DUPLICATE_KEY,
    )


def test_integers_are_exact_uint64() raises:
    var exact = parse_raw_header(
        '{"a":{"dtype":"U8","shape":[9007199254740993],'
        '"data_offsets":[0,18446744073709551615]}}'.as_bytes()
    )
    assert_equal(exact.tensors[0].shape[0], UInt64(9007199254740993))
    assert_equal(exact.tensors[0].end, UInt64.MAX)

    _assert_header_error(
        (
            '{"a":{"dtype":"U8","shape":[18446744073709551616],'
            '"data_offsets":[0,0]}}'
        ),
        SafeTensorErrorKind.VALIDATION_OVERFLOW,
    )


def test_invalid_integer_forms_are_rejected() raises:
    for token in ["-1", "+1", "01", "1.0", "1e0", "1E3"]:
        _assert_header_error(
            '{"a":{"dtype":"U8","shape":[' + token + '],"data_offsets":[0,0]}}',
            SafeTensorErrorKind.INVALID_SHAPE,
        )
    _assert_header_error(
        '{"a":{"dtype":"U8","shape":[0],"data_offsets":[-1,0]}}',
        SafeTensorErrorKind.INVALID_OFFSETS,
    )


def test_schema_and_padding_errors() raises:
    _assert_header_error("[]", SafeTensorErrorKind.INVALID_HEADER_START)
    _assert_header_error("{}\t", SafeTensorErrorKind.INVALID_HEADER_PADDING)
    _assert_header_error("{} {}", SafeTensorErrorKind.INVALID_HEADER_PADDING)
    _assert_header_error(
        '{"__metadata__":{"version":1}}',
        SafeTensorErrorKind.INVALID_FIELD_TYPE,
    )
    _assert_header_error(
        '{"a":{"dtype":"U8","shape":[1],"data_offsets":[0,1],"extra":null}}',
        SafeTensorErrorKind.UNKNOWN_FIELD,
    )
    _assert_header_error(
        '{"a":{"dtype":"U8","shape":[1]}}',
        SafeTensorErrorKind.MISSING_FIELD,
    )
    _assert_header_error(
        '{"a":{"dtype":"U8","shape":[1],"data_offsets":[0]}}',
        SafeTensorErrorKind.INVALID_OFFSETS,
    )


def test_invalid_strings_and_utf8_are_controlled() raises:
    _assert_header_error(
        '{"\\x":' + _descriptor() + "}",
        SafeTensorErrorKind.INVALID_JSON,
    )
    _assert_header_error(
        '{"\\ud800":' + _descriptor() + "}",
        SafeTensorErrorKind.INVALID_JSON,
    )
    _assert_header_error(
        '{"__metadata__":{"control":"line\nbreak"}}',
        SafeTensorErrorKind.INVALID_JSON,
    )

    var bad_utf8 = [
        Byte(0x7B),
        Byte(0x22),
        Byte(0xFF),
        Byte(0x22),
        Byte(0x3A),
        Byte(0x7B),
        Byte(0x7D),
        Byte(0x7D),
    ]
    var raised = False
    try:
        _ = parse_raw_header(bad_utf8)
    except error:
        raised = True
        assert_equal(error.kind, SafeTensorErrorKind.INVALID_UTF8)
    assert_true(raised)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
