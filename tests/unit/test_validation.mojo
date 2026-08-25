"""Validated metadata invariant tests."""

from std.testing import TestSuite, assert_equal, assert_false, assert_true

from safetensors import (
    RawSafeTensorMetadata,
    RawTensorInfo,
    SafeDType,
    SafeTensorErrorKind,
    decode_header_length,
    parse_metadata,
    parse_metadata_from_header,
    validate_metadata,
)


def _raw(tensors: List[RawTensorInfo]) -> RawSafeTensorMetadata:
    return RawSafeTensorMetadata(Dict[String, String](), tensors.copy())


def _assert_validation_error(
    raw: RawSafeTensorMetadata,
    data_length: UInt64,
    expected: SafeTensorErrorKind,
) raises:
    var raised = False
    try:
        _ = validate_metadata(raw, data_length)
    except error:
        raised = True
        assert_equal(error.kind, expected)
    assert_true(raised)


def _complete_file(header: String, data: List[UInt8]) -> List[UInt8]:
    var result = List[UInt8]()
    var header_length = UInt64(header.byte_length())
    for index in range(8):
        result.append(
            UInt8((header_length >> UInt64(index * 8)) & UInt64(0xFF))
        )
    for value in header.as_bytes():
        result.append(value)
    for value in data:
        result.append(value)
    return result^


def test_scalar_zero_shape_and_offset_order() raises:
    var tensors: List[RawTensorInfo] = [
        RawTensorInfo("second", "U16", [UInt64(1)], 4, 6),
        RawTensorInfo("scalar", "F32", List[UInt64](), 0, 4),
        RawTensorInfo("empty", "U64", [UInt64.MAX, 0, UInt64.MAX], 6, 6),
    ]
    var metadata = validate_metadata(_raw(tensors), 6)
    assert_equal(len(metadata), 3)
    assert_false(metadata.is_empty())
    assert_true(metadata.contains("scalar"))
    assert_equal(metadata.offset_names()[0], "scalar")
    assert_equal(metadata.offset_names()[1], "second")
    assert_equal(metadata.offset_names()[2], "empty")

    var scalar = metadata.info("scalar")
    assert_equal(scalar.dtype, SafeDType.F32)
    assert_equal(scalar.element_count, UInt64(1))
    assert_equal(scalar.bit_length, UInt64(32))
    assert_equal(scalar.byte_length, UInt64(4))

    var empty = metadata.info("empty")
    assert_equal(empty.element_count, UInt64(0))
    assert_equal(empty.byte_length, UInt64(0))


def test_validated_metadata_returns_copies() raises:
    var user_metadata = Dict[String, String]()
    user_metadata["producer"] = "test"
    var raw = RawSafeTensorMetadata(
        user_metadata^,
        [RawTensorInfo("value", "U8", [UInt64(1)], 0, 1)],
    )
    var metadata = validate_metadata(raw, 1)

    var names = metadata.names()
    names[0] = "changed"
    assert_equal(metadata.names()[0], "value")

    var info = metadata.info("value")
    info.shape[0] = 99
    assert_equal(metadata.info("value").shape[0], UInt64(1))

    var copied_metadata = metadata.user_metadata()
    copied_metadata["producer"] = "changed"
    assert_equal(metadata.metadata_value("producer").value(), "test")


def test_shape_and_bit_size_failures() raises:
    _assert_validation_error(
        _raw([RawTensorInfo("a", "U8", [UInt64.MAX, 2], 0, 0)]),
        0,
        SafeTensorErrorKind.VALIDATION_OVERFLOW,
    )
    _assert_validation_error(
        _raw([RawTensorInfo("a", "U64", [UInt64.MAX], 0, 0)]),
        0,
        SafeTensorErrorKind.VALIDATION_OVERFLOW,
    )
    _assert_validation_error(
        _raw([RawTensorInfo("a", "F4", [UInt64(1)], 0, 0)]),
        0,
        SafeTensorErrorKind.MISALIGNED_SLICE,
    )
    _assert_validation_error(
        _raw([RawTensorInfo("a", "U16", [UInt64(1)], 0, 1)]),
        1,
        SafeTensorErrorKind.INVALID_TENSOR_SIZE,
    )


def test_offset_and_coverage_failures() raises:
    _assert_validation_error(
        _raw([RawTensorInfo("a", "U8", [UInt64(1)], 1, 0)]),
        1,
        SafeTensorErrorKind.INVALID_OFFSETS,
    )
    _assert_validation_error(
        _raw([RawTensorInfo("a", "U8", [UInt64(2)], 0, 2)]),
        1,
        SafeTensorErrorKind.INVALID_OFFSETS,
    )
    _assert_validation_error(
        _raw(
            [
                RawTensorInfo("a", "U8", [UInt64(1)], 0, 1),
                RawTensorInfo("b", "U8", [UInt64(1)], 2, 3),
            ]
        ),
        3,
        SafeTensorErrorKind.INCOMPLETE_DATA_COVERAGE,
    )
    _assert_validation_error(
        _raw(
            [
                RawTensorInfo("a", "U8", [UInt64(2)], 0, 2),
                RawTensorInfo("b", "U8", [UInt64(2)], 1, 3),
            ]
        ),
        3,
        SafeTensorErrorKind.INCOMPLETE_DATA_COVERAGE,
    )
    _assert_validation_error(
        _raw([RawTensorInfo("a", "U8", [UInt64(1)], 0, 1)]),
        2,
        SafeTensorErrorKind.INCOMPLETE_DATA_COVERAGE,
    )
    _assert_validation_error(
        _raw(
            [
                RawTensorInfo("payload", "U8", [UInt64(2)], 0, 2),
                RawTensorInfo("empty", "U8", [UInt64(0)], 1, 1),
            ]
        ),
        2,
        SafeTensorErrorKind.INCOMPLETE_DATA_COVERAGE,
    )
    _assert_validation_error(
        _raw(List[RawTensorInfo]()),
        1,
        SafeTensorErrorKind.INCOMPLETE_DATA_COVERAGE,
    )


def test_duplicate_and_reserved_names_are_revalidated() raises:
    _assert_validation_error(
        _raw(
            [
                RawTensorInfo("a", "U8", [UInt64(0)], 0, 0),
                RawTensorInfo("a", "U8", [UInt64(0)], 0, 0),
            ]
        ),
        0,
        SafeTensorErrorKind.DUPLICATE_KEY,
    )
    _assert_validation_error(
        _raw([RawTensorInfo("__metadata__", "U8", [UInt64(0)], 0, 0)]),
        0,
        SafeTensorErrorKind.INVALID_METADATA,
    )


def test_length_prefix_and_complete_buffer() raises:
    var explicit_little_endian = [
        UInt8(0x08),
        0x07,
        0x06,
        0x05,
        0x04,
        0x03,
        0x02,
        0x01,
    ]
    assert_equal(
        decode_header_length(explicit_little_endian),
        UInt64(0x0102030405060708),
    )

    var header = '{"value":{"dtype":"U8","shape":[3],"data_offsets":[0,3]}}'
    var file = _complete_file(header, [UInt8(1), 2, 3])
    assert_equal(decode_header_length(file), UInt64(header.byte_length()))
    var metadata = parse_metadata(file)
    assert_equal(metadata.data_start(), UInt64(8 + header.byte_length()))
    assert_equal(metadata.data_length(), UInt64(3))
    assert_equal(metadata.info("value").byte_length, UInt64(3))


def test_isolated_header_compatibility_and_strict_mode() raises:
    var header = "\t{}\n"
    var compatible = parse_metadata_from_header(header.as_bytes(), 0)
    assert_true(compatible.is_empty())

    var raised = False
    try:
        _ = parse_metadata_from_header(header.as_bytes(), 0, strict=True)
    except error:
        raised = True
        assert_equal(error.kind, SafeTensorErrorKind.INVALID_HEADER_START)
    assert_true(raised)


def test_length_prefix_failures() raises:
    var short = [UInt8(0), 0, 0, 0, 0, 0, 0]
    var raised = False
    try:
        _ = parse_metadata(short)
    except error:
        raised = True
        assert_equal(error.kind, SafeTensorErrorKind.HEADER_TOO_SMALL)
    assert_true(raised)

    var small_limit_file = _complete_file("{}", List[UInt8]())
    raised = False
    try:
        _ = parse_metadata(small_limit_file, max_header_bytes=1)
    except error:
        raised = True
        assert_equal(error.kind, SafeTensorErrorKind.HEADER_TOO_LARGE)
    assert_true(raised)

    var truncated = [UInt8(20), 0, 0, 0, 0, 0, 0, 0, 0x7B, 0x7D]
    raised = False
    try:
        _ = parse_metadata(truncated)
    except error:
        raised = True
        assert_equal(error.kind, SafeTensorErrorKind.INVALID_HEADER_LENGTH)
    assert_true(raised)

    var overflowing = List[UInt8]()
    for _ in range(8):
        overflowing.append(0xFF)
    raised = False
    try:
        _ = parse_metadata(overflowing, max_header_bytes=UInt64.MAX)
    except error:
        raised = True
        assert_equal(error.kind, SafeTensorErrorKind.INVALID_HEADER_LENGTH)
    assert_true(raised)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
