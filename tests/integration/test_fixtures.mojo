"""Conformance fixture integration tests."""

from std.pathlib import Path
from std.testing import TestSuite, assert_equal, assert_true

from safetensors import SafeDType, SafeTensorErrorKind, parse_metadata


def _read_fixture(group: String, name: String) raises -> List[UInt8]:
    return (
        Path("fixtures")
        .joinpath(group)
        .joinpath(name + ".safetensors")
        .read_bytes()
    )


def _assert_malformed(name: String, expected: SafeTensorErrorKind) raises:
    var contents = _read_fixture("malformed", name)
    var raised = False
    try:
        _ = parse_metadata(contents)
    except error:
        raised = True
        assert_equal(error.kind, expected)
    assert_true(raised)


def test_every_valid_fixture_parses() raises:
    for name in [
        "aligned_scalar_i64",
        "canonical_writer",
        "empty_archive",
        "float8_scalars",
        "high_rank",
        "metadata_only",
        "multiple_empty_boundaries",
        "reference_f32",
        "reordered_offsets",
        "scalar_i64",
        "space_padding",
        "subbyte",
        "unicode",
        "zero_dimension",
    ]:
        _ = parse_metadata(_read_fixture("valid", name))


def test_reference_and_specialized_valid_fixtures() raises:
    var reference = parse_metadata(_read_fixture("valid", "reference_f32"))
    var weights = reference.info("weights")
    assert_equal(weights.dtype, SafeDType.F32)
    assert_equal(weights.shape[0], UInt64(2))
    assert_equal(weights.shape[1], UInt64(2))
    assert_equal(weights.byte_length, UInt64(16))

    var reordered = parse_metadata(_read_fixture("valid", "reordered_offsets"))
    assert_equal(reordered.offset_names()[0], "first")
    assert_equal(reordered.offset_names()[1], "second")

    var subbyte = parse_metadata(_read_fixture("valid", "subbyte"))
    assert_equal(subbyte.info("f4").byte_length, UInt64(1))
    assert_equal(subbyte.info("f6").byte_length, UInt64(3))

    var unicode = parse_metadata(_read_fixture("valid", "unicode"))
    assert_true(unicode.contains("wëight😊"))
    assert_equal(unicode.metadata_value("author").value(), "Zoë")


def test_malformed_header_fixtures() raises:
    _assert_malformed("header_too_small", SafeTensorErrorKind.HEADER_TOO_SMALL)
    _assert_malformed("header_too_large", SafeTensorErrorKind.HEADER_TOO_LARGE)
    _assert_malformed(
        "truncated_header", SafeTensorErrorKind.INVALID_HEADER_LENGTH
    )
    _assert_malformed(
        "invalid_header_start", SafeTensorErrorKind.INVALID_HEADER_START
    )
    _assert_malformed("invalid_utf8", SafeTensorErrorKind.INVALID_UTF8)
    _assert_malformed("invalid_json", SafeTensorErrorKind.INVALID_JSON)
    _assert_malformed(
        "trailing_non_space", SafeTensorErrorKind.INVALID_HEADER_PADDING
    )
    _assert_malformed(
        "trailing_second_value", SafeTensorErrorKind.INVALID_HEADER_PADDING
    )


def test_malformed_duplicate_fixtures() raises:
    _assert_malformed("duplicate_top_level", SafeTensorErrorKind.DUPLICATE_KEY)
    _assert_malformed(
        "duplicate_top_level_decoded", SafeTensorErrorKind.DUPLICATE_KEY
    )
    _assert_malformed(
        "duplicate_metadata_decoded", SafeTensorErrorKind.DUPLICATE_KEY
    )
    _assert_malformed(
        "duplicate_descriptor_decoded", SafeTensorErrorKind.DUPLICATE_KEY
    )


def test_malformed_schema_and_integer_fixtures() raises:
    _assert_malformed(
        "metadata_is_not_object", SafeTensorErrorKind.INVALID_METADATA
    )
    _assert_malformed(
        "metadata_value_not_string", SafeTensorErrorKind.INVALID_FIELD_TYPE
    )
    _assert_malformed("missing_dtype", SafeTensorErrorKind.MISSING_FIELD)
    _assert_malformed("missing_shape", SafeTensorErrorKind.MISSING_FIELD)
    _assert_malformed("missing_offsets", SafeTensorErrorKind.MISSING_FIELD)
    _assert_malformed(
        "unknown_descriptor_field", SafeTensorErrorKind.UNKNOWN_FIELD
    )
    _assert_malformed(
        "wrong_dtype_type", SafeTensorErrorKind.INVALID_FIELD_TYPE
    )
    _assert_malformed("negative_integer", SafeTensorErrorKind.INVALID_SHAPE)
    _assert_malformed("fractional_integer", SafeTensorErrorKind.INVALID_SHAPE)
    _assert_malformed("exponent_integer", SafeTensorErrorKind.INVALID_SHAPE)
    _assert_malformed("leading_zero_integer", SafeTensorErrorKind.INVALID_SHAPE)
    _assert_malformed(
        "overflowing_integer", SafeTensorErrorKind.VALIDATION_OVERFLOW
    )
    _assert_malformed("unknown_dtype", SafeTensorErrorKind.UNSUPPORTED_DTYPE)
    _assert_malformed(
        "reserved_name_conflict", SafeTensorErrorKind.INVALID_FIELD_TYPE
    )


def test_malformed_validation_fixtures() raises:
    _assert_malformed(
        "shape_product_overflow", SafeTensorErrorKind.VALIDATION_OVERFLOW
    )
    _assert_malformed(
        "bit_length_overflow", SafeTensorErrorKind.VALIDATION_OVERFLOW
    )
    _assert_malformed(
        "subbyte_not_byte_aligned", SafeTensorErrorKind.INVALID_TENSOR_SIZE
    )
    _assert_malformed("begin_after_end", SafeTensorErrorKind.INVALID_OFFSETS)
    _assert_malformed(
        "offset_outside_data", SafeTensorErrorKind.INVALID_OFFSETS
    )
    _assert_malformed(
        "wrong_tensor_size", SafeTensorErrorKind.INVALID_TENSOR_SIZE
    )
    _assert_malformed(
        "overlapping_ranges", SafeTensorErrorKind.INCOMPLETE_DATA_COVERAGE
    )
    _assert_malformed(
        "gap_between_ranges", SafeTensorErrorKind.INCOMPLETE_DATA_COVERAGE
    )
    _assert_malformed(
        "trailing_unindexed_data",
        SafeTensorErrorKind.INCOMPLETE_DATA_COVERAGE,
    )
    _assert_malformed(
        "data_without_tensors", SafeTensorErrorKind.INCOMPLETE_DATA_COVERAGE
    )


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
