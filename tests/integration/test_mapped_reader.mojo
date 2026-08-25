"""Memory-mapped reader integration tests."""

from std.os import remove
from std.pathlib import Path
from std.tempfile import NamedTemporaryFile
from std.testing import TestSuite, assert_equal, assert_true

from safetensors import (
    SafeDType,
    SafeTensorErrorKind,
    map_safetensors,
)
from safetensors.io.mapped_reader import _ReadOnlyMapping


def _fixture(group: String, name: String) -> String:
    return String(
        Path("fixtures").joinpath(group).joinpath(name + ".safetensors")
    )


def _temporary_copy(source: String) raises -> String:
    var temporary = NamedTemporaryFile(mode="w", delete=False)
    var temporary_path = temporary.name.copy()
    temporary.write_bytes(Path(source).read_bytes())
    temporary.close()
    return temporary_path^


def _proc_maps_contains(path: String) raises -> Bool:
    var bytes = Path("/proc/self/maps").read_bytes()
    return path in String(from_utf8=Span(bytes))


def _assert_mapping_visible_in_scope(path: String) raises:
    var mapped = map_safetensors(path)
    var view = mapped.tensor_bytes("weights")
    assert_true(_proc_maps_contains(path))
    assert_equal(view[0], UInt8(0))


def _assert_map_error(
    path: String,
    expected: SafeTensorErrorKind,
) raises:
    var raised = False
    try:
        _ = map_safetensors(path)
    except error:
        raised = True
        assert_equal(error.kind, expected)
    assert_true(raised)


def test_mapped_metadata_and_file_length() raises:
    var path = _fixture("valid", "reference_f32")
    var mapped = map_safetensors(path)
    var metadata = mapped.metadata()

    assert_equal(mapped.file_length(), UInt64(len(Path(path).read_bytes())))
    assert_equal(len(metadata), 1)
    assert_equal(metadata.names()[0], "weights")
    assert_equal(metadata.info("weights").dtype, SafeDType.F32)
    assert_equal(metadata.info("weights").byte_length, UInt64(16))
    assert_equal(metadata.data_length(), UInt64(16))
    assert_true(metadata.data_start() < mapped.file_length())
    assert_equal(
        metadata.metadata_value("producer").value(),
        "Python safetensors reference",
    )


def test_empty_archive_maps() raises:
    var mapped = map_safetensors(_fixture("valid", "empty_archive"))
    var metadata = mapped.metadata()

    assert_equal(len(metadata), 0)
    assert_equal(metadata.data_length(), UInt64(0))
    assert_equal(metadata.data_start(), mapped.file_length())


def test_exact_reordered_and_repeated_views() raises:
    var mapped = map_safetensors(_fixture("valid", "reordered_offsets"))

    var second = mapped.tensor_bytes("second")
    assert_equal(len(second), 2)
    assert_equal(second[0], UInt8(0x34))
    assert_equal(second[1], UInt8(0x12))

    var first = mapped.tensor_bytes("first")
    assert_equal(len(first), 4)
    assert_equal(first[0], UInt8(0x80))
    assert_equal(first[1], UInt8(0x00))
    assert_equal(first[2], UInt8(0x01))
    assert_equal(first[3], UInt8(0x7F))

    var repeated = mapped.tensor_bytes("first")
    assert_equal(len(repeated), 4)
    assert_equal(repeated[0], UInt8(0x80))
    assert_equal(repeated[1], UInt8(0x00))
    assert_equal(repeated[2], UInt8(0x01))
    assert_equal(repeated[3], UInt8(0x7F))


def test_scalar_view() raises:
    var mapped = map_safetensors(_fixture("valid", "scalar_i64"))
    var scalar = mapped.tensor_bytes("scalar")

    assert_equal(len(scalar), 8)
    assert_equal(scalar[0], UInt8(0xD6))
    for index in range(1, 8):
        assert_equal(scalar[index], UInt8(0xFF))


def test_zero_byte_views() raises:
    var zero_dimension = map_safetensors(_fixture("valid", "zero_dimension"))
    var empty = zero_dimension.tensor_bytes("empty")
    assert_equal(len(empty), 0)

    var boundaries = map_safetensors(
        _fixture("valid", "multiple_empty_boundaries")
    )
    var empty_before = boundaries.tensor_bytes("empty_before")
    var payload = boundaries.tensor_bytes("payload")
    var empty_after = boundaries.tensor_bytes("empty_after")

    assert_equal(len(empty_before), 0)
    assert_equal(len(payload), 3)
    assert_equal(payload[0], UInt8(1))
    assert_equal(payload[1], UInt8(2))
    assert_equal(payload[2], UInt8(3))
    assert_equal(len(empty_after), 0)


def test_missing_tensor_name_is_typed() raises:
    var mapped = map_safetensors(_fixture("valid", "reference_f32"))
    var raised = False
    try:
        _ = mapped.tensor_bytes("missing")
    except error:
        raised = True
        assert_equal(error.kind, SafeTensorErrorKind.TENSOR_NOT_FOUND)
    assert_true(raised)


def test_missing_and_malformed_files_are_typed() raises:
    _assert_map_error(
        "fixtures/does-not-exist.safetensors",
        SafeTensorErrorKind.IO_ERROR,
    )
    _assert_map_error(
        _fixture("malformed", "header_too_small"),
        SafeTensorErrorKind.HEADER_TOO_SMALL,
    )
    _assert_map_error(
        _fixture("malformed", "truncated_header"),
        SafeTensorErrorKind.INVALID_HEADER_LENGTH,
    )
    _assert_map_error(
        _fixture("malformed", "header_too_large"),
        SafeTensorErrorKind.HEADER_TOO_LARGE,
    )
    _assert_map_error(
        _fixture("malformed", "overlapping_ranges"),
        SafeTensorErrorKind.INCOMPLETE_DATA_COVERAGE,
    )


def test_mapping_failure_is_io_error() raises:
    var device = open("/dev/null", "r")
    var raised = False
    try:
        _ = _ReadOnlyMapping(device, 1)
    except error:
        raised = True
        assert_equal(error.kind, SafeTensorErrorKind.IO_ERROR)
    assert_true(raised)


def test_mapping_is_unmapped_at_scope_exit() raises:
    var temporary_path = _temporary_copy(_fixture("valid", "reference_f32"))
    assert_true(not _proc_maps_contains(temporary_path))
    _assert_mapping_visible_in_scope(temporary_path)
    assert_true(not _proc_maps_contains(temporary_path))
    remove(temporary_path)


def test_mapping_respects_custom_header_limit() raises:
    var raised = False
    try:
        _ = map_safetensors(
            _fixture("valid", "reference_f32"), max_header_bytes=1
        )
    except error:
        raised = True
        assert_equal(error.kind, SafeTensorErrorKind.HEADER_TOO_LARGE)
    assert_true(raised)


def test_unlink_does_not_invalidate_mapping() raises:
    var temporary_path = _temporary_copy(_fixture("valid", "reordered_offsets"))
    var mapped = map_safetensors(temporary_path)
    remove(temporary_path)

    var first = mapped.tensor_bytes("first")
    assert_equal(len(first), 4)
    assert_equal(first[0], UInt8(0x80))
    assert_equal(first[1], UInt8(0x00))
    assert_equal(first[2], UInt8(0x01))
    assert_equal(first[3], UInt8(0x7F))


def test_path_replacement_does_not_redirect_mapping() raises:
    var temporary_path = _temporary_copy(_fixture("valid", "reordered_offsets"))
    var replacement = Path(_fixture("valid", "empty_archive")).read_bytes()
    var mapped = map_safetensors(temporary_path)
    remove(temporary_path)
    Path(temporary_path).write_bytes(replacement)

    var first = mapped.tensor_bytes("first")
    assert_equal(len(first), 4)
    assert_equal(first[0], UInt8(0x80))
    assert_equal(first[1], UInt8(0x00))
    assert_equal(first[2], UInt8(0x01))
    assert_equal(first[3], UInt8(0x7F))
    remove(temporary_path)


def test_growth_before_view_is_io_error() raises:
    var temporary_path = _temporary_copy(_fixture("valid", "reference_f32"))
    var mapped = map_safetensors(temporary_path)

    var appender = open(temporary_path, "a")
    var extra = [UInt8(0)]
    appender.write_all(extra)
    appender.close()

    var raised = False
    try:
        _ = mapped.tensor_bytes("weights")
    except error:
        raised = True
        assert_equal(error.kind, SafeTensorErrorKind.IO_ERROR)
    assert_true(raised)
    remove(temporary_path)


def test_truncation_before_view_is_io_error_without_dereference() raises:
    var temporary_path = _temporary_copy(_fixture("valid", "reference_f32"))
    var mapped = map_safetensors(temporary_path)

    var truncator = open(temporary_path, "w")
    truncator.close()
    assert_equal(len(Path(temporary_path).read_bytes()), 0)

    var raised = False
    try:
        _ = mapped.tensor_bytes("weights")
    except error:
        raised = True
        assert_equal(error.kind, SafeTensorErrorKind.IO_ERROR)
    assert_true(raised)
    remove(temporary_path)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
