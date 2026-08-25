from std.os import remove
from std.pathlib import Path
from std.tempfile import NamedTemporaryFile
from std.testing import TestSuite, assert_equal, assert_true

from safetensors import (
    SafeDType,
    SafeTensorErrorKind,
    open_safetensors,
)


def _fixture(group: String, name: String) -> String:
    return String(
        Path("fixtures").joinpath(group).joinpath(name + ".safetensors")
    )


def _assert_open_error(
    path: String,
    expected: SafeTensorErrorKind,
) raises:
    var raised = False
    try:
        _ = open_safetensors(path)
    except error:
        raised = True
        assert_equal(error.kind, expected)
    assert_true(raised)


def test_metadata_only_opening() raises:
    var path = _fixture("valid", "reference_f32")
    var reader = open_safetensors(path)
    var metadata = reader.metadata()

    assert_equal(reader.file_length(), UInt64(len(Path(path).read_bytes())))
    assert_equal(len(metadata), 1)
    assert_equal(metadata.names()[0], "weights")
    assert_equal(metadata.info("weights").dtype, SafeDType.F32)
    assert_equal(metadata.info("weights").byte_length, UInt64(16))
    assert_equal(metadata.data_length(), UInt64(16))
    assert_true(metadata.data_start() < reader.file_length())


def test_opening_preserves_core_error_kinds() raises:
    _assert_open_error(
        _fixture("malformed", "header_too_small"),
        SafeTensorErrorKind.HEADER_TOO_SMALL,
    )
    _assert_open_error(
        _fixture("malformed", "truncated_header"),
        SafeTensorErrorKind.INVALID_HEADER_LENGTH,
    )
    _assert_open_error(
        _fixture("malformed", "header_too_large"),
        SafeTensorErrorKind.HEADER_TOO_LARGE,
    )
    _assert_open_error(
        _fixture("malformed", "overlapping_ranges"),
        SafeTensorErrorKind.INCOMPLETE_DATA_COVERAGE,
    )


def test_missing_file_is_io_error() raises:
    _assert_open_error(
        "fixtures/does-not-exist.safetensors",
        SafeTensorErrorKind.IO_ERROR,
    )


def test_opening_respects_custom_header_limit() raises:
    var raised = False
    try:
        _ = open_safetensors(
            _fixture("valid", "reference_f32"), max_header_bytes=1
        )
    except error:
        raised = True
        assert_equal(error.kind, SafeTensorErrorKind.HEADER_TOO_LARGE)
    assert_true(raised)


def test_exact_reads_are_positional_and_repeatable() raises:
    var reader = open_safetensors(_fixture("valid", "reordered_offsets"))
    var second = List[UInt8](length=2, fill=0)
    reader.read_tensor_into("second", second)
    assert_equal(second, [UInt8(0x34), 0x12])

    var first = List[UInt8](length=4, fill=0)
    reader.read_tensor_into("first", first)
    assert_equal(first, [UInt8(0x80), 0x00, 0x01, 0x7F])

    first = List[UInt8](length=4, fill=0)
    reader.read_tensor_into("first", first)
    assert_equal(first, [UInt8(0x80), 0x00, 0x01, 0x7F])


def test_zero_byte_tensor_reads() raises:
    var reader = open_safetensors(
        _fixture("valid", "multiple_empty_boundaries")
    )
    var destination = List[UInt8]()
    reader.read_tensor_into("empty_before", destination)
    reader.read_tensor_into("empty_after", destination)


def test_owned_tensor_loads() raises:
    var scalar_reader = open_safetensors(_fixture("valid", "scalar_i64"))
    assert_equal(
        scalar_reader.load_tensor("scalar"),
        [
            UInt8(0xD6),
            0xFF,
            0xFF,
            0xFF,
            0xFF,
            0xFF,
            0xFF,
            0xFF,
        ],
    )

    var reordered_reader = open_safetensors(
        _fixture("valid", "reordered_offsets")
    )
    assert_equal(
        reordered_reader.load_tensor("first"),
        [UInt8(0x80), 0x00, 0x01, 0x7F],
    )
    assert_equal(
        reordered_reader.load_tensor("second"),
        [UInt8(0x34), 0x12],
    )

    var empty_reader = open_safetensors(_fixture("valid", "zero_dimension"))
    assert_equal(len(empty_reader.load_tensor("empty")), 0)


def test_read_rejects_missing_tensor_and_wrong_destination_size() raises:
    var reader = open_safetensors(_fixture("valid", "reference_f32"))
    var destination = List[UInt8](length=15, fill=0)
    var raised = False
    try:
        reader.read_tensor_into("weights", destination)
    except error:
        raised = True
        assert_equal(error.kind, SafeTensorErrorKind.DESTINATION_SIZE_MISMATCH)
    assert_true(raised)

    destination = List[UInt8](length=17, fill=0)
    raised = False
    try:
        reader.read_tensor_into("weights", destination)
    except error:
        raised = True
        assert_equal(error.kind, SafeTensorErrorKind.DESTINATION_SIZE_MISMATCH)
    assert_true(raised)

    destination = List[UInt8](length=16, fill=0)
    raised = False
    try:
        reader.read_tensor_into("missing", destination)
    except error:
        raised = True
        assert_equal(error.kind, SafeTensorErrorKind.TENSOR_NOT_FOUND)
    assert_true(raised)


def test_post_open_truncation_is_io_error() raises:
    var contents = Path(_fixture("valid", "reference_f32")).read_bytes()
    var temporary = NamedTemporaryFile(mode="w", delete=False)
    var temporary_path = temporary.name.copy()
    temporary.write_bytes(contents)
    temporary.close()
    var reader = open_safetensors(temporary_path)

    var truncator = open(temporary_path, "w")
    truncator.close()
    assert_equal(len(Path(temporary_path).read_bytes()), 0)

    var destination = List[UInt8](length=16, fill=0)
    var raised = False
    try:
        reader.read_tensor_into("weights", destination)
    except error:
        raised = True
        assert_equal(error.kind, SafeTensorErrorKind.IO_ERROR)
    assert_true(raised)
    remove(temporary_path)


def test_post_open_growth_is_io_error() raises:
    var contents = Path(_fixture("valid", "reference_f32")).read_bytes()
    var temporary = NamedTemporaryFile(mode="w", delete=False)
    var temporary_path = temporary.name.copy()
    temporary.write_bytes(contents)
    temporary.close()
    var reader = open_safetensors(temporary_path)

    var appender = open(temporary_path, "a")
    var extra = [UInt8(0)]
    appender.write_all(extra)
    appender.close()

    var destination = List[UInt8](length=16, fill=0)
    var raised = False
    try:
        reader.read_tensor_into("weights", destination)
    except error:
        raised = True
        assert_equal(error.kind, SafeTensorErrorKind.IO_ERROR)
    assert_true(raised)
    remove(temporary_path)


def test_path_replacement_does_not_redirect_reader() raises:
    var contents = Path(_fixture("valid", "reordered_offsets")).read_bytes()
    var replacement = Path(_fixture("valid", "empty_archive")).read_bytes()
    var temporary = NamedTemporaryFile(mode="w", delete=False)
    var temporary_path = temporary.name.copy()
    temporary.write_bytes(contents)
    temporary.close()

    var reader = open_safetensors(temporary_path)
    remove(temporary_path)
    Path(temporary_path).write_bytes(replacement)

    assert_equal(
        reader.load_tensor("first"),
        [UInt8(0x80), 0x00, 0x01, 0x7F],
    )
    remove(temporary_path)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
