"""Safe native typed-view integration tests."""

from std.os import remove
from std.pathlib import Path
from std.tempfile import NamedTemporaryFile
from std.testing import TestSuite, assert_equal, assert_true

from safetensors import (
    MappedSafeTensorFile,
    SafeTensorErrorKind,
    map_safetensors,
)


def _fixture(name: String) -> String:
    return String(
        Path("fixtures").joinpath("valid").joinpath(name + ".safetensors")
    )


def _temporary_copy(source: String) raises -> String:
    var temporary = NamedTemporaryFile(mode="w", delete=False)
    var temporary_path = temporary.name.copy()
    temporary.write_bytes(Path(source).read_bytes())
    temporary.close()
    return temporary_path^


def _assert_view_error[
    dtype: DType,
](
    mapped: MappedSafeTensorFile,
    name: String,
    expected: SafeTensorErrorKind,
) raises:
    var raised = False
    try:
        _ = mapped.tensor_view[dtype](name)
    except error:
        raised = True
        assert_equal(error.kind, expected)
    assert_true(raised)


def test_native_float_integer_and_repeated_views() raises:
    var mapped = map_safetensors(_fixture("reference_f32"))
    var raw = mapped.tensor_bytes("weights")
    var values = mapped.tensor_view[DType.float32]("weights")
    var repeated = mapped.tensor_view[DType.float32]("weights")

    assert_equal(len(raw), 16)
    assert_equal(len(values), 4)
    assert_equal(values[0], Float32(1.0))
    assert_equal(values[1], Float32(-2.0))
    assert_equal(values[2], Float32(3.5))
    assert_equal(values[3], Float32(4.25))
    assert_equal(repeated[2], Float32(3.5))
    assert_equal(raw[0], UInt8(0))

    var unicode = map_safetensors(_fixture("unicode"))
    var halves = unicode.tensor_view[DType.float16]("wëight😊")
    assert_equal(len(halves), 2)
    assert_equal(halves[0], Float16(1.0))
    assert_equal(halves[1], Float16(-2.0))

    var reordered = map_safetensors(_fixture("reordered_offsets"))
    var integers = reordered.tensor_view[DType.int8]("first")
    assert_equal(len(integers), 4)
    assert_equal(integers[0], Int8(-128))
    assert_equal(integers[1], Int8(0))
    assert_equal(integers[2], Int8(1))
    assert_equal(integers[3], Int8(127))


def test_aligned_scalar_view() raises:
    var mapped = map_safetensors(_fixture("aligned_scalar_i64"))
    var metadata = mapped.metadata()
    assert_equal(metadata.data_start() % 8, UInt64(0))
    assert_equal(len(metadata.info("scalar").shape), 0)

    var scalar = mapped.tensor_view[DType.int64]("scalar")
    assert_equal(len(scalar), 1)
    assert_equal(scalar[0], Int64(-42))


def test_float8_wire_encodings_have_exact_native_views() raises:
    var mapped = map_safetensors(_fixture("float8_scalars"))

    var e5m2 = mapped.tensor_view[DType.float8_e5m2]("e5m2")
    var e4m3 = mapped.tensor_view[DType.float8_e4m3fn]("e4m3")
    var e8m0 = mapped.tensor_view[DType.float8_e8m0fnu]("e8m0")
    var e4m3fnuz = mapped.tensor_view[DType.float8_e4m3fnuz]("e4m3fnuz")
    var e5m2fnuz = mapped.tensor_view[DType.float8_e5m2fnuz]("e5m2fnuz")

    assert_equal(len(e5m2), 1)
    assert_equal(Float32(e5m2[0]), Float32(1.0))
    assert_equal(Float32(e4m3[0]), Float32(1.0))
    assert_equal(Float32(e8m0[0]), Float32(1.0))
    assert_equal(Float32(e4m3fnuz[0]), Float32(1.0))
    assert_equal(Float32(e5m2fnuz[0]), Float32(1.0))


def test_dtype_failures_are_typed() raises:
    var mapped = map_safetensors(_fixture("reference_f32"))
    _assert_view_error[DType.uint32](
        mapped, "weights", SafeTensorErrorKind.DTYPE_MISMATCH
    )
    _assert_view_error[DType.bool](
        mapped, "weights", SafeTensorErrorKind.UNSUPPORTED_DTYPE
    )
    _assert_view_error[DType.float32](
        mapped, "missing", SafeTensorErrorKind.TENSOR_NOT_FOUND
    )

    var subbyte = map_safetensors(_fixture("subbyte"))
    _assert_view_error[DType.uint8](
        subbyte, "f4", SafeTensorErrorKind.UNSUPPORTED_DTYPE
    )

    var boolean = map_safetensors(_fixture("space_padding"))
    _assert_view_error[DType.uint8](
        boolean, "padded", SafeTensorErrorKind.UNSUPPORTED_DTYPE
    )


def test_actual_address_alignment_and_raw_fallback() raises:
    var two_byte = map_safetensors(_fixture("reordered_offsets"))
    var two_info = two_byte.metadata().info("second")
    assert_true((two_byte.metadata().data_start() + two_info.begin) % 2 != 0)
    _assert_view_error[DType.uint16](
        two_byte, "second", SafeTensorErrorKind.MISALIGNED_TENSOR
    )
    assert_equal(len(two_byte.tensor_bytes("second")), 2)

    var four_byte = map_safetensors(_fixture("high_rank"))
    assert_true(four_byte.metadata().data_start() % 4 != 0)
    _assert_view_error[DType.float32](
        four_byte, "rank16", SafeTensorErrorKind.MISALIGNED_TENSOR
    )
    assert_equal(len(four_byte.tensor_bytes("rank16")), 4)

    var eight_byte = map_safetensors(_fixture("scalar_i64"))
    assert_true(eight_byte.metadata().data_start() % 8 != 0)
    _assert_view_error[DType.int64](
        eight_byte, "scalar", SafeTensorErrorKind.MISALIGNED_TENSOR
    )
    assert_equal(len(eight_byte.tensor_bytes("scalar")), 8)


def test_empty_typed_views_skip_address_constraints() raises:
    var zero_dimension = map_safetensors(_fixture("zero_dimension"))
    var empty = zero_dimension.tensor_view[DType.float32]("empty")
    assert_equal(len(empty), 0)
    _assert_view_error[DType.int32](
        zero_dimension, "empty", SafeTensorErrorKind.DTYPE_MISMATCH
    )

    var boundaries = map_safetensors(_fixture("multiple_empty_boundaries"))
    var metadata = boundaries.metadata()
    var before_info = metadata.info("empty_before")
    var after_info = metadata.info("empty_after")
    assert_true((metadata.data_start() + before_info.begin) % 4 != 0)
    assert_true((metadata.data_start() + after_info.begin) % 2 != 0)

    var before = boundaries.tensor_view[DType.float32]("empty_before")
    var after = boundaries.tensor_view[DType.int16]("empty_after")
    assert_equal(len(before), 0)
    assert_equal(len(after), 0)


def test_file_growth_before_typed_view_is_io_error() raises:
    var temporary_path = _temporary_copy(_fixture("reference_f32"))
    var mapped = map_safetensors(temporary_path)

    var appender = open(temporary_path, "a")
    var extra = [UInt8(0)]
    appender.write_all(extra)
    appender.close()

    _assert_view_error[DType.float32](
        mapped, "weights", SafeTensorErrorKind.IO_ERROR
    )
    remove(temporary_path)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
