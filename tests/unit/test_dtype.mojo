"""SafeDType wire-model tests."""

from std.testing import (
    TestSuite,
    assert_equal,
    assert_true,
)

from safetensors.errors import SafeTensorErrorKind
from safetensors.format.dtype import SafeDType


def _check_dtype(
    wire_name: String,
    expected: SafeDType,
    bits: UInt64,
    alignment: UInt64,
) raises:
    var dtype = SafeDType.from_wire_name(wire_name)
    assert_equal(dtype, expected)
    assert_equal(dtype.wire_name(), wire_name)
    assert_equal(dtype.bits_per_element(), bits)
    assert_equal(dtype.required_alignment(), alignment)
    assert_equal(dtype.is_byte_aligned(), bits % 8 == 0)


def test_all_wire_dtypes() raises:
    _check_dtype("BOOL", SafeDType.BOOL, 8, 1)
    _check_dtype("F4", SafeDType.F4, 4, 1)
    _check_dtype("F6_E2M3", SafeDType.F6_E2M3, 6, 1)
    _check_dtype("F6_E3M2", SafeDType.F6_E3M2, 6, 1)
    _check_dtype("U8", SafeDType.U8, 8, 1)
    _check_dtype("I8", SafeDType.I8, 8, 1)
    _check_dtype("F8_E5M2", SafeDType.F8_E5M2, 8, 1)
    _check_dtype("F8_E4M3", SafeDType.F8_E4M3, 8, 1)
    _check_dtype("F8_E8M0", SafeDType.F8_E8M0, 8, 1)
    _check_dtype("F8_E4M3FNUZ", SafeDType.F8_E4M3FNUZ, 8, 1)
    _check_dtype("F8_E5M2FNUZ", SafeDType.F8_E5M2FNUZ, 8, 1)
    _check_dtype("I16", SafeDType.I16, 16, 2)
    _check_dtype("U16", SafeDType.U16, 16, 2)
    _check_dtype("F16", SafeDType.F16, 16, 2)
    _check_dtype("BF16", SafeDType.BF16, 16, 2)
    _check_dtype("I32", SafeDType.I32, 32, 4)
    _check_dtype("U32", SafeDType.U32, 32, 4)
    _check_dtype("F32", SafeDType.F32, 32, 4)
    _check_dtype("C64", SafeDType.C64, 64, 8)
    _check_dtype("F64", SafeDType.F64, 64, 8)
    _check_dtype("I64", SafeDType.I64, 64, 8)
    _check_dtype("U64", SafeDType.U64, 64, 8)
    assert_true(not SafeDType.F4.is_byte_aligned())


def test_unknown_dtype_is_typed() raises:
    var raised = False
    try:
        _ = SafeDType.from_wire_name("f32")
    except error:
        raised = True
        assert_equal(error.kind, SafeTensorErrorKind.UNSUPPORTED_DTYPE)
    assert_true(raised)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
