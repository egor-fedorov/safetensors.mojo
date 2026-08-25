from std.testing import (
    TestSuite,
    assert_equal,
    assert_false,
    assert_true,
)

from safetensors.checked import (
    checked_add_u64,
    checked_decimal_append,
    checked_mul_u64,
    checked_product_u64,
    checked_sub_u64,
    checked_u64_to_int,
)
from safetensors.dtype import SafeDType
from safetensors.errors import SafeTensorErrorKind


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


def test_unknown_dtype_is_typed() raises:
    var raised = False
    try:
        _ = SafeDType.from_wire_name("f32")
    except error:
        raised = True
        assert_equal(error.kind, SafeTensorErrorKind.UNSUPPORTED_DTYPE)
    assert_true(raised)


def test_checked_arithmetic_boundaries() raises:
    assert_equal(checked_add_u64(UInt64.MAX - 1, 1), UInt64.MAX)
    assert_equal(checked_sub_u64(1, 1), UInt64(0))
    assert_equal(checked_mul_u64(UInt64.MAX, 1), UInt64.MAX)
    assert_equal(checked_product_u64(List[UInt64]()), UInt64(1))
    assert_equal(
        checked_product_u64([UInt64.MAX, UInt64(0), UInt64.MAX]),
        UInt64(0),
    )

    var add_raised = False
    try:
        _ = checked_add_u64(UInt64.MAX, 1)
    except error:
        add_raised = True
        assert_equal(error.kind, SafeTensorErrorKind.VALIDATION_OVERFLOW)
    assert_true(add_raised)

    var sub_raised = False
    try:
        _ = checked_sub_u64(0, 1)
    except error:
        sub_raised = True
        assert_equal(error.kind, SafeTensorErrorKind.VALIDATION_OVERFLOW)
    assert_true(sub_raised)

    var mul_raised = False
    try:
        _ = checked_mul_u64(UInt64.MAX, 2)
    except error:
        mul_raised = True
        assert_equal(error.kind, SafeTensorErrorKind.VALIDATION_OVERFLOW)
    assert_true(mul_raised)


def test_checked_native_conversion() raises:
    assert_equal(checked_u64_to_int(UInt64(Int.MAX)), Int.MAX)
    var raised = False
    try:
        _ = checked_u64_to_int(UInt64(Int.MAX) + 1)
    except error:
        raised = True
        assert_equal(error.kind, SafeTensorErrorKind.VALIDATION_OVERFLOW)
    assert_true(raised)


def test_checked_decimal_append() raises:
    var value: UInt64 = 0
    for digit in [UInt8(9), 0, 0, 7, 1, 9, 9, 2, 5, 4, 7, 4, 0, 9, 9, 3]:
        value = checked_decimal_append(value, digit)
    assert_equal(value, UInt64(9007199254740993))

    var raised = False
    try:
        _ = checked_decimal_append(UInt64.MAX, 9)
    except error:
        raised = True
        assert_equal(error.kind, SafeTensorErrorKind.VALIDATION_OVERFLOW)
    assert_true(raised)

    assert_false(SafeDType.F4.is_byte_aligned())


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
