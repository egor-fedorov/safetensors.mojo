"""Checked-arithmetic boundary tests."""

from std.testing import TestSuite, assert_equal, assert_true

from safetensors.format.checked import (
    checked_add_u64,
    checked_decimal_append,
    checked_mul_u64,
    checked_product_u64,
    checked_sub_u64,
    checked_u64_to_int,
)
from safetensors.errors import SafeTensorErrorKind


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


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
