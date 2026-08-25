"""Checked arithmetic for every value controlled by a Safetensors file."""

from .errors import SafeTensorError, SafeTensorErrorKind, make_error


def checked_add_u64(lhs: UInt64, rhs: UInt64) raises SafeTensorError -> UInt64:
    """Adds two UInt64 values or raises ValidationOverflow."""
    if rhs > UInt64.MAX - lhs:
        raise make_error(
            SafeTensorErrorKind.VALIDATION_OVERFLOW,
            "UInt64 addition overflow",
        )
    return lhs + rhs


def checked_sub_u64(lhs: UInt64, rhs: UInt64) raises SafeTensorError -> UInt64:
    """Subtracts two UInt64 values or raises ValidationOverflow."""
    if rhs > lhs:
        raise make_error(
            SafeTensorErrorKind.VALIDATION_OVERFLOW,
            "UInt64 subtraction underflow",
        )
    return lhs - rhs


def checked_mul_u64(lhs: UInt64, rhs: UInt64) raises SafeTensorError -> UInt64:
    """Multiplies two UInt64 values or raises ValidationOverflow."""
    if lhs != 0 and rhs > UInt64.MAX // lhs:
        raise make_error(
            SafeTensorErrorKind.VALIDATION_OVERFLOW,
            "UInt64 multiplication overflow",
        )
    return lhs * rhs


def checked_product_u64(
    values: List[UInt64],
) raises SafeTensorError -> UInt64:
    """Returns the checked product of values, using one as the identity."""
    for value in values:
        if value == 0:
            return 0
    var result: UInt64 = 1
    for value in values:
        result = checked_mul_u64(result, value)
    return result


def checked_u64_to_int(value: UInt64) raises SafeTensorError -> Int:
    """Converts UInt64 to the native index type without truncation."""
    if value > UInt64(Int.MAX):
        raise make_error(
            SafeTensorErrorKind.VALIDATION_OVERFLOW,
            "value does not fit in the native index type",
        )
    return Int(value)


def checked_decimal_append(
    value: UInt64, digit: UInt8
) raises SafeTensorError -> UInt64:
    """Appends one numeric decimal digit without intermediate floating point."""
    if digit > 9:
        raise make_error(
            SafeTensorErrorKind.INVALID_JSON,
            "decimal digit is outside the range 0 through 9",
        )
    return checked_add_u64(checked_mul_u64(value, 10), UInt64(digit))
