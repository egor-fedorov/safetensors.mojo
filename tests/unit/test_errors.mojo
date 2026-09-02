"""Stable diagnostic spelling tests for sharding error categories."""

from std.testing import TestSuite, assert_equal

from safetensors import SafeTensorErrorKind


def test_sharding_error_codes_are_stable() raises:
    assert_equal(
        SafeTensorErrorKind.PATH_TRAVERSAL, SafeTensorErrorKind(UInt8(22))
    )
    assert_equal(
        SafeTensorErrorKind.INDEX_TOO_LARGE, SafeTensorErrorKind(UInt8(26))
    )
    assert_equal(
        SafeTensorErrorKind.INVALID_INDEX, SafeTensorErrorKind(UInt8(27))
    )
    assert_equal(
        SafeTensorErrorKind.SHARD_MISMATCH, SafeTensorErrorKind(UInt8(28))
    )
    assert_equal(
        SafeTensorErrorKind.TOTAL_SIZE_MISMATCH,
        SafeTensorErrorKind(UInt8(29)),
    )
    assert_equal(
        SafeTensorErrorKind.SHARD_LIMIT_EXCEEDED,
        SafeTensorErrorKind(UInt8(30)),
    )
    assert_equal(
        SafeTensorErrorKind.INDEX_ENTRY_LIMIT_EXCEEDED,
        SafeTensorErrorKind(UInt8(31)),
    )
    assert_equal(SafeTensorErrorKind.PATH_TRAVERSAL.code(), "PathTraversal")
    assert_equal(SafeTensorErrorKind.INDEX_TOO_LARGE.code(), "IndexTooLarge")
    assert_equal(SafeTensorErrorKind.INVALID_INDEX.code(), "InvalidIndex")
    assert_equal(SafeTensorErrorKind.SHARD_MISMATCH.code(), "ShardMismatch")
    assert_equal(
        SafeTensorErrorKind.TOTAL_SIZE_MISMATCH.code(), "TotalSizeMismatch"
    )
    assert_equal(
        SafeTensorErrorKind.SHARD_LIMIT_EXCEEDED.code(),
        "ShardLimitExceeded",
    )
    assert_equal(
        SafeTensorErrorKind.INDEX_ENTRY_LIMIT_EXCEEDED.code(),
        "IndexEntryLimitExceeded",
    )


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
