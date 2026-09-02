"""Platform ABI tests for descriptor-relative shard resolution."""

from std.os import remove, stat
from std.sys import CompilationTarget
from std.tempfile import NamedTemporaryFile
from std.testing import TestSuite, assert_equal, assert_true

from safetensors.errors import SafeTensorErrorKind
from safetensors.sharding._file_status import (
    _darwin_state_from_status,
    _descriptor_state,
)


def _darwin_status(mode: UInt16, length: UInt64) -> List[UInt64]:
    var status = List[UInt64](length=18, fill=0)
    status[0] = UInt64(0x12345678) | (UInt64(mode) << 32)
    status[1] = UInt64(0x1020304050607080)
    status[10] = UInt64(1_725_000_000)
    status[11] = UInt64(987_654_321)
    status[12] = length
    return status^


def test_darwin_stat_fields_use_the_arm64_abi_offsets() raises:
    var status = _darwin_status(UInt16(0o100600), UInt64(4096))
    var state = _darwin_state_from_status(status, SafeTensorErrorKind.IO_ERROR)

    assert_equal(state.identity.device_major, UInt32(0x12345678))
    assert_equal(state.identity.device_minor, UInt32(0))
    assert_equal(state.identity.inode, UInt64(0x1020304050607080))
    assert_true(state.identity.has_birth_time)
    assert_equal(state.identity.birth_time_seconds, UInt64(1_725_000_000))
    assert_equal(state.identity.birth_time_nanoseconds, UInt32(987_654_321))
    assert_equal(state.length, UInt64(4096))


def test_darwin_stat_rejects_symlinks_with_the_index_hint() raises:
    var status = _darwin_status(UInt16(0o120777), UInt64(0))
    var raised = False
    try:
        _ = _darwin_state_from_status(
            status,
            SafeTensorErrorKind.PATH_TRAVERSAL,
            index_symlink_hint=True,
        )
    except error:
        raised = True
        assert_equal(error.kind, SafeTensorErrorKind.PATH_TRAVERSAL)
    assert_true(raised)


def test_darwin_stat_rejects_negative_lengths_and_invalid_times() raises:
    var negative = _darwin_status(UInt16(0o100600), UInt64.MAX)
    var raised = False
    try:
        _ = _darwin_state_from_status(negative, SafeTensorErrorKind.IO_ERROR)
    except error:
        raised = True
        assert_equal(error.kind, SafeTensorErrorKind.IO_ERROR)
    assert_true(raised)

    var invalid_time = _darwin_status(UInt16(0o100600), UInt64(1))
    invalid_time[11] = UInt64(1_000_000_000)
    raised = False
    try:
        _ = _darwin_state_from_status(
            invalid_time, SafeTensorErrorKind.IO_ERROR
        )
    except error:
        raised = True
        assert_equal(error.kind, SafeTensorErrorKind.IO_ERROR)
    assert_true(raised)


def test_native_descriptor_state_matches_public_stat() raises:
    var temporary = NamedTemporaryFile(mode="w", delete=False)
    var path = temporary.name.copy()
    var contents = [UInt8(1), 2, 3, 4, 5]
    temporary.write_bytes(contents)
    temporary.close()

    var public_state = stat(path)
    var file = open(path, "r")
    var descriptor_state = _descriptor_state(file, SafeTensorErrorKind.IO_ERROR)
    file.close()
    remove(path)

    comptime if CompilationTarget.is_macos():
        assert_equal(
            Int(descriptor_state.identity.device_major), public_state.st_dev
        )
    assert_equal(Int(descriptor_state.identity.inode), public_state.st_ino)
    assert_equal(Int(descriptor_state.length), public_state.st_size)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
