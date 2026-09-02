"""Platform contract tests for portable I/O primitives."""

from std.ffi import c_int, external_call
from std.os import remove, stat
from std.tempfile import NamedTemporaryFile
from std.testing import (
    TestSuite,
    assert_equal,
    assert_false,
    assert_true,
)

from safetensors.io._atomic_file import _open_exclusive
from safetensors.io._platform import (
    _AT_FDCWD,
    _AT_SYMLINK_NOFOLLOW,
    _GETENTROPY_MAX_BYTES,
    _IS_LINUX,
    _IS_MACOS,
    _IS_X86,
    _O_CLOEXEC,
    _O_CREAT,
    _O_DIRECTORY,
    _O_EXCL,
    _O_NOFOLLOW,
    _O_NONBLOCK,
    _O_RDONLY,
    _O_WRONLY,
    _fill_os_random,
)


def _unused_path() raises -> String:
    var temporary = NamedTemporaryFile(mode="w", delete=False)
    var path = temporary.name.copy()
    temporary.close()
    remove(path)
    return path^


def test_platform_constants_match_the_host_abi() raises:
    assert_true(_IS_LINUX != _IS_MACOS)
    assert_equal(_O_RDONLY, 0x0000)
    assert_equal(_O_WRONLY, 0x0001)
    comptime if _IS_LINUX:
        assert_equal(_AT_FDCWD, -100)
        assert_equal(_AT_SYMLINK_NOFOLLOW, 0x0100)
        assert_equal(_O_CREAT, 0x0040)
        assert_equal(_O_EXCL, 0x0080)
        assert_equal(_O_CLOEXEC, 0x80000)
        assert_equal(_O_NONBLOCK, 0x0800)
        comptime if _IS_X86:
            assert_equal(_O_DIRECTORY, 0x10000)
            assert_equal(_O_NOFOLLOW, 0x20000)
        else:
            # Mojo 1.0 supports AArch64 as its non-x86 Linux target.
            assert_equal(_O_DIRECTORY, 0x4000)
            assert_equal(_O_NOFOLLOW, 0x8000)
    elif _IS_MACOS:
        assert_equal(_AT_FDCWD, -2)
        assert_equal(_AT_SYMLINK_NOFOLLOW, 0x0020)
        assert_equal(_O_CREAT, 0x0200)
        assert_equal(_O_EXCL, 0x0800)
        assert_equal(_O_CLOEXEC, 0x1000000)
        assert_equal(_O_NONBLOCK, 0x0004)
        assert_equal(_O_DIRECTORY, 0x00100000)
        assert_equal(_O_NOFOLLOW, 0x0100)


def test_os_entropy_fills_small_buffers_and_rejects_oversize_requests() raises:
    var bytes = List[UInt8](length=32, fill=0xA5)
    assert_true(_fill_os_random(Span(bytes)))
    var changed = False
    for byte in bytes:
        if byte != 0xA5:
            changed = True
            break
    assert_true(changed)

    var oversized = List[UInt8](length=_GETENTROPY_MAX_BYTES + 1, fill=0)
    assert_false(_fill_os_random(Span(oversized)))


def test_exclusive_open_sets_mode_and_close_on_exec() raises:
    var path = _unused_path()
    var descriptor = _open_exclusive(path)
    assert_true(descriptor >= 0)
    var descriptor_flags = Int(
        external_call["fcntl", c_int, num_fixed_args=2](
            c_int(descriptor), c_int(1)
        )
    )
    var duplicate = _open_exclusive(path)

    var file = FileHandle()
    file.handle = descriptor
    file.close()
    var mode = stat(path).st_mode & 0o777
    remove(path)

    assert_true(descriptor_flags >= 0)
    assert_equal(descriptor_flags & 1, 1)
    assert_true(duplicate < 0)
    assert_equal(mode, 0o600)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
