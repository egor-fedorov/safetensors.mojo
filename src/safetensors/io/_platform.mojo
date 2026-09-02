"""Internal compile-time platform values and portable OS primitives."""

from std.ffi import c_int, c_size_t, external_call
from std.sys import CompilationTarget
from std.sys.info import platform_map


comptime _IS_LINUX = CompilationTarget.is_linux()
comptime _IS_MACOS = CompilationTarget.is_macos()
comptime _IS_X86 = CompilationTarget.is_x86()

comptime _O_RDONLY = 0x0000
comptime _O_WRONLY = 0x0001
comptime _O_CREAT = platform_map[T=Int, "O_CREAT", linux=0x0040, macos=0x0200]()
comptime _O_EXCL = platform_map[T=Int, "O_EXCL", linux=0x0080, macos=0x0800]()
comptime _O_CLOEXEC = platform_map[
    T=Int, "O_CLOEXEC", linux=0x80000, macos=0x1000000
]()
comptime _O_NONBLOCK = platform_map[
    T=Int, "O_NONBLOCK", linux=0x0800, macos=0x0004
]()
comptime _O_DIRECTORY = platform_map[
    T=Int,
    "O_DIRECTORY",
    linux=(0x10000 if _IS_X86 else 0x4000),
    macos=0x00100000,
]()
comptime _O_NOFOLLOW = platform_map[
    T=Int,
    "O_NOFOLLOW",
    linux=(0x20000 if _IS_X86 else 0x8000),
    macos=0x0100,
]()
comptime _AT_FDCWD = platform_map[T=Int, "AT_FDCWD", linux=-100, macos=-2]()
comptime _AT_SYMLINK_NOFOLLOW = platform_map[
    T=Int, "AT_SYMLINK_NOFOLLOW", linux=0x0100, macos=0x0020
]()

comptime _GETENTROPY_MAX_BYTES = 256


def _fill_os_random[
    origin: MutOrigin
](destination: Span[UInt8, origin]) -> Bool:
    """Fills one small span from the operating system's entropy source."""
    if len(destination) == 0:
        return True
    if len(destination) > _GETENTROPY_MAX_BYTES:
        return False
    return (
        external_call["getentropy", c_int](
            destination.unsafe_ptr(), c_size_t(len(destination))
        )
        == 0
    )
