"""Platform file-identity queries for safe sharded archive resolution."""

from std.ffi import c_int, external_call
from std.stat import S_ISLNK, S_ISREG

from safetensors.errors import SafeTensorError, SafeTensorErrorKind, make_error
from safetensors.io._platform import (
    _AT_SYMLINK_NOFOLLOW,
    _IS_LINUX,
    _IS_MACOS,
)


comptime _LINUX_AT_EMPTY_PATH = 0x1000
comptime _LINUX_STATX_BASIC_STATS = 0x07FF
comptime _LINUX_STATX_BTIME = 0x0800
comptime _LINUX_STATX_REQUIRED = 0x0301
comptime _LINUX_STATX_BUFFER_WORDS = 32
comptime _LINUX_STATX_MASK_OFFSET = 0
comptime _LINUX_STATX_MODE_OFFSET = 28
comptime _LINUX_STATX_INODE_OFFSET = 32
comptime _LINUX_STATX_SIZE_OFFSET = 40
comptime _LINUX_STATX_BTIME_SECONDS_OFFSET = 80
comptime _LINUX_STATX_BTIME_NANOSECONDS_OFFSET = 88
comptime _LINUX_STATX_DEVICE_MAJOR_OFFSET = 136
comptime _LINUX_STATX_DEVICE_MINOR_OFFSET = 140

comptime _DARWIN_STAT_BUFFER_WORDS = 18
comptime _DARWIN_STAT_DEVICE_OFFSET = 0
comptime _DARWIN_STAT_MODE_OFFSET = 4
comptime _DARWIN_STAT_INODE_OFFSET = 8
comptime _DARWIN_STAT_BTIME_SECONDS_OFFSET = 80
comptime _DARWIN_STAT_BTIME_NANOSECONDS_OFFSET = 88
comptime _DARWIN_STAT_SIZE_OFFSET = 96


@fieldwise_init
struct _FileIdentity(Copyable, Movable):
    """Stable device and inode identity for one filesystem object."""

    var device_major: UInt32
    var device_minor: UInt32
    var inode: UInt64
    var has_birth_time: Bool
    var birth_time_seconds: UInt64
    var birth_time_nanoseconds: UInt32


@fieldwise_init
struct _DescriptorState(Copyable, Movable):
    """Identity and non-negative byte length of one regular file."""

    var identity: _FileIdentity
    var length: UInt64


def _same_identity(left: _FileIdentity, right: _FileIdentity) -> Bool:
    if (
        left.device_major != right.device_major
        or left.device_minor != right.device_minor
        or left.inode != right.inode
    ):
        return False
    if left.has_birth_time != right.has_birth_time:
        return False
    if left.has_birth_time:
        return (
            left.birth_time_seconds == right.birth_time_seconds
            and left.birth_time_nanoseconds == right.birth_time_nanoseconds
        )
    return True


def _index_symlink_error() -> SafeTensorError:
    return make_error(
        SafeTensorErrorKind.PATH_TRAVERSAL,
        (
            "index-controlled shard is a symbolic link; use "
            "open_sharded_safetensors or map_sharded_safetensors for "
            "caller-trusted shard paths"
        ),
    )


def _load_u16(words: List[UInt64], offset: Int) -> UInt16:
    var shift = UInt64((offset % 8) * 8)
    return UInt16((words[offset // 8] >> shift) & UInt64(0xFFFF))


def _load_u32(words: List[UInt64], offset: Int) -> UInt32:
    var shift = UInt64((offset % 8) * 8)
    return UInt32((words[offset // 8] >> shift) & UInt64(0xFFFFFFFF))


def _load_u64(words: List[UInt64], offset: Int) -> UInt64:
    return words[offset // 8]


def _linux_descriptor_state(
    file: FileHandle,
    non_regular_kind: SafeTensorErrorKind,
    index_symlink_hint: Bool = False,
) raises SafeTensorError -> _DescriptorState:
    # Mojo 1.0 has no public fstat wrapper, and its high-level stat_result is
    # not C-ABI-compatible with struct stat. Linux statx with AT_EMPTY_PATH
    # provides the same descriptor-only query through a stable 256-byte UAPI.
    # UInt64 storage guarantees the alignment required by the kernel structure;
    # all consumed fields have fixed offsets in Linux's public statx ABI. Both
    # supported Linux architectures are little-endian, matching these loads.
    var empty_path = String("")
    var status = List[UInt64](length=_LINUX_STATX_BUFFER_WORDS, fill=0)
    var result = external_call["statx", c_int](
        c_int(file.handle),
        empty_path.as_c_string_slice().unsafe_ptr(),
        c_int(_LINUX_AT_EMPTY_PATH),
        c_int(_LINUX_STATX_BASIC_STATS | _LINUX_STATX_BTIME),
        status.unsafe_ptr(),
    )
    if result != 0:
        raise make_error(
            SafeTensorErrorKind.IO_ERROR,
            "file descriptor status query failed",
        )
    var mask = _load_u32(status, _LINUX_STATX_MASK_OFFSET)
    if mask & UInt32(_LINUX_STATX_REQUIRED) != UInt32(_LINUX_STATX_REQUIRED):
        raise make_error(
            SafeTensorErrorKind.IO_ERROR,
            "file descriptor status query omitted required fields",
        )
    var mode = Int(_load_u16(status, _LINUX_STATX_MODE_OFFSET))
    if index_symlink_hint and S_ISLNK(mode):
        raise _index_symlink_error()
    if not S_ISREG(mode):
        raise make_error(
            non_regular_kind,
            "opened object is not a regular file",
        )
    var has_birth_time = mask & UInt32(_LINUX_STATX_BTIME) != 0
    return _DescriptorState(
        _FileIdentity(
            _load_u32(status, _LINUX_STATX_DEVICE_MAJOR_OFFSET),
            _load_u32(status, _LINUX_STATX_DEVICE_MINOR_OFFSET),
            _load_u64(status, _LINUX_STATX_INODE_OFFSET),
            has_birth_time,
            _load_u64(status, _LINUX_STATX_BTIME_SECONDS_OFFSET),
            _load_u32(status, _LINUX_STATX_BTIME_NANOSECONDS_OFFSET),
        ),
        _load_u64(status, _LINUX_STATX_SIZE_OFFSET),
    )


def _darwin_state_from_status(
    status: List[UInt64],
    non_regular_kind: SafeTensorErrorKind,
    index_symlink_hint: Bool = False,
) raises SafeTensorError -> _DescriptorState:
    """Decodes the fixed little-endian arm64 Darwin stat fields we consume."""
    var mode = Int(_load_u16(status, _DARWIN_STAT_MODE_OFFSET))
    if index_symlink_hint and S_ISLNK(mode):
        raise _index_symlink_error()
    if not S_ISREG(mode):
        raise make_error(
            non_regular_kind,
            "opened object is not a regular file",
        )

    # st_size is signed. A set sign bit denotes an invalid negative length;
    # this also keeps every later conversion to the native Int checked.
    var length = _load_u64(status, _DARWIN_STAT_SIZE_OFFSET)
    if length > UInt64(Int.MAX):
        raise make_error(
            SafeTensorErrorKind.IO_ERROR,
            "file status reported a negative or unsupported byte length",
        )
    var birth_nanoseconds = _load_u64(
        status, _DARWIN_STAT_BTIME_NANOSECONDS_OFFSET
    )
    if birth_nanoseconds >= UInt64(1_000_000_000):
        raise make_error(
            SafeTensorErrorKind.IO_ERROR,
            "file status reported an invalid birth time",
        )
    return _DescriptorState(
        _FileIdentity(
            _load_u32(status, _DARWIN_STAT_DEVICE_OFFSET),
            0,
            _load_u64(status, _DARWIN_STAT_INODE_OFFSET),
            True,
            _load_u64(status, _DARWIN_STAT_BTIME_SECONDS_OFFSET),
            UInt32(birth_nanoseconds),
        ),
        length,
    )


def _darwin_descriptor_state(
    file: FileHandle,
    non_regular_kind: SafeTensorErrorKind,
) raises SafeTensorError -> _DescriptorState:
    # Mojo 1.0 does not expose fstat. UInt64 storage supplies the 8-byte
    # alignment and exact 144-byte size of Darwin arm64's public struct stat.
    var status = List[UInt64](length=_DARWIN_STAT_BUFFER_WORDS, fill=0)
    var result = external_call["fstat", c_int](
        c_int(file.handle), status.unsafe_ptr()
    )
    if result != 0:
        raise make_error(
            SafeTensorErrorKind.IO_ERROR,
            "file descriptor status query failed",
        )
    return _darwin_state_from_status(status, non_regular_kind)


def _darwin_path_state(
    directory: FileHandle,
    basename: String,
) raises SafeTensorError -> _DescriptorState:
    # fstatat classifies the final directory entry without following it or
    # opening a FIFO/device. The subsequent openat result is identity-checked.
    var owned_basename = basename
    var status = List[UInt64](length=_DARWIN_STAT_BUFFER_WORDS, fill=0)
    var result = external_call["fstatat", c_int](
        c_int(directory.handle),
        owned_basename.as_c_string_slice().unsafe_ptr(),
        status.unsafe_ptr(),
        c_int(_AT_SYMLINK_NOFOLLOW),
    )
    if result != 0:
        raise make_error(
            SafeTensorErrorKind.IO_ERROR,
            "index-controlled shard status query failed",
        )
    return _darwin_state_from_status(
        status,
        SafeTensorErrorKind.PATH_TRAVERSAL,
        index_symlink_hint=True,
    )


def _descriptor_state(
    file: FileHandle,
    non_regular_kind: SafeTensorErrorKind,
    index_symlink_hint: Bool = False,
) raises SafeTensorError -> _DescriptorState:
    comptime if _IS_LINUX:
        return _linux_descriptor_state(
            file, non_regular_kind, index_symlink_hint
        )
    elif _IS_MACOS:
        return _darwin_descriptor_state(file, non_regular_kind)
    else:
        raise make_error(
            SafeTensorErrorKind.IO_ERROR,
            "the current operating system is unsupported",
        )
