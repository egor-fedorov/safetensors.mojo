"""Linux descriptor-relative path resolution for local sharded archives."""

from std.ffi import c_int, external_call
from std.os import SEEK_SET
from std.os.path import split
from std.stat import S_ISREG
from std.sys._libc_errno import ErrNo, get_errno

from safetensors.errors import SafeTensorError, SafeTensorErrorKind, make_error
from safetensors.format.checked import checked_u64_to_int
from safetensors.sharding.index_parser import (
    DEFAULT_MAX_INDEX_BYTES,
    _ParsedIndex,
    _parse_index,
)


comptime _AT_FDCWD = -100
comptime _O_RDONLY = 0x0000
comptime _O_NONBLOCK = 0x0800
comptime _O_DIRECTORY = 0x10000
comptime _O_NOFOLLOW = 0x20000
comptime _O_CLOEXEC = 0x80000
comptime _O_PATH = 0x200000
comptime _AT_EMPTY_PATH = 0x1000
comptime _STATX_BASIC_STATS = 0x07FF
comptime _STATX_BTIME = 0x0800
comptime _STATX_REQUIRED = 0x0301
comptime _STATX_BUFFER_WORDS = 32
comptime _STATX_MASK_OFFSET = 0
comptime _STATX_MODE_OFFSET = 28
comptime _STATX_INODE_OFFSET = 32
comptime _STATX_SIZE_OFFSET = 40
comptime _STATX_BTIME_SECONDS_OFFSET = 80
comptime _STATX_BTIME_NANOSECONDS_OFFSET = 88
comptime _STATX_DEVICE_MAJOR_OFFSET = 136
comptime _STATX_DEVICE_MINOR_OFFSET = 140


@fieldwise_init
struct _FileIdentity(Copyable, Movable):
    """Stable device and inode identity observed through an open descriptor."""

    var device_major: UInt32
    var device_minor: UInt32
    var inode: UInt64
    var has_birth_time: Bool
    var birth_time_seconds: UInt64
    var birth_time_nanoseconds: UInt32


@fieldwise_init
struct _DescriptorState(Copyable, Movable):
    """Identity and non-negative byte length of one regular file descriptor."""

    var identity: _FileIdentity
    var length: UInt64


struct _OpenedIndex(Movable):
    """Parsed index state retaining its lexical parent directory descriptor."""

    var directory: FileHandle
    var parsed: _ParsedIndex

    def __init__(out self, var directory: FileHandle, var parsed: _ParsedIndex):
        self.directory = directory^
        self.parsed = parsed^


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


def _load_u16(words: List[UInt64], offset: Int) -> UInt16:
    var shift = UInt64((offset % 8) * 8)
    return UInt16((words[offset // 8] >> shift) & UInt64(0xFFFF))


def _load_u32(words: List[UInt64], offset: Int) -> UInt32:
    var shift = UInt64((offset % 8) * 8)
    return UInt32((words[offset // 8] >> shift) & UInt64(0xFFFFFFFF))


def _load_u64(words: List[UInt64], offset: Int) -> UInt64:
    return words[offset // 8]


def _descriptor_state(
    file: FileHandle,
    non_regular_kind: SafeTensorErrorKind,
) raises SafeTensorError -> _DescriptorState:
    # Mojo 1.0 has no public fstat wrapper, and its high-level stat_result is
    # not C-ABI-compatible with struct stat. Linux statx with AT_EMPTY_PATH
    # provides the same descriptor-only query through a stable 256-byte UAPI.
    # UInt64 storage guarantees the alignment required by the kernel structure;
    # all consumed fields have fixed offsets in Linux's public statx ABI. The
    # supported linux-64 target is little-endian, matching the word loads below.
    var empty_path = String("")
    var status = List[UInt64](length=_STATX_BUFFER_WORDS, fill=0)
    var result = external_call["statx", c_int](
        c_int(file.handle),
        empty_path.as_c_string_slice().unsafe_ptr(),
        c_int(_AT_EMPTY_PATH),
        c_int(_STATX_BASIC_STATS | _STATX_BTIME),
        status.unsafe_ptr(),
    )
    if result != 0:
        raise make_error(
            SafeTensorErrorKind.IO_ERROR,
            "file descriptor status query failed",
        )
    var mask = _load_u32(status, _STATX_MASK_OFFSET)
    if mask & UInt32(_STATX_REQUIRED) != UInt32(_STATX_REQUIRED):
        raise make_error(
            SafeTensorErrorKind.IO_ERROR,
            "file descriptor status query omitted required fields",
        )
    var mode = Int(_load_u16(status, _STATX_MODE_OFFSET))
    if not S_ISREG(mode):
        raise make_error(
            non_regular_kind,
            "opened object is not a regular file",
        )
    var has_birth_time = mask & UInt32(_STATX_BTIME) != 0
    return _DescriptorState(
        _FileIdentity(
            _load_u32(status, _STATX_DEVICE_MAJOR_OFFSET),
            _load_u32(status, _STATX_DEVICE_MINOR_OFFSET),
            _load_u64(status, _STATX_INODE_OFFSET),
            has_birth_time,
            _load_u64(status, _STATX_BTIME_SECONDS_OFFSET),
            _load_u32(status, _STATX_BTIME_NANOSECONDS_OFFSET),
        ),
        _load_u64(status, _STATX_SIZE_OFFSET),
    )


def _contains_nul(value: String) -> Bool:
    for byte in value.as_bytes():
        if byte == 0:
            return True
    return False


def _open_at(
    directory_fd: Int,
    name: String,
    flags: Int,
    nofollow_is_path_traversal: Bool = False,
) raises SafeTensorError -> FileHandle:
    if _contains_nul(name):
        raise make_error(
            SafeTensorErrorKind.IO_ERROR,
            "trusted path contains an embedded NUL byte",
        )
    var owned_name = name
    var descriptor = external_call["openat", c_int, num_fixed_args=3](
        c_int(directory_fd),
        owned_name.as_c_string_slice(),
        c_int(flags),
    )
    if descriptor < 0:
        var error_number = get_errno()
        if nofollow_is_path_traversal and error_number == ErrNo.ELOOP:
            raise make_error(
                SafeTensorErrorKind.PATH_TRAVERSAL,
                "index-controlled shard must not be a symbolic link",
            )
        raise make_error(SafeTensorErrorKind.IO_ERROR, "file open failed")

    var file = FileHandle()
    file.handle = Int(descriptor)
    return file^


def _open_trusted_regular(path: String) raises SafeTensorError -> FileHandle:
    """Opens a caller-trusted path, following symlinks but never blocking."""
    var file = _open_at(
        _AT_FDCWD,
        path,
        _O_RDONLY | _O_NONBLOCK | _O_CLOEXEC,
    )
    _ = _descriptor_state(file, SafeTensorErrorKind.IO_ERROR)
    return file^


def _validate_shard_basename(name: String) raises SafeTensorError:
    """Requires one safe index-controlled `.safetensors` basename."""
    if name == "" or name == "." or name == "..":
        raise make_error(
            SafeTensorErrorKind.PATH_TRAVERSAL,
            "index-controlled shard name must be one basename",
        )
    if not name.endswith(".safetensors"):
        raise make_error(
            SafeTensorErrorKind.PATH_TRAVERSAL,
            "index-controlled shard name must end in .safetensors",
        )
    var bytes = name.as_bytes()
    for index in range(len(bytes)):
        var byte = bytes[index]
        if (
            byte < 0x20
            or byte == 0x7F
            or byte == 0x2F
            or byte == 0x5C
            or byte == 0x3A
            or (
                byte == 0xC2
                and index + 1 < len(bytes)
                and bytes[index + 1] >= 0x80
                and bytes[index + 1] <= 0x9F
            )
        ):
            raise make_error(
                SafeTensorErrorKind.PATH_TRAVERSAL,
                "index-controlled shard name contains a forbidden byte",
            )


def _open_index_shard(
    directory: FileHandle, basename: String
) raises SafeTensorError -> FileHandle:
    """Opens an untrusted shard basename beneath an anchored directory."""
    _validate_shard_basename(basename)
    # O_PATH obtains a side-effect-free descriptor for every existing final
    # object, including sockets, FIFOs, devices, directories, and symlinks.
    # Classify that pinned object before attempting to open it for reading.
    var candidate = _open_at(
        directory.handle,
        basename,
        _O_PATH | _O_NOFOLLOW | _O_CLOEXEC,
    )
    var expected = _descriptor_state(
        candidate, SafeTensorErrorKind.PATH_TRAVERSAL
    )
    var file = _open_at(
        directory.handle,
        basename,
        _O_RDONLY | _O_NONBLOCK | _O_NOFOLLOW | _O_CLOEXEC,
        nofollow_is_path_traversal=True,
    )
    var actual = _descriptor_state(file, SafeTensorErrorKind.PATH_TRAVERSAL)
    if not _same_identity(expected.identity, actual.identity):
        raise make_error(
            SafeTensorErrorKind.IO_ERROR,
            "index-controlled shard changed while it was opened",
        )
    return file^


def _open_index_document(
    index_path: String,
    max_index_bytes: UInt64 = DEFAULT_MAX_INDEX_BYTES,
) raises SafeTensorError -> _OpenedIndex:
    """Parses an index while retaining its lexical directory anchor."""
    var components = split(index_path)
    var parent = components[0]
    var basename = components[1]
    if parent == "":
        parent = "."
    if basename == "":
        raise make_error(
            SafeTensorErrorKind.IO_ERROR,
            "index path has no file basename",
        )

    var directory = _open_at(
        _AT_FDCWD,
        parent,
        _O_RDONLY | _O_NONBLOCK | _O_DIRECTORY | _O_CLOEXEC,
    )
    var index_file = _open_at(
        directory.handle,
        basename,
        _O_RDONLY | _O_NONBLOCK | _O_CLOEXEC,
    )
    var before = _descriptor_state(index_file, SafeTensorErrorKind.IO_ERROR)
    if before.length > max_index_bytes:
        raise make_error(
            SafeTensorErrorKind.INDEX_TOO_LARGE,
            "index file exceeds the configured byte limit",
        )
    var size = checked_u64_to_int(before.length)
    var contents: List[UInt8]
    try:
        _ = index_file.seek(0, SEEK_SET)
        contents = index_file.read_bytes(size)
    except:
        raise make_error(
            SafeTensorErrorKind.IO_ERROR,
            "index file read failed",
        )
    if len(contents) != size:
        raise make_error(
            SafeTensorErrorKind.IO_ERROR,
            "index file changed or was truncated while reading",
        )
    var after = _descriptor_state(index_file, SafeTensorErrorKind.IO_ERROR)
    if (
        not _same_identity(before.identity, after.identity)
        or before.length != after.length
    ):
        raise make_error(
            SafeTensorErrorKind.IO_ERROR,
            "index file changed while reading",
        )
    var parsed = _parse_index(contents)
    return _OpenedIndex(directory^, parsed^)
