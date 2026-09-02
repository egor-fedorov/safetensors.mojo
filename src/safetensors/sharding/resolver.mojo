"""POSIX descriptor-relative path resolution for local sharded archives."""

from std.ffi import c_int, external_call
from std.os import SEEK_SET
from std.os.path import split
from std.sys._libc_errno import ErrNo, get_errno

from safetensors.errors import SafeTensorError, SafeTensorErrorKind, make_error
from safetensors.format.checked import checked_u64_to_int
from safetensors.io._platform import (
    _AT_FDCWD,
    _IS_LINUX,
    _IS_MACOS,
    _O_CLOEXEC,
    _O_DIRECTORY,
    _O_NOFOLLOW,
    _O_NONBLOCK,
    _O_RDONLY,
)
from safetensors.sharding._file_status import (
    _DescriptorState,
    _darwin_path_state,
    _descriptor_state,
    _index_symlink_error,
    _same_identity,
)
from safetensors.sharding.index_parser import (
    DEFAULT_MAX_INDEX_BYTES,
    DEFAULT_MAX_INDEX_ENTRIES,
    DEFAULT_MAX_SHARDS,
    _ParsedIndex,
    _parse_index,
)


comptime _LINUX_O_PATH = 0x200000


struct _OpenedIndex(Movable):
    """Parsed index state retaining its lexical parent directory descriptor."""

    var directory: FileHandle
    var parsed: _ParsedIndex

    def __init__(out self, var directory: FileHandle, var parsed: _ParsedIndex):
        self.directory = directory^
        self.parsed = parsed^


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
            raise _index_symlink_error()
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
    var expected: _DescriptorState
    var file: FileHandle
    comptime if _IS_LINUX:
        # O_PATH pins every existing final object without triggering FIFO,
        # device, or socket open behavior. Keep candidate alive until the real
        # open completes so the classified object remains pinned throughout
        # pathname resolution.
        var candidate = _open_at(
            directory.handle,
            basename,
            _LINUX_O_PATH | _O_NOFOLLOW | _O_CLOEXEC,
        )
        expected = _descriptor_state(
            candidate,
            SafeTensorErrorKind.PATH_TRAVERSAL,
            index_symlink_hint=True,
        )
        file = _open_at(
            directory.handle,
            basename,
            _O_RDONLY | _O_NONBLOCK | _O_NOFOLLOW | _O_CLOEXEC,
            nofollow_is_path_traversal=True,
        )
        # Mojo may destroy owned values after their last use rather than at the
        # closing brace. Recheck the pinned descriptor after openat so its
        # lifetime provably covers the pathname reopen.
        var pinned = _descriptor_state(
            candidate,
            SafeTensorErrorKind.PATH_TRAVERSAL,
            index_symlink_hint=True,
        )
        if not _same_identity(expected.identity, pinned.identity):
            raise make_error(
                SafeTensorErrorKind.IO_ERROR,
                "pinned index-controlled shard identity changed",
            )
    elif _IS_MACOS:
        # Darwin has no O_PATH. A no-follow fstatat preflight rejects unsafe
        # object kinds without opening them; openat below is non-blocking and
        # no-follow, and its descriptor identity must match this observation.
        expected = _darwin_path_state(directory, basename)
        file = _open_at(
            directory.handle,
            basename,
            _O_RDONLY | _O_NONBLOCK | _O_NOFOLLOW | _O_CLOEXEC,
            nofollow_is_path_traversal=True,
        )
    else:
        raise make_error(
            SafeTensorErrorKind.IO_ERROR,
            "the current operating system is unsupported",
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
    max_index_entries: UInt64 = DEFAULT_MAX_INDEX_ENTRIES,
    max_shards: UInt64 = DEFAULT_MAX_SHARDS,
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
    var parsed = _parse_index(contents, max_index_entries, max_shards)
    return _OpenedIndex(directory^, parsed^)
