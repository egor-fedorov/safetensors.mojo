"""Internal sibling-temporary-file replacement transaction."""

from std.ffi import c_int, external_call
from std.os import remove
from std.os.path import basename, dirname
from std.pathlib import Path
from std.sys._libc_errno import ErrNo, get_errno

from safetensors.errors import SafeTensorError
from safetensors.io._file import _io_error
from safetensors.io._platform import (
    _O_CLOEXEC,
    _O_CREAT,
    _O_EXCL,
    _O_WRONLY,
    _fill_os_random,
)


comptime _TEMP_ATTEMPTS = 128


def _open_exclusive(path: String) -> Int:
    var owned_path = path
    return Int(
        external_call["open", c_int, num_fixed_args=2](
            owned_path.as_c_string_span().ptr(),
            c_int(_O_WRONLY | _O_CREAT | _O_EXCL | _O_CLOEXEC),
            c_int(0o600),
        )
    )


def _rename(source: String, destination: String) -> Bool:
    var owned_source = source
    var owned_destination = destination
    return (
        external_call["rename", c_int](
            owned_source.as_c_string_span().ptr(),
            owned_destination.as_c_string_span().ptr(),
        )
        == 0
    )


def _random_nonce() raises SafeTensorError -> UInt64:
    """Returns one OS-random nonce for an opaque temporary basename."""
    var bytes = List[UInt8](length=8, fill=0)
    var destination = Span(bytes)
    if not _fill_os_random(destination):
        raise _io_error("temporary file randomness failed")

    var nonce: UInt64 = 0
    for index in range(len(bytes)):
        nonce |= UInt64(bytes[index]) << UInt64(index * 8)
    return nonce


struct _AtomicFile(Movable):
    """An uncommitted sibling file removed automatically on failure."""

    var _destination: String
    var _temporary: String
    var _file: FileHandle
    var _committed: Bool

    def __init__(
        out self,
        destination: String,
        temporary: String,
        var file: FileHandle,
    ):
        self._destination = destination.copy()
        self._temporary = temporary.copy()
        self._file = file^
        self._committed = False

    def __deinit__(deinit self):
        if not self._committed:
            try:
                self._file.close()
            except:
                pass
            try:
                remove(self._temporary)
            except:
                pass

    def write_all[
        origin: Origin
    ](mut self, bytes: Span[UInt8, origin]) raises SafeTensorError:
        """Writes one complete span or returns a typed I/O error."""
        try:
            self._file.write_all(bytes)
        except:
            raise _io_error("temporary file write failed")

    def commit(mut self) raises SafeTensorError:
        """Closes and atomically renames the sibling file into place."""
        try:
            self._file.close()
        except:
            raise _io_error("temporary file close failed")

        if not _rename(self._temporary, self._destination):
            raise _io_error("atomic destination replacement failed")

        self._committed = True


def _create_atomic_file(
    destination: String,
) raises SafeTensorError -> _AtomicFile:
    """Creates one exclusive sibling temporary file for a destination."""
    if "\x00" in destination:
        raise _io_error("destination path contains an embedded NUL byte")
    var parent = dirname(destination)
    if parent == "":
        parent = "."
    var name = basename(destination)
    if name == "":
        raise _io_error("destination path has no file name")
    # Distinct leading punctuation also separates the two entries on macOS
    # volumes with case-insensitive names.
    var prefix = ".st-" if name.startswith(".~") else ".~st-"
    for _ in range(_TEMP_ATTEMPTS):
        var temporary = String(
            Path(parent).joinpath(prefix + String(_random_nonce()))
        )
        var descriptor = _open_exclusive(temporary)
        if descriptor >= 0:
            var file = FileHandle()
            file.handle = descriptor
            return _AtomicFile(destination, temporary, file^)
        if get_errno() != ErrNo.EEXIST:
            raise _io_error("sibling temporary file creation failed")
    raise _io_error("sibling temporary file name attempts were exhausted")
