"""Internal Linux sibling-temporary-file replacement transaction."""

from std.ffi import c_int, c_long, c_size_t, external_call
from std.os import remove
from std.os.path import basename, dirname, lexists
from std.pathlib import Path

from safetensors.errors import SafeTensorError
from safetensors.io._file import _io_error


comptime _O_WRONLY = 0x0001
comptime _O_CREAT = 0x0040
comptime _O_EXCL = 0x0080
comptime _O_CLOEXEC = 0x80000
comptime _TEMP_ATTEMPTS = 128


def _open_exclusive(path: String) -> Int:
    var owned_path = path
    return Int(
        external_call["open", c_int, num_fixed_args=2](
            owned_path.as_c_string_slice().unsafe_ptr(),
            c_int(_O_WRONLY | _O_CREAT | _O_EXCL | _O_CLOEXEC),
            c_int(0o600),
        )
    )


def _rename(source: String, destination: String) -> Bool:
    var owned_source = source
    var owned_destination = destination
    return (
        external_call["rename", c_int](
            owned_source.as_c_string_slice().unsafe_ptr(),
            owned_destination.as_c_string_slice().unsafe_ptr(),
        )
        == 0
    )


def _random_nonce() raises SafeTensorError -> UInt64:
    """Returns one OS-random nonce for an opaque temporary basename."""
    var bytes = List[UInt8](length=8, fill=0)
    var destination = Span(bytes)
    var count = Int(
        external_call["getrandom", c_long](
            destination.unsafe_ptr(),
            c_size_t(len(destination)),
            c_int(0),
        )
    )
    if count != len(destination):
        raise _io_error("temporary file randomness failed")

    var nonce: UInt64 = 0
    for index in range(len(bytes)):
        nonce |= UInt64(bytes[index]) << UInt64(index * 8)
    return nonce


def _temporary_path(destination: String) raises SafeTensorError -> String:
    var parent = dirname(destination)
    if parent == "":
        parent = "."
    var name = basename(destination)
    if name == "":
        raise _io_error("destination path has no file name")

    for attempt in range(_TEMP_ATTEMPTS):
        var nonce = _random_nonce()
        var candidate_name = (
            "."
            + name
            + ".safetensors-mojo-"
            + String(nonce)
            + "-"
            + String(attempt)
            + ".tmp"
        )
        var candidate = String(Path(parent).joinpath(candidate_name))
        if not lexists(candidate):
            return candidate^
    raise _io_error("unable to choose a sibling temporary file name")


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
    for _ in range(_TEMP_ATTEMPTS):
        var temporary = _temporary_path(destination)
        var descriptor = _open_exclusive(temporary)
        if descriptor >= 0:
            var file = FileHandle()
            file.handle = descriptor
            return _AtomicFile(destination, temporary, file^)
        if not lexists(temporary):
            raise _io_error("sibling temporary file creation failed")
    raise _io_error("sibling temporary file name attempts were exhausted")
