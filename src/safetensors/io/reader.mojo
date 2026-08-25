"""Metadata-only local file opening for validated Safetensors readers."""

from std.os import SEEK_SET

from safetensors.errors import SafeTensorError, SafeTensorErrorKind, make_error
from safetensors.format.checked import checked_add_u64, checked_u64_to_int
from safetensors.format.model import SafeTensorMetadata
from safetensors.format.parser import DEFAULT_MAX_HEADER_BYTES
from safetensors.io._file import (
    _ValidatedFile,
    _io_error,
    _open_validated_file,
    _require_file_length,
    _seek,
)


def _read_exact_into[
    origin: MutOrigin
](file: FileHandle, destination: Span[UInt8, origin],) raises SafeTensorError:
    var filled = 0
    while filled < len(destination):
        var remaining = len(destination) - filled
        var count: Int
        try:
            count = file.read(destination[filled:])
        except:
            raise _io_error("file read failed")
        if count <= 0:
            raise _io_error("unexpected EOF while reading tensor data")
        if count > remaining:
            raise _io_error("file read returned an invalid byte count")
        filled += count


struct SafeTensorReader(Movable):
    """An owned local file handle paired with validated Safetensors metadata.

    A reader has one shared seek cursor and does not support concurrent reads.
    """

    var _opened: _ValidatedFile

    def __init__(out self, var opened: _ValidatedFile):
        """Internal hook for open_safetensors-owned validated state.

        Callers must use open_safetensors instead of constructing a reader
        directly.
        """
        self._opened = opened^

    def metadata(self) -> SafeTensorMetadata:
        """Returns a validated metadata copy for the opened file."""
        return self._opened.metadata.copy()

    def file_length(self) -> UInt64:
        """Returns the complete file length observed during opening."""
        return self._opened.file_length

    def read_tensor_into[
        origin: MutOrigin
    ](
        mut self,
        name: String,
        destination: Span[UInt8, origin],
    ) raises SafeTensorError:
        """Reads one named tensor into an exactly sized caller-owned buffer.

        The destination may be partially modified when an I/O failure occurs.
        Reads on the same reader must not execute concurrently.
        """
        var info = self._opened.metadata.info(name)
        if UInt64(len(destination)) != info.byte_length:
            raise make_error(
                SafeTensorErrorKind.DESTINATION_SIZE_MISMATCH,
                "destination length does not match the tensor byte length",
            )

        var absolute_begin = checked_add_u64(
            self._opened.metadata.data_start(), info.begin
        )
        var absolute_end = checked_add_u64(
            self._opened.metadata.data_start(), info.end
        )
        if absolute_end > self._opened.file_length:
            raise make_error(
                SafeTensorErrorKind.INVALID_OFFSETS,
                "absolute tensor offsets extend beyond the opened file",
            )
        var seek_position = checked_u64_to_int(absolute_begin)

        _require_file_length(self._opened.file, self._opened.file_length)
        _ = _seek(self._opened.file, seek_position, SEEK_SET)
        _read_exact_into(self._opened.file, destination)
        _require_file_length(self._opened.file, self._opened.file_length)

    def load_tensor(
        mut self, name: String
    ) raises SafeTensorError -> List[UInt8]:
        """Allocates and returns the exact raw wire bytes of one tensor."""
        var info = self._opened.metadata.info(name)
        var size = checked_u64_to_int(info.byte_length)
        var destination = List[UInt8](length=size, fill=0)
        self.read_tensor_into(name, destination)
        return destination^


def open_safetensors(
    path: String,
    max_header_bytes: UInt64 = DEFAULT_MAX_HEADER_BYTES,
    strict: Bool = False,
) raises SafeTensorError -> SafeTensorReader:
    """Opens and fully validates a file under the selected header policy.

    Strict mode additionally requires canonical boundary whitespace and a
    closed tensor descriptor schema.
    """
    var opened = _open_validated_file(path, max_header_bytes, strict)
    return SafeTensorReader(opened^)
