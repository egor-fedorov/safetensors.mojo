"""Metadata-only local file opening for validated Safetensors readers."""

from std.os import SEEK_END, SEEK_SET

from .checked import checked_add_u64, checked_sub_u64, checked_u64_to_int
from .errors import SafeTensorError, SafeTensorErrorKind, make_error
from .model import SafeTensorMetadata
from .parser import (
    DEFAULT_MAX_HEADER_BYTES,
    decode_header_length,
    parse_metadata_from_header,
)


def _io_error(message: String) -> SafeTensorError:
    return make_error(SafeTensorErrorKind.IO_ERROR, message)


def _seek(
    file: FileHandle, offset: Int, whence: UInt8
) raises SafeTensorError -> UInt64:
    try:
        return file.seek(offset, whence)
    except:
        raise _io_error("file seek failed")


def _file_length(file: FileHandle) raises SafeTensorError -> UInt64:
    return _seek(file, 0, SEEK_END)


def _require_file_length(
    file: FileHandle, expected: UInt64
) raises SafeTensorError:
    if _file_length(file) != expected:
        raise _io_error("file length changed after metadata validation")


def _read_owned_exact(
    file: FileHandle,
    size: Int,
    short_read_kind: SafeTensorErrorKind,
    short_read_message: String,
) raises SafeTensorError -> List[UInt8]:
    var contents: List[UInt8]
    try:
        contents = file.read_bytes(size)
    except:
        raise _io_error("file read failed")
    if len(contents) != size:
        raise make_error(short_read_kind, short_read_message)
    return contents^


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

    var _file: FileHandle
    var _metadata: SafeTensorMetadata
    var _file_length: UInt64

    def __init__(
        out self,
        var file: FileHandle,
        var metadata: SafeTensorMetadata,
        file_length: UInt64,
    ):
        """Internal hook for open_safetensors-owned validated state.

        Callers must use open_safetensors instead of constructing a reader
        directly.
        """
        self._file = file^
        self._metadata = metadata^
        self._file_length = file_length

    def metadata(self) -> SafeTensorMetadata:
        """Returns a validated metadata copy for the opened file."""
        return self._metadata.copy()

    def file_length(self) -> UInt64:
        """Returns the complete file length observed during opening."""
        return self._file_length

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
        var info = self._metadata.info(name)
        if UInt64(len(destination)) != info.byte_length:
            raise make_error(
                SafeTensorErrorKind.DESTINATION_SIZE_MISMATCH,
                "destination length does not match the tensor byte length",
            )

        var absolute_begin = checked_add_u64(
            self._metadata.data_start(), info.begin
        )
        var absolute_end = checked_add_u64(
            self._metadata.data_start(), info.end
        )
        if absolute_end > self._file_length:
            raise make_error(
                SafeTensorErrorKind.INVALID_OFFSETS,
                "absolute tensor offsets extend beyond the opened file",
            )
        var seek_position = checked_u64_to_int(absolute_begin)

        _require_file_length(self._file, self._file_length)
        _ = _seek(self._file, seek_position, SEEK_SET)
        _read_exact_into(self._file, destination)
        _require_file_length(self._file, self._file_length)

    def load_tensor(
        mut self, name: String
    ) raises SafeTensorError -> List[UInt8]:
        """Allocates and returns the exact raw wire bytes of one tensor."""
        var info = self._metadata.info(name)
        var size = checked_u64_to_int(info.byte_length)
        var destination = List[UInt8](length=size, fill=0)
        self.read_tensor_into(name, destination)
        return destination^


def open_safetensors(
    path: String,
    max_header_bytes: UInt64 = DEFAULT_MAX_HEADER_BYTES,
) raises SafeTensorError -> SafeTensorReader:
    """Opens a local file and reads only its length prefix and JSON header."""
    var file: FileHandle
    try:
        file = open(path, "r")
    except:
        raise _io_error("file open failed")

    var file_length = _file_length(file)
    if file_length < 8:
        raise make_error(
            SafeTensorErrorKind.HEADER_TOO_SMALL,
            "at least 8 bytes are required for the header length prefix",
        )

    _ = _seek(file, 0, SEEK_SET)
    var prefix = _read_owned_exact(
        file,
        8,
        SafeTensorErrorKind.HEADER_TOO_SMALL,
        "the header length prefix was truncated while opening the file",
    )
    var header_length = decode_header_length(prefix)
    if header_length > max_header_bytes:
        raise make_error(
            SafeTensorErrorKind.HEADER_TOO_LARGE,
            "declared header length exceeds the configured limit",
        )

    var data_start: UInt64
    try:
        data_start = checked_add_u64(8, header_length)
    except:
        raise make_error(
            SafeTensorErrorKind.INVALID_HEADER_LENGTH,
            "header length overflows the file offset",
        )
    if data_start > file_length:
        raise make_error(
            SafeTensorErrorKind.INVALID_HEADER_LENGTH,
            "declared header extends beyond the opened file",
        )

    var header_size: Int
    try:
        header_size = checked_u64_to_int(header_length)
    except:
        raise make_error(
            SafeTensorErrorKind.INVALID_HEADER_LENGTH,
            "header length does not fit the native index type",
        )

    var header = _read_owned_exact(
        file,
        header_size,
        SafeTensorErrorKind.INVALID_HEADER_LENGTH,
        "the declared header was truncated while opening the file",
    )
    var data_length = checked_sub_u64(file_length, data_start)
    var metadata = parse_metadata_from_header(header, data_length, data_start)

    _require_file_length(file, file_length)

    return SafeTensorReader(file^, metadata^, file_length)
