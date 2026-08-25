"""Shared validated-file ownership and local I/O helpers."""

from std.os import SEEK_END, SEEK_SET

from safetensors.errors import SafeTensorError, SafeTensorErrorKind, make_error
from safetensors.format.checked import (
    checked_add_u64,
    checked_sub_u64,
    checked_u64_to_int,
)
from safetensors.format.model import SafeTensorMetadata
from safetensors.format.parser import (
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


struct _ValidatedFile(Movable):
    """An owned file handle paired with metadata validated from that handle."""

    var file: FileHandle
    var metadata: SafeTensorMetadata
    var file_length: UInt64

    def __init__(
        out self,
        var file: FileHandle,
        var metadata: SafeTensorMetadata,
        file_length: UInt64,
    ):
        self.file = file^
        self.metadata = metadata^
        self.file_length = file_length


def _open_validated_file(
    path: String,
    max_header_bytes: UInt64 = DEFAULT_MAX_HEADER_BYTES,
) raises SafeTensorError -> _ValidatedFile:
    """Opens one descriptor and validates metadata read from that descriptor."""
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

    return _ValidatedFile(file^, metadata^, file_length)
