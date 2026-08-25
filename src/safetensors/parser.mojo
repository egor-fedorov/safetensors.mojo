"""Safetensors length-prefix and in-memory metadata parsing."""

from .checked import checked_add_u64, checked_sub_u64, checked_u64_to_int
from .errors import SafeTensorError, SafeTensorErrorKind, make_error
from .json_parser import parse_raw_header
from .model import RawSafeTensorMetadata, SafeTensorMetadata
from .validation import validate_metadata


comptime DEFAULT_MAX_HEADER_BYTES = UInt64(100_000_000)


def decode_header_length[
    origin: Origin
](prefix: Span[UInt8, origin],) raises SafeTensorError -> UInt64:
    """Decodes the first eight bytes as an unsigned little-endian length."""
    if len(prefix) < 8:
        raise make_error(
            SafeTensorErrorKind.HEADER_TOO_SMALL,
            "at least 8 bytes are required for the header length prefix",
        )

    var value: UInt64 = 0
    for index in range(8):
        value |= UInt64(prefix[index]) << UInt64(index * 8)
    return value


def parse_metadata_from_header[
    origin: Origin
](
    header: Span[UInt8, origin],
    data_length: UInt64,
    data_start: UInt64 = 0,
) raises SafeTensorError -> SafeTensorMetadata:
    """Parses an isolated header with an optional caller-supplied data origin.
    """
    var raw = parse_raw_header(header)
    return validate_metadata(raw, data_length, data_start)


def parse_metadata[
    origin: Origin
](
    buffer: Span[UInt8, origin],
    max_header_bytes: UInt64 = DEFAULT_MAX_HEADER_BYTES,
) raises SafeTensorError -> SafeTensorMetadata:
    """Parses validated metadata from a complete in-memory file buffer."""
    var header_length = decode_header_length(buffer)
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
            "header length overflows the complete-buffer offset",
        )

    var available = UInt64(len(buffer))
    if data_start > available:
        raise make_error(
            SafeTensorErrorKind.INVALID_HEADER_LENGTH,
            "declared header extends beyond the available buffer",
        )

    var data_start_index: Int
    try:
        data_start_index = checked_u64_to_int(data_start)
    except:
        raise make_error(
            SafeTensorErrorKind.INVALID_HEADER_LENGTH,
            "header length does not fit the native index type",
        )

    var data_length = checked_sub_u64(available, data_start)
    return parse_metadata_from_header(
        buffer[8:data_start_index], data_length, data_start
    )
