"""Compile-only contract for every root-level public export."""

from safetensors import (
    DEFAULT_MAX_HEADER_BYTES,
    MappedSafeTensorFile,
    RawSafeTensorMetadata,
    RawTensorInfo,
    SafeDType,
    SafeTensorError,
    SafeTensorErrorKind,
    SafeTensorMetadata,
    SafeTensorReader,
    TensorInfo,
    decode_header_length,
    map_safetensors,
    open_safetensors,
    parse_metadata,
    parse_metadata_from_header,
    parse_raw_header,
    validate_metadata,
)


def main():
    print(DEFAULT_MAX_HEADER_BYTES)
