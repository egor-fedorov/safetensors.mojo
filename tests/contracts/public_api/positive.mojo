"""Compile-only contract for every root-level public export."""

from safetensors import (
    DEFAULT_MAX_HEADER_BYTES,
    MappedSafeTensorFile,
    RawSafeTensorMetadata,
    RawTensorInfo,
    SafeDType,
    SafeTensorData,
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
    save_safetensors,
    validate_metadata,
)


def _writer_contract() raises:
    var tensors: List[SafeTensorData] = [
        SafeTensorData("tensor", SafeDType.U8, [UInt64(1)], [UInt8(7)])
    ]
    save_safetensors("compile-contract.safetensors", tensors)


def main():
    print(DEFAULT_MAX_HEADER_BYTES)
