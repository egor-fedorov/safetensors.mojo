"""Strict Safetensors parsing, validation, local reads, and atomic writes."""

from safetensors.errors import SafeTensorError, SafeTensorErrorKind
from safetensors.format import (
    DEFAULT_MAX_HEADER_BYTES,
    RawSafeTensorMetadata,
    RawTensorInfo,
    SafeDType,
    SafeTensorData,
    SafeTensorMetadata,
    TensorInfo,
    decode_header_length,
    parse_metadata,
    parse_metadata_from_header,
    parse_raw_header,
    validate_metadata,
)
from safetensors.io import (
    MappedSafeTensorFile,
    SafeTensorReader,
    map_safetensors,
    open_safetensors,
    save_safetensors,
)
