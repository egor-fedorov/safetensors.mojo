"""Strict Safetensors parsing, validation, and local byte access."""

from safetensors.errors import SafeTensorError, SafeTensorErrorKind
from safetensors.format import (
    DEFAULT_MAX_HEADER_BYTES,
    RawSafeTensorMetadata,
    RawTensorInfo,
    SafeDType,
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
)
