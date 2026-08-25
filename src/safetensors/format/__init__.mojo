"""Runtime-independent Safetensors parsing, validation, and write planning."""

from safetensors.format.dtype import SafeDType
from safetensors.format.json_parser import parse_raw_header
from safetensors.format.model import (
    RawSafeTensorMetadata,
    RawTensorInfo,
    SafeTensorMetadata,
    SafeTensorData,
    TensorInfo,
)
from safetensors.format.parser import (
    DEFAULT_MAX_HEADER_BYTES,
    decode_header_length,
    parse_metadata,
    parse_metadata_from_header,
)
from safetensors.format.validation import validate_metadata
