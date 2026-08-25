"""Strict Safetensors format-core parsing and metadata validation."""

from .dtype import SafeDType
from .errors import SafeTensorError, SafeTensorErrorKind
from .json_parser import parse_raw_header
from .model import (
    RawSafeTensorMetadata,
    RawTensorInfo,
    SafeTensorMetadata,
    TensorInfo,
)
from .parser import (
    DEFAULT_MAX_HEADER_BYTES,
    decode_header_length,
    parse_metadata,
    parse_metadata_from_header,
)
from .validation import validate_metadata
