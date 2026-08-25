"""Strict Safetensors parsing, validation, and local byte access."""

from .dtype import SafeDType
from .errors import SafeTensorError, SafeTensorErrorKind
from .json_parser import parse_raw_header
from .mapped_reader import MappedSafeTensorFile, map_safetensors
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
from .reader import SafeTensorReader, open_safetensors
from .validation import validate_metadata
