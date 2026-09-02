"""Validated Safetensors parsing, local reads, mappings, and atomic writes."""

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
from safetensors.sharding import (
    DEFAULT_MAX_INDEX_BYTES,
    DEFAULT_MAX_SHARDS,
    MappedShardedSafeTensorArchive,
    ShardedSafeTensorMetadata,
    ShardedSafeTensorReader,
    ShardedTensorInfo,
    map_safetensors_index,
    map_sharded_safetensors,
    open_safetensors_index,
    open_sharded_safetensors,
)
