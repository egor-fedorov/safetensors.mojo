"""Internal sharded archive parsing, validation, and local access."""

from safetensors.sharding.index_parser import (
    DEFAULT_MAX_INDEX_BYTES,
    DEFAULT_MAX_INDEX_ENTRIES,
    DEFAULT_MAX_SHARDS,
)
from safetensors.sharding.model import (
    ShardedSafeTensorMetadata,
    ShardedTensorInfo,
)
from safetensors.sharding.mapped_reader import (
    MappedShardedSafeTensorArchive,
    map_safetensors_index,
    map_sharded_safetensors,
)
from safetensors.sharding.reader import (
    ShardedSafeTensorReader,
    open_safetensors_index,
    open_sharded_safetensors,
)
