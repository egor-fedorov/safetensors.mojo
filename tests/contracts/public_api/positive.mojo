"""Compile-only contract for every root-level public export."""

from safetensors import (
    DEFAULT_MAX_HEADER_BYTES,
    DEFAULT_MAX_INDEX_BYTES,
    DEFAULT_MAX_INDEX_ENTRIES,
    DEFAULT_MAX_SHARDS,
    MappedSafeTensorFile,
    MappedShardedSafeTensorArchive,
    RawSafeTensorMetadata,
    RawTensorInfo,
    SafeDType,
    SafeTensorData,
    SafeTensorError,
    SafeTensorErrorKind,
    SafeTensorMetadata,
    SafeTensorReader,
    ShardedSafeTensorMetadata,
    ShardedSafeTensorReader,
    ShardedTensorInfo,
    TensorInfo,
    decode_header_length,
    map_safetensors,
    map_safetensors_index,
    map_sharded_safetensors,
    open_safetensors,
    open_safetensors_index,
    open_sharded_safetensors,
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


def _strict_reader_contract() raises:
    var header: List[UInt8] = [UInt8(ord("{")), UInt8(ord("}"))]
    _ = parse_raw_header(header, strict=True)
    _ = parse_metadata_from_header(header, 0, strict=True)

    var archive: List[UInt8] = [
        UInt8(2),
        UInt8(0),
        UInt8(0),
        UInt8(0),
        UInt8(0),
        UInt8(0),
        UInt8(0),
        UInt8(0),
        UInt8(ord("{")),
        UInt8(ord("}")),
    ]
    _ = parse_metadata(archive, strict=True)
    _ = open_safetensors("compile-contract.safetensors", strict=True)
    _ = map_safetensors("compile-contract.safetensors", strict=True)


def _sharded_reader_contract(paths: List[String]) raises:
    var reader: ShardedSafeTensorReader = open_sharded_safetensors(
        paths, max_shards=DEFAULT_MAX_SHARDS, strict=True
    )
    var metadata: ShardedSafeTensorMetadata = reader.metadata()
    var info: ShardedTensorInfo = metadata.info("tensor")
    _ = metadata.shard_grouped_names()
    _ = reader.load_tensor(info.name)

    var indexed: ShardedSafeTensorReader = open_safetensors_index(
        "compile-contract.safetensors.index.json",
        max_index_bytes=DEFAULT_MAX_INDEX_BYTES,
        max_index_entries=DEFAULT_MAX_INDEX_ENTRIES,
        strict=True,
    )
    _ = indexed.metadata()

    var mapped: MappedShardedSafeTensorArchive = map_sharded_safetensors(paths)
    _ = mapped.tensor_bytes("tensor")
    var mapped_index: MappedShardedSafeTensorArchive = map_safetensors_index(
        "compile-contract.safetensors.index.json",
        max_index_entries=DEFAULT_MAX_INDEX_ENTRIES,
    )
    _ = mapped_index.metadata()


def main():
    print(DEFAULT_MAX_HEADER_BYTES)
    print(
        DEFAULT_MAX_INDEX_BYTES,
        DEFAULT_MAX_INDEX_ENTRIES,
        DEFAULT_MAX_SHARDS,
    )
