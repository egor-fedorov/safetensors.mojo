"""Cross-shard scanning, deduplication, and aggregate validation."""

from std.builtin.sort import sort
from std.collections import Dict, List

from safetensors.errors import SafeTensorError, SafeTensorErrorKind, make_error
from safetensors.format.model import SafeTensorMetadata, TensorInfo
from safetensors.format.parser import DEFAULT_MAX_HEADER_BYTES
from safetensors.io._file import _ValidatedFile, _validate_open_file
from safetensors.sharding.index_parser import (
    DEFAULT_MAX_SHARDS,
    _ParsedIndex,
)
from safetensors.sharding.model import (
    ShardedSafeTensorMetadata,
    ShardedTensorInfo,
)
from safetensors.sharding.resolver import (
    _FileIdentity,
    _descriptor_state,
    _open_index_shard,
    _open_trusted_regular,
    _same_identity,
    _validate_shard_basename,
)


@fieldwise_init
struct _ShardSpec(Copyable, Movable):
    """Immutable reopening and validation snapshot for one unique shard."""

    var path: String
    var display_name: String
    var identity: _FileIdentity
    var file_length: UInt64
    var metadata: SafeTensorMetadata


@fieldwise_init
struct _ArchiveScan(Movable):
    """Fully validated aggregate state without retained shard descriptors."""

    var shards: List[_ShardSpec]
    var metadata: ShardedSafeTensorMetadata
    var tensor_to_shard: Dict[String, Int]


@fieldwise_init
struct _AggregateState(Movable):
    """Aggregate metadata paired with its tensor routing lookup."""

    var metadata: ShardedSafeTensorMetadata
    var tensor_to_shard: Dict[String, Int]


def _string_less(left: String, right: String) capturing -> Bool:
    return left < right


def _find_identity(shards: List[_ShardSpec], identity: _FileIdentity) -> Int:
    for index in range(len(shards)):
        if _same_identity(shards[index].identity, identity):
            return index
    return -1


def _same_tensor_info(left: TensorInfo, right: TensorInfo) -> Bool:
    return (
        left.name == right.name
        and left.dtype == right.dtype
        and left.shape == right.shape
        and left.begin == right.begin
        and left.end == right.end
        and left.element_count == right.element_count
        and left.bit_length == right.bit_length
        and left.byte_length == right.byte_length
    )


def _same_metadata(left: SafeTensorMetadata, right: SafeTensorMetadata) -> Bool:
    if (
        left.data_start() != right.data_start()
        or left.data_length() != right.data_length()
        or len(left) != len(right)
    ):
        return False

    var left_tensors = left.tensors_in_offset_order()
    var right_tensors = right.tensors_in_offset_order()
    for index in range(len(left_tensors)):
        if not _same_tensor_info(left_tensors[index], right_tensors[index]):
            return False

    var left_user = left.user_metadata()
    var right_user = right.user_metadata()
    if len(left_user) != len(right_user):
        return False
    for key in left_user:
        var right_value = right_user.get(key)
        if not right_value:
            return False
        var left_value = left_user.get(key)
        if not left_value or left_value.value() != right_value.value():
            return False
    return True


def _as_sharded_info(info: TensorInfo, shard: String) -> ShardedTensorInfo:
    return ShardedTensorInfo(
        info.name.copy(),
        info.dtype,
        info.shape.copy(),
        info.begin,
        info.end,
        info.element_count,
        info.bit_length,
        info.byte_length,
        shard.copy(),
    )


def _append_opened_shard(
    var file: FileHandle,
    path: String,
    display_name: String,
    mut shards: List[_ShardSpec],
    max_header_bytes: UInt64,
    max_shards: UInt64,
    strict: Bool,
) raises SafeTensorError -> Int:
    var state = _descriptor_state(file, SafeTensorErrorKind.IO_ERROR)
    var existing = _find_identity(shards, state.identity)
    if existing >= 0:
        return existing
    if UInt64(len(shards)) >= max_shards:
        raise make_error(
            SafeTensorErrorKind.SHARD_LIMIT_EXCEEDED,
            "unique shard count exceeds the configured limit",
        )

    var opened = _validate_open_file(file^, max_header_bytes, strict)
    var after = _descriptor_state(opened.file, SafeTensorErrorKind.IO_ERROR)
    if (
        not _same_identity(state.identity, after.identity)
        or state.length != after.length
        or opened.file_length != state.length
    ):
        raise make_error(
            SafeTensorErrorKind.IO_ERROR,
            "shard changed while its header was validated",
        )
    shards.append(
        _ShardSpec(
            path.copy(),
            display_name.copy(),
            state.identity.copy(),
            state.length,
            opened.metadata.copy(),
        )
    )
    return len(shards) - 1


def _build_explicit_aggregate(
    shards: List[_ShardSpec],
) raises SafeTensorError -> _AggregateState:
    var tensors = List[ShardedTensorInfo]()
    var shard_names = List[String]()
    var tensor_to_shard = Dict[String, Int]()
    for shard_index in range(len(shards)):
        shard_names.append(shards[shard_index].display_name.copy())
        var infos = shards[shard_index].metadata.tensors_in_offset_order()
        for info in infos:
            if info.name in tensor_to_shard:
                raise make_error(
                    SafeTensorErrorKind.SHARD_MISMATCH,
                    "tensor name appears in more than one shard",
                )
            tensor_to_shard[info.name.copy()] = shard_index
            tensors.append(
                _as_sharded_info(info, shards[shard_index].display_name)
            )

    var metadata = ShardedSafeTensorMetadata(tensors^, shard_names^)
    return _AggregateState(metadata^, tensor_to_shard^)


def _build_index_aggregate(
    shards: List[_ShardSpec],
    parsed: _ParsedIndex,
    alias_to_shard: Dict[String, Int],
) raises SafeTensorError -> _AggregateState:
    var tensors = List[ShardedTensorInfo]()
    var shard_names = List[String]()
    var tensor_to_shard = Dict[String, Int]()
    for shard_index in range(len(shards)):
        shard_names.append(shards[shard_index].display_name.copy())
        var infos = shards[shard_index].metadata.tensors_in_offset_order()
        for info in infos:
            if info.name in tensor_to_shard:
                raise make_error(
                    SafeTensorErrorKind.SHARD_MISMATCH,
                    "tensor name appears physically in more than one shard",
                )
            var declared_name = parsed.weight_map.get(info.name)
            if not declared_name:
                raise make_error(
                    SafeTensorErrorKind.SHARD_MISMATCH,
                    "a shard tensor is omitted from weight_map",
                )
            var declared_index = alias_to_shard.get(declared_name.value())
            if not declared_index or declared_index.value() != shard_index:
                raise make_error(
                    SafeTensorErrorKind.SHARD_MISMATCH,
                    "weight_map routes a tensor to the wrong shard",
                )
            tensor_to_shard[info.name.copy()] = shard_index
            tensors.append(
                _as_sharded_info(info, shards[shard_index].display_name)
            )

    for tensor_name in parsed.weight_map:
        if tensor_name not in tensor_to_shard:
            raise make_error(
                SafeTensorErrorKind.SHARD_MISMATCH,
                "weight_map names a tensor absent from its shard",
            )

    var metadata = ShardedSafeTensorMetadata(
        tensors^,
        shard_names^,
        parsed.declared_total_size,
    )
    if (
        parsed.declared_total_size
        and parsed.declared_total_size.value() != metadata.total_size()
    ):
        raise make_error(
            SafeTensorErrorKind.TOTAL_SIZE_MISMATCH,
            "metadata.total_size disagrees with tensor payload bytes",
        )
    return _AggregateState(metadata^, tensor_to_shard^)


def _scan_explicit_paths(
    paths: List[String],
    max_header_bytes: UInt64 = DEFAULT_MAX_HEADER_BYTES,
    max_shards: UInt64 = DEFAULT_MAX_SHARDS,
    strict: Bool = False,
) raises SafeTensorError -> _ArchiveScan:
    """Validates and deduplicates a non-empty caller-trusted path list."""
    if len(paths) == 0:
        raise make_error(
            SafeTensorErrorKind.INVALID_INDEX,
            "at least one shard path is required",
        )
    var shards = List[_ShardSpec]()
    for path in paths:
        var file = _open_trusted_regular(path)
        _ = _append_opened_shard(
            file^,
            path,
            path,
            shards,
            max_header_bytes,
            max_shards,
            strict,
        )
    var aggregate = _build_explicit_aggregate(shards)
    return _ArchiveScan(
        shards^, aggregate.metadata.copy(), aggregate.tensor_to_shard.copy()
    )


def _index_shard_names(
    parsed: _ParsedIndex,
) raises SafeTensorError -> List[String]:
    var names = List[String]()
    var seen = Dict[String, Bool]()
    for tensor_name in parsed.weight_map:
        var maybe_name = parsed.weight_map.get(tensor_name)
        if not maybe_name:
            raise make_error(
                SafeTensorErrorKind.INVALID_INDEX,
                "weight_map entry disappeared during validation",
            )
        var shard_name = maybe_name.value().copy()
        _validate_shard_basename(shard_name)
        if shard_name not in seen:
            seen[shard_name.copy()] = True
            names.append(shard_name^)
    sort[T=String, cmp_fn=_string_less](names)
    return names^


def _scan_index_shards(
    directory: FileHandle,
    parsed: _ParsedIndex,
    max_header_bytes: UInt64 = DEFAULT_MAX_HEADER_BYTES,
    max_shards: UInt64 = DEFAULT_MAX_SHARDS,
    strict: Bool = False,
) raises SafeTensorError -> _ArchiveScan:
    """Validates every unique shard and exact index-to-file consistency."""
    var shard_paths = _index_shard_names(parsed)
    var shards = List[_ShardSpec]()
    var alias_to_shard = Dict[String, Int]()
    for basename in shard_paths:
        var file = _open_index_shard(directory, basename)
        var shard_index = _append_opened_shard(
            file^,
            basename,
            basename,
            shards,
            max_header_bytes,
            max_shards,
            strict,
        )
        alias_to_shard[basename.copy()] = shard_index

    var aggregate = _build_index_aggregate(shards, parsed, alias_to_shard)
    return _ArchiveScan(
        shards^, aggregate.metadata.copy(), aggregate.tensor_to_shard.copy()
    )


def _reopen_validated_shard(
    spec: _ShardSpec,
    directory: FileHandle,
    anchored: Bool,
    max_header_bytes: UInt64,
    strict: Bool,
) raises SafeTensorError -> _ValidatedFile:
    """Reopens and compares one shard with its construction snapshot."""
    var file: FileHandle
    if anchored:
        file = _open_index_shard(directory, spec.path)
    else:
        file = _open_trusted_regular(spec.path)
    var state = _descriptor_state(file, SafeTensorErrorKind.IO_ERROR)
    if (
        not _same_identity(state.identity, spec.identity)
        or state.length != spec.file_length
    ):
        raise make_error(
            SafeTensorErrorKind.IO_ERROR,
            "shard identity or length changed after archive validation",
        )
    var opened = _validate_open_file(file^, max_header_bytes, strict)
    if opened.file_length != spec.file_length or not _same_metadata(
        opened.metadata, spec.metadata
    ):
        raise make_error(
            SafeTensorErrorKind.IO_ERROR,
            "shard metadata changed after archive validation",
        )
    return opened^
