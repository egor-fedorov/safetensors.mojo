"""Eager origin-bound mappings for validated local shard sets."""

from std.collections import Dict, List

from safetensors.errors import SafeTensorError, SafeTensorErrorKind, make_error
from safetensors.format.checked import checked_u64_to_int
from safetensors.format.parser import DEFAULT_MAX_HEADER_BYTES
from safetensors.io._file import _require_file_length
from safetensors.io.mapped_reader import MappedSafeTensorFile, _ReadOnlyMapping
from safetensors.sharding.archive import (
    _ArchiveScan,
    _reopen_validated_shard,
    _scan_explicit_paths,
    _scan_index_shards,
)
from safetensors.sharding.index_parser import (
    DEFAULT_MAX_INDEX_BYTES,
    DEFAULT_MAX_INDEX_ENTRIES,
    DEFAULT_MAX_SHARDS,
)
from safetensors.sharding.model import ShardedSafeTensorMetadata
from safetensors.sharding.resolver import _open_index_document


def _map_scanned_shards(
    scan: _ArchiveScan,
    directory: FileHandle,
    anchored: Bool,
    max_header_bytes: UInt64,
    strict: Bool,
) raises SafeTensorError -> List[MappedSafeTensorFile]:
    var mappings = List[MappedSafeTensorFile]()
    for index in range(len(scan.shards)):
        var opened = _reopen_validated_shard(
            scan.shards[index],
            directory,
            anchored,
            max_header_bytes,
            strict,
        )
        var mapping_length = checked_u64_to_int(opened.file_length)
        var mapping = _ReadOnlyMapping(opened.file, mapping_length)
        _require_file_length(opened.file, opened.file_length)
        mappings.append(MappedSafeTensorFile(opened^, mapping^))
    return mappings^


struct MappedShardedSafeTensorArchive(Movable):
    """An immutable aggregate owning one whole-file mapping per unique shard."""

    var _metadata: ShardedSafeTensorMetadata
    var _tensor_to_shard: Dict[String, Int]
    var _mappings: List[MappedSafeTensorFile]

    def __init__(
        out self,
        scan: _ArchiveScan,
        var mappings: List[MappedSafeTensorFile],
    ):
        """Internal hook for an eagerly mapped validated shard scan."""
        self._metadata = scan.metadata.copy()
        self._tensor_to_shard = scan.tensor_to_shard.copy()
        self._mappings = mappings^

    def metadata(self) -> ShardedSafeTensorMetadata:
        """Returns a copy of the validated aggregate metadata."""
        return self._metadata.copy()

    def _route(self, name: String) raises SafeTensorError -> Int:
        var maybe_shard = self._tensor_to_shard.get(name)
        if not maybe_shard:
            raise make_error(
                SafeTensorErrorKind.TENSOR_NOT_FOUND,
                "tensor not found",
            )
        return maybe_shard.value()

    def _bind_span_to_owner[
        origin: ImmOrigin,
        element: AnyType,
        inner_origin: ImmOrigin,
    ](
        ref[origin] self,
        inner: Span[element, inner_origin],
    ) -> Span[
        element, origin
    ]:
        """Binds one immutable shard-element span to the aggregate owner.

        This is the only origin bridge for List interior origins. The mapping
        list is fully populated before construction completes and is never
        mutated while the archive is alive, so each element remains owned by
        this outer archive for the complete outer borrow.
        """
        var pointer = inner.unsafe_ptr().unsafe_origin_cast[origin]()
        return Span[element, origin](unsafe_ptr=pointer, length=len(inner))

    def tensor_bytes[
        origin: ImmOrigin
    ](
        ref[origin] self,
        name: String,
    ) raises SafeTensorError -> Span[
        UInt8, origin
    ]:
        """Returns an immutable zero-copy view from the routed shard."""
        var shard_index = self._route(name)
        var inner = self._mappings[shard_index].tensor_bytes(name)
        return self._bind_span_to_owner(inner)

    def tensor_view[
        origin: ImmOrigin,
        //,
        dtype: DType,
    ](
        ref[origin] self,
        name: String,
    ) raises SafeTensorError -> Span[
        Scalar[dtype], origin
    ]:
        """Returns an exact immutable native scalar view from one shard."""
        var shard_index = self._route(name)
        var inner = self._mappings[shard_index].tensor_view[dtype](name)
        return self._bind_span_to_owner(inner)


def map_sharded_safetensors(
    paths: List[String],
    max_header_bytes: UInt64 = DEFAULT_MAX_HEADER_BYTES,
    max_shards: UInt64 = DEFAULT_MAX_SHARDS,
    strict: Bool = False,
) raises SafeTensorError -> MappedShardedSafeTensorArchive:
    """Validates and eagerly maps a caller-trusted shard list."""
    var scan = _scan_explicit_paths(paths, max_header_bytes, max_shards, strict)
    var no_directory = FileHandle()
    var mappings = _map_scanned_shards(
        scan,
        no_directory,
        False,
        max_header_bytes,
        strict,
    )
    return MappedShardedSafeTensorArchive(scan, mappings^)


def map_safetensors_index(
    index_path: String,
    max_index_bytes: UInt64 = DEFAULT_MAX_INDEX_BYTES,
    max_index_entries: UInt64 = DEFAULT_MAX_INDEX_ENTRIES,
    max_header_bytes: UInt64 = DEFAULT_MAX_HEADER_BYTES,
    max_shards: UInt64 = DEFAULT_MAX_SHARDS,
    strict: Bool = False,
) raises SafeTensorError -> MappedShardedSafeTensorArchive:
    """Validates and eagerly maps an anchored shard index."""
    var opened_index = _open_index_document(
        index_path, max_index_bytes, max_index_entries, max_shards
    )
    var scan = _scan_index_shards(
        opened_index.directory,
        opened_index.parsed,
        max_header_bytes,
        max_shards,
        strict,
    )
    var mappings = _map_scanned_shards(
        scan,
        opened_index.directory,
        True,
        max_header_bytes,
        strict,
    )
    return MappedShardedSafeTensorArchive(scan, mappings^)
