"""Buffered random-access reader for validated local shard sets."""

from std.collections import Dict, List

from safetensors.errors import SafeTensorError, SafeTensorErrorKind, make_error
from safetensors.format.checked import checked_u64_to_int
from safetensors.format.parser import DEFAULT_MAX_HEADER_BYTES
from safetensors.io.reader import SafeTensorReader
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


struct ShardedSafeTensorReader(Movable):
    """A validated aggregate with at most one active shard reader."""

    var _scan: _ArchiveScan
    var _directory: FileHandle
    var _anchored: Bool
    var _max_header_bytes: UInt64
    var _strict: Bool
    var _active: List[SafeTensorReader]
    var _active_shard: Int

    def __init__(
        out self,
        var scan: _ArchiveScan,
        var directory: FileHandle,
        anchored: Bool,
        max_header_bytes: UInt64,
        strict: Bool,
    ):
        """Internal hook for a completely validated shard scan."""
        self._scan = scan^
        self._directory = directory^
        self._anchored = anchored
        self._max_header_bytes = max_header_bytes
        self._strict = strict
        self._active = List[SafeTensorReader]()
        self._active_shard = -1

    def metadata(self) -> ShardedSafeTensorMetadata:
        """Returns a copy of the validated aggregate metadata."""
        return self._scan.metadata.copy()

    def _route(self, name: String) raises SafeTensorError -> Int:
        var maybe_shard = self._scan.tensor_to_shard.get(name)
        if not maybe_shard:
            raise make_error(
                SafeTensorErrorKind.TENSOR_NOT_FOUND,
                "tensor not found",
            )
        return maybe_shard.value()

    def _activate(mut self, shard_index: Int) raises SafeTensorError:
        if self._active_shard == shard_index and len(self._active) == 1:
            return
        self._active = List[SafeTensorReader]()
        self._active_shard = -1
        var opened = _reopen_validated_shard(
            self._scan.shards[shard_index],
            self._directory,
            self._anchored,
            self._max_header_bytes,
            self._strict,
        )
        self._active.append(SafeTensorReader(opened^))
        self._active_shard = shard_index

    def read_tensor_into[
        origin: MutOrigin
    ](
        mut self,
        name: String,
        destination: Span[UInt8, origin],
    ) raises SafeTensorError:
        """Reads one globally named tensor into an exactly sized buffer."""
        var shard_index = self._route(name)
        self._activate(shard_index)
        self._active[0].read_tensor_into(name, destination)

    def load_tensor(
        mut self, name: String
    ) raises SafeTensorError -> List[UInt8]:
        """Allocates and returns the exact raw wire bytes of one tensor."""
        var info = self._scan.metadata.info(name)
        var destination = List[UInt8](
            length=checked_u64_to_int(info.byte_length), fill=0
        )
        self.read_tensor_into(name, destination)
        return destination^


def open_sharded_safetensors(
    paths: List[String],
    max_header_bytes: UInt64 = DEFAULT_MAX_HEADER_BYTES,
    max_shards: UInt64 = DEFAULT_MAX_SHARDS,
    strict: Bool = False,
) raises SafeTensorError -> ShardedSafeTensorReader:
    """Opens a caller-trusted shard list, following path symlinks."""
    var scan = _scan_explicit_paths(paths, max_header_bytes, max_shards, strict)
    var no_directory = FileHandle()
    return ShardedSafeTensorReader(
        scan^, no_directory^, False, max_header_bytes, strict
    )


def open_safetensors_index(
    index_path: String,
    max_index_bytes: UInt64 = DEFAULT_MAX_INDEX_BYTES,
    max_index_entries: UInt64 = DEFAULT_MAX_INDEX_ENTRIES,
    max_header_bytes: UInt64 = DEFAULT_MAX_HEADER_BYTES,
    max_shards: UInt64 = DEFAULT_MAX_SHARDS,
    strict: Bool = False,
) raises SafeTensorError -> ShardedSafeTensorReader:
    """Opens an index with anchored, no-symlink shard resolution."""
    var opened_index = _open_index_document(
        index_path, max_index_bytes, max_index_entries, max_shards
    )
    var directory = opened_index.directory^
    opened_index.directory = FileHandle()
    var scan = _scan_index_shards(
        directory,
        opened_index.parsed,
        max_header_bytes,
        max_shards,
        strict,
    )
    return ShardedSafeTensorReader(
        scan^,
        directory^,
        True,
        max_header_bytes,
        strict,
    )
