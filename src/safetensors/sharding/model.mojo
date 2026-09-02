"""Validated aggregate metadata for local sharded Safetensors archives."""

from std.builtin.sort import sort

from safetensors.errors import SafeTensorError, SafeTensorErrorKind, make_error
from safetensors.format.checked import checked_add_u64
from safetensors.format.dtype import SafeDType


@fieldwise_init
struct ShardedTensorInfo(Copyable, Movable, Writable):
    """One validated tensor descriptor paired with its physical shard."""

    var name: String
    var dtype: SafeDType
    var shape: List[UInt64]
    var begin: UInt64
    var end: UInt64
    var element_count: UInt64
    var bit_length: UInt64
    var byte_length: UInt64
    var shard: String


def _tensor_name_less(
    left: ShardedTensorInfo, right: ShardedTensorInfo
) capturing -> Bool:
    return left.name < right.name


def _shard_name_less(left: String, right: String) capturing -> Bool:
    return left < right


struct ShardedSafeTensorMetadata(Copyable, Movable, Sized, Writable):
    """Validated metadata for one global namespace spanning local shards."""

    var _tensors_by_name: Dict[String, Int]
    var _tensors_by_name_order: List[ShardedTensorInfo]
    var _shard_names: List[String]
    var _total_size: UInt64
    var _declared_total_size: Optional[UInt64]

    def __init__(
        out self,
        var tensors: List[ShardedTensorInfo],
        var shard_names: List[String],
        declared_total_size: Optional[UInt64] = None,
    ) raises SafeTensorError:
        """Indexes aggregate state after shard validation has completed."""
        sort[T=ShardedTensorInfo, cmp_fn=_tensor_name_less](tensors)
        sort[T=String, cmp_fn=_shard_name_less](shard_names)

        var tensors_by_name = Dict[String, Int]()
        var total_size: UInt64 = 0
        for index in range(len(tensors)):
            var name = tensors[index].name.copy()
            if name in tensors_by_name:
                raise make_error(
                    SafeTensorErrorKind.SHARD_MISMATCH,
                    "tensor name appears in more than one shard",
                )
            tensors_by_name[name] = index
            total_size = checked_add_u64(total_size, tensors[index].byte_length)

        self._tensors_by_name = tensors_by_name^
        self._tensors_by_name_order = tensors^
        self._shard_names = shard_names^
        self._total_size = total_size
        self._declared_total_size = declared_total_size

    def __len__(self) -> Int:
        """Returns the number of tensors across every shard."""
        return len(self._tensors_by_name_order)

    def len(self) -> Int:
        """Returns the number of tensors across every shard."""
        return len(self._tensors_by_name_order)

    def is_empty(self) -> Bool:
        """Returns whether the aggregate tensor namespace is empty."""
        return len(self._tensors_by_name_order) == 0

    def contains(self, name: String) -> Bool:
        """Returns whether the exact decoded tensor name exists."""
        return name in self._tensors_by_name

    def names(self) -> List[String]:
        """Returns decoded tensor names in lexicographic order."""
        var result = List[String]()
        for index in range(len(self._tensors_by_name_order)):
            result.append(self._tensors_by_name_order[index].name.copy())
        return result^

    def info(self, name: String) raises SafeTensorError -> ShardedTensorInfo:
        """Returns a descriptor copy without exposing validation state."""
        var maybe_index = self._tensors_by_name.get(name)
        if not maybe_index:
            raise make_error(
                SafeTensorErrorKind.TENSOR_NOT_FOUND,
                "tensor not found",
            )
        return self._tensors_by_name_order[maybe_index.value()].copy()

    def shard_names(self) -> List[String]:
        """Returns unique shard identifiers in lexicographic order."""
        return self._shard_names.copy()

    def total_size(self) -> UInt64:
        """Returns the checked sum of all tensor payload byte lengths."""
        return self._total_size

    def declared_total_size(self) -> Optional[UInt64]:
        """Returns the optional exact size declared by an index document."""
        return self._declared_total_size
