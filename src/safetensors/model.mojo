"""Raw and validated metadata models for Safetensors headers."""

from .dtype import SafeDType
from .errors import SafeTensorError, SafeTensorErrorKind, make_error


@fieldwise_init
struct RawTensorInfo(Copyable, Movable, Writable):
    """An unvalidated tensor descriptor decoded directly from JSON."""

    var name: String
    var dtype_name: String
    var shape: List[UInt64]
    var begin: UInt64
    var end: UInt64


@fieldwise_init
struct RawSafeTensorMetadata(Copyable, Movable, Writable):
    """A decoded header whose dtype, shape, and offset invariants are unchecked.
    """

    var user_metadata: Dict[String, String]
    var tensors: List[RawTensorInfo]


@fieldwise_init
struct TensorInfo(Copyable, Movable, Writable):
    """An immutable-by-copy descriptor with validation-derived sizes."""

    var name: String
    var dtype: SafeDType
    var shape: List[UInt64]
    var begin: UInt64
    var end: UInt64
    var element_count: UInt64
    var bit_length: UInt64
    var byte_length: UInt64


struct SafeTensorMetadata(Copyable, Movable, Sized, Writable):
    """Validated metadata with protected invariants and name/offset indexes."""

    var _user_metadata: Dict[String, String]
    var _tensors_by_name: Dict[String, Int]
    var _tensors_in_offset_order: List[TensorInfo]
    var _data_start: UInt64
    var _data_length: UInt64

    def __init__(
        out self,
        user_metadata: Dict[String, String],
        tensors_in_offset_order: List[TensorInfo],
        data_start: UInt64,
        data_length: UInt64,
    ) raises SafeTensorError:
        """Internal hook that indexes descriptors validated by the caller.

        Use parse_metadata or validate_metadata instead of calling this directly.
        """
        var tensors_by_name = Dict[String, Int]()
        for index in range(len(tensors_in_offset_order)):
            var tensor_name = tensors_in_offset_order[index].name.copy()
            if tensor_name in tensors_by_name:
                raise make_error(
                    SafeTensorErrorKind.DUPLICATE_KEY,
                    "duplicate decoded tensor name",
                )
            tensors_by_name[tensor_name] = index

        self._user_metadata = user_metadata.copy()
        self._tensors_by_name = tensors_by_name^
        self._tensors_in_offset_order = tensors_in_offset_order.copy()
        self._data_start = data_start
        self._data_length = data_length

    def __len__(self) -> Int:
        """Returns the number of tensors in the header."""
        return len(self._tensors_in_offset_order)

    def len(self) -> Int:
        """Returns the number of tensors in the header."""
        return len(self._tensors_in_offset_order)

    def is_empty(self) -> Bool:
        """Returns whether the header contains no tensors."""
        return len(self._tensors_in_offset_order) == 0

    def contains(self, name: String) -> Bool:
        """Returns whether a tensor with the exact decoded name exists."""
        return name in self._tensors_by_name

    def names(self) -> List[String]:
        """Returns tensor names in deterministic data-offset order."""
        return self.offset_names()

    def offset_names(self) -> List[String]:
        """Returns tensor names in validated data-offset order."""
        var result = List[String]()
        for index in range(len(self._tensors_in_offset_order)):
            result.append(self._tensors_in_offset_order[index].name.copy())
        return result^

    def info(self, name: String) raises SafeTensorError -> TensorInfo:
        """Returns a descriptor copy, preserving the metadata's invariants."""
        var maybe_index = self._tensors_by_name.get(name)
        if not maybe_index:
            raise make_error(
                SafeTensorErrorKind.TENSOR_NOT_FOUND,
                "tensor not found",
            )
        return self._tensors_in_offset_order[maybe_index.value()].copy()

    def tensors_in_offset_order(self) -> List[TensorInfo]:
        """Returns descriptor copies sorted by validated data offset."""
        return self._tensors_in_offset_order.copy()

    def user_metadata(self) -> Dict[String, String]:
        """Returns a copy of user metadata from the reserved JSON entry."""
        return self._user_metadata.copy()

    def metadata_value(self, key: String) -> Optional[String]:
        """Returns a copied user-metadata value when the key exists."""
        return self._user_metadata.get(key)

    def data_start(self) -> UInt64:
        """Returns the supplied data origin, absolute for complete buffers."""
        return self._data_start

    def data_length(self) -> UInt64:
        """Returns the validated tensor data-buffer length."""
        return self._data_length
