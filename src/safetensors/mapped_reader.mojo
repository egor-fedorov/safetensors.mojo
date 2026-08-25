"""Linux read-only mappings with origin-bound Safetensors byte views."""

from std.ffi import c_int, c_long, c_size_t, external_call

from .checked import checked_add_u64, checked_sub_u64, checked_u64_to_int
from .errors import SafeTensorError, SafeTensorErrorKind, make_error
from .model import SafeTensorMetadata
from .parser import DEFAULT_MAX_HEADER_BYTES
from .reader import (
    _ValidatedFile,
    _io_error,
    _open_validated_file,
    _require_file_length,
)


comptime _PROT_READ = 1
comptime _MAP_PRIVATE = 2


struct _ReadOnlyMapping(Movable):
    """An owned whole-file mapping released exactly once during destruction."""

    var _pointer: Pointer[UInt8, ImmUntrackedOrigin]
    var _length: Int

    def __init__(
        out self, file: FileHandle, length: Int
    ) raises SafeTensorError:
        var pointer = external_call["mmap", Pointer[UInt8, ImmUntrackedOrigin]](
            Optional[Pointer[UInt8, MutUntrackedOrigin]](None),
            c_size_t(length),
            c_int(_PROT_READ),
            c_int(_MAP_PRIVATE),
            c_int(file.handle),
            c_long(0),
        )
        if Int(pointer) == -1:
            raise _io_error("file mapping failed")
        self._pointer = pointer
        self._length = length

    def __deinit__(deinit self):
        _ = external_call["munmap", c_int](
            self._pointer, c_size_t(self._length)
        )


struct MappedSafeTensorFile(Movable):
    """An owned read-only mapping with validated Safetensors metadata.

    The backing file must remain unchanged from the call that created this
    value until this value and every borrowed view are dead. External
    truncation can cause an operating-system fault when mapped memory is
    accessed.
    """

    var _opened: _ValidatedFile
    var _mapping: _ReadOnlyMapping

    def __init__(
        out self,
        var opened: _ValidatedFile,
        var mapping: _ReadOnlyMapping,
    ):
        """Internal hook for state created by map_safetensors."""
        self._opened = opened^
        self._mapping = mapping^

    def metadata(self) -> SafeTensorMetadata:
        """Returns a validated metadata copy for the mapped file."""
        return self._opened.metadata.copy()

    def file_length(self) -> UInt64:
        """Returns the complete file length observed during mapping."""
        return self._opened.file_length

    def tensor_bytes[
        origin: ImmOrigin
    ](
        ref[origin] self,
        name: String,
    ) raises SafeTensorError -> Span[
        UInt8, origin
    ]:
        """Returns an immutable zero-copy view of one tensor's wire bytes.

        The returned span cannot remain usable after this mapping owner is
        consumed. The backing file must not be modified while the span is used.
        """
        var info = self._opened.metadata.info(name)
        var absolute_begin = checked_add_u64(
            self._opened.metadata.data_start(), info.begin
        )
        var absolute_end = checked_add_u64(
            self._opened.metadata.data_start(), info.end
        )
        if absolute_end < absolute_begin:
            raise make_error(
                SafeTensorErrorKind.INVALID_OFFSETS,
                "mapped tensor range ends before it begins",
            )
        var absolute_length = checked_sub_u64(absolute_end, absolute_begin)
        if absolute_length != info.byte_length:
            raise make_error(
                SafeTensorErrorKind.INVALID_OFFSETS,
                "mapped tensor range disagrees with validated byte length",
            )
        if absolute_end > self._opened.file_length:
            raise make_error(
                SafeTensorErrorKind.INVALID_OFFSETS,
                "mapped tensor range extends beyond the opened file",
            )
        if absolute_end > UInt64(self._mapping._length):
            raise make_error(
                SafeTensorErrorKind.INVALID_OFFSETS,
                "mapped tensor range extends beyond the mapped region",
            )

        var begin_index = checked_u64_to_int(absolute_begin)
        var byte_length = checked_u64_to_int(info.byte_length)
        _require_file_length(self._opened.file, self._opened.file_length)

        var pointer_offset = begin_index
        if byte_length == 0:
            pointer_offset = 0
        var pointer = self._mapping._pointer.unsafe_offset(
            pointer_offset
        ).unsafe_origin_cast[origin]()
        return Span[UInt8, origin](unsafe_ptr=pointer, length=byte_length)


def map_safetensors(
    path: String,
    max_header_bytes: UInt64 = DEFAULT_MAX_HEADER_BYTES,
) raises SafeTensorError -> MappedSafeTensorFile:
    """Validates and maps one local Safetensors file without copying payloads.

    The caller must keep the backing file unchanged from before this call
    begins until the returned owner and every view borrowed from it are dead.
    """
    var opened = _open_validated_file(path, max_header_bytes)
    var mapping_length = checked_u64_to_int(opened.file_length)
    var mapping = _ReadOnlyMapping(opened.file, mapping_length)
    _require_file_length(opened.file, opened.file_length)
    return MappedSafeTensorFile(opened^, mapping^)
