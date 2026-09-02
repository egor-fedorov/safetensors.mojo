"""Stable error categories and typed errors for Safetensors parsing."""


@fieldwise_init
struct SafeTensorErrorKind(Equatable, ImplicitlyCopyable, Writable):
    """A stable, machine-readable Safetensors error category."""

    var _value: UInt8

    comptime HEADER_TOO_SMALL = Self(0)
    comptime HEADER_TOO_LARGE = Self(1)
    comptime INVALID_HEADER_LENGTH = Self(2)
    comptime INVALID_HEADER_START = Self(3)
    comptime INVALID_UTF8 = Self(4)
    comptime INVALID_JSON = Self(5)
    comptime DUPLICATE_KEY = Self(6)
    comptime INVALID_METADATA = Self(7)
    comptime MISSING_FIELD = Self(8)
    comptime INVALID_FIELD_TYPE = Self(9)
    comptime UNSUPPORTED_DTYPE = Self(10)
    comptime INVALID_SHAPE = Self(11)
    comptime INVALID_OFFSETS = Self(12)
    comptime INVALID_TENSOR_SIZE = Self(13)
    comptime INCOMPLETE_DATA_COVERAGE = Self(14)
    comptime VALIDATION_OVERFLOW = Self(15)
    comptime MISALIGNED_TENSOR = Self(16)
    comptime MISALIGNED_SLICE = Self(17)
    comptime TENSOR_NOT_FOUND = Self(18)
    comptime DESTINATION_SIZE_MISMATCH = Self(19)
    comptime UNSUPPORTED_ENDIANNESS = Self(20)
    comptime IO_ERROR = Self(21)
    comptime PATH_TRAVERSAL = Self(22)
    comptime UNKNOWN_FIELD = Self(23)
    comptime INVALID_HEADER_PADDING = Self(24)
    comptime DTYPE_MISMATCH = Self(25)
    comptime INDEX_TOO_LARGE = Self(26)
    comptime INVALID_INDEX = Self(27)
    comptime SHARD_MISMATCH = Self(28)
    comptime TOTAL_SIZE_MISMATCH = Self(29)
    comptime SHARD_LIMIT_EXCEEDED = Self(30)

    def code(self) -> String:
        """Returns the stable spelling used in diagnostics and tests."""
        if self == Self.HEADER_TOO_SMALL:
            return "HeaderTooSmall"
        if self == Self.HEADER_TOO_LARGE:
            return "HeaderTooLarge"
        if self == Self.INVALID_HEADER_LENGTH:
            return "InvalidHeaderLength"
        if self == Self.INVALID_HEADER_START:
            return "InvalidHeaderStart"
        if self == Self.INVALID_UTF8:
            return "InvalidUtf8"
        if self == Self.INVALID_JSON:
            return "InvalidJson"
        if self == Self.DUPLICATE_KEY:
            return "DuplicateKey"
        if self == Self.INVALID_METADATA:
            return "InvalidMetadata"
        if self == Self.MISSING_FIELD:
            return "MissingField"
        if self == Self.INVALID_FIELD_TYPE:
            return "InvalidFieldType"
        if self == Self.UNSUPPORTED_DTYPE:
            return "UnsupportedDType"
        if self == Self.INVALID_SHAPE:
            return "InvalidShape"
        if self == Self.INVALID_OFFSETS:
            return "InvalidOffsets"
        if self == Self.INVALID_TENSOR_SIZE:
            return "InvalidTensorSize"
        if self == Self.INCOMPLETE_DATA_COVERAGE:
            return "IncompleteDataCoverage"
        if self == Self.VALIDATION_OVERFLOW:
            return "ValidationOverflow"
        if self == Self.MISALIGNED_TENSOR:
            return "MisalignedTensor"
        if self == Self.MISALIGNED_SLICE:
            return "MisalignedSlice"
        if self == Self.TENSOR_NOT_FOUND:
            return "TensorNotFound"
        if self == Self.DESTINATION_SIZE_MISMATCH:
            return "DestinationSizeMismatch"
        if self == Self.UNSUPPORTED_ENDIANNESS:
            return "UnsupportedEndianness"
        if self == Self.IO_ERROR:
            return "IoError"
        if self == Self.PATH_TRAVERSAL:
            return "PathTraversal"
        if self == Self.UNKNOWN_FIELD:
            return "UnknownField"
        if self == Self.INVALID_HEADER_PADDING:
            return "InvalidHeaderPadding"
        if self == Self.DTYPE_MISMATCH:
            return "DTypeMismatch"
        if self == Self.INDEX_TOO_LARGE:
            return "IndexTooLarge"
        if self == Self.INVALID_INDEX:
            return "InvalidIndex"
        if self == Self.SHARD_MISMATCH:
            return "ShardMismatch"
        if self == Self.TOTAL_SIZE_MISMATCH:
            return "TotalSizeMismatch"
        if self == Self.SHARD_LIMIT_EXCEEDED:
            return "ShardLimitExceeded"
        return "UnknownErrorKind"

    def write_to(self, mut writer: Some[Writer]):
        """Writes the stable error-kind code."""
        writer.write(self.code())


@fieldwise_init
struct SafeTensorError(Copyable, Movable, Writable):
    """A typed Safetensors failure with bounded human-readable context."""

    var kind: SafeTensorErrorKind
    var message: String

    def diagnostic(self) -> String:
        """Returns a stable-prefix diagnostic suitable for text boundaries."""
        if self.message == "":
            return "safetensors:" + self.kind.code()
        return "safetensors:" + self.kind.code() + ": " + self.message

    def write_to(self, mut writer: Some[Writer]):
        """Writes the stable-prefix diagnostic."""
        writer.write(self.diagnostic())


def make_error(kind: SafeTensorErrorKind, message: String) -> SafeTensorError:
    """Constructs a typed Safetensors error."""
    return SafeTensorError(kind, message)
