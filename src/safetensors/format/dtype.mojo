"""Safetensors wire-format dtype definitions."""

from safetensors.errors import SafeTensorError, SafeTensorErrorKind, make_error


@fieldwise_init
struct SafeDType(Equatable, ImplicitlyCopyable, Writable):
    """A dtype from the Safetensors wire format, independent of runtime dtypes.
    """

    var _value: UInt8

    comptime BOOL = Self(0)
    comptime F4 = Self(1)
    comptime F6_E2M3 = Self(2)
    comptime F6_E3M2 = Self(3)
    comptime U8 = Self(4)
    comptime I8 = Self(5)
    comptime F8_E5M2 = Self(6)
    comptime F8_E4M3 = Self(7)
    comptime F8_E8M0 = Self(8)
    comptime F8_E4M3FNUZ = Self(9)
    comptime F8_E5M2FNUZ = Self(10)
    comptime I16 = Self(11)
    comptime U16 = Self(12)
    comptime F16 = Self(13)
    comptime BF16 = Self(14)
    comptime I32 = Self(15)
    comptime U32 = Self(16)
    comptime F32 = Self(17)
    comptime C64 = Self(18)
    comptime F64 = Self(19)
    comptime I64 = Self(20)
    comptime U64 = Self(21)

    @staticmethod
    def from_wire_name(wire_name: String) raises SafeTensorError -> SafeDType:
        """Parses one exact, case-sensitive Safetensors dtype name."""
        if wire_name == "BOOL":
            return SafeDType.BOOL
        if wire_name == "F4":
            return SafeDType.F4
        if wire_name == "F6_E2M3":
            return SafeDType.F6_E2M3
        if wire_name == "F6_E3M2":
            return SafeDType.F6_E3M2
        if wire_name == "U8":
            return SafeDType.U8
        if wire_name == "I8":
            return SafeDType.I8
        if wire_name == "F8_E5M2":
            return SafeDType.F8_E5M2
        if wire_name == "F8_E4M3":
            return SafeDType.F8_E4M3
        if wire_name == "F8_E8M0":
            return SafeDType.F8_E8M0
        if wire_name == "F8_E4M3FNUZ":
            return SafeDType.F8_E4M3FNUZ
        if wire_name == "F8_E5M2FNUZ":
            return SafeDType.F8_E5M2FNUZ
        if wire_name == "I16":
            return SafeDType.I16
        if wire_name == "U16":
            return SafeDType.U16
        if wire_name == "F16":
            return SafeDType.F16
        if wire_name == "BF16":
            return SafeDType.BF16
        if wire_name == "I32":
            return SafeDType.I32
        if wire_name == "U32":
            return SafeDType.U32
        if wire_name == "F32":
            return SafeDType.F32
        if wire_name == "C64":
            return SafeDType.C64
        if wire_name == "F64":
            return SafeDType.F64
        if wire_name == "I64":
            return SafeDType.I64
        if wire_name == "U64":
            return SafeDType.U64
        raise make_error(
            SafeTensorErrorKind.UNSUPPORTED_DTYPE,
            "unsupported Safetensors dtype",
        )

    def wire_name(self) -> String:
        """Returns the exact name stored in a Safetensors JSON header."""
        if self == Self.BOOL:
            return "BOOL"
        if self == Self.F4:
            return "F4"
        if self == Self.F6_E2M3:
            return "F6_E2M3"
        if self == Self.F6_E3M2:
            return "F6_E3M2"
        if self == Self.U8:
            return "U8"
        if self == Self.I8:
            return "I8"
        if self == Self.F8_E5M2:
            return "F8_E5M2"
        if self == Self.F8_E4M3:
            return "F8_E4M3"
        if self == Self.F8_E8M0:
            return "F8_E8M0"
        if self == Self.F8_E4M3FNUZ:
            return "F8_E4M3FNUZ"
        if self == Self.F8_E5M2FNUZ:
            return "F8_E5M2FNUZ"
        if self == Self.I16:
            return "I16"
        if self == Self.U16:
            return "U16"
        if self == Self.F16:
            return "F16"
        if self == Self.BF16:
            return "BF16"
        if self == Self.I32:
            return "I32"
        if self == Self.U32:
            return "U32"
        if self == Self.F32:
            return "F32"
        if self == Self.C64:
            return "C64"
        if self == Self.F64:
            return "F64"
        if self == Self.I64:
            return "I64"
        if self == Self.U64:
            return "U64"
        return "UNKNOWN"

    def bits_per_element(self) -> UInt64:
        """Returns the exact number of wire-format bits per element."""
        if self == Self.F4:
            return 4
        if self == Self.F6_E2M3 or self == Self.F6_E3M2:
            return 6
        if (
            self == Self.BOOL
            or self == Self.U8
            or self == Self.I8
            or self == Self.F8_E5M2
            or self == Self.F8_E4M3
            or self == Self.F8_E8M0
            or self == Self.F8_E4M3FNUZ
            or self == Self.F8_E5M2FNUZ
        ):
            return 8
        if (
            self == Self.I16
            or self == Self.U16
            or self == Self.F16
            or self == Self.BF16
        ):
            return 16
        if self == Self.I32 or self == Self.U32 or self == Self.F32:
            return 32
        if (
            self == Self.C64
            or self == Self.F64
            or self == Self.I64
            or self == Self.U64
        ):
            return 64
        return 0

    def is_byte_aligned(self) -> Bool:
        """Returns whether one element occupies a whole number of bytes."""
        return self.bits_per_element() % 8 == 0

    def required_alignment(self) -> UInt64:
        """Returns the natural byte alignment used for packed wire data."""
        var bits = self.bits_per_element()
        if bits <= 8:
            return 1
        return bits // 8

    def write_to(self, mut writer: Some[Writer]):
        """Writes the exact wire-format dtype name."""
        writer.write(self.wire_name())


def parse_dtype(wire_name: String) raises SafeTensorError -> SafeDType:
    """Parses a case-sensitive Safetensors dtype name."""
    return SafeDType.from_wire_name(wire_name)


def bits_per_element(dtype: SafeDType) -> UInt64:
    """Returns the exact number of wire-format bits per element."""
    return dtype.bits_per_element()


def is_byte_aligned(dtype: SafeDType) -> Bool:
    """Returns whether one element occupies a whole number of bytes."""
    return dtype.is_byte_aligned()


def required_alignment(dtype: SafeDType) -> UInt64:
    """Returns the natural byte alignment used for packed wire data."""
    return dtype.required_alignment()
