# Format Core

The format core owns Safetensors framing, JSON decoding, wire dtypes, raw
metadata, semantic validation, validated metadata, and serialization planning.
It does not perform filesystem I/O or interpret tensor payloads through a
tensor runtime.

The supported entry points are re-exported from the root `safetensors` package.
Direct imports from nested modules and direct construction of validated state
are implementation details.

## Framing and bounds

A complete Safetensors file begins with an unsigned 8-byte little-endian header
length. `decode_header_length` requires at least eight bytes and decodes those
bytes without a signed or floating-point intermediate.

`parse_metadata` then:

1. enforces the configured header limit, which defaults to 100,000,000 bytes;
2. computes `8 + header_length` with checked `UInt64` arithmetic;
3. verifies that the declared header is inside the complete buffer;
4. converts the data-section origin to native `Int` with a checked conversion;
5. derives the remaining data length; and
6. parses and fully validates the isolated header.

`parse_metadata_from_header` accepts an already isolated header, a data length,
and an optional data origin. Local readers use the same entry point after
reading the prefix and header from a retained file handle.

## Schema-directed JSON parser

The pure-Mojo parser reads the Safetensors schema directly instead of building
a generic JSON document. It validates the complete declared header as UTF-8
before parsing and applies these rules in both reading modes:

- JSON string escapes and surrogate pairs are decoded before semantic keys are
  compared.
- Duplicate decoded keys are rejected at the root, inside `__metadata__`, and
  inside tensor descriptors. Escape-equivalent spellings are duplicates.
- User metadata values must be strings.
- Tensor descriptors must contain `dtype`, `shape`, and exactly two
  `data_offsets` values.
- Known dimensions and offsets are parsed directly as `UInt64`. Their grammar
  is `0` or a nonzero decimal digit followed by decimal digits, with no sign,
  fraction, exponent, or leading zero.
- Root keys other than `__metadata__` are decoded tensor names whose values
  must be descriptor objects.

Exact integer parsing never passes through `Float64`, so values above `2^53`
remain exact and overflow is reported instead of rounded.

### Compatible and strict reading

Compatibility mode is the default. Strict mode is available at every public
parsing and opening entry point.

| Policy | Compatible mode | Strict mode |
| --- | --- | --- |
| Leading header whitespace | JSON space, tab, LF, or CR | Byte zero must be `{` |
| Trailing header whitespace | JSON space, tab, LF, or CR | ASCII space only |
| Unknown descriptor fields | Consume one complete valid JSON value | Raise `UnknownField` |

The compatible unknown-value skipper validates strings, number grammar,
literals, arrays, and objects without retaining their values. Array and object
nesting is limited to 128. Descriptor-level unknown field names are decoded and
duplicate-checked; keys nested inside an ignored object are intentionally not
retained or duplicate-checked.

Strict mode controls boundary whitespace and whether the descriptor schema is
closed. It is not a general canonical-JSON checker. Both modes apply identical
dtype, shape, size, offset, and complete-coverage validation.

## Raw and validated metadata

The type boundary makes untrusted parsing state explicit:

| Type | Meaning |
| --- | --- |
| `RawTensorInfo` | Decoded dtype text, shape, and relative offsets with no semantic-validity claim |
| `RawSafeTensorMetadata` | Decoded user metadata and raw tensor descriptors |
| `TensorInfo` | Exact `SafeDType` plus validation-derived element, bit, and byte counts |
| `SafeTensorMetadata` | Fully validated descriptors, name index, offset order, data origin, and data length |

`SafeTensorMetadata` stores tensors in deterministic data-offset order and
indexes them by decoded name. Its supported collection and descriptor accessors
return copies, so callers using the public interface do not mutate
validation-protected state. Mojo 1.0 does not enforce field visibility; direct
field access and mutation remain unsupported.

## Wire dtypes

`SafeDType` is an exact, case-sensitive model of the 22 recognized wire
encodings:

```text
BOOL, F4, F6_E2M3, F6_E3M2, U8, I8,
F8_E5M2, F8_E4M3, F8_E8M0, F8_E4M3FNUZ, F8_E5M2FNUZ,
I16, U16, F16, BF16, I32, U32, F32, C64, F64, I64, U64
```

The model records wire names and bits per element without depending on Mojo
runtime scalar types. It also exposes a dtype byte-alignment helper for layout
policy; semantic file validation does not require aligned tensor offsets.
This allows the format core to validate packed and otherwise unsupported
native-view encodings.

## Checked semantic validation

Arithmetic on file-controlled dimensions, data offsets, lengths, and counts
uses checked helpers for addition, subtraction, multiplication, shape
products, decimal integer accumulation, and native-index conversion. Parser
cursors instead advance within an already bounded native span. Shape products
treat an empty shape as one scalar and return zero when any dimension is zero
before multiplying the other dimensions.

For every raw tensor, validation:

1. rejects the reserved name and duplicate decoded names;
2. resolves the exact wire dtype;
3. computes element, bit, and byte counts with checked arithmetic;
4. requires the complete tensor bit length to be byte-addressable;
5. requires `begin <= end <= data_length`;
6. requires `end - begin` to equal the computed byte length; and
7. sorts descriptors by `(begin, end, name)` and requires gap-free,
   overlap-free coverage of the complete data section.

The last rule also rejects an unindexed data tail and non-empty data with no
tensors. Offsets stay relative to the data section; `data_start` records the
absolute origin supplied by the complete-buffer or local-file parser.

Public format operations report a typed `SafeTensorErrorKind`. Consumers should
branch on the kind rather than the contextual message text.

## Serialization boundary

Canonical ordering, checked offsets, JSON encoding, and header padding are pure
format operations. The format layer produces a complete serialization plan;
the I/O layer alone creates a temporary file and commits it. This preserves the
runtime-independent core and ensures the writer can validate the entire layout
before filesystem mutation. See [Writer](writer.md).

## Decision history

- [ADR-001](../decisions/001-json-parser.md) selected the schema-directed
  pure-Mojo parser, exact integer handling, and decoded-key duplicate checks.
- [ADR-006](../decisions/006-compatible-header-reading.md) superseded ADR-001's
  default boundary-whitespace and unknown-field policy while retaining strict
  mode.
