# ADR-006: Read Reference-Compatible Headers by Default

- Status: Accepted
- Date: 2026-08-25
- Current architecture: [Format core](../architecture/format-core.md)

## Context

ADR-001 deliberately began with a narrow reader policy: byte zero had to be
`{`, only ASCII space could follow the root object, and every tensor descriptor
field had to be known. That policy follows the literal Safetensors format text,
but it is narrower than the reference implementation.

Safetensors 0.8 accepts all four JSON whitespace bytes before and after the
root object. Its upstream tests explain the leading case: a writer may pad the
header to align the data section to a page boundary. Its descriptor model also
does not deny unknown fields, so additive descriptor extensions are ignored.
A reader that rejects either form cannot open files accepted by the reference
implementation and can become incompatible as the descriptor schema grows.

Compatibility must not weaken the security properties established by ADR-001.
Known dimensions and offsets still need exact integer parsing and checked
arithmetic, decoded duplicate keys must remain visible, ignored values must be
fully validated as JSON, and hostile nesting must remain bounded. The writer
also needs one canonical output even when the reader accepts equivalent forms.

## Decision

Reference-compatible reading is the default. The previous canonical-header
policy remains available through a `strict: Bool = False` argument on every
public parsing and opening entry point:

```mojo
parse_raw_header(header, strict=False)
parse_metadata_from_header(header, data_length, data_start=0, strict=False)
parse_metadata(buffer, max_header_bytes=DEFAULT_MAX_HEADER_BYTES, strict=False)
open_safetensors(path, max_header_bytes=DEFAULT_MAX_HEADER_BYTES, strict=False)
map_safetensors(path, max_header_bytes=DEFAULT_MAX_HEADER_BYTES, strict=False)
```

The argument is threaded through the shared parser, so in-memory, buffered,
and memory-mapped access cannot silently apply different policies.

### Compatible mode

With `strict=False`, the reader:

- accepts bytes `0x20`, `0x09`, `0x0A`, and `0x0D` before the root object and
  after it within the declared header length;
- requires all leading and trailing bytes to be JSON whitespace and still
  rejects any second value or other trailing content;
- ignores unknown fields inside a tensor descriptor only after validating and
  consuming exactly one complete JSON value; and
- continues to reject duplicate decoded descriptor keys, including unknown
  names and escape-equivalent spellings.

Ignored values may be strings, numbers, booleans, null, arrays, or objects.
They are consumed by a schema-local skipper rather than a generic JSON object
model. Numbers are checked lexically against the JSON number grammar and are
never represented as `Float64` or accumulated into an integer. Ignored string
values are UTF-8 and escape validated without allocating a decoded string.
Keys inside an ignored object use the same no-allocation string validator and
are not retained. JSON permits readers to define duplicate-member behavior,
and no value inside an ignored field affects Safetensors semantics; retaining
those keys would allow a large extension object to amplify memory use. Decoded
duplicate detection remains mandatory at the root, in `__metadata__`, and in
the tensor descriptor itself, where names do affect semantics.

Array and object nesting in ignored values is limited to 128 levels. The depth
guard runs before incrementing the level, and all cursor movement remains
bounded by the already native-sized header span. Invalid syntax, an incomplete
value, or excessive nesting raises `InvalidJson`.

Compatibility applies only to unknown descriptor fields and insignificant
header whitespace. It does not relax the types of `dtype`, `shape`, or
`data_offsets`; exact unsigned integer handling; UTF-8 validation; duplicate
checks at every schema level where a decoded key affects Safetensors semantics;
dtype recognition; tensor byte-size rules; checked offset arithmetic; or
gap-free, non-overlapping coverage validation.

### Strict mode and canonical writing

With `strict=True`, byte zero must be `{`, only ASCII space may follow the root
object, and an unknown descriptor field raises `UnknownField`. This mode is an
explicit canonical-schema policy, not a claim that compatible files are
invalid Safetensors archives.

Writing is unchanged. `save_safetensors` emits no leading whitespace, uses
only ASCII spaces for 8-byte header padding, and emits exactly `dtype`, `shape`,
and `data_offsets` for each descriptor. Reader tolerance therefore does not
create multiple project-defined output forms.

## Consequences

- Files accepted by the Safetensors 0.8 whitespace and descriptor-extension
  behavior are accepted by default.
- Applications that need the original closed descriptor schema and canonical
  padding can opt into it consistently across every access path.
- Additive descriptor fields no longer make every existing package build
  immediately unable to read a newer archive.
- The project owns a bounded JSON-value skipper in addition to its
  schema-directed parser and must test every value form and hostile nesting.
- `strict` does not disable metadata validation and is not a permissive mode
  for malformed known fields, offsets, shapes, or dtypes.

## References

- [ADR-001: Use a Schema-Directed Pure-Mojo JSON Parser](001-json-parser.md)
- [Safetensors 0.8.0 header-whitespace tests](https://github.com/huggingface/safetensors/blob/v0.8.0/safetensors/src/tensor.rs)
- [Safetensors 0.8.0 format description](https://github.com/huggingface/safetensors/blob/v0.8.0/README.md#format)
