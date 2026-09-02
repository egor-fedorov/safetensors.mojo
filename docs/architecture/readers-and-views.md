# Readers and Views

The local read APIs share one validation pipeline but offer two payload-access
models: explicit buffered copies and POSIX zero-copy memory-mapped views.

| Operation | Result | Payload copy | Ownership |
| --- | --- | ---: | --- |
| `reader.metadata()` | Validated metadata | No | Independent copy |
| `reader.read_tensor_into()` | Raw wire bytes | Yes | Caller buffer |
| `reader.load_tensor()` | `List[UInt8]` | Yes | New owned allocation |
| `mapped.tensor_bytes()` | `Span[UInt8, origin]` | No | Borrowed from mapping |
| `mapped.tensor_view[DType]()` | Flat native scalar span | No | Borrowed from mapping |

The root `safetensors` package exports `SafeTensorReader`,
`MappedSafeTensorFile`, `open_safetensors`, and `map_safetensors`. Internal
mapping and typed-view helpers are not public API.

## Shared opening pipeline

Both access models open the path once and retain that same file handle. The
shared helper:

1. records the file length;
2. reads exactly the 8-byte prefix and declared header;
3. enforces the configured header limit and checked framing bounds;
4. parses and completely validates metadata against the remaining data length;
5. retains the handle, validated metadata, and opening length; and
6. rechecks the same handle's length before returning.

The tensor data section is not read during this process. Renaming, replacing,
or unlinking the original path cannot redirect later access because the reader
or mapping remains attached to the opened file identity.

Compatibility mode is the default, with `strict=True` available for canonical
boundary whitespace and a closed descriptor schema. This choice never weakens
semantic validation. See [Format core](format-core.md).

## Buffered reader

`SafeTensorReader` is movable but not copyable and owns the retained handle.
One instance has one seek cursor, so concurrent or reentrant tensor reads on
the same instance are unsupported. Independent readers have independent
cursors.

`read_tensor_into()`:

- requires an exactly sized mutable destination;
- computes the absolute payload range with checked arithmetic;
- checks that range against the validated opening length;
- checks the current file length before and after the read;
- seeks through the retained handle and loops until the destination is full;
  and
- may partially modify the destination if an intervening I/O failure occurs.

`load_tensor()` allocates exactly the validated payload size and delegates to
`read_tensor_into()`. After a successful return, that list is independent of
the reader. The operation is not an authenticated or transactional snapshot:
same-length in-place file mutation cannot be detected.

## Whole-file mapping

`MappedSafeTensorFile` owns both the retained validated file and a read-only
whole-file mapping. On every supported platform the mapping uses
`PROT_READ | MAP_PRIVATE`, keeps an immutable pointer and checked native length,
and calls `munmap` exactly once when its owner is destroyed.

Mapping the complete file avoids per-tensor page-alignment arithmetic. It
reserves virtual address space for the full file, while the operating system
loads pages on demand.

`tensor_bytes()`:

1. looks up the validated tensor descriptor;
2. computes its absolute begin and end with checked arithmetic;
3. reconfirms byte length and mapping bounds;
4. converts offsets and lengths to native `Int` with checked conversions;
5. rechecks the retained file's current length; and
6. returns an immutable span whose Mojo origin is tied to the mapping owner.

An empty tensor produces a zero-length origin-bound span based at the mapping
base and does not dereference tensor data.

Mojo's origin and ownership rules prevent a returned view from escaping its
owner, becoming mutable, being used after the owner is consumed, or outliving
the mapping. Multiple immutable views may coexist. The API does not promise
that concurrent accessor calls on one owner are safe.

## Exact native typed views

`tensor_view[DType]()` reinterprets validated mapped bytes without copying or
converting them. It returns a flat immutable `Span[Scalar[dtype], origin]`;
logical dimensions remain available through metadata.

Exact native views support:

- `U8` and `I8`;
- `F8_E5M2`, `F8_E4M3`, `F8_E8M0`, `F8_E4M3FNUZ`, and
  `F8_E5M2FNUZ`;
- `I16`, `U16`, `F16`, and `BF16`;
- `I32`, `U32`, and `F32`; and
- `I64`, `U64`, and `F64`.

The following dtypes remain available only as raw bytes:

| Dtype | Reason |
| --- | --- |
| `BOOL` | Payload bytes are not validated as canonical zero or one values |
| `F4`, `F6_E2M3`, `F6_E3M2` | Packed sub-byte elements have no ordinary scalar stride |
| `C64` | Mojo 1.0 has no corresponding scalar `DType` |

The typed accessor applies checks in this order:

1. resolve the requested compile-time Mojo dtype through an explicit whitelist;
2. require an exact match with the tensor's validated `SafeDType`;
3. obtain the already checked raw byte span;
4. compute scalar size times element count with checked `UInt64` arithmetic;
5. require agreement with validated and actual raw byte lengths;
6. convert the element count to native `Int` with a checked conversion;
7. for non-empty multi-byte values, require a little-endian host;
8. for every non-empty tensor, verify the actual mapped pointer alignment; and
9. perform the bitcast only after all checks pass.

An unsupported requested or stored encoding raises `UnsupportedDType`. Two
supported but different encodings raise `DTypeMismatch`. Address alignment is
an accessor constraint, not a file-validity constraint: a valid unaligned
tensor is still available through `tensor_bytes()`.

Empty typed views still require an exact supported dtype and consistent sizes,
but skip endianness and address-alignment checks because no element can be
dereferenced.

## External mutation warning

> A read-only private mapping and an immutable Mojo span do not make the
> underlying file an immutable snapshot. Keep the backing file stable for the
> entire lifetime of the mapping and every borrowed view.

Length changes detected before or after a buffered read, or before creation of
a mapped view, are reported as `IoError`. Those checks cannot close races after
a view has been returned. In particular:

- neither access model detects same-length in-place mutation;
- mapped pages can reflect such mutation despite `MAP_PRIVATE`; and
- external truncation followed by mapped dereference can terminate the process
  with `SIGBUS`, which is not a catchable Mojo error.

Use the buffered API when owned bytes are required. Use mapped views only when
the file's external lifecycle is controlled for the complete borrow lifetime.

## Current limitations

- There is no authenticity, integrity, checksum, or snapshot guarantee.
- There is no mutable mapping, manual unmapping, or shared mapping ownership.
- Typed access does not byte-swap, decode packed types, or allocate a fallback.
- Typed views are flat spans rather than multidimensional tensor containers.
- The library does not provide slicing or MAX adapters. Multi-file behavior is
  documented separately in [Sharded readers](sharded-readers.md).

## Decision history

- [ADR-002](../decisions/002-local-reader.md) defines retained-handle buffered
  reads.
- [ADR-003](../decisions/003-memory-mapped-reader.md) defines mapping ownership
  and the external-mutation model.
- [ADR-004](../decisions/004-native-typed-views.md) defines exact native views,
  alignment, and endianness rules.
- [ADR-008](../decisions/008-supported-platforms.md) extends the mapping backend
  to the supported Linux and Apple silicon targets.
