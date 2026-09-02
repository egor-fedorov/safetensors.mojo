# ADR-005: Write Canonical Files Through an Atomic Sibling Transaction

- Status: Accepted
- Date: 2026-08-25
- Current architecture: [Writer](../architecture/writer.md)

> Platform-scope addendum: this record preserves the version 0.4 Linux design
> and its original `getrandom` implementation.
> [ADR-008](008-supported-platforms.md) extends the transaction to
> `linux-aarch64` and `osx-arm64` and replaces that internal random-source
> boundary with portable `getentropy`. Canonical format planning and the public
> atomic-visibility contract are unchanged.

## Context

Version 0.4 adds the first write path to a package whose existing public API is
read-only. The writer must produce deterministic Safetensors files, stream
tensor payloads without first assembling the complete archive in memory, and
replace a local destination without exposing a partially written file.

The writer must preserve the runtime-independent format core. It accepts raw
Safetensors wire bytes rather than values from MAX or another tensor runtime,
and it must apply checked arithmetic to every caller-controlled shape, size,
offset, header length, file length, and native-width conversion. The package is
pinned to Mojo 1.0.0 and currently distributed only for `linux-64`.

Determinism requires more than valid JSON. `Dict` iteration order cannot define
the archive, input tensor order must not change the output, JSON strings need
one fixed encoding, and padding must be explicit. Compatibility with the
Safetensors 0.8 reference implementation is a semantic requirement, but that
implementation stores user metadata in a hash map and therefore cannot be the
byte-for-byte oracle for every valid metadata map.

Atomic replacement also requires an explicit operating-system boundary. A
writer must not truncate an existing path because a live
`MappedSafeTensorFile` may still map the old inode. All validation and layout
work must finish before the writer creates a temporary file, and every failure
before the final rename must leave the destination entry unchanged.

## Mojo 1.0.0 capability check

Compiler and runtime probes established that:

- `FileHandle.write_all()` accepts a byte span, loops over partial `write(2)`
  results, and raises when it cannot complete the write;
- `FileHandle.write_bytes()` is unsuitable because its `Writer` contract
  aborts instead of raising on a write failure;
- the standard `NamedTemporaryFile` chooses a name with an existence check
  before opening it and exposes the same aborting byte-write path, so it does
  not provide the exclusive, error-controlled transaction required here;
- the standard file API supports only `r`, `w`, `rw`, and `a` modes and has no
  exclusive-create mode; and
- the Mojo standard library has no file-replacement rename operation.

The implementation therefore needs a small Linux libc boundary for OS-random
temporary names, exclusive file creation, and the final rename. Once the
exclusive descriptor is owned by a `FileHandle`, all payload writes use its
raising `write_all()` API.

## Candidate designs

1. Return a complete `List[UInt8]` archive and let the caller write it. This is
   simple, but duplicates the complete tensor payload and does not provide
   atomic replacement.
2. Expose a stateful writer with `begin`, `add_tensor`, and `finish` operations.
   This permits incremental production but creates partially completed state,
   ordering-dependent output, cleanup obligations, and a much larger public
   contract.
3. Accept callbacks or heterogeneous borrowed spans. This can avoid ownership
   transfer, but it adds Mojo-origin and reentrancy constraints that are not
   needed for a raw local-file writer.
4. Accept one list of raw tensor entries, build and validate a bounded header,
   then write that header and the entry-owned payloads sequentially through one
   atomic sibling-file transaction.

## Decision

The public writer is deliberately one-shot and consists of one owned raw entry
type and one function:

```mojo
@fieldwise_init
struct SafeTensorData(Movable):
    var name: String
    var dtype: SafeDType
    var shape: List[UInt64]
    var data: List[UInt8]


def save_safetensors(
    path: String,
    tensors: List[SafeTensorData],
    user_metadata: Dict[String, String] = Dict[String, String](),
) raises SafeTensorError: ...
```

The signature above defines the intended API shape. Production modules must
contain complete compiled implementations rather than placeholders. The call
borrows both input collections. Each `SafeTensorData` entry owns its name,
shape, and payload lists and is movable but not copyable, so the model does not
silently duplicate tensor payloads.

There is no public serialization-to-bytes function, stateful writer, header
builder, append mode, or caller-controlled offset model. The writer computes
all offsets. Internal header planning and encoding are not exported.

### Raw tensor contract

Every one of the 22 `SafeDType` values is accepted: `BOOL`, `F4`, `F6_E2M3`,
`F6_E3M2`, `U8`, `I8`, all five float8 encodings, `I16`, `U16`, `F16`, `BF16`,
`I32`, `U32`, `F32`, `C64`, `F64`, `I64`, and `U64`.

`data` is already packed C-order Safetensors wire data in little-endian byte
order. The writer neither interprets values nor converts native endianness,
alignment, strides, or dtypes. It does not scan boolean payloads for canonical
zero and one values. Sub-byte dtypes are accepted only when the complete tensor
bit length is byte-addressable, preserving the format validator's existing
rule.

Before any filesystem mutation, the writer rejects:

- an invalid manually constructed `SafeDType`;
- the reserved tensor name `__metadata__`;
- duplicate exact decoded tensor names;
- a checked shape product or bit-length overflow;
- a non-byte-addressable tensor bit length;
- payload bytes whose length differs from the dtype and shape; and
- a header or complete-file layout that cannot be represented without
  overflow or a checked native-width conversion.

### Canonical layout

The same semantic input produces the same file bytes independently of input
list order and `Dict` iteration order.

Tensors are sorted first by descending stable `SafeDType` serialization rank
and then by ascending exact tensor name. The complete rank is:

```text
U64, I64, F64, C64, F32, U32, I32, BF16, F16, U16, I16,
F8_E5M2FNUZ, F8_E4M3FNUZ, F8_E8M0, F8_E4M3, F8_E5M2,
I8, U8, F6_E3M2, F6_E2M3, F4, BOOL
```

This matches the Safetensors 0.8 dtype-descending writer order. Higher-alignment
payloads precede lower-alignment payloads, while the exact name is the stable
tie breaker. Zero-length tensors follow the same order and may share an offset
boundary.

The header uses this fixed representation:

- `__metadata__` is omitted for an empty metadata map and otherwise appears as
  the first top-level member;
- metadata keys are sorted in ascending exact decoded-key order;
- tensor descriptors follow canonical tensor/data order;
- descriptor fields are exactly `dtype`, `shape`, and `data_offsets`, in that
  order;
- JSON is compact, with no insignificant whitespace inside the value;
- integers use the shortest unsigned base-10 spelling;
- strings emit non-ASCII Unicode as validated UTF-8, escape `"` as `\"` and
  `\` as `\\`, use
  `\b`, `\t`, `\n`, `\f`, and `\r` for those controls, and use lowercase
  `\u00xx` escapes for the remaining bytes below `0x20`; `/` is not escaped;
- the JSON object is followed only by ASCII spaces until the declared header
  length is a multiple of eight; and
- the 8-byte prefix is that padded header length encoded as little-endian
  `UInt64`.

The fixed maximum declared header length is 100,000,000 bytes, including space
padding. `save_safetensors` has no option to increase it. Header construction
is bounded as bytes are appended rather than checked only after an unbounded
allocation.

All tensor offsets are relative to the data section and form complete,
gap-free coverage in canonical tensor order. Shape products, bit lengths, byte
lengths, cumulative offsets, prefix-plus-header length, total file length, JSON
encoded lengths, padding, and every potentially narrowing native conversion
are explicitly bounded or use checked helpers. Nonnegative native `List`
lengths widen into `UInt64`, and padding converted back to `Int` is bounded
below eight. No caller-controlled integer is first represented as floating
point or converted by truncation.

### Format and I/O layering

The runtime-independent `safetensors.format` layer owns:

- writer-input and layout validation;
- canonical ordering;
- checked offset planning; and
- the schema-specific JSON header encoder.

It imports no operating-system, Python, MAX, or tensor-runtime API. The
`safetensors.io` layer owns the local-file transaction and depends on the
format plan. The root `safetensors` package re-exports only `SafeTensorData` and
`save_safetensors` as the supported writer surface; nested writer modules are
internal.

### Atomic sibling transaction

The complete validated plan and header are built before the first filesystem
mutation. The destination's parent directory must already exist; the writer
does not create it.

The I/O layer then:

1. Generates an opaque temporary basename from Linux `getrandom(2)` in the
   destination directory and uses `open(2)` with
   `O_WRONLY | O_CREAT | O_EXCL | O_CLOEXEC` and mode `0600`. Name collisions
   are retried with a new nonce. Exclusive creation prevents following or
   overwriting an attacker-created entry.
2. Adopts the returned descriptor into one `FileHandle` and calls
   `write_all()` for the 8-byte prefix, bounded header, and each tensor payload
   in canonical order.
3. Explicitly closes the handle. A write or close failure maps to `IoError`,
   removes the sibling temporary entry on a best-effort basis, and leaves the
   destination unchanged.
4. Calls Linux libc `rename(2)` with the temporary path and destination path.
   This successful rename is the only commit point and the function performs
   no fallible work afterward.

The committed file has mode `0600`; existing destination permissions are not
preserved. If the destination is a symbolic link, the directory entry for the
link is replaced and its target is not followed. Renaming over an ordinary
existing file atomically replaces that entry. A missing parent, a destination
directory, an unsupported filesystem operation, or any random-source, open,
write, close, or rename failure is `IoError`. Cleanup is best-effort and does
not replace the original failure.

No lock serializes concurrent writers. Each writes a distinct sibling file,
each completed rename is atomic, and the last successful rename wins. Readers
see either the complete previous inode or one complete replacement, never the
partially written temporary file. An already open reader or mapping remains
attached to its previous inode.

The guarantee is atomic namespace visibility on the current Linux platform,
not crash durability. Version 0.4 does not call `fsync` on the file or parent
directory and does not promise survival across a process, kernel, power, or
storage failure. Closing before rename drains Mojo and descriptor state but is
not a durable-storage barrier.

### Errors

The writer adds no error kinds and does not renumber the stable error values.
It reuses:

- `UnsupportedDType` for an invalid dtype value;
- `DuplicateKey` for duplicate tensor names;
- `InvalidMetadata` for the reserved tensor name;
- `MisalignedSlice` when a complete sub-byte tensor's bit length is not
  divisible by eight, matching the reference implementation;
- `InvalidTensorSize` when the supplied payload byte length does not match the
  dtype and shape;
- `ValidationOverflow` for checked-layout arithmetic failures;
- `HeaderTooLarge` for the fixed padded-header limit; and
- `IoError` for local randomness, temporary-file creation, write, close, or
  rename failures.

The unused `PathTraversal` kind was removed because the writer receives its
destination path directly from the caller and Safetensors descriptors do not
contain paths. Its former ordinal 22 remains reserved so later stable numeric
values are unchanged.

The first validation failure is reported before a temporary file is created.
Once temporary creation succeeds, any failure before commit attempts cleanup;
the destination remains unchanged even if cleanup cannot remove the orphaned
temporary entry.

## Verification requirements

Implementation verification includes:

- an independently generated exact golden covering metadata, scalar,
  zero-length, multi-tensor, Unicode/control escaping, alignment classes, and
  stable name tie-breaking;
- a byte-exact reference-generated matrix covering every dtype exposed by the
  Safetensors 0.8 serializer across byte-addressable scalar, vector,
  multidimensional, and zero-element shapes, plus semantic
  reference-deserializer coverage for both recognized F6 encodings that
  serializer cannot produce;
- permutation tests proving tensor input and metadata insertion order do not
  change bytes, plus focused empty, metadata-only, sub-byte, all-dtype, and
  checked-overflow planning tests;
- round trips through the Mojo parser, buffered reader, mapping, and an exact
  typed view;
- live pinned Safetensors 0.8 tests that reproduce the exact matrix, load the
  independent golden, and compare metadata, dtype, shape, and payload
  semantics;
- integration tests for unchanged destinations after preflight failure,
  ordinary inode replacement, symbolic-link entry replacement, mode `0600`, a
  missing parent, and temporary cleanup after a failed rename;
- a root public API compile contract; and
- clean-installed package smoke coverage using `SafeTensorData` and
  `save_safetensors`.

## Deliberate exclusions

Version 0.4 does not add serialization to an owned complete byte buffer, a
stateful or incremental writer API, append/update-in-place behavior,
caller-provided offsets, typed-value encoding, byte swapping, mmap writes,
slicing, sharded indexes, MAX adapters, tensor-runtime adapters, callbacks,
scatter/gather input, compression, checksums, signatures, encryption, locking,
cross-platform replacement, or crash-durability guarantees.

## Consequences

- The same semantic input has one project-defined byte representation.
- Tensor payload memory is owned by each raw input entry and borrowed for the
  one-shot call; the writer does not allocate a second complete archive before
  writing it.
- Existing files and mappings are never truncated in place.
- Validation failures cannot create temporary files or alter the destination.
- Pre-commit I/O failures can leave an orphan only when cleanup itself fails;
  they cannot expose a partial destination.
- Linux libc remains isolated to the I/O layer while format planning stays
  pure Mojo and runtime-independent.
- A future borrowed, incremental, durable, or non-Linux writer requires a
  separate decision rather than silently expanding this contract.

## References

- [ADR-001: Use a Schema-Directed Pure-Mojo JSON Parser](001-json-parser.md)
- [ADR-002: Retain a File Handle for Local Random-Access Reads](002-local-reader.md)
- [ADR-003: Own Read-Only Mappings and Return Origin-Bound Byte Views](003-memory-mapped-reader.md)
- [ADR-008: Support Three Native Mojo Platforms](008-supported-platforms.md)
- [Mojo 1.0.0 `FileHandle`](https://mojolang.org/docs/std/io/file/FileHandle/)
- [Safetensors 0.8.0 serialization](https://github.com/huggingface/safetensors/blob/v0.8.0/safetensors/src/tensor.rs#L209-L359)
- [Linux `getrandom(2)`](https://man7.org/linux/man-pages/man2/getrandom.2.html)
- [Linux `open(2)`](https://man7.org/linux/man-pages/man2/open.2.html)
- [Linux `rename(2)`](https://man7.org/linux/man-pages/man2/rename.2.html)
