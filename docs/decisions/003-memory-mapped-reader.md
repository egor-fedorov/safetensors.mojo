# ADR-003: Own Read-Only Mappings and Return Origin-Bound Byte Views

- Status: Accepted
- Date: 2026-08-25
- Current architecture: [Readers and views](../architecture/readers-and-views.md)

## Context

Version 0.3 adds zero-copy access to local tensor bytes without changing the
runtime-independent format core or the cursor-based reader introduced in
version 0.2. A mapped API must keep the mapping alive for every returned view,
must never expose mutable access to a read-only file, and must preserve
schema-directed parsing, metadata validation, and checked arithmetic.

The package currently targets `linux-64` and is pinned to Mojo 1.0.0. That
standard library exposes the Unix descriptor through `FileHandle.handle`, but
does not provide a memory-mapping wrapper. A mapping therefore requires a
small internal libc FFI boundary.

Memory mapping also has a different external-mutation contract from owned
reads. `PROT_READ | MAP_PRIVATE` prevents writes through this API, but it does
not create an immutable snapshot of the backing file. Same-length writes may
become visible, and access after external truncation can terminate the process
with `SIGBUS` instead of producing a catchable Mojo error.

## Candidate check

The following designs and Mojo capabilities were evaluated before
implementation:

1. Read the complete file into a `List[UInt8]` and return borrowed spans. This
   gives a safe owned snapshot, but duplicates the tensor payload and is not a
   memory-mapped access path.
2. Add mapping state and borrowed methods to `SafeTensorReader`. This would mix
   cursor-based reads, which check file length around each operation, with
   mapping semantics that cannot provide the same post-return guarantee.
3. Map each requested tensor independently. File offsets would need page
   alignment and every returned view would require an independent mapping
   owner, adding offset arithmetic, system calls, and lifetime state.
4. Introduce a separate owner for one whole-file, read-only private mapping.
   Mapping from offset zero avoids page-offset alignment arithmetic and gives
   every tensor view one common owner.

Mojo 1.0.0 compiler probes established that:

- `std.os` has no `mmap` or `munmap` API;
- `std.ffi.external_call` can call libc `mmap` and `munmap` using `c_int`,
  `c_long`, and `c_size_t` with the descriptor from `FileHandle.handle`;
- an internal `Pointer[UInt8, ImmUntrackedOrigin]` can be recast inside the
  borrowed accessor to an immutable origin tied to the owning value;
- `Span[UInt8, origin]` can be constructed from that pointer without copying;
  and
- consuming the owner and then using such a span is rejected by the compiler.

`Pointer`, `ImmOrigin`, and the `Span(unsafe_ptr=..., length=...)` constructor
are the non-deprecated Mojo 1.0.0 spellings. Production code must compile with
warnings treated as errors at this boundary.

## Decision

The mapped public API will be additive:

```mojo
struct MappedSafeTensorFile(Movable):
    def metadata(self) -> SafeTensorMetadata: ...
    def file_length(self) -> UInt64: ...
    def tensor_bytes[origin: ImmOrigin](
        ref[origin] self, name: String
    ) raises SafeTensorError -> Span[UInt8, origin]: ...

def map_safetensors(
    path: String,
    max_header_bytes: UInt64 = DEFAULT_MAX_HEADER_BYTES,
    strict: Bool = False,
) raises SafeTensorError -> MappedSafeTensorFile: ...
```

The signatures above define the intended API shape; production modules must
contain complete compiled implementations, not placeholders.

`SafeTensorReader` and `open_safetensors` remain available and keep their
existing semantics. `MappedSafeTensorFile` is movable but not copyable. It
owns:

- one retained read-only `FileHandle` for descriptor identity and point-in-time
  length checks;
- one underscore-prefixed movable mapping region, treated as an implementation
  detail, that unmaps exactly once during destruction;
- validated metadata; and
- the file-length snapshot observed during opening.

Opening reads and validates the prefix and JSON header through the retained
descriptor before mapping. It then maps the complete validated file from that
same descriptor at offset zero with `PROT_READ | MAP_PRIVATE`. It never
validates one path instance and reopens the path for mapping. Invalid physical
files are rejected before a zero-length `mmap` call, and mapping is supported
only on the current `linux-64` package platform.

The `strict` argument selects the shared header policy defined by ADR-006. It
does not weaken dtype, shape, size, offset, or complete-coverage validation.

The mapping region stores a read-only
`Pointer[UInt8, ImmUntrackedOrigin]` and a checked native `Int` length. Its
destructor calls `munmap`; because destruction cannot return a normal
`SafeTensorError`, cleanup failure is not part of the public error surface.
The libc result must be compared with `MAP_FAILED`, whose pointer value is
`(void *) -1` rather than null, before constructing or committing the owning
region. A failed mapping raises `IoError`.

Mojo 1.0.0 does not enforce field visibility. The region and pointer fields are
underscore-prefixed implementation details, and supported public methods and
exports never return the raw pointer or an untracked-origin span. Direct field
access or unsafe pointer casting by a caller is outside the supported API.

`tensor_bytes` performs an exact decoded-name lookup. It calculates absolute
begin and end offsets from `metadata.data_start()` and the validated relative
offsets, checks those values against the mapped length, and converts offsets
and byte counts to `Int` only through checked helpers. Pointer offsetting occurs
only after those checks. The returned `Span[UInt8, origin]` is immutable and its
origin is the `ref` borrow of the public owner, so Mojo prevents the owner from
being consumed while a subsequently used view is live. Multiple immutable
views may coexist. Zero-byte tensors return an empty origin-bound span without
dereferencing tensor data.

The mapped type's supported methods do not return a whole-file span, a mutable
pointer, manual unmapping, shared ownership, slicing, or tensor-runtime
adapters.

All parser and validator errors retain their existing kinds. File open, seek,
length-change, mapping, and unsupported-file failures use `IoError`. Missing
names use `TensorNotFound`; checked native-width failures use
`ValidationOverflow`; and a defensive range inconsistency after validation
uses `InvalidOffsets`.

### External mutation contract

The backing inode must not be modified, truncated, or rewritten from before
`map_safetensors()` begins until its returned `MappedSafeTensorFile` and every
view borrowed from it are dead. This includes header reading, validation, and
the `mmap` call; otherwise a same-length rewrite could make validated metadata
describe different mapped bytes. A length check before returning a view can
reject a change that has already happened, but it cannot close the race after
the view is returned. Callers that need to avoid mmap fault exposure should use
`SafeTensorReader` with caller-owned buffers or make an owned copy. The reader
detects ordinary observed length changes, but neither access path provides an
integrity guarantee or detects every same-length modification. Authenticity
and snapshot requirements need a trusted stable source or an independent owned
snapshot.

Renaming, unlinking, or replacing the path does not redirect an existing
mapping because validation and mapping use the retained descriptor. In-place
same-length modification remains observable or unobservable according to the
operating system; the API makes no snapshot or integrity guarantee.

### Typed-view boundary

Raw byte views require neither native alignment nor host-endian
reinterpretation. Format validation must continue to accept otherwise valid
files whose tensor addresses are not naturally aligned.

Native typed views are defined separately by
[ADR-004](004-native-typed-views.md). It specifies the exact
`SafeDType`-to-Mojo-`DType` whitelist, mismatch behavior, actual-address
alignment checks, little-endian host policy, zero-element representation, and
the raw-only treatment of encodings without a safe native scalar mapping.
Alignment or endianness limitations of the typed accessor do not become
Safetensors format-validity rules.

## Verification requirements

Implementation must include:

- runtime parity tests for metadata, exact tensor bytes, scalars, zero-byte
  tensors, reordered offsets, repeated views, missing names, missing files,
  malformed inputs, and mapping failures;
- path-replacement and unlink tests proving that the mapping remains attached
  to the opened file;
- controlled growth and truncation tests only before obtaining a view;
- a positive compiler fixture for normal origin-bound use;
- negative compiler fixtures proving that the owner cannot be copied and that
  a view cannot be mutated, escape its owner, or remain usable after the owner
  is consumed; and
- a scoped destruction smoke test for automatic cleanup.

Tests must not dereference a mapping after truncating its backing file in the
main test process, because the expected operating-system behavior can be
`SIGBUS` rather than a Mojo exception.

## Consequences

- Tensor bytes can be accessed without copying them into a Mojo allocation.
- Header parsing still allocates only in proportion to the bounded header.
- One mapping reserves virtual address space for the complete file while pages
  remain demand-loaded by the operating system.
- The retained descriptor and mapping are released automatically with their
  non-copyable owner.
- Mojo origins express the owner/view relationship without exposing unsafe
  pointer ownership to callers.
- The existing reader avoids mapped-memory fault exposure and reports ordinary
  observed length changes through controlled read errors, without claiming a
  stable snapshot.
- Supporting macOS or Windows requires a separately compiled and tested
  mapping backend; this decision does not claim those platforms.

## References

- [ADR-006: Read Reference-Compatible Headers by Default](006-compatible-header-reading.md)
- [Mojo 1.0.0 `FileHandle`](https://mojolang.org/docs/std/io/file/FileHandle/)
- [Mojo 1.0.0 `external_call`](https://mojolang.org/docs/std/ffi/external_call/)
- [Mojo 1.0.0 `Pointer`](https://mojolang.org/docs/std/memory/pointer/Pointer/)
- [Mojo 1.0.0 `Span`](https://mojolang.org/docs/std/collections/span/Span/)
- [Mojo origins](https://mojolang.org/docs/std/origin/)
- [Safetensors 0.8.0 format](https://github.com/huggingface/safetensors/blob/v0.8.0/README.md#format)
- [Linux `mmap(2)`](https://man7.org/linux/man-pages/man2/mmap.2.html)
