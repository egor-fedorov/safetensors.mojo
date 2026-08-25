# ADR-002: Retain a File Handle for Local Random-Access Reads

- Status: Accepted
- Date: 2026-08-25

## Context

Version 0.2 adds local file reads without changing the runtime-independent
format core. Opening a file must parse and validate its metadata without
loading tensor data. Later operations must read one named tensor either into a
caller-owned byte buffer or into a newly allocated owned byte list.

The reader must preserve the existing rules for schema-directed parsing,
complete data coverage, bounded headers, and checked arithmetic. It must also
define what happens when a path or an already opened file changes between
metadata parsing and tensor reads.

Mojo 1.0.0 provides a movable `FileHandle` with `seek()`, owned `read_bytes()`,
and `read()` into a mutable `Span`. The handle owns its operating-system file
descriptor and closes it during destruction.

## Candidate designs

1. Read the complete file with `Path.read_bytes()` and reuse the in-memory
   parser. This is simple but violates metadata-only opening and duplicates the
   complete tensor payload in memory.
2. Store a path and reopen it for every tensor read. This supports positional
   reads, but a path replacement can silently redirect a validated reader to a
   different file.
3. Retain one read-only `FileHandle`, validate metadata against a file-length
   snapshot, and seek on that handle for every tensor read. This avoids
   whole-file loading and path-replacement redirection.
4. Memory-map the file. This can provide borrowed views, but introduces origin,
   alignment, endianness, mapping-lifetime, and platform-policy decisions that
   belong to the separate v0.3 milestone.

## Decision

The public file API consists of:

- `open_safetensors(path, max_header_bytes, strict=False)` returning a movable
  `SafeTensorReader`;
- `SafeTensorReader.metadata()` returning a validated metadata copy;
- `SafeTensorReader.file_length()` returning the length observed at opening;
- `SafeTensorReader.read_tensor_into(name, destination)` filling an exactly
  sized caller-owned mutable byte span; and
- `SafeTensorReader.load_tensor(name)` explicitly allocating and returning one
  tensor's bytes.

`open_safetensors` accepts a `String` path and retains the read-only handle.
Its `strict` argument selects the shared header policy defined by ADR-006; it
does not weaken shape, dtype, offset, size, or coverage validation. The reader
seeks to the end to obtain a `UInt64` file-length snapshot, then reads only the
8-byte prefix and the declared JSON header. It validates tensor offsets against
the remaining file length. A second end seek must observe the same length before
opening succeeds, so growth or truncation during metadata opening is rejected.

Every offset and length derived from the file remains `UInt64` until an I/O API
requires the native signed `Int` type. Addition, subtraction, multiplication,
and conversion use the checked format-core helpers. No file-controlled value is
first represented as floating point or converted with truncation.

Named reads use the validated tensor descriptor and checked addition of its
relative offset to the absolute data origin. The destination length must equal
the validated tensor byte length. The implementation seeks on the retained
handle and loops over `FileHandle.read(Span)` until the destination is full.
Zero-byte tensors still perform the checked lookup and seek but do not read.

Operating-system open, seek, and read failures map to `IoError`. An unexpected
EOF after successful metadata opening also maps to `IoError`, because it means
the opened file no longer matches the validated length snapshot. Unknown tensor
names remain `TensorNotFound`, and an incorrect caller buffer remains
`DestinationSizeMismatch`. Format, schema, dtype, shape, offset, and overflow
failures preserve their existing categories.

The reader owns its handle and is movable but not copyable. Destruction closes
the handle. One reader has one seek cursor, so concurrent or reentrant reads on
the same instance are unsupported. Independent readers may be used when callers
need independent cursors.

Retaining a handle prevents a later path replacement from changing the file
object being read. It cannot detect in-place modification that preserves the
same file length. Callers requiring authenticity, integrity, or a stable
external snapshot must provide those guarantees separately.

The reader returns raw wire bytes. It does not reinterpret dtypes, byte-swap,
provide typed views, map memory, slice tensors, resolve shards, write files, or
integrate with MAX or another tensor runtime.

## Consequences

- Metadata opening uses memory proportional to the header rather than the full
  file.
- Caller-owned reads can reuse storage and avoid an additional allocation.
- Owned loads are explicit and allocate exactly one validated tensor payload.
- Path replacement cannot redirect an already opened reader.
- Reads on one reader are serialized by its shared file cursor.
- Post-open truncation is reported when it prevents an exact read, while
  same-length in-place mutation remains outside the format's guarantees.
- Borrowed views and typed interpretation remain available for later,
  separately reviewed milestones.

## References

- [ADR-006: Read Reference-Compatible Headers by Default](006-compatible-header-reading.md)
- [Mojo 1.0.0 `FileHandle`](https://mojolang.org/docs/std/io/file/FileHandle/)
- [Safetensors format](https://github.com/huggingface/safetensors/blob/main/README.md#format)
