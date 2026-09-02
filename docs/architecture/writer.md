# Writer

The writer is a deterministic one-shot API for local Safetensors files. It
accepts owned raw wire bytes, validates and plans the entire archive, streams
the planned sections to an exclusive sibling temporary file, and atomically
replaces the destination entry.

`SafeTensorData` owns a tensor name, `SafeDType`, shape, and byte list. It is
movable but not copyable. `save_safetensors(path, tensors, user_metadata)` is
the public write operation. Payload bytes must already use packed C-order
little-endian Safetensors representation; the writer performs no typed-value
conversion.

## End-to-end flow

```text
SafeTensorData and user metadata
  -> validate all inputs
  -> choose canonical order and checked offsets
  -> encode bounded compact JSON and padding
  -> encode the 8-byte little-endian header length
  -> create an exclusive sibling temporary file
  -> write prefix, header, and payloads in planned order
  -> close
  -> rename as the commit point
```

Format planning is implemented in the runtime-independent format layer. The
I/O layer owns prefix encoding and transaction orchestration, while the POSIX
libc boundary and compile-time flag selection are isolated in internal
platform and atomic-file helpers.

## Complete preflight

Before creating a temporary file, planning rejects:

- reserved or duplicate tensor names;
- an invalid directly constructed `SafeDType`;
- shape-product or bit-length overflow;
- complete tensor payloads whose bit lengths are not byte-addressable;
- payload lengths that do not exactly match dtype and shape; and
- a header that exceeds the configured header limit;
- overflow in cumulative offsets or total file length.

A zero dimension makes the element count zero before other dimensions are
multiplied. Tensor offsets are generated cumulatively, so the resulting data
section is gap-free and overlap-free by construction.

If preflight fails, the filesystem is unchanged.

## Canonical representation

The writer emits one deterministic representation for equivalent inputs:

- tensors are ordered by descending recognized `SafeDType` ordinal, then by
  ascending decoded name;
- user-metadata keys are ordered lexicographically;
- `__metadata__` appears first when metadata is non-empty;
- every tensor descriptor contains exactly `dtype`, `shape`, and
  `data_offsets`;
- JSON is compact and uses one string-escaping policy;
- the header is padded with zero to seven ASCII spaces to a multiple of eight
  bytes; and
- the header length is stored as an unsigned 8-byte little-endian integer.

Compatible reader behavior does not change writer output. The writer continues
to emit canonical boundary padding and a closed descriptor schema.

## Atomic local-file transaction

The transaction creates a random sibling temporary name using `getentropy`,
opens it with the platform's `O_EXCL | O_CLOEXEC` flags, and requests mode
`0600`. The effective permissions remain subject to the process `umask`.

All planned sections are written with complete-write loops. The temporary file
is explicitly closed before `rename` atomically replaces the destination
directory entry. Until that rename succeeds, an existing destination remains
unchanged. The transaction destructor makes a best-effort attempt to close and
remove an uncommitted temporary file after a failure.

The destination's parent directory must already exist. Replacing a symlink
replaces the symlink entry rather than following its target. Readers that
opened the previous destination remain attached to its previous inode.
Concurrent writers are not locked; the last successful rename determines the
visible destination.

This is an atomic visibility guarantee, not a crash-durability guarantee. The
writer does not call `fsync` for either the file or parent directory.

## Failure boundaries

| Phase | Observable result |
| --- | --- |
| Preflight validation | No filesystem mutation |
| Temporary-file write or close failure | Destination unchanged; temporary cleanup is best effort |
| Rename failure | Destination unchanged; temporary cleanup is best effort |
| Successful rename | Complete new file becomes visible atomically |

Writer failures use the same typed `SafeTensorErrorKind` interface as the read
path. Relevant categories distinguish duplicate or reserved names, unsupported
dtypes, overflow, non-byte-addressable payloads, tensor-size mismatches,
oversized headers, and filesystem I/O failures.

## Current limitations

- There is no incremental, append, or in-place update API.
- There is no API that assembles and returns a complete archive in memory.
- There is no typed encoding, byte swapping, or tensor-runtime integration.
- There are no memory-mapped writes or transaction guarantees beyond the
  supported POSIX platforms.
- There is no writer locking or crash-durability promise.

## Decision history

- [ADR-005](../decisions/005-atomic-writer.md) defines canonical planning and
  atomic sibling replacement.
- [ADR-006](../decisions/006-compatible-header-reading.md) separates tolerant
  reading from canonical writing.
- [ADR-008](../decisions/008-supported-platforms.md) defines the supported
  native targets and the portable transaction boundary.
