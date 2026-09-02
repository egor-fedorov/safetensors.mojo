# Sharded Readers

Sharded readers present several validated Safetensors files as one deterministic
tensor namespace. They preserve the two single-file access models: buffered
readers return owned bytes, while mapped archives return origin-bound raw or
native typed views.

| Entry point | Path source | Symlink policy | Payload access |
| --- | --- | --- | --- |
| `open_sharded_safetensors(paths, ...)` | Trusted caller list | Allowed | Buffered copies |
| `map_sharded_safetensors(paths, ...)` | Trusted caller list | Allowed | Eager mapped views |
| `open_safetensors_index(index_path, ...)` | Untrusted `weight_map` values | Shard symlinks rejected | Buffered copies |
| `map_safetensors_index(index_path, ...)` | Untrusted `weight_map` values | Shard symlinks rejected | Eager mapped views |

All supported types and entry points are exported by the root `safetensors`
package. Nested sharding modules are implementation details.

## Aggregate metadata

`ShardedSafeTensorMetadata` contains one global decoded-name index and the
validated descriptor for every tensor. Its supported operations are `len()`,
`is_empty()`, `contains()`, `names()`, `info()`, `shard_names()`,
`total_size()`, and `declared_total_size()`. Collection and descriptor accessors
return independent values rather than mutable aliases into validation state.

`ShardedTensorInfo` combines the per-file `TensorInfo` data with the shard
identifier selected during aggregate validation. Names and shard names are
deterministically lexicographically ordered. `total_size()` is the checked sum
of tensor payload byte lengths. It excludes each file's 8-byte prefix, JSON
header, padding, and filesystem allocation. `declared_total_size()` is present
only when an index supplied `metadata.total_size`.

An archive must contain at least one unique shard, but its tensor namespace may
be empty when its single-file metadata is valid. One unique shard is allowed.
Exact path repetitions and aliases of the same file are coalesced using Linux
device and inode identity, strengthened with birth time when `statx` reports
it; the first path spelling supplies its display identifier.

## Index parsing

A shard index is a UTF-8 JSON object with this semantic shape:

```json
{
  "metadata": {"total_size": 1024},
  "weight_map": {
    "decoder.bias": "model-00001-of-00002.safetensors",
    "decoder.weight": "model-00002-of-00002.safetensors"
  }
}
```

`weight_map` is required and non-empty. Its decoded keys are tensor names and
its values are shard basenames. `metadata` is optional; `total_size`, when
present, must be an exact non-negative `UInt64` JSON integer. Unknown root and
metadata fields are forward-compatible: the parser validates and consumes one
complete JSON value without retaining it. The nesting limit for ignored values
is 128.

The parser validates the complete document and detects duplicate decoded keys
at the root, inside `metadata`, and inside `weight_map`. Escape-equivalent keys
therefore cannot bypass duplicate detection. Integers are accumulated with
checked arithmetic and never represented as `Float64`. `strict` does not apply
to this document; it only selects compatible or canonical parsing for each
referenced Safetensors header.

The default `DEFAULT_MAX_INDEX_BYTES` limit is 100,000,000 bytes. The default
`DEFAULT_MAX_SHARDS` limit is 256 unique file identities. Callers may lower or
raise either limit explicitly. The existing `DEFAULT_MAX_HEADER_BYTES` limit
is enforced independently for every shard.

## Filesystem trust boundaries

The explicit-list APIs accept trusted application paths. Symlinks are followed,
which is required for common cache layouts. The resolved descriptor must refer
to a regular file. Passing strings copied from an untrusted index to these APIs
would cross the documented trust boundary and is unsupported.

For an index API, the caller-supplied `index_path` is trusted but its decoded
`weight_map` values are not. Opening follows this sequence:

```text
trusted index path
  -> open lexical parent directory
  -> open index basename relative to parent (final symlink allowed)
  -> parse and validate weight_map basenames
  -> open each shard relative to retained parent (no final symlink)
  -> require a regular-file descriptor
```

The directory descriptor fixes the resolution root before parsing. If the
trusted index file is itself a symlink, shard names still resolve beside the
lexical index path supplied by the caller, not beside the symlink target.

Each untrusted shard value must be one non-empty basename ending in
`.safetensors`. The resolver rejects `.`, `..`, slash, backslash, colon, NUL,
ASCII and Unicode C1 controls, and path-like absolute, drive, UNC, or URL
spellings. It first obtains an `O_PATH | O_NOFOLLOW` descriptor and classifies
that pinned object with `statx(AT_EMPTY_PATH)`. It opens only a regular object
for reading with `O_NONBLOCK | O_NOFOLLOW | O_CLOEXEC`, then requires both
descriptor identities to agree. This prevents directory escape, shard symlink
traversal, and blocking on an attacker-controlled FIFO or socket.

Use explicit trusted paths for a Hugging Face cache snapshot whose visible
shards are symlinks. An index-controlled shard symlink intentionally produces
`PathTraversal`; this is the security boundary working as designed.

## Aggregate validation

Before either access object is returned, every unique shard header passes the
complete single-file Safetensors parser and semantic validator. The aggregate
builder then requires exact consistency:

- each tensor named by `weight_map` exists in its declared shard;
- no referenced shard contains a tensor omitted from `weight_map`;
- a tensor name appears physically in only one unique shard;
- the declared route and the physical shard agree; and
- declared `metadata.total_size`, when present, equals the checked aggregate
  tensor payload size.

Explicit-list construction scans all supplied shard headers and builds routes
from physical contents. It rejects duplicate tensor names across unique shard
identities. It has no declared routing table or size to cross-check. Files not
referenced by the list or index are unrelated and ignored.

File-derived lengths, offsets, sums, indexes, and native-width conversions use
checked arithmetic. Per-file dtype, shape, tensor-size, offset, and gap-free
coverage rules remain unchanged.

## Buffered sharded reader

`ShardedSafeTensorReader` offers `metadata()`, `read_tensor_into()`, and
`load_tensor()` with the same ownership behavior as `SafeTensorReader`.
Construction opens shard headers sequentially and closes those temporary file
handles after recording validated metadata, Linux device/inode identity,
optional birth time, and file length. An index-based owner retains the
lexical-parent directory descriptor; an explicit-list owner retains trusted
path spellings.

At runtime the owner keeps at most one active shard reader. A request in the
same shard reuses it. Switching shards closes the previous reader, reopens the
required shard through its original trust path, fully revalidates its header,
and compares file identity, length, and validated metadata with the opening
snapshot. Replacement or incompatible mutation raises `IoError` before tensor
bytes are returned.

As with `SafeTensorReader`, the result is not an authenticated snapshot.
Reopening detects a changed metadata snapshot, but same-inode, same-length
in-place content modification while a shard remains active cannot be detected.
On filesystems that do not report birth time, pathological deletion and
immediate inode reuse can also evade the replacement check. One reader's shared
seek cursor does not support concurrent or reentrant reads.

## Eager mapped archive

`MappedShardedSafeTensorArchive` offers `metadata()`, `tensor_bytes()`, and
`tensor_view[DType]()` across the global tensor namespace. It eagerly retains
one `MappedSafeTensorFile` for every unique shard. This consumes one open file
descriptor and one whole-file virtual mapping per shard, while payload pages
remain demand-loaded by the operating system.

Eager ownership is what makes views from different shards coexist safely. The
internal mapping list is immutable after construction. After an underlying
mapped reader performs its existing name, range, length, dtype, endianness,
and alignment checks, one audited wrapper uses `unsafe_origin_cast` to tie the
result's interior origin to the immutable outer archive borrow. Public raw and
typed spans therefore retain the same `Span[..., origin]` lifetime contract as
single-file mapped spans: they cannot outlive or consume their owner and are
never mutable.

The external-mutation warning for a single mapped file applies independently
to every shard. In-place same-length mutation is not an integrity violation the
library can detect, and truncation followed by dereference may terminate the
process with `SIGBUS`. Keep every backing shard stable for the complete archive
and borrowed-view lifetime.

## Error surface

Sharded readers add these machine-readable error codes:

| Code | Meaning |
| --- | --- |
| `IndexTooLarge` | Index input exceeds the configured byte limit |
| `InvalidIndex` | Structurally or semantically invalid index content not covered by a narrower parser error |
| `PathTraversal` | An index-controlled shard name or object violates the resolution policy |
| `ShardMismatch` | Declared routing and physical shard contents disagree, or shard tensor names conflict |
| `TotalSizeMismatch` | Declared and computed tensor payload sizes differ |
| `ShardLimitExceeded` | Unique shard count exceeds the configured limit |

Specific JSON, UTF-8, duplicate-key, field, overflow, Safetensors validation,
tensor lookup, destination-size, dtype, alignment, endianness, and I/O errors
retain their existing codes.

## Current limitations

- Mapping and descriptor-relative resolution currently target Linux.
- Remote Hub resolution and downloading are not provided.
- There is no index writer or automatic shard planner.
- There is no tensor slicing, MAX adapter, runtime tensor conversion, checksum,
  authentication, or immutable file snapshot.
- Mapped access intentionally favors stable origin-bound views over lazy
  descriptor use; raise `max_shards` only with the process descriptor budget in
  mind.

## Decision history

- [ADR-007](../decisions/007-sharded-readers.md) defines the separate path
  trust contracts, index validation, resource models, and mapped ownership
  boundary.
- [ADR-002](../decisions/002-local-reader.md) defines the underlying buffered
  reader contract.
- [ADR-003](../decisions/003-memory-mapped-reader.md) and
  [ADR-004](../decisions/004-native-typed-views.md) define the underlying mapped
  byte and native typed views.
