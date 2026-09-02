# ADR-007: Separate Trusted Shard Lists from Untrusted Index Resolution

- Status: Accepted
- Date: 2026-09-02
- Current architecture: [Sharded readers](../architecture/sharded-readers.md)

> Platform-scope addendum: this record defines the sharding trust and ownership
> contracts and documents their original Linux resolver.
> [ADR-008](008-supported-platforms.md) adds the Darwin resolver while
> preserving the same basename, symlink, regular-file, identity, and
> nonblocking-open guarantees.

## Context

Large model checkpoints commonly split tensors across several Safetensors
files and describe that split with a `*.safetensors.index.json` file. The
library needs to expose the same validated metadata, buffered reads, raw mapped
bytes, and native typed views across those files without weakening the
single-file guarantees established by ADR-002 through ADR-004.

There are two distinct sources of shard paths:

1. an explicit list supplied by the application; and
2. decoded `weight_map` values from an index file.

The first is part of the caller's trust boundary. The second is untrusted file
content that becomes a filesystem lookup. Treating both sources identically
would either reject useful deployments or allow an index to escape its own
directory. In particular, Hugging Face cache snapshots commonly expose shard
files through symlinks to blob storage, while an untrusted index must not be
allowed to select a symlink or a path such as `../../../etc/shadow`.

The mapped API also has a Mojo 1.0 ownership constraint. A tensor span borrowed
directly from an element of `List[MappedSafeTensorFile]` carries the element's
interior origin rather than the outer archive's origin. Mutating a lazy cache
would invalidate that interior origin, and the compiler cannot prove that a
returned span remains valid for the outer archive's complete borrow.

## Alternatives considered

### Require an index for every sharded archive

This preserves the standard file layout and its `total_size` cross-check, but
cannot safely follow cache-layout shard symlinks under the untrusted path
policy. It also excludes applications that already have an authoritative list
of local shard paths.

### Accept only a caller-supplied list

Scanning every supplied header can construct the tensor-to-file map without an
index. This removes index-controlled path traversal and matches APIs that
receive paths from a model loader. It loses standard index validation,
including declared routing and `metadata.total_size`, and does not directly
open downloaded index layouts.

### Lazily map shards

This limits open descriptors and mappings, but insertion into or replacement
within the mapping collection conflicts with origin-bound spans that may still
be live. Adding reference-counted or individually boxed owners would enlarge
the unsafe and public ownership surface beyond the current library design.

## Decision

Both entry forms are supported, with separate trust contracts:

```mojo
open_sharded_safetensors(paths, max_header_bytes, max_shards, strict)
map_sharded_safetensors(paths, max_header_bytes, max_shards, strict)
open_safetensors_index(index_path, max_index_bytes, max_index_entries, max_header_bytes, max_shards, strict)
map_safetensors_index(index_path, max_index_bytes, max_index_entries, max_header_bytes, max_shards, strict)
```

The imported package remains `safetensors`; sharding does not introduce a new
public namespace. The root facade also exports `ShardedTensorInfo`,
`ShardedSafeTensorMetadata`, `ShardedSafeTensorReader`, and
`MappedShardedSafeTensorArchive`.

### Explicit path trust boundary

Every path passed to an explicit-list entry point is trusted application input.
The operating system may follow symlinks in those paths. This supports layouts
such as Hugging Face cache snapshots, where the visible shard filenames are
symlinks into a content-addressed blob directory. The resolved object must
still be a regular file, every shard is parsed and semantically validated, and
file identities are deduplicated by Linux device and inode identity, strengthened
with birth time when the filesystem reports it.

An application must never copy untrusted `weight_map` strings into this API to
bypass index resolution. The distinction is an explicit security boundary, not
two equivalent spellings of the same operation.

### Index path trust boundary

The `index_path` argument itself is trusted. Its final component may be a
symlink. The implementation first opens the lexical parent directory and then
opens the index basename relative to that directory descriptor, following the
trusted final symlink. The parent directory descriptor remains the resolution
anchor: shard names are resolved beside the path the caller supplied, not
beside a symlink target elsewhere.

Every decoded `weight_map` value is untrusted. It must be exactly one basename
ending in `.safetensors`. Empty names, `.`, `..`, `/`, `\`, colon, NUL, ASCII
and Unicode C1 controls, absolute paths, drive paths, UNC paths, and URL-like
forms are rejected as `PathTraversal`. A shard is first pinned relative to the
anchored directory with `openat(O_PATH | O_NOFOLLOW | O_CLOEXEC)` and classified
through descriptor-only `statx(AT_EMPTY_PATH)`. After that regular-file
preflight, the pathname is opened again with
`O_NONBLOCK | O_NOFOLLOW | O_CLOEXEC`; the resulting descriptor must also be a
regular file and its identity must match the preflight descriptor. A shard
symlink selected through an index is therefore rejected even when its target
would be a regular file, including when a pathname changes to a symlink between
the two opens.

`O_NONBLOCK` prevents an attacker-controlled FIFO from blocking the process;
the regular-file check prevents it and every other non-regular object from
being treated as a shard. A security-policy violation is `PathTraversal`.
Ordinary absence, permissions, and I/O failures remain `IoError`.

### Index JSON and aggregate validation

The index parser is a bounded, pure-Mojo, schema-directed parser. It requires a
non-empty root `weight_map` object from decoded tensor names to validated shard
basenames. `metadata` is optional. If `metadata.total_size` is present, it is
parsed directly as an exact `UInt64`; it never passes through `Float64`.

Unknown root and metadata values are accepted for forward compatibility only
after one complete JSON value has been syntactically validated and skipped.
Decoded duplicate keys are rejected at the root, inside `metadata`, and inside
`weight_map`, including escape-equivalent spellings. UTF-8, integer grammar,
cursor movement, nesting, input size, and every file-derived arithmetic
operation remain bounded or checked. The ignored-value nesting limit is 128.

Every unique referenced shard is opened and its complete Safetensors header is
validated before an archive is returned. Aggregate validation requires:

- every `weight_map` tensor to exist in exactly its declared shard;
- every tensor physically present in a referenced shard to appear in the map;
- no tensor name to appear physically in more than one referenced shard;
- no tensor to be routed to a different shard than the map declares; and
- declared `metadata.total_size`, when present, to equal the checked sum of
  tensor payload byte lengths across all unique shards.

`total_size` excludes prefixes, JSON headers, header padding, and filesystem
allocation. Extra files in the directory are outside the archive and ignored.
Explicit-list opening applies the same per-shard validation and rejects global
duplicate tensor names, but has no declared routing or `total_size` to compare.

The shard count defaults to a maximum of 256 distinct decoded index shard-name
strings or unique physical identities from a trusted explicit list. The index
byte limit defaults to 100,000,000, and the `weight_map` entry limit defaults to
1,000,000. Entry and distinct-value counting happen during parsing; the latter
therefore precedes basename validation as well as every shard open. All three
limits are caller-configurable. Physical file identities are deduplicated after
opening. At least one unique shard is required; a one-file sharded archive is
valid. The existing header limit applies independently to each shard. `strict`
controls only the Safetensors header policy from ADR-006; index JSON syntax and
the index security policy are always enforced.

### Buffered and mapped ownership

`ShardedSafeTensorReader` validates shard headers sequentially during
construction, closes those temporary handles, and stores validated metadata,
file identity, length, and the information required to reopen each shard. The
identity contains Linux device and inode values plus birth time when the
filesystem reports it through `statx`. An index-based reader retains the
anchored directory descriptor. At runtime it keeps at most one active
`SafeTensorReader`. Switching shards reopens and fully revalidates the file,
then compares identity, length, and the metadata snapshot before serving bytes.
An observed replacement is reported as `IoError`. Reopening detects a changed
metadata snapshot, but same-inode, same-length in-place content mutation while
a shard remains active is undetectable, as with the single-file buffered
reader. On a filesystem that omits birth time, pathological inode reuse can
also evade this best-effort replacement check.

`MappedShardedSafeTensorArchive` eagerly opens and maps every unique shard.
This costs one descriptor and one whole-file virtual mapping per shard, but its
owner collection is immutable after construction and multiple tensor views
from different shards may coexist. Each underlying mapped reader performs the
existing bounds, dtype, alignment, endianness, and external-length checks.

One small internal wrapper casts the already checked span's interior origin to
the immutable outer archive origin with `unsafe_origin_cast`. This is an
audited lifetime bridge, not a general pointer escape: the mapping list is
never mutated after construction, every mapping is owned by the outer archive,
and the public return type remains `Span[..., origin]` borrowed from that
archive. Compiler contracts must prove that views cannot escape, become
mutable, survive owner consumption, or coexist with an owner copy.

### Errors

The removed `PathTraversal` ordinal is restored without renumbering existing
kinds. Sharded APIs add `IndexTooLarge`, `IndexEntryLimitExceeded`,
`InvalidIndex`, `ShardMismatch`, `TotalSizeMismatch`, and
`ShardLimitExceeded`. Existing parser, UTF-8, duplicate-key, missing-field,
field-type, validation-overflow, dtype, offset, and I/O errors retain their
established meanings.

## Consequences

- Standard index files receive exact cross-shard routing and size validation.
- Trusted applications can open symlink-based caches through explicit paths
  without weakening index-controlled path resolution.
- Buffered access bounds descriptor use after construction to the anchored
  directory descriptor plus at most one active shard reader.
- Mapped access preserves the single-file zero-copy and typed-view API, at the
  cost of one descriptor and mapping per unique shard.
- File aliases do not multiply resource use or permit inconsistent duplicate
  validation because identity, not path spelling, defines uniqueness.
- Index parsing and filesystem resolution introduce a larger hostile-input
  surface that requires dedicated malformed fixtures, compiler contracts, and
  deterministic fuzzing.

## Out of scope

This decision does not add remote Hub downloads, an index writer, automatic
shard planning, tensor slicing, MAX or runtime adapters, checksums, integrity
authentication, or cross-platform mapping backends.

## References

- [ADR-002: Retain a File Handle for Local Random-Access Reads](002-local-reader.md)
- [ADR-003: Own Read-Only Mappings and Return Origin-Bound Byte Views](003-memory-mapped-reader.md)
- [ADR-004: Expose Exact Aligned Native Scalar Views](004-native-typed-views.md)
- [ADR-006: Read Reference-Compatible Headers by Default](006-compatible-header-reading.md)
- [ADR-008: Support Three Native Mojo Platforms](008-supported-platforms.md)
- [Linux `openat(2)`](https://man7.org/linux/man-pages/man2/openat.2.html)
- [Linux `statx(2)`](https://man7.org/linux/man-pages/man2/statx.2.html)
