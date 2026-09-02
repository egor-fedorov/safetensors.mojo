# Architecture

This section describes the current architecture of `safetensors.mojo`. It is
the maintained entry point for understanding how the implementation works
today. The numbered [architecture decision records](../decisions/README.md)
preserve why important choices were made and how those choices evolved.

## Layers

| Layer | Responsibility |
| --- | --- |
| `safetensors` | Supported public import facade |
| `safetensors.format` | Runtime-independent framing, JSON parsing, wire dtypes, validation, and serialization planning |
| `safetensors.io` | Retained file handles, buffered reads, Linux mappings, native views, and atomic local-file writes |
| `safetensors.sharding` | Aggregate metadata, index validation, anchored shard resolution, and multi-file readers |

Dependencies point inward:

```text
public facade
   |-- format core
   |-- local I/O ----> format core
   `-- sharding -----> local I/O ----> format core
```

The format core has no filesystem, Python, MAX, or tensor-runtime dependency.
Linux-specific FFI is isolated inside the local I/O and sharding layers.
Importing from the root `safetensors` package is the supported public API;
nested modules remain an implementation detail.

## Read path

Both buffered and mapped readers share the same open-and-validation path:

```text
retained file handle
  -> 8-byte little-endian header length
  -> bounded UTF-8 JSON header
  -> raw metadata
  -> complete semantic validation
  -> validated metadata
  -> buffered bytes or origin-bound mapped views
```

Only the prefix and declared header are read while opening a local file. Tensor
payloads are copied on demand by the buffered reader or exposed through a
read-only whole-file mapping. See [Readers and views](readers-and-views.md).

Sharded readers apply that pipeline to every unique referenced file and build
one exact tensor namespace. Index-controlled shard names resolve through an
anchored directory descriptor, while explicitly supplied paths belong to a
separate caller-trusted boundary. See [Sharded readers](sharded-readers.md).

## Write path

The writer separates pure planning from filesystem mutation:

```text
owned raw tensor inputs
  -> complete validation and canonical layout planning
  -> compact padded JSON header
  -> exclusive sibling temporary file
  -> prefix, header, and payload writes
  -> close and atomic rename
```

No temporary file is created until the complete archive layout has passed
validation. See [Writer](writer.md).

## Cross-cutting invariants

- File-controlled dimensions, data offsets, lengths, products, decimal integer
  accumulation, and narrowing conversions use checked arithmetic.
- Raw decoded metadata and validated metadata are distinct types. High-level
  local-file APIs expose fully validated metadata; low-level format APIs also
  expose the explicit raw parsing stage. Supported validated accessors return
  copies.
- Reader compatibility is intentionally broader than canonical writer output.
  Compatibility mode never relaxes dtype, shape, size, offset, or complete
  coverage validation.
- Tensor payloads cross the public API as raw Safetensors wire bytes unless a
  mapped tensor satisfies the exact native-view contract.
- Public failures use typed `SafeTensorErrorKind` values. Message text provides
  context but is not the machine-readable interface.
- Local mapping, descriptor-relative shard resolution, descriptor-only `statx`
  inspection, and atomic replacement currently depend on Linux facilities.

## Detailed design

- [Format core](format-core.md): framing, JSON policy, metadata models, wire
  dtypes, checked arithmetic, and semantic validation.
- [Readers and views](readers-and-views.md): retained handles, buffered reads,
  memory mappings, ownership, typed views, and external mutation constraints.
- [Sharded readers](sharded-readers.md): aggregate validation, standard index
  parsing, filesystem trust boundaries, buffered shard switching, and eager
  mapped ownership.
- [Writer](writer.md): canonical layout planning, one-shot serialization, and
  atomic local-file replacement.

For historical context and supersession relationships, see the
[decision index](../decisions/README.md).
