# Roadmap

This roadmap records project direction, not delivery dates or compatibility
promises. Until version 1.0, minor releases may refine the public API when Mojo
language or packaging constraints require it.

## v0.1.x — Format core (released)

The released line contains only the runtime-independent Safetensors format
core:

- strict header parsing and checked validation;
- raw and validated metadata models;
- all recognized wire-format dtypes;
- deterministic valid and malformed fixtures; and
- Mojo 1.0.0 Pixi and Conda packaging.

Maintenance releases in this line should focus on correctness, compatibility,
documentation, and additional malformed-input coverage.

## v0.2.x — Local random-access reader (released)

- Metadata-only local file opening with an owned read-only handle.
- Exact named-tensor reads into caller-owned byte buffers.
- Explicit owned loads of raw tensor bytes.
- Detection of ordinary file-length changes around validated reads.

## v0.3.x — Borrowed read-only views (current development)

- Read-only memory mapping.
- Mojo-origin-bound byte views.
- Alignment and endianness checks for supported typed views.

## Candidate later milestones

### v0.4.x — Writer

- Deterministic serialization.
- Streaming writes with checked layout arithmetic.
- Atomic file replacement and reference-implementation compatibility tests.

## Exploratory work

Slicing, sharded indexes, and tensor-runtime adapters remain separate optional
layers. They should not introduce dependencies into the format core.
