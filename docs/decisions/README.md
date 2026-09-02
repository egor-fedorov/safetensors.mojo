# Architecture Decision Records

Architecture decision records (ADRs) preserve the context, alternatives, and
tradeoffs behind important design choices. They are historical records, not the
primary description of the current code.

Start with the [architecture guide](../architecture/README.md) to understand
how the library works today. Follow an ADR link when the reasoning or an older
policy matters. Accepted decision rationale is not rewritten to make history
look current; a later decision supersedes it explicitly.

## Index

| ID | Decision | Status | Current architecture |
| --- | --- | --- | --- |
| [ADR-001](001-json-parser.md) | Use a schema-directed pure-Mojo JSON parser | Superseded in part by ADR-006 | [Format core](../architecture/format-core.md) |
| [ADR-002](002-local-reader.md) | Retain a file handle for local random-access reads | Accepted | [Readers and views](../architecture/readers-and-views.md) |
| [ADR-003](003-memory-mapped-reader.md) | Own read-only mappings and return origin-bound byte views | Accepted | [Readers and views](../architecture/readers-and-views.md) |
| [ADR-004](004-native-typed-views.md) | Expose exact aligned native scalar views | Accepted | [Readers and views](../architecture/readers-and-views.md) |
| [ADR-005](005-atomic-writer.md) | Write canonical files through an atomic sibling transaction | Accepted | [Writer](../architecture/writer.md) |
| [ADR-006](006-compatible-header-reading.md) | Read reference-compatible headers by default | Accepted | [Format core](../architecture/format-core.md) |
| [ADR-007](007-sharded-readers.md) | Separate trusted shard lists from untrusted index resolution | Accepted | [Sharded readers](../architecture/sharded-readers.md) |

ADR-001 remains authoritative for the pure-Mojo parser, decoded-key duplicate
detection, exact integer parsing, and checked arithmetic. ADR-006 supersedes
only its default boundary-whitespace and unknown-descriptor-field policy.

## Maintenance convention

New records use the next zero-padded identifier and begin with a status and
date. When behavior changes, add or supersede a record, update this index, and
update the relevant architecture page. Do not renumber existing records:
stable identifiers keep discussions and external links meaningful.
