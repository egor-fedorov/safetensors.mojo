# ADR-001: Use a Schema-Directed Pure-Mojo JSON Parser

- Status: Accepted
- Date: 2026-08-25

## Context

The Safetensors header is UTF-8 JSON, but its security properties require more
than decoding a JSON object into a dictionary. Object keys must remain visible
until duplicates are checked after JSON string decoding. Shape dimensions and
data offsets must also be parsed as exact unsigned integers without first
passing through `Float64`.

The parser must remain independent of Python, MAX, and a tensor runtime. The
runtime environment is pinned to Mojo 1.0.0.

## Candidate check

The following candidates were evaluated before implementation:

1. Mojo 1.0.0 standard library JSON support. Compiler probes for both
   `from std import json` and `import json` fail because no JSON package is
   available.
2. Python's standard `json` module. It can be configured with pair and integer
   callbacks, but using it would introduce a Python runtime dependency into the
   core and would prevent a pure-Mojo parser.
3. EmberJson 0.3.4. A probe on Mojo 1.0.0 rejected
   `{"a":1,"\u0061":2}` with `Duplicate key: a`, demonstrating duplicate
   detection after key decoding. It also represented `9007199254740993` as an
   exact integer instead of converting it to `Float64`. However, inspection of
   its generic DOM integer path found unchecked intermediate `UInt64`
   accumulation before post-validation. A file-controlled integer can
   therefore overflow before the parser checks its range, which conflicts with
   the requirement that every value derived from the file use checked
   arithmetic.
4. A schema-directed parser implemented in this repository. Its object and
   number handling can enforce both requirements before values enter the
   metadata model.

## Decision

The format core uses a schema-directed, pure-Mojo parser for the Safetensors
header rather than a general JSON object model.

The parser:

- validates the complete header as UTF-8;
- requires the first header byte to be `{`;
- implements all JSON string escapes, `\uXXXX`, and UTF-16 surrogate pairs;
- compares fully decoded keys before inserting them into any dictionary;
- rejects duplicate keys at the top level, in `__metadata__`, and in every
  tensor descriptor, including lexical equivalents such as `a` and `\u0061`;
- parses shape dimensions and offsets directly into `UInt64` using checked
  decimal accumulation;
- accepts integer tokens matching only `0` or `[1-9][0-9]*` and rejects signs,
  fractions, exponent notation, leading zeroes, and values above `UInt64.MAX`;
- requires exactly one top-level object and permits only byte `0x20` after that
  object within the declared header length; and
- rejects unknown tensor descriptor fields in the initial strict format-core
  implementation.

The parser is not a reusable general-purpose JSON API. Its nesting is limited
by the Safetensors schema, so untrusted input cannot create an unbounded
recursive object tree.

All arithmetic derived from file-controlled values uses an explicit
guard-before-operation check. This is required because Mojo 1.0.0 integer
arithmetic wraps on overflow and does not provide the removed SIMD
`add_with_overflow`, `sub_with_overflow`, or `mul_with_overflow` methods.

## Consequences

- Duplicate decoded keys cannot be silently overwritten by a dictionary.
- Integers above the exact range of `Float64` retain their exact value or fail
  with a controlled range error.
- The core has no JSON, Python, MAX, or tensor-runtime dependency.
- The project owns the implementation and test burden for JSON strings,
  Unicode escapes, whitespace, punctuation, and error reporting.
- Accepting future descriptor extensions will require an explicit follow-up
  decision and a bounded JSON-value skipping strategy.
