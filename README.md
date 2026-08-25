# safetensors.mojo

`safetensors.mojo` is an early, pure-Mojo implementation of the Safetensors
file format. It provides a strict runtime-independent format core and a local
random-access reader plus Linux memory-mapped views for raw tensor bytes; it
does not load values into a tensor runtime.

> **Disclaimer:** `safetensors.mojo` is an independent implementation of the
> Safetensors file format for Mojo and is not affiliated with or endorsed by
> Hugging Face.

## Naming

The project uses distinct names at each packaging layer:

- Repository and project brand: `safetensors.mojo`
- Pixi/Conda distribution: `safetensors-mojo`
- Importable Mojo package: `safetensors`

The project does not create or use a `mojo.safetensors` namespace.

## Current scope

The current API provides:

- the independent `SafeDType` wire-format model;
- raw and validated metadata structures;
- decoding of the 8-byte little-endian header length;
- strict UTF-8 and Safetensors JSON header parsing;
- decoded-key duplicate detection;
- exact unsigned integer parsing without a floating-point intermediate;
- checked shape, bit-length, byte-length, offset, and full-coverage
  validation;
- valid and malformed compatibility fixtures;
- metadata-only opening through an owned read-only file handle;
- exact named-tensor reads into caller-owned byte buffers;
- explicit owned loads of one tensor payload;
- Linux whole-file read-only private mappings; and
- immutable named-tensor byte spans whose Mojo origins are tied to the mapping
  owner.

Unknown tensor descriptor fields are rejected in this strict initial
implementation. See
[ADR-001](docs/decisions/001-json-parser.md) for the parser decision and
[ADR-002](docs/decisions/002-local-reader.md) for the local-reader design.
[ADR-003](docs/decisions/003-memory-mapped-reader.md) defines mapping ownership
and external-mutation constraints. Future work is tracked in
[ROADMAP.md](ROADMAP.md).

## Usage

The Conda package is published for Linux x86-64 on the project's Prefix.dev
channel. Add that channel before the Modular and conda-forge channels, then
install the distribution:

```toml
[workspace]
channels = [
  "https://prefix.dev/egor-fedorov/safetensors-mojo",
  "https://conda.modular.com/max",
  "conda-forge",
]
platforms = ["linux-64"]

[dependencies]
safetensors-mojo = "==0.2.0"
```

The installed Mojo package is imported as `safetensors`.

Open a local file without loading its tensor data, inspect the validated
metadata, and explicitly load one tensor as owned raw wire bytes:

```mojo
from safetensors import open_safetensors


def main() raises:
    var reader = open_safetensors("model.safetensors")
    var metadata = reader.metadata()
    var info = metadata.info("weights")
    var bytes = reader.load_tensor("weights")

    print(info.dtype, info.shape, info.byte_length)
    print("loaded bytes:", len(bytes))
```

To reuse caller-owned storage, pass a mutable byte buffer whose length exactly
matches the validated tensor byte length:

```mojo
var destination = List[UInt8](length=16, fill=0)
reader.read_tensor_into("weights", destination)
```

`SafeTensorReader` owns one file handle and is movable but not copyable. Calls
on one reader share its seek cursor and must not execute concurrently. Opening
reads only the 8-byte prefix and declared JSON header; tensor data is read only
by `read_tensor_into()` or `load_tensor()`.

### Unreleased v0.3 mapped API

The following API is available on the `main` branch for the future v0.3.0
release. It is not included in the published v0.2.0 package shown above.

```mojo
from safetensors import map_safetensors


def main() raises:
    var mapped = map_safetensors("model.safetensors")
    var info = mapped.metadata().info("weights")
    var bytes = mapped.tensor_bytes("weights")

    print(info.dtype, info.shape, info.byte_length)
    print("mapped bytes:", len(bytes), bytes[0])
```

`MappedSafeTensorFile` owns one Linux `PROT_READ | MAP_PRIVATE` whole-file
mapping and is movable but not copyable. `tensor_bytes()` returns an immutable
`Span[UInt8]` without copying the payload. Its Mojo origin prevents subsequent
use after the mapping owner is consumed.

The backing inode must remain unchanged from before `map_safetensors()` starts
until the mapping owner and every borrowed span are dead. A length check before
returning a span catches an already-observed growth or truncation, but cannot
eliminate the later race: dereferencing pages after external truncation can
terminate the process with `SIGBUS`. Renaming, unlinking, or replacing the path
does not redirect an existing mapping.

The format core remains available for caller-owned buffers containing a
complete `.safetensors` file:

```mojo
from std.pathlib import Path

from safetensors import parse_metadata


def main() raises:
    var bytes = Path("model.safetensors").read_bytes()
    var metadata = parse_metadata(bytes)

    for name in metadata.names():
        var info = metadata.info(name)
        print(name, info.dtype, info.shape, info.begin, info.end)
```

The parser validates ranges against the remaining data-buffer length but
neither copies nor interprets tensor data. File-reader results are raw wire
bytes. Wire data is defined as packed C-order and little-endian.

Validated metadata accessors return copies. Mojo 1.0 does not enforce field
visibility, so underscore-prefixed fields and direct `SafeTensorMetadata` or
`SafeTensorReader` or `MappedSafeTensorFile` construction are implementation
details. Mutating or constructing this state outside the public parsing and
opening functions is unsupported and can invalidate the validated-state
contract. The supported API is exported from the root `safetensors` package;
nested module paths are internal and may change between releases.

## Deliberate limitations

Mapped access currently exposes only borrowed raw byte spans on Linux. This
milestone does not implement native typed tensor views, writers, slicing,
sharding, MAX adapters, or other tensor-runtime adapters. The parser and local
access APIs validate data ranges but do not interpret tensor values.

Safetensors prevents arbitrary code execution through its data format, but it
does not provide authenticity, integrity, signatures, encryption, or protection
against in-place mutation. A retained reader handle prevents path replacement
from redirecting later reads and detects ordinary file-length changes around a
read, but it cannot detect same-length changes to already opened file contents.
A read-only private mapping is likewise not an immutable snapshot and requires
a stable backing file for its entire lifetime.

## Development

The supported toolchain is Mojo 1.0.0 on Linux x86-64. Pixi installs the exact
compiler version and the Python-only development dependencies used to generate
reference fixtures. The generated `.mojoc` package is compiler-version-specific
and must be consumed with Mojo 1.0.0.

The repository is organized by responsibility:

```text
src/safetensors/
  format/       # Runtime-independent parsing and validation
  io/           # Buffered and memory-mapped local-file access
tests/
  unit/         # Focused format-core behavior
  integration/  # Fixture and local-I/O behavior
  contracts/    # Compile-time public and ownership contracts
  tooling/      # Python tooling tests
tools/
  checks/       # Formatting and test orchestration
  fixtures/     # Reproducible fixture generation
  packaging/    # Clean-install package smoke tests
  release/      # Release-policy and channel preflight checks
```

Mojo production modules use absolute `safetensors.*` imports. The `format`
subpackage depends only on shared errors, `io` depends on the format core, and
the root package re-exports the supported API from both subpackages.

```text
pixi install
pixi run check
pixi run all
```

`pixi run check` verifies formatting, runs the Python tooling and compile-time
API contract tests, compiles the importable package, runs the Mojo tests, and
checks that fixtures are reproducible. `pixi run all` additionally builds the
`safetensors-mojo` Conda package and verifies it in a clean Pixi workspace.
Individual tasks include `compile`, `test`, `format-check`, `fixtures-check`,
and `package-build`.

Release artifacts are published by the tag workflow after a clean package
installation test. Maintainer setup and the release checklist are documented
in [docs/releasing.md](docs/releasing.md).

To rewrite Mojo sources with the canonical formatter, run `pixi run format`.
To regenerate the committed fixture corpus, run `pixi run fixtures`.

## License

Licensed under the Apache License, Version 2.0. See [LICENSE](LICENSE).
