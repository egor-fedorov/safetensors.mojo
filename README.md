# safetensors.mojo

`safetensors.mojo` is an early, pure-Mojo implementation of the core
Safetensors file format. The current milestone parses and validates metadata
from an in-memory byte buffer; it does not load tensor values into a tensor
runtime.

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

The format core provides:

- the independent `SafeDType` wire-format model;
- raw and validated metadata structures;
- decoding of the 8-byte little-endian header length;
- strict UTF-8 and Safetensors JSON header parsing;
- decoded-key duplicate detection;
- exact unsigned integer parsing without a floating-point intermediate;
- checked shape, bit-length, byte-length, offset, and full-coverage
  validation; and
- valid and malformed compatibility fixtures.

Unknown tensor descriptor fields are rejected in this strict initial
implementation. See
[ADR-001](docs/decisions/001-json-parser.md) for the parser decision.
Future work is tracked in [ROADMAP.md](ROADMAP.md).

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
safetensors-mojo = "==0.1.0"
```

The installed Mojo package is imported as `safetensors`.

Pass a caller-owned byte buffer containing a complete `.safetensors` file:

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

The parser reads the length prefix and header only. It validates ranges against
the remaining data-buffer length but neither copies nor interprets tensor data.
Wire data is defined as packed C-order and little-endian.

Validated metadata accessors return copies. Mojo 1.0 does not enforce field
visibility, so underscore-prefixed fields and direct `SafeTensorMetadata`
construction are implementation details; mutating them is unsupported and can
invalidate the validated-state contract.

## Deliberate limitations

This milestone does not implement file readers, memory mapping, tensor data
views, writers, slicing, sharding, MAX adapters, or other tensor-runtime
adapters. The parser validates data ranges but does not inspect tensor values.

Safetensors prevents arbitrary code execution through its data format, but it
does not provide authenticity, integrity, signatures, encryption, or protection
against an untrusted file being replaced while another component uses it.

## Development

The supported toolchain is Mojo 1.0.0 on Linux x86-64. Pixi installs the exact
compiler version and the Python-only development dependencies used to generate
reference fixtures. The generated `.mojoc` package is compiler-version-specific
and must be consumed with Mojo 1.0.0.

```text
pixi install
pixi run check
pixi run all
```

`pixi run check` verifies formatting, compiles the importable package, runs the
tests, and checks that fixtures are reproducible. `pixi run all` additionally
builds the `safetensors-mojo` Conda package. Individual tasks include
`compile`, `test`, `format-check`, `fixtures-check`, and `package-build`.

Release artifacts are published by the tag workflow after a clean package
installation test. Maintainer setup and the release checklist are documented
in [docs/releasing.md](docs/releasing.md).

To rewrite Mojo sources with the canonical formatter, run `pixi run format`.
To regenerate the committed fixture corpus, run `pixi run fixtures`.

## License

Licensed under the Apache License, Version 2.0. See [LICENSE](LICENSE).
