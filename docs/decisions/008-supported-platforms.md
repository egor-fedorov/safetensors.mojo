# ADR-008: Support Three Native Mojo Platforms

- Status: Accepted
- Date: 2026-09-02
- Current architecture: [Architecture](../architecture/README.md)

## Context

Versions through 0.6 distributed one `linux-64` package. The format core is
operating-system-independent, but memory mapping, atomic replacement, and the
index-controlled shard resolver were implemented directly against Linux libc
constants and facilities. Version 0.7 adds `linux-aarch64` and `osx-arm64`
without changing the public Safetensors API or weakening its filesystem trust
boundaries.

Mojo `.mojoc` packages are compiler- and target-specific. Packaging also runs
compiled programs to verify a clean installation, so resolving dependencies
for another target is not enough to establish that an artifact works. The
release pipeline needs a native build and verification policy as well as
portable source code.

Darwin does not provide Linux `O_PATH` or `statx`. A portable resolver cannot
pretend that their flag values or structures are shared, but it must retain the
security properties of ADR-007: an index-controlled name is one basename, its
final symlink is rejected, a FIFO cannot block the process, the opened object
is regular, and a pathname replacement between classification and open is
detected.

## Decision

### Supported targets

The package supports exactly these Pixi/Conda platforms with Mojo 1.0.0:

| Platform | Native host baseline |
| --- | --- |
| `linux-64` | Linux, glibc 2.34 or later, and an x86-64-v3 CPU |
| `linux-aarch64` | Linux, glibc 2.34 or later, and an ARM64 Neoverse N1-class or newer CPU |
| `osx-arm64` | Apple silicon running macOS 15 or later, with Xcode or Xcode Command Line Tools 16 or later |

Linux development also requires a C compiler usable as the linker. The project
does not publish `osx-64`: Mojo 1.0 supports macOS on Apple silicon only.
Windows, including a native Windows package, is outside this target matrix.

Each package must be built, tested, and clean-install-smoked on a native runner
for that target. Cross-building one package platform from another host is not
a supported release path. Publication is an aggregate step and may proceed
only after every platform artifact has completed its native checks.

### Compile-time platform boundary

The runtime-independent `safetensors.format` layer remains identical across
targets. Internal I/O and sharding modules select operating-system constants
and implementations at compile time through Mojo's `CompilationTarget` and
`platform_map` facilities. Unsupported targets fail at compilation rather than
silently selecting Linux values.

This platform boundary is internal. The root `safetensors` facade, error kinds,
validated metadata, buffered-reader behavior, mapped-view ownership, and
writer signatures do not vary by operating system.

### Memory mapping

Linux on both architectures and Darwin use the POSIX `mmap` and `munmap`
interface with a whole-file `PROT_READ | MAP_PRIVATE` mapping. The mapping owner
and every returned span keep the origin contract from ADR-003 and ADR-004.
Platform expansion does not change the external-mutation warning: an in-place
write can become visible, and dereferencing after external truncation can
terminate the process with `SIGBUS`.

### Atomic writer

The sibling-file transaction uses `getentropy` for its small random nonce on
Linux and Darwin. It creates the temporary file with the platform-specific
values for `O_WRONLY | O_CREAT | O_EXCL | O_CLOEXEC`, requests mode `0600`,
writes and closes the complete file, and calls `rename` as the only commit
point. Both source and destination are in the same directory.

The guarantees from ADR-005 are unchanged: preflight completes before
filesystem mutation, exclusive creation does not overwrite an attacker-created
temporary entry, failure before rename leaves the destination unchanged, and
a successful rename exposes one complete inode. This remains an atomic
namespace-visibility guarantee, not a crash-durability guarantee; neither the
file nor its parent directory is synchronized with `fsync`.

### Descriptor-relative shard resolution

Both operating-system implementations retain the lexical parent directory
descriptor for an index and resolve every validated `weight_map` basename
relative to it. Trusted explicit paths may follow symlinks; index-controlled
shard entries may not. Device and inode identity are compared across opens and
strengthened with birth time when the operating system reports it.

On Linux, the resolver:

1. opens the final directory entry with `O_PATH | O_NOFOLLOW | O_CLOEXEC`;
2. classifies that pinned descriptor with `statx(AT_EMPTY_PATH)`;
3. rejects symlinks and non-regular objects;
4. opens the entry again with `O_NONBLOCK | O_NOFOLLOW | O_CLOEXEC`; and
5. uses descriptor-only `statx` to require the same regular-file identity.

On Darwin, the resolver:

1. classifies the final directory entry with
   `fstatat(AT_SYMLINK_NOFOLLOW)`;
2. rejects symlinks and non-regular objects;
3. opens the entry with `O_NONBLOCK | O_NOFOLLOW | O_CLOEXEC`;
4. classifies the returned descriptor with `fstat`; and
5. requires the preflight and opened regular-file identities to agree.

The Darwin preflight is pathname-based rather than a pinned descriptor. The
second open still rejects a replacement symlink, `O_NONBLOCK` prevents a FIFO
from blocking, and the post-open type and identity comparison rejects a swap
to another regular file. Hard links remain outside the path-traversal defense
on both operating systems, as documented by ADR-007.

### Verification policy

The common format, reader, mapping, typed-view, writer, sharding, fixture, and
package smoke contracts apply to all three native jobs. Tests that inspect
Linux-only process state such as `/proc/self/maps` or `/proc/self/fd` must be
conditional diagnostics rather than the only assertion of public behavior.
Platform-specific resolver tests must exercise symlink rejection,
non-regular-file rejection without blocking, and replacement detection through
the implementation selected for that host.

Deterministic fuzzing remains architecture-independent at the format boundary.
It may run in a dedicated job rather than multiplying the complete generated
corpus across every release build; native compile, unit, integration, contract,
and clean-install checks remain mandatory for each published target.

## Consequences

- One source package and public API cover x86-64 Linux, ARM64 Linux, and Apple
  silicon macOS while preserving native artifacts.
- Operating-system constants, stat structures, and resolver mechanics stay
  isolated from format parsing and validation.
- The Linux resolver keeps its descriptor-pinned `O_PATH` preflight; Darwin
  provides equivalent public guarantees through `fstatat`, nonblocking
  `openat`, and `fstat` identity comparison.
- Native release jobs cost more runner time, but a successful aggregate release
  represents an executable test on every advertised package target.
- Adding another operating system or architecture requires explicit source,
  native CI, clean-install, and release support; adding a name to the Pixi
  platform list alone is insufficient.

## Out of scope

This decision does not add `osx-64`, native Windows, cross-compilation,
cross-platform file locking, crash durability, immutable file snapshots,
remote Hub access, or tensor-runtime adapters.

## References

- [ADR-003: Own Read-Only Mappings and Return Origin-Bound Byte Views](003-memory-mapped-reader.md)
- [ADR-004: Expose Exact Aligned Native Scalar Views](004-native-typed-views.md)
- [ADR-005: Write Canonical Files Through an Atomic Sibling Transaction](005-atomic-writer.md)
- [ADR-007: Separate Trusted Shard Lists from Untrusted Index Resolution](007-sharded-readers.md)
- [Mojo 1.0 system requirements](https://mojolang.org/docs/requirements/)
- [Mojo 1.0 `CompilationTarget` source](https://github.com/modular/modular/blob/mojo/v1.0.0/mojo/stdlib/std/sys/info.mojo)
