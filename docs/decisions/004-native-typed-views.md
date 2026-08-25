# ADR-004: Expose Exact Aligned Native Scalar Views

- Status: Accepted
- Date: 2026-08-25

## Context

`MappedSafeTensorFile.tensor_bytes()` returns immutable, origin-bound raw bytes
without copying tensor payloads. Callers can inspect those bytes safely with
respect to the mapping owner's Mojo origin, but interpreting them as native
numeric values still requires an unsafe pointer cast outside the package.

A typed zero-copy accessor must preserve four independent invariants:

1. the requested Mojo scalar encoding must exactly match the tensor's
   `SafeDType`;
2. every file-derived size, count, and offset must remain checked before pointer
   arithmetic or conversion to the native `Int` width;
3. a non-empty typed pointer must satisfy the native scalar's actual alignment;
   and
4. multi-byte little-endian wire values must not be exposed as native values on
   a big-endian host.

These are accessor constraints, not Safetensors format-validity constraints. A
valid archive may place a tensor at an address unsuitable for a native typed
view, and raw byte access must continue to work for that archive.

The package is pinned to Mojo 1.0.0 and currently distributed only for
`linux-64`. The decision nevertheless makes endianness explicit so a future
platform expansion cannot silently reinterpret little-endian wire bytes using
the wrong native byte order.

## Candidate check

The following designs were evaluated:

1. Keep only `Span[UInt8]`. This avoids unsafe code in the package but requires
   each caller to reproduce dtype, range, alignment, endianness, and origin
   handling.
2. Decode into an owned `List[T]`. This can repair alignment and byte order, but
   it copies the payload and is a different operation from the v0.3 borrowed
   zero-copy milestone.
3. Return a generic `Span[T]` for any trivial type. Native size and alignment
   alone do not prove that an arbitrary type has a Safetensors wire encoding.
4. Add one method per scalar type. This can be safe, but duplicates one contract
   across many public names and does not compose well with generic callers.
5. Add one `DType`-parameterized method with an exact internal whitelist. The
   caller chooses a compile-time native scalar representation, while the method
   compares it with runtime metadata before casting.

Mojo 1.0.0 compiler probes with warnings treated as errors established that:

- an inferred owner origin and explicit `DType` parameter produce an immutable
  `Span[Scalar[dtype], origin]`;
- `Span.unsafe_ptr()` followed by `Pointer.unsafe_bitcast[Scalar[dtype]]()`
  retains the span's origin;
- `size_of` and `align_of` from `std.sys` report the native scalar layout;
- `is_little_endian` from `std.sys` exposes the target byte order;
- typed-span mutation, escape to `ImmStaticOrigin`, and use after consuming the
  owner are rejected by the compiler; and
- a runtime metadata dtype cannot be used as a parameter value, so the caller's
  requested dtype must be compile-time while the exact-match check is runtime.

Probes also showed why predicates are insufficient. Mojo integral types include
widths that have no Safetensors encoding, Safetensors `F4` packs two elements in
one byte even though a Mojo float4 scalar occupies one addressable byte, and
`Scalar[DType.bool]` interprets arbitrary bytes by their low bit rather than
preserving a validated Safetensors boolean representation.

## Decision

`MappedSafeTensorFile` gains one additive public method:

```mojo
def tensor_view[
    origin: ImmOrigin,
    //,
    dtype: DType,
](
    ref[origin] self,
    name: String,
) raises SafeTensorError -> Span[Scalar[dtype], origin]: ...
```

The signature above defines the public API shape. Production modules contain a
complete compiled implementation, not a placeholder. A caller selects a native
scalar at compile time, for example:

```mojo
var values = mapped.tensor_view[DType.float32]("weights")
```

The returned view is flat. Its logical shape remains available through
validated metadata; the accessor does not create a multidimensional container
or a tensor-runtime value.

### Exact dtype whitelist

The initial native mappings are:

| Safetensors dtype | Mojo dtype |
| --- | --- |
| `U8` | `DType.uint8` |
| `I8` | `DType.int8` |
| `F8_E5M2` | `DType.float8_e5m2` |
| `F8_E4M3` | `DType.float8_e4m3fn` |
| `F8_E8M0` | `DType.float8_e8m0fnu` |
| `F8_E4M3FNUZ` | `DType.float8_e4m3fnuz` |
| `F8_E5M2FNUZ` | `DType.float8_e5m2fnuz` |
| `I16` | `DType.int16` |
| `U16` | `DType.uint16` |
| `F16` | `DType.float16` |
| `BF16` | `DType.bfloat16` |
| `I32` | `DType.int32` |
| `U32` | `DType.uint32` |
| `F32` | `DType.float32` |
| `F64` | `DType.float64` |
| `I64` | `DType.int64` |
| `U64` | `DType.uint64` |

The five float8 mappings use the corresponding one-byte Mojo 1.0.0 encodings.
They require no byte-order conversion because each element occupies one byte.

Typed views do not initially support:

- `BOOL`, because format validation does not scan payloads for canonical zero
  and one bytes and a Mojo bool reinterpretation would not preserve arbitrary
  input-byte semantics;
- `F4`, `F6_E2M3`, or `F6_E3M2`, because their packed element boundaries cannot
  be represented by ordinary scalar pointer increments; or
- `C64`, because Mojo 1.0.0 has no corresponding `DType` scalar and this project
  will not promise the ABI of a separate complex struct.

An unsupported requested Mojo specialization raises `UnsupportedDType`. A
recognized but unsupported tensor dtype also raises `UnsupportedDType`. When
both sides are supported but differ, including equal-width pairs such as `F32`
and `U32`, the accessor raises the new `DTypeMismatch` kind. The new kind is
appended to the stable error values so existing numeric values do not change.

### Validation and cast order

The accessor follows this order:

1. Look up the validated `TensorInfo` and resolve the requested dtype through
   the exact whitelist.
2. Call `tensor_bytes()` to reuse its checked absolute-offset arithmetic,
   mapping bounds, retained-file length check, and origin-bound pointer.
3. Multiply the file-derived element count by native `size_of` through checked
   `UInt64` arithmetic and require exact agreement with both validated and raw
   byte lengths.
4. Convert the file-derived element count to `Int` only through the checked
   conversion helper.
5. For a non-empty multi-byte tensor, require a little-endian host.
6. For a non-empty tensor, require the actual raw pointer address to be divisible
   by `align_of[Scalar[dtype]]()`. Checking only a relative file offset is not
   the public safety contract, even though a whole-file mapping is page-aligned
   on the current platform.
7. Only after every check, reinterpret the byte pointer with `unsafe_bitcast`
   and construct the immutable typed span with the checked element count.

A zero-element tensor still requires a supported exact dtype. It skips
endianness and tensor-address alignment checks because no element can be
dereferenced. `tensor_bytes()` already represents an empty tensor with the
mapping's aligned base pointer, which can be safely recast into a zero-length
origin-bound span.

The first failing boundary determines the error. Missing names remain
`TensorNotFound`; raw range inconsistencies remain `InvalidOffsets`; observed
file-length changes remain `IoError`; arithmetic failures remain
`ValidationOverflow`; defensive native-size disagreement is
`InvalidTensorSize`; incompatible byte order is `UnsupportedEndianness`; and an
unaligned non-empty address is `MisalignedTensor`. `MisalignedSlice` remains
reserved because slicing is outside this decision.

### Ownership and mutation

The typed span has the same origin as the immutable borrow of
`MappedSafeTensorFile`. It cannot outlive, be used after consuming, or provide
mutable access to the mapping owner. Multiple raw and typed immutable views may
coexist.

The external-mutation contract from ADR-003 is unchanged. A typed view is not a
snapshot: same-length changes may become visible, and later access after
external truncation may terminate the process with `SIGBUS`.

## Verification requirements

Implementation must include:

- runtime value tests for aligned integer, floating-point, and exact float8
  wire views;
- exact dtype mismatch and unsupported wire-dtype tests;
- two-, four-, and eight-byte actual-address misalignment tests that also prove
  raw byte access remains available;
- aligned and logically misaligned zero-element tests;
- a testable forced-big-endian policy branch plus the real host check;
- checked size and native element-count boundary tests;
- positive compilation of every whitelisted specialization;
- negative compiler contracts for mutation, owner escape, and use after owner
  consumption;
- deterministic fixture generation for an aligned scalar and float8 values;
- clean-installed package smoke coverage through the root public API.

Tests must not dereference a mapping after truncating its backing file. The
existing ADR-003 `SIGBUS` restriction applies equally to typed spans.

## Consequences

- Supported mapped tensors can be read as native scalars without payload copies
  or caller-owned unsafe casts.
- Dynamic code must still branch to a compile-time dtype specialization after
  inspecting metadata.
- Valid unaligned archives retain byte access but intentionally do not gain a
  typed view.
- Big-endian hosts can use non-empty one-byte typed views; multi-byte typed
  views require a future copied byte-swapping API.
- Packed values, unchecked booleans, and complex values remain available as raw
  bytes without overstating native representation guarantees.
- Adding another mapping later requires an explicit ABI and wire-encoding
  review rather than merely matching byte widths.

## References

- [ADR-003: Own Read-Only Mappings and Return Origin-Bound Byte Views](003-memory-mapped-reader.md)
- [Mojo 1.0.0 `DType`](https://mojolang.org/docs/types/dtype/)
- [Mojo 1.0.0 `Pointer`](https://mojolang.org/docs/std/memory/pointer/Pointer/)
- [Mojo 1.0.0 `Span`](https://mojolang.org/docs/std/collections/span/Span/)
- [Mojo 1.0.0 system information](https://mojolang.org/docs/std/sys/info/)
- [Safetensors 0.8.0 format](https://github.com/huggingface/safetensors/blob/v0.8.0/README.md#format)
