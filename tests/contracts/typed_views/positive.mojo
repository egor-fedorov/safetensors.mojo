"""Positive typed-view dtype and ownership contract."""

from safetensors import MappedSafeTensorFile, map_safetensors


def instantiate_every_supported_dtype(mapped: MappedSafeTensorFile) raises:
    _ = mapped.tensor_view[DType.uint8]("weights")
    _ = mapped.tensor_view[DType.int8]("weights")
    _ = mapped.tensor_view[DType.float8_e5m2]("weights")
    _ = mapped.tensor_view[DType.float8_e4m3fn]("weights")
    _ = mapped.tensor_view[DType.float8_e8m0fnu]("weights")
    _ = mapped.tensor_view[DType.float8_e4m3fnuz]("weights")
    _ = mapped.tensor_view[DType.float8_e5m2fnuz]("weights")
    _ = mapped.tensor_view[DType.int16]("weights")
    _ = mapped.tensor_view[DType.uint16]("weights")
    _ = mapped.tensor_view[DType.float16]("weights")
    _ = mapped.tensor_view[DType.bfloat16]("weights")
    _ = mapped.tensor_view[DType.int32]("weights")
    _ = mapped.tensor_view[DType.uint32]("weights")
    _ = mapped.tensor_view[DType.float32]("weights")
    _ = mapped.tensor_view[DType.float64]("weights")
    _ = mapped.tensor_view[DType.int64]("weights")
    _ = mapped.tensor_view[DType.uint64]("weights")


def main() raises:
    var archive: MappedSafeTensorFile = map_safetensors(
        "fixtures/valid/reference_f32.safetensors"
    )
    instantiate_every_supported_dtype(archive)
    var raw = archive.tensor_bytes("weights")
    var values = archive.tensor_view[DType.float32]("weights")
    print(len(raw), len(values), values[0])
