"""Must fail because a borrowed typed view cannot escape its owner."""

from safetensors import map_safetensors


def escaped_values() raises -> Span[Float32, ImmStaticOrigin]:
    var archive = map_safetensors("fixtures/valid/reference_f32.safetensors")
    return archive.tensor_view[DType.float32]("weights")


def main() raises:
    print(len(escaped_values()))
