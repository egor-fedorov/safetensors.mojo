"""Must fail because mapped typed views are immutable."""

from safetensors import map_safetensors


def main() raises:
    var archive = map_safetensors("fixtures/valid/reference_f32.safetensors")
    var values = archive.tensor_view[DType.float32]("weights")
    values[0] = 0.0
