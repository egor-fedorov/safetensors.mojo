"""Must fail because mapped byte views are immutable."""

from safetensors import map_safetensors


def main() raises:
    var archive = map_safetensors("fixtures/valid/reference_f32.safetensors")
    var bytes = archive.tensor_bytes("weights")
    bytes[0] = 0
