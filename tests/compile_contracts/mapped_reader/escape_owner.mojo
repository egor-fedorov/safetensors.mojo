from safetensors import map_safetensors


def escaped_bytes() raises -> Span[UInt8, ImmStaticOrigin]:
    var archive = map_safetensors("fixtures/valid/reference_f32.safetensors")
    return archive.tensor_bytes("weights")


def main() raises:
    print(len(escaped_bytes()))
