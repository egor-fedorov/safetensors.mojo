from safetensors import map_safetensors


def main() raises:
    var archive = map_safetensors("fixtures/valid/reference_f32.safetensors")
    var duplicate = archive.copy()
    print(len(duplicate.tensor_bytes("weights")))
