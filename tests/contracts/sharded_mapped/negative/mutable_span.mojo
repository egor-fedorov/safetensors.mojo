"""Must fail because sharded mapped byte views are immutable."""

from safetensors import map_safetensors_index


def main() raises:
    var archive = map_safetensors_index(
        "fixtures/sharded/valid/multiple/model.safetensors.index.json"
    )
    var bytes = archive.tensor_bytes("alpha")
    bytes[0] = 0
