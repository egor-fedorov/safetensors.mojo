"""Must fail because a mapped sharded archive cannot be copied."""

from safetensors import map_safetensors_index


def main() raises:
    var archive = map_safetensors_index(
        "fixtures/sharded/valid/multiple/model.safetensors.index.json"
    )
    var duplicate = archive.copy()
    print(len(duplicate.tensor_bytes("alpha")))
