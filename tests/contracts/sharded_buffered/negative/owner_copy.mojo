"""Must fail because a buffered sharded reader cannot be copied."""

from safetensors import open_safetensors_index


def main() raises:
    var reader = open_safetensors_index(
        "fixtures/sharded/valid/multiple/model.safetensors.index.json"
    )
    var duplicate = reader.copy()
    print(duplicate.metadata().len())
