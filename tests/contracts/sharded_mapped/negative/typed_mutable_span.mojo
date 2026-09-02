"""Must fail because typed sharded mapped views are immutable."""

from safetensors import map_safetensors_index


def main() raises:
    var archive = map_safetensors_index(
        "fixtures/sharded/valid/reference/model.safetensors.index.json"
    )
    var values = archive.tensor_view[DType.float32]("beta")
    values[0] = 0.0
