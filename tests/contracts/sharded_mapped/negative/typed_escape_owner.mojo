"""Must fail because a typed sharded view cannot escape its owner."""

from safetensors import map_safetensors_index


def escaped_values() raises -> Span[Float32, ImmStaticOrigin]:
    var archive = map_safetensors_index(
        "fixtures/sharded/valid/reference/model.safetensors.index.json"
    )
    return archive.tensor_view[DType.float32]("beta")


def main() raises:
    print(len(escaped_values()))
