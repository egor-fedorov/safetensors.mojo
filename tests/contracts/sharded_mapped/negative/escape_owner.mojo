"""Must fail because a sharded view cannot escape its aggregate owner."""

from safetensors import map_safetensors_index


def escaped_bytes() raises -> Span[UInt8, ImmStaticOrigin]:
    var archive = map_safetensors_index(
        "fixtures/sharded/valid/multiple/model.safetensors.index.json"
    )
    return archive.tensor_bytes("alpha")


def main() raises:
    print(len(escaped_bytes()))
