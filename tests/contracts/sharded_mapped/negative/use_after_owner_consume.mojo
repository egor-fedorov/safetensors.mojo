"""Must fail because a sharded view cannot survive owner consumption."""

from safetensors import MappedShardedSafeTensorArchive, map_safetensors_index


def consume(var archive: MappedShardedSafeTensorArchive):
    pass


def main() raises:
    var archive = map_safetensors_index(
        "fixtures/sharded/valid/multiple/model.safetensors.index.json"
    )
    var bytes = archive.tensor_bytes("alpha")
    consume(archive^)
    print(bytes[0])
