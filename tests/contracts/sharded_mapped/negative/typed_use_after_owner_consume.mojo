"""Must fail because a typed sharded view cannot outlive its owner."""

from safetensors import MappedShardedSafeTensorArchive, map_safetensors_index


def consume(var archive: MappedShardedSafeTensorArchive):
    pass


def main() raises:
    var archive = map_safetensors_index(
        "fixtures/sharded/valid/reference/model.safetensors.index.json"
    )
    var values = archive.tensor_view[DType.float32]("beta")
    consume(archive^)
    print(values[0])
