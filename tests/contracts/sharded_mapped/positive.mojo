"""Positive mapped-sharded ownership contract."""

from safetensors import MappedShardedSafeTensorArchive, map_safetensors_index


def main() raises:
    var archive: MappedShardedSafeTensorArchive = map_safetensors_index(
        "fixtures/sharded/valid/reference/model.safetensors.index.json"
    )
    var alpha = archive.tensor_bytes("alpha")
    var beta = archive.tensor_view[DType.float32]("beta")
    print(alpha[0], beta[0])
