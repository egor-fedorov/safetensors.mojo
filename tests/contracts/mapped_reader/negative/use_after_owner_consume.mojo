"""Must fail because a view cannot outlive a consumed owner."""

from safetensors import MappedSafeTensorFile, map_safetensors


def consume(var archive: MappedSafeTensorFile):
    pass


def main() raises:
    var archive = map_safetensors("fixtures/valid/reference_f32.safetensors")
    var bytes = archive.tensor_bytes("weights")
    consume(archive^)
    print(bytes[0])
