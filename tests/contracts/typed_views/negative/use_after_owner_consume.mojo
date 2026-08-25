"""Must fail because a typed view cannot outlive a consumed owner."""

from safetensors import MappedSafeTensorFile, map_safetensors


def consume(var archive: MappedSafeTensorFile):
    pass


def main() raises:
    var archive = map_safetensors("fixtures/valid/reference_f32.safetensors")
    var values = archive.tensor_view[DType.float32]("weights")
    consume(archive^)
    print(values[0])
