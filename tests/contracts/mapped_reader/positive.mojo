"""Positive mapped-reader ownership contract."""

from safetensors import MappedSafeTensorFile, map_safetensors


def main() raises:
    var archive: MappedSafeTensorFile = map_safetensors(
        "fixtures/valid/reference_f32.safetensors"
    )
    var bytes = archive.tensor_bytes("weights")
    print(len(bytes), bytes[0])
