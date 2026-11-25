"""Print the shape and first value of a named F32 tensor in a local checkpoint."""

from std.sys import argv

from safetensors import map_safetensors


def main() raises:
    var args = argv()
    if len(args) != 3:
        raise Error("usage: inspect_f32.mojo <checkpoint.safetensors> <tensor>")
    var archive = map_safetensors(String(args[1]))
    var name = String(args[2])
    var info = archive.metadata().info(name)
    var values = archive.tensor_view[DType.float32](name)
    print(name, info.dtype.wire_name(), info.shape)
    if len(values) > 0:
        print("first value:", values[0])
