"""End-to-end API smoke test for an installed package."""

from std.os import remove
from std.pathlib import Path
from std.testing import assert_equal, assert_true

from safetensors import (
    SafeDType,
    decode_header_length,
    map_safetensors,
    open_safetensors,
    parse_metadata,
)


def main() raises:
    var header = '{"tensor":{"dtype":"U8","shape":[2],"data_offsets":[0,2]}}'
    var header_length = UInt64(header.byte_length())
    var contents = List[UInt8]()

    for index in range(8):
        contents.append(
            UInt8((header_length >> UInt64(index * 8)) & UInt64(0xFF))
        )
    for byte in header.as_bytes():
        contents.append(byte)
    contents.append(17)
    contents.append(29)

    assert_equal(decode_header_length(contents), header_length)
    var metadata = parse_metadata(contents)
    assert_equal(len(metadata), 1)
    assert_true(metadata.contains("tensor"))

    var tensor = metadata.info("tensor")
    assert_equal(tensor.dtype, SafeDType.U8)
    assert_equal(tensor.dtype.wire_name(), "U8")
    assert_equal(tensor.shape[0], UInt64(2))
    assert_equal(tensor.element_count, UInt64(2))
    assert_equal(tensor.byte_length, UInt64(2))

    var path = "package-smoke.safetensors"
    Path(path).write_bytes(contents)
    var reader = open_safetensors(path)
    assert_equal(reader.metadata().info("tensor").byte_length, UInt64(2))
    assert_equal(reader.load_tensor("tensor"), [UInt8(17), 29])

    var mapped = map_safetensors(path)
    assert_equal(mapped.metadata().info("tensor").byte_length, UInt64(2))
    var view = mapped.tensor_bytes("tensor")
    assert_equal(len(view), 2)
    assert_equal(view[0], UInt8(17))
    assert_equal(view[1], UInt8(29))
    remove(path)

    print("safetensors-mojo package smoke test passed")
