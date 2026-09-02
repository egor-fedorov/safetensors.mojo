"""End-to-end API smoke test for an installed package."""

from std.os import remove
from std.testing import assert_equal, assert_true

from safetensors import (
    SafeDType,
    SafeTensorData,
    decode_header_length,
    map_safetensors,
    map_safetensors_index,
    open_safetensors,
    open_safetensors_index,
    parse_metadata,
    save_safetensors,
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
    var tensors: List[SafeTensorData] = [
        SafeTensorData("tensor", SafeDType.U8, [UInt64(2)], [UInt8(17), 29])
    ]
    var user_metadata = Dict[String, String]()
    user_metadata["producer"] = "package smoke test"
    save_safetensors(path, tensors, user_metadata)
    var reader = open_safetensors(path)
    assert_equal(reader.metadata().info("tensor").byte_length, UInt64(2))
    assert_equal(
        reader.metadata().metadata_value("producer").value(),
        "package smoke test",
    )
    assert_equal(reader.load_tensor("tensor"), [UInt8(17), 29])

    var mapped = map_safetensors(path)
    assert_equal(mapped.metadata().info("tensor").byte_length, UInt64(2))
    var view = mapped.tensor_bytes("tensor")
    assert_equal(len(view), 2)
    assert_equal(view[0], UInt8(17))
    assert_equal(view[1], UInt8(29))
    var typed = mapped.tensor_view[DType.uint8]("tensor")
    assert_equal(len(typed), 2)
    assert_equal(typed[0], UInt8(17))
    assert_equal(typed[1], UInt8(29))
    remove(path)

    var first_shard = "package-smoke-00001.safetensors"
    var second_shard = "package-smoke-00002.safetensors"
    var index_path = "package-smoke.safetensors.index.json"
    save_safetensors(
        first_shard,
        [SafeTensorData("left", SafeDType.U8, [UInt64(2)], [UInt8(3), 5])],
    )
    save_safetensors(
        second_shard,
        [
            SafeTensorData(
                "right",
                SafeDType.U16,
                [UInt64(1)],
                [UInt8(0x34), 0x12],
            )
        ],
    )
    var index_document = (
        '{"metadata":{"total_size":4},"weight_map":{'
        '"left":"package-smoke-00001.safetensors",'
        '"right":"package-smoke-00002.safetensors"}}'
    )
    var index_file = open(index_path, "w")
    index_file.write_all(index_document.as_bytes())
    index_file.close()

    var sharded = open_safetensors_index(index_path)
    assert_equal(sharded.metadata().names(), ["left", "right"])
    assert_equal(sharded.load_tensor("left"), [UInt8(3), 5])
    assert_equal(sharded.load_tensor("right"), [UInt8(0x34), 0x12])

    var mapped_sharded = map_safetensors_index(index_path)
    var right = mapped_sharded.tensor_view[DType.uint16]("right")
    assert_equal(len(right), 1)
    assert_equal(right[0], UInt16(0x1234))

    remove(index_path)
    remove(first_shard)
    remove(second_shard)

    print("safetensors-mojo package smoke test passed")
