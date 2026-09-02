"""Deterministic aggregate metadata traversal-order tests."""

from std.testing import TestSuite, assert_equal

from safetensors import (
    SafeDType,
    ShardedSafeTensorMetadata,
    ShardedTensorInfo,
)


def _info(name: String, shard: String) -> ShardedTensorInfo:
    return ShardedTensorInfo(
        name,
        SafeDType.U8,
        [UInt64(1)],
        0,
        1,
        1,
        8,
        1,
        shard,
    )


def test_global_and_shard_grouped_orders_are_independent() raises:
    var metadata = ShardedSafeTensorMetadata(
        [
            _info("aardvark", "shard-z.safetensors"),
            _info("zebra", "shard-a.safetensors"),
            _info("middle", "shard-z.safetensors"),
            _info("beta", "shard-a.safetensors"),
        ],
        ["shard-z.safetensors", "shard-a.safetensors"],
    )

    assert_equal(metadata.names(), ["aardvark", "beta", "middle", "zebra"])
    assert_equal(
        metadata.shard_grouped_names(),
        ["beta", "zebra", "aardvark", "middle"],
    )
    assert_equal(
        metadata.shard_names(),
        ["shard-a.safetensors", "shard-z.safetensors"],
    )


def test_shard_grouped_names_returns_an_independent_copy() raises:
    var metadata = ShardedSafeTensorMetadata(
        [_info("alpha", "shard.safetensors")],
        ["shard.safetensors"],
    )
    var names = metadata.shard_grouped_names()
    names[0] = "changed"
    assert_equal(metadata.shard_grouped_names(), ["alpha"])


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
