"""Survival harness for hostile Safetensors shard-index documents.

A dedicated canonical index first verifies both public entry points. Every
numbered hostile case then sits beside valid local shards and is exercised
through those same APIs. Accepted tensors are fully dereferenced so mapping and
buffered-read paths cannot be optimized away. For numbered cases, rejection is
an equally valid outcome and the only assertion is process survival.

Usage:
    mojo run -I src tests/fuzz/index_fuzz_harness.mojo <corpus-directory>
"""

from std.pathlib import Path
from std.sys import argv, exit
from std.testing import assert_equal

from safetensors import (
    DEFAULT_MAX_INDEX_ENTRIES,
    DEFAULT_MAX_SHARDS,
    map_safetensors_index,
    open_safetensors_index,
)


comptime _VALID_INDEX_NAME = "valid.safetensors.index.json"
comptime _EXPECTED_BUFFERED_CHECKSUM = 560
comptime _EXPECTED_MAPPED_CHECKSUM = 568


def _case_name(index: Int) -> String:
    var digits = String(index)
    while digits.byte_length() < 6:
        digits = "0" + digits
    return "case" + digits + ".safetensors.index.json"


def _touch_mapped(
    path: String,
    max_index_entries: UInt64 = DEFAULT_MAX_INDEX_ENTRIES,
    max_shards: UInt64 = DEFAULT_MAX_SHARDS,
) raises -> Int:
    var checksum = 0
    var archive = map_safetensors_index(
        path,
        max_index_entries=max_index_entries,
        max_shards=max_shards,
    )
    var metadata = archive.metadata()
    for name in metadata.names():
        var raw = archive.tensor_bytes(name)
        for index in range(len(raw)):
            checksum += Int(raw[index])

        try:
            var unsigned_values = archive.tensor_view[DType.uint8](name)
            for index in range(len(unsigned_values)):
                checksum += Int(unsigned_values[index])
        except:
            pass

        try:
            var signed_values = archive.tensor_view[DType.int16](name)
            for index in range(len(signed_values)):
                checksum += Int(signed_values[index] != 0)
        except:
            pass

        try:
            var float_values = archive.tensor_view[DType.float32](name)
            for index in range(len(float_values)):
                checksum += Int(float_values[index] != 0)
        except:
            pass
    return checksum


def _touch_buffered(
    path: String,
    max_index_entries: UInt64 = DEFAULT_MAX_INDEX_ENTRIES,
    max_shards: UInt64 = DEFAULT_MAX_SHARDS,
) raises -> Int:
    var checksum = 0
    var reader = open_safetensors_index(
        path,
        max_index_entries=max_index_entries,
        max_shards=max_shards,
    )
    var metadata = reader.metadata()
    for name in metadata.shard_grouped_names():
        var loaded = reader.load_tensor(name)
        for index in range(len(loaded)):
            checksum += Int(loaded[index])
    return checksum


def main() raises:
    var arguments = argv()
    if len(arguments) != 2:
        print("usage: index_fuzz_harness <corpus-directory>")
        exit(2)

    var corpus = Path(String(arguments[1]))
    var count = Int(corpus.joinpath("count.txt").read_text().strip())
    if count <= 0:
        print("error: index fuzz corpus must contain at least one case")
        exit(2)

    var valid_path = String(corpus.joinpath(_VALID_INDEX_NAME))
    assert_equal(_touch_buffered(valid_path), _EXPECTED_BUFFERED_CHECKSUM)
    assert_equal(_touch_mapped(valid_path), _EXPECTED_MAPPED_CHECKSUM)

    var buffered_ok = 0
    var mapped_ok = 0
    var entry_limited_buffered_ok = 0
    var entry_limited_mapped_ok = 0
    var shard_limited_buffered_ok = 0
    var shard_limited_mapped_ok = 0
    var checksum = 0
    for index in range(count):
        var path = String(corpus.joinpath(_case_name(index)))
        try:
            checksum += _touch_buffered(path)
            buffered_ok += 1
        except:
            pass

        try:
            checksum += _touch_mapped(path)
            mapped_ok += 1
        except:
            pass

        try:
            checksum += _touch_buffered(path, max_index_entries=1)
            entry_limited_buffered_ok += 1
        except:
            pass

        try:
            checksum += _touch_mapped(path, max_index_entries=1)
            entry_limited_mapped_ok += 1
        except:
            pass

        try:
            checksum += _touch_buffered(path, max_shards=1)
            shard_limited_buffered_ok += 1
        except:
            pass

        try:
            checksum += _touch_mapped(path, max_shards=1)
            shard_limited_mapped_ok += 1
        except:
            pass

    print("canonical smoke:    buffered and mapped accepted")
    print("cases:              ", count)
    print("buffered accepted:  ", buffered_ok)
    print("mapped accepted:    ", mapped_ok)
    print("entry-limit buffered:", entry_limited_buffered_ok)
    print("entry-limit mapped:  ", entry_limited_mapped_ok)
    print("shard-limit buffered:", shard_limited_buffered_ok)
    print("shard-limit mapped:  ", shard_limited_mapped_ok)
    print("touch checksum:     ", checksum)
    print("survived every index case without a panic, fault, or hang")
