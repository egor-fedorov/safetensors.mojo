"""Eager mapped sharded-reader integration tests."""

from std.os import remove
from std.pathlib import Path
from std.tempfile import TemporaryDirectory
from std.testing import TestSuite, assert_equal, assert_true

from safetensors import (
    SafeTensorErrorKind,
    map_safetensors_index,
    map_sharded_safetensors,
)


def _index(group: String, name: String) -> String:
    return (
        "fixtures/sharded/"
        + group
        + "/"
        + name
        + "/model.safetensors.index.json"
    )


def _proc_maps_contains(path: String) raises -> Bool:
    var contents = Path("/proc/self/maps").read_bytes()
    return path in String(from_utf8=Span(contents))


def _assert_shards_mapped(first: String, second: String) raises:
    var archive = map_sharded_safetensors([first, second])
    var alpha = archive.tensor_bytes("alpha")
    var beta = archive.tensor_bytes("beta")
    assert_true(_proc_maps_contains(first))
    assert_true(_proc_maps_contains(second))
    assert_equal(alpha[0], UInt8(1))
    assert_equal(beta[0], UInt8(0xFE))


def test_raw_views_from_different_shards_coexist() raises:
    var archive = map_safetensors_index(_index("valid", "multiple"))
    var alpha = archive.tensor_bytes("alpha")
    var beta = archive.tensor_bytes("beta")
    assert_equal(len(alpha), 3)
    assert_equal(len(beta), 4)
    assert_equal(alpha[0], UInt8(1))
    assert_equal(alpha[2], UInt8(3))
    assert_equal(beta[0], UInt8(0xFE))
    assert_equal(beta[3], UInt8(0x01))


def test_exact_typed_views_route_across_reference_shards() raises:
    var archive = map_safetensors_index(_index("valid", "reference"))
    var alpha = archive.tensor_view[DType.uint8]("alpha")
    var beta = archive.tensor_view[DType.float32]("beta")
    var gamma = archive.tensor_view[DType.uint16]("gamma")
    assert_equal(len(alpha), 4)
    assert_equal(alpha[3], UInt8(255))
    assert_equal(len(beta), 2)
    assert_equal(beta[0], Float32(1.5))
    assert_equal(beta[1], Float32(-2.25))
    assert_equal(len(gamma), 2)
    assert_equal(gamma[0], UInt16(0x1234))
    assert_equal(gamma[1], UInt16(0xABCD))


def test_explicit_mapping_accepts_cache_style_symlink() raises:
    var path = "fixtures/sharded/security/symlink-shard/shard.safetensors"
    var target = (
        "fixtures/sharded/security/symlink-shard/target/real.safetensors"
    )
    var archive = map_sharded_safetensors([path, target], max_shards=1)
    assert_equal(archive.metadata().shard_names(), [path])
    var bytes = archive.tensor_bytes("alpha")
    assert_equal(len(bytes), 3)
    assert_equal(bytes[0], UInt8(1))


def test_index_mapping_preserves_security_and_limits() raises:
    var raised = False
    try:
        _ = map_safetensors_index(_index("security", "symlink-shard"))
    except error:
        raised = True
        assert_equal(error.kind, SafeTensorErrorKind.PATH_TRAVERSAL)
    assert_true(raised)

    raised = False
    try:
        _ = map_safetensors_index(_index("valid", "multiple"), max_shards=1)
    except error:
        raised = True
        assert_equal(error.kind, SafeTensorErrorKind.SHARD_LIMIT_EXCEEDED)
    assert_true(raised)

    raised = False
    try:
        _ = map_safetensors_index(
            _index("valid", "multiple"), max_index_entries=1
        )
    except error:
        raised = True
        assert_equal(error.kind, SafeTensorErrorKind.INDEX_ENTRY_LIMIT_EXCEEDED)
    assert_true(raised)


def test_missing_tensor_and_dtype_mismatch_are_typed() raises:
    var archive = map_safetensors_index(_index("valid", "multiple"))
    var raised = False
    try:
        _ = archive.tensor_bytes("missing")
    except error:
        raised = True
        assert_equal(error.kind, SafeTensorErrorKind.TENSOR_NOT_FOUND)
    assert_true(raised)

    raised = False
    try:
        _ = archive.tensor_view[DType.float32]("alpha")
    except error:
        raised = True
        assert_equal(error.kind, SafeTensorErrorKind.DTYPE_MISMATCH)
    assert_true(raised)


def test_every_shard_mapping_is_released_at_scope_exit() raises:
    with TemporaryDirectory() as temporary:
        var source = "fixtures/sharded/valid/multiple/"
        var first = temporary + "/shard-a.safetensors"
        var second = temporary + "/shard-b.safetensors"
        Path(first).write_bytes(
            Path(source + "shard-a.safetensors").read_bytes()
        )
        Path(second).write_bytes(
            Path(source + "shard-b.safetensors").read_bytes()
        )
        assert_true(not _proc_maps_contains(first))
        assert_true(not _proc_maps_contains(second))
        _assert_shards_mapped(first, second)
        assert_true(not _proc_maps_contains(first))
        assert_true(not _proc_maps_contains(second))


def test_unlink_does_not_redirect_existing_shard_mapping() raises:
    with TemporaryDirectory() as temporary:
        var source = "fixtures/sharded/valid/multiple/"
        var index_path = temporary + "/model.safetensors.index.json"
        var first = temporary + "/shard-a.safetensors"
        var second = temporary + "/shard-b.safetensors"
        Path(index_path).write_bytes(
            Path(source + "model.safetensors.index.json").read_bytes()
        )
        Path(first).write_bytes(
            Path(source + "shard-a.safetensors").read_bytes()
        )
        Path(second).write_bytes(
            Path(source + "shard-b.safetensors").read_bytes()
        )
        var archive = map_safetensors_index(index_path)
        remove(first)
        var alpha = archive.tensor_bytes("alpha")
        var beta = archive.tensor_bytes("beta")
        assert_equal(alpha[0], UInt8(1))
        assert_equal(beta[0], UInt8(0xFE))


def test_growth_before_sharded_view_is_io_error() raises:
    with TemporaryDirectory() as temporary:
        var source = "fixtures/sharded/valid/multiple/"
        var first = temporary + "/shard-a.safetensors"
        var second = temporary + "/shard-b.safetensors"
        Path(first).write_bytes(
            Path(source + "shard-a.safetensors").read_bytes()
        )
        Path(second).write_bytes(
            Path(source + "shard-b.safetensors").read_bytes()
        )
        var archive = map_sharded_safetensors([first, second])
        var appender = open(second, "a")
        var extra = [UInt8(0)]
        appender.write_all(extra)
        appender.close()
        var raised = False
        try:
            _ = archive.tensor_bytes("beta")
        except error:
            raised = True
            assert_equal(error.kind, SafeTensorErrorKind.IO_ERROR)
        assert_true(raised)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
