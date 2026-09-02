"""Descriptor-resolution security regression tests."""

from std.testing import TestSuite, assert_equal, assert_true

from safetensors import (
    SafeTensorErrorKind,
    open_safetensors_index,
    open_sharded_safetensors,
)
from safetensors.sharding.resolver import _validate_shard_basename


def _with_embedded_nul(path: String) raises -> String:
    var bytes = List[UInt8]()
    for byte in path.as_bytes():
        bytes.append(byte)
    bytes.append(0)
    for byte in ".ignored".as_bytes():
        bytes.append(byte)
    return String(from_utf8=Span(bytes))


def test_explicit_shard_path_rejects_embedded_nul() raises:
    var path = _with_embedded_nul(
        "fixtures/sharded/valid/single/model.safetensors"
    )
    var raised = False
    try:
        _ = open_sharded_safetensors([path])
    except error:
        raised = True
        assert_equal(error.kind, SafeTensorErrorKind.IO_ERROR)
    assert_true(raised)


def test_index_path_components_reject_embedded_nul() raises:
    for path in [
        _with_embedded_nul(
            "fixtures/sharded/valid/single/model.safetensors.index.json"
        ),
        _with_embedded_nul("fixtures/sharded/valid/single")
        + "/model.safetensors.index.json",
    ]:
        var raised = False
        try:
            _ = open_safetensors_index(path)
        except error:
            raised = True
            assert_equal(error.kind, SafeTensorErrorKind.IO_ERROR)
        assert_true(raised)


def test_index_shard_basename_rejects_unicode_control() raises:
    var bytes = List[UInt8]()
    for byte in "shard".as_bytes():
        bytes.append(byte)
    bytes.append(0xC2)
    bytes.append(0x85)
    for byte in ".safetensors".as_bytes():
        bytes.append(byte)
    var name = String(from_utf8=Span(bytes))

    var raised = False
    try:
        _validate_shard_basename(name)
    except error:
        raised = True
        assert_equal(error.kind, SafeTensorErrorKind.PATH_TRAVERSAL)
    assert_true(raised)


def test_index_shard_symlink_diagnostic_names_trusted_apis() raises:
    var raised = False
    try:
        _ = open_safetensors_index(
            "fixtures/sharded/security/symlink-shard/"
            "model.safetensors.index.json"
        )
    except error:
        raised = True
        assert_equal(error.kind, SafeTensorErrorKind.PATH_TRAVERSAL)
        assert_equal(
            error.message,
            (
                "index-controlled shard is a symbolic link; use "
                "open_sharded_safetensors or map_sharded_safetensors for "
                "caller-trusted shard paths"
            ),
        )
        assert_equal(
            error.diagnostic(),
            (
                "safetensors:PathTraversal: index-controlled shard is a "
                "symbolic link; use open_sharded_safetensors or "
                "map_sharded_safetensors for caller-trusted shard paths"
            ),
        )
    assert_true(raised)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
