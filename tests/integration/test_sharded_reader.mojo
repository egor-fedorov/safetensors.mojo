"""Buffered sharded-reader integration tests."""

from std.ffi import c_int, external_call
from std.os import listdir, makedirs, remove, stat
from std.pathlib import Path
from std.tempfile import TemporaryDirectory
from std.testing import TestSuite, assert_equal, assert_false, assert_true

from safetensors import (
    SafeDType,
    SafeTensorErrorKind,
    open_safetensors_index,
    open_sharded_safetensors,
)


def _index(group: String, name: String) -> String:
    return (
        "fixtures/sharded/"
        + group
        + "/"
        + name
        + "/model.safetensors.index.json"
    )


def _assert_index_error(
    group: String,
    name: String,
    expected: SafeTensorErrorKind,
) raises:
    var raised = False
    try:
        _ = open_safetensors_index(_index(group, name))
    except error:
        raised = True
        assert_equal(error.kind, expected)
    assert_true(raised)


def _rename(source: String, destination: String) raises:
    var owned_source = source
    var owned_destination = destination
    var result = external_call["rename", c_int](
        owned_source.as_c_string_slice(),
        owned_destination.as_c_string_slice(),
    )
    assert_equal(result, 0)


def _replace_ascii_bytes(
    mut contents: List[UInt8], old: String, replacement: String
) raises:
    var old_bytes = old.as_bytes()
    var replacement_bytes = replacement.as_bytes()
    assert_equal(len(old_bytes), len(replacement_bytes))
    var found = -1
    for start in range(len(contents) - len(old_bytes) + 1):
        var matches = True
        for offset in range(len(old_bytes)):
            if contents[start + offset] != old_bytes[offset]:
                matches = False
                break
        if matches:
            assert_equal(found, -1)
            found = start
    assert_true(found >= 0)
    for offset in range(len(replacement_bytes)):
        contents[found + offset] = replacement_bytes[offset]


def _bind_unix_socket(path: String) raises -> FileHandle:
    var descriptor = external_call["socket", c_int](
        c_int(1), c_int(1 | 0x80000), c_int(0)
    )
    assert_true(descriptor >= 0)
    var socket_file = FileHandle()
    socket_file.handle = Int(descriptor)

    # Linux sockaddr_un stores a 2-byte family followed by a 108-byte path.
    var address = List[UInt8](length=110, fill=0)
    address[0] = 1
    address[1] = 0
    var path_bytes = path.as_bytes()
    assert_true(len(path_bytes) <= 107)
    for index in range(len(path_bytes)):
        address[index + 2] = path_bytes[index]
    var result = external_call["bind", c_int](
        c_int(descriptor),
        address.unsafe_ptr(),
        c_int(len(path_bytes) + 3),
    )
    assert_equal(result, 0)
    return socket_file^


def _copy_multiple_archive(temporary: String) raises:
    var source = "fixtures/sharded/valid/multiple/"
    Path(temporary + "/model.safetensors.index.json").write_bytes(
        Path(source + "model.safetensors.index.json").read_bytes()
    )
    Path(temporary + "/shard-a.safetensors").write_bytes(
        Path(source + "shard-a.safetensors").read_bytes()
    )
    Path(temporary + "/shard-b.safetensors").write_bytes(
        Path(source + "shard-b.safetensors").read_bytes()
    )


def test_index_metadata_and_cross_shard_reads() raises:
    var reader = open_safetensors_index(_index("valid", "multiple"))
    var metadata = reader.metadata()
    assert_equal(metadata.len(), 3)
    assert_equal(metadata.names(), ["alpha", "beta", "empty"])
    assert_equal(
        metadata.shard_names(),
        ["shard-a.safetensors", "shard-b.safetensors"],
    )
    assert_equal(metadata.total_size(), UInt64(7))
    assert_true(metadata.declared_total_size())
    assert_equal(metadata.declared_total_size().value(), UInt64(7))
    assert_equal(metadata.info("alpha").dtype, SafeDType.U8)
    assert_equal(metadata.info("alpha").shard, "shard-a.safetensors")
    assert_equal(metadata.info("beta").shard, "shard-b.safetensors")

    assert_equal(reader.load_tensor("alpha"), [UInt8(1), 2, 3])
    assert_equal(reader.load_tensor("beta"), [UInt8(0xFE), 0xFF, 0x2C, 0x01])
    assert_equal(reader.load_tensor("alpha"), [UInt8(1), 2, 3])
    assert_equal(len(reader.load_tensor("empty")), 0)


def test_index_symlink_is_trusted() raises:
    var reader = open_safetensors_index(_index("valid", "index-symlink"))
    assert_equal(reader.load_tensor("alpha"), [UInt8(1), 2, 3])


def test_explicit_paths_follow_symlinks_and_deduplicate_identity() raises:
    var symlink = "fixtures/sharded/security/symlink-shard/shard.safetensors"
    var target = (
        "fixtures/sharded/security/symlink-shard/target/real.safetensors"
    )
    var reader = open_sharded_safetensors([symlink, target], max_shards=1)
    assert_equal(reader.metadata().shard_names(), [symlink])
    assert_equal(reader.load_tensor("alpha"), [UInt8(1), 2, 3])


def test_explicit_paths_route_across_unique_shards() raises:
    var root = "fixtures/sharded/valid/multiple/"
    var reader = open_sharded_safetensors(
        [root + "shard-a.safetensors", root + "shard-b.safetensors"]
    )
    assert_equal(reader.load_tensor("alpha"), [UInt8(1), 2, 3])
    assert_equal(reader.load_tensor("beta"), [UInt8(0xFE), 0xFF, 0x2C, 0x01])


def test_explicit_paths_allow_empty_archive_and_reject_duplicate_tensors() raises:
    var empty = open_sharded_safetensors(
        ["fixtures/valid/empty_archive.safetensors"]
    )
    assert_true(empty.metadata().is_empty())
    assert_equal(empty.metadata().total_size(), UInt64(0))

    var duplicate_root = (
        "fixtures/sharded/malformed/duplicate-tensor-across-shards/"
    )
    var raised = False
    try:
        _ = open_sharded_safetensors(
            [
                duplicate_root + "shard-a.safetensors",
                duplicate_root + "shard-b.safetensors",
            ]
        )
    except error:
        raised = True
        assert_equal(error.kind, SafeTensorErrorKind.SHARD_MISMATCH)
    assert_true(raised)


def test_index_rejects_untrusted_path_spellings_and_shard_symlink() raises:
    for name in [
        "empty-filename",
        "dot",
        "dot-dot",
        "parent-traversal",
        "nested-path",
        "backslash-path",
        "absolute-posix",
        "windows-drive",
        "windows-unc",
        "url",
        "colon",
        "nul",
        "control",
        "wrong-suffix",
        "symlink-shard",
    ]:
        _assert_index_error(
            "security", name, SafeTensorErrorKind.PATH_TRAVERSAL
        )


def test_index_aggregate_failures_are_typed() raises:
    for name in [
        "wrong-route",
        "omitted-tensor",
        "ghost-tensor",
        "duplicate-tensor-across-shards",
    ]:
        _assert_index_error(
            "malformed", name, SafeTensorErrorKind.SHARD_MISMATCH
        )
    _assert_index_error(
        "malformed",
        "total-size-mismatch",
        SafeTensorErrorKind.TOTAL_SIZE_MISMATCH,
    )
    _assert_index_error(
        "malformed", "missing-shard", SafeTensorErrorKind.IO_ERROR
    )
    _assert_index_error(
        "malformed", "malformed-shard", SafeTensorErrorKind.HEADER_TOO_SMALL
    )


def test_index_parser_fixture_failures_are_typed() raises:
    _assert_index_error(
        "malformed", "invalid-json", SafeTensorErrorKind.INVALID_JSON
    )
    _assert_index_error(
        "malformed", "invalid-utf8", SafeTensorErrorKind.INVALID_UTF8
    )
    _assert_index_error(
        "malformed", "missing-weight-map", SafeTensorErrorKind.MISSING_FIELD
    )
    _assert_index_error(
        "malformed", "empty-weight-map", SafeTensorErrorKind.INVALID_INDEX
    )
    _assert_index_error(
        "malformed", "excessive-nesting", SafeTensorErrorKind.INVALID_JSON
    )
    for name in [
        "duplicate-root-decoded",
        "duplicate-metadata-decoded",
        "duplicate-weight-map-decoded",
    ]:
        _assert_index_error(
            "malformed", name, SafeTensorErrorKind.DUPLICATE_KEY
        )


def test_index_schema_fixture_failures_are_typed() raises:
    for name in [
        "weight-map-not-object",
        "weight-map-value-not-string",
        "metadata-not-object",
        "total-size-negative",
        "total-size-fractional",
        "total-size-string",
        "total-size-boolean",
        "total-size-null",
        "total-size-exponent",
    ]:
        _assert_index_error(
            "malformed", name, SafeTensorErrorKind.INVALID_FIELD_TYPE
        )
    _assert_index_error(
        "malformed",
        "total-size-overflow",
        SafeTensorErrorKind.VALIDATION_OVERFLOW,
    )


def test_configured_limits_and_empty_explicit_list() raises:
    var raised = False
    try:
        _ = open_safetensors_index(
            _index("valid", "multiple"), max_index_bytes=1
        )
    except error:
        raised = True
        assert_equal(error.kind, SafeTensorErrorKind.INDEX_TOO_LARGE)
    assert_true(raised)

    raised = False
    try:
        _ = open_safetensors_index(
            _index("valid", "single"), max_header_bytes=1
        )
    except error:
        raised = True
        assert_equal(error.kind, SafeTensorErrorKind.HEADER_TOO_LARGE)
    assert_true(raised)

    raised = False
    try:
        _ = open_safetensors_index(_index("valid", "multiple"), max_shards=1)
    except error:
        raised = True
        assert_equal(error.kind, SafeTensorErrorKind.SHARD_LIMIT_EXCEEDED)
    assert_true(raised)

    raised = False
    try:
        _ = open_sharded_safetensors(List[String]())
    except error:
        raised = True
        assert_equal(error.kind, SafeTensorErrorKind.INVALID_INDEX)
    assert_true(raised)


def test_missing_tensor_and_destination_size_errors_are_preserved() raises:
    var reader = open_safetensors_index(_index("valid", "multiple"))
    var raised = False
    try:
        _ = reader.load_tensor("missing")
    except error:
        raised = True
        assert_equal(error.kind, SafeTensorErrorKind.TENSOR_NOT_FOUND)
    assert_true(raised)

    var too_short = List[UInt8](length=2, fill=0)
    raised = False
    try:
        reader.read_tensor_into("alpha", too_short)
    except error:
        raised = True
        assert_equal(error.kind, SafeTensorErrorKind.DESTINATION_SIZE_MISMATCH)
    assert_true(raised)


def test_single_shard_without_declared_size() raises:
    var reader = open_safetensors_index(_index("valid", "single"))
    assert_equal(reader.metadata().len(), 1)
    assert_false(reader.metadata().declared_total_size())
    assert_equal(reader.load_tensor("alpha"), [UInt8(1), 2, 3])


def test_buffered_switch_rejects_replaced_shard() raises:
    with TemporaryDirectory() as temporary:
        var source = "fixtures/sharded/valid/multiple/"
        var index_path = temporary + "/model.safetensors.index.json"
        var shard_b = temporary + "/shard-b.safetensors"
        _copy_multiple_archive(temporary)

        var reader = open_safetensors_index(index_path)
        assert_equal(reader.load_tensor("alpha"), [UInt8(1), 2, 3])
        var retained_old_path = temporary + "/retained-old-shard-b"
        _rename(shard_b, retained_old_path)
        Path(shard_b).write_bytes(
            Path(source + "shard-b.safetensors").read_bytes()
        )
        var raised = False
        try:
            _ = reader.load_tensor("beta")
        except error:
            raised = True
            assert_equal(error.kind, SafeTensorErrorKind.IO_ERROR)
        assert_true(raised)


def test_buffered_switch_rejects_grown_shard() raises:
    with TemporaryDirectory() as temporary:
        var index_path = temporary + "/model.safetensors.index.json"
        var shard_a = temporary + "/shard-a.safetensors"
        _copy_multiple_archive(temporary)

        var reader = open_safetensors_index(index_path)
        _ = reader.load_tensor("beta")
        var appender = open(shard_a, "a")
        var extra = [UInt8(0)]
        appender.write_all(extra)
        appender.close()
        var raised = False
        try:
            _ = reader.load_tensor("alpha")
        except error:
            raised = True
            assert_equal(error.kind, SafeTensorErrorKind.IO_ERROR)
        assert_true(raised)


def test_buffered_switch_rejects_truncated_shard() raises:
    with TemporaryDirectory() as temporary:
        var index_path = temporary + "/model.safetensors.index.json"
        var shard_a = temporary + "/shard-a.safetensors"
        _copy_multiple_archive(temporary)

        var reader = open_safetensors_index(index_path)
        _ = reader.load_tensor("beta")
        var truncator = open(shard_a, "w")
        truncator.close()
        var raised = False
        try:
            _ = reader.load_tensor("alpha")
        except error:
            raised = True
            assert_equal(error.kind, SafeTensorErrorKind.IO_ERROR)
        assert_true(raised)


def test_buffered_reopen_rejects_same_inode_metadata_mutation() raises:
    with TemporaryDirectory() as temporary:
        var source = "fixtures/sharded/valid/multiple/"
        var shard = temporary + "/shard-a.safetensors"
        Path(shard).write_bytes(
            Path(source + "shard-a.safetensors").read_bytes()
        )
        var reader = open_sharded_safetensors([shard])
        var mutated = Path(shard).read_bytes()
        var original_length = len(mutated)
        var before = stat(shard)
        _replace_ascii_bytes(mutated, "alpha", "omega")
        Path(shard).write_bytes(mutated)
        var after = stat(shard)
        assert_equal(after.st_dev, before.st_dev)
        assert_equal(after.st_ino, before.st_ino)
        assert_equal(len(Path(shard).read_bytes()), original_length)

        var raised = False
        try:
            _ = reader.load_tensor("alpha")
        except error:
            raised = True
            assert_equal(error.kind, SafeTensorErrorKind.IO_ERROR)
        assert_true(raised)


def test_buffered_reader_retains_at_most_one_active_shard() raises:
    var reader = open_safetensors_index(_index("valid", "multiple"))
    var inactive_count = len(listdir("/proc/self/fd"))
    _ = reader.load_tensor("alpha")
    var active_ceiling = len(listdir("/proc/self/fd"))
    assert_true(active_ceiling <= inactive_count + 1)
    _ = reader.load_tensor("beta")
    assert_true(len(listdir("/proc/self/fd")) <= active_ceiling)
    _ = reader.load_tensor("alpha")
    assert_true(len(listdir("/proc/self/fd")) <= active_ceiling)


def test_buffered_index_keeps_lexical_directory_anchor_after_rename() raises:
    with TemporaryDirectory() as temporary:
        var source = "fixtures/sharded/valid/single/"
        var archive = temporary + "/archive"
        var moved = temporary + "/moved"
        makedirs(archive)
        Path(archive + "/model.safetensors.index.json").write_bytes(
            Path(source + "model.safetensors.index.json").read_bytes()
        )
        Path(archive + "/model.safetensors").write_bytes(
            Path(source + "model.safetensors").read_bytes()
        )
        var reader = open_safetensors_index(
            archive + "/model.safetensors.index.json"
        )
        _rename(archive, moved)
        assert_equal(reader.load_tensor("alpha"), [UInt8(1), 2, 3])


def test_non_regular_shards_never_block_or_open() raises:
    with TemporaryDirectory() as temporary:
        var directory_shard = temporary + "/directory.safetensors"
        makedirs(directory_shard)
        var index_path = temporary + "/model.safetensors.index.json"
        var document = '{"weight_map":{"alpha":"directory.safetensors"}}'
        Path(index_path).write_bytes(List(document.as_bytes()))
        var raised = False
        try:
            _ = open_safetensors_index(index_path)
        except error:
            raised = True
            assert_equal(error.kind, SafeTensorErrorKind.PATH_TRAVERSAL)
        assert_true(raised)

        var fifo = temporary + "/pipe.safetensors"
        var fifo_name = fifo
        var result = external_call["mkfifo", c_int](
            fifo_name.as_c_string_slice(), c_int(0o600)
        )
        assert_equal(result, 0)
        document = '{"weight_map":{"alpha":"pipe.safetensors"}}'
        Path(index_path).write_bytes(List(document.as_bytes()))
        raised = False
        try:
            _ = open_safetensors_index(index_path)
        except error:
            raised = True
            assert_equal(error.kind, SafeTensorErrorKind.PATH_TRAVERSAL)
        assert_true(raised)

        raised = False
        try:
            _ = open_sharded_safetensors([fifo])
        except error:
            raised = True
            assert_equal(error.kind, SafeTensorErrorKind.IO_ERROR)
        assert_true(raised)
        remove(fifo)

        var socket_path = temporary + "/socket.safetensors"
        var socket_file = _bind_unix_socket(socket_path)
        document = '{"weight_map":{"alpha":"socket.safetensors"}}'
        Path(index_path).write_bytes(List(document.as_bytes()))
        raised = False
        try:
            _ = open_safetensors_index(index_path)
        except error:
            raised = True
            assert_equal(error.kind, SafeTensorErrorKind.PATH_TRAVERSAL)
        assert_true(raised)
        socket_file.close()
        remove(socket_path)


def test_strict_applies_to_shard_headers() raises:
    var raised = False
    try:
        _ = open_sharded_safetensors(
            ["fixtures/valid/unknown_descriptor_fields.safetensors"],
            strict=True,
        )
    except error:
        raised = True
        assert_equal(error.kind, SafeTensorErrorKind.UNKNOWN_FIELD)
    assert_true(raised)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
