"""Deterministic atomic local-writer integration tests."""

from std.os import listdir, mkdir, remove, rmdir, stat, symlink
from std.os.path import basename, dirname, isdir, islink
from std.pathlib import Path
from std.tempfile import NamedTemporaryFile
from std.testing import TestSuite, assert_equal, assert_true

from safetensors import (
    SafeDType,
    SafeTensorData,
    SafeTensorErrorKind,
    map_safetensors,
    open_safetensors,
    save_safetensors,
)


def _temporary_path() raises -> String:
    var temporary = NamedTemporaryFile(mode="w", delete=False)
    var path = temporary.name.copy()
    temporary.close()
    return path^


def _writer_sibling_count(destination: String) raises -> Int:
    var parent = dirname(destination)
    if parent == "":
        parent = "."
    var prefix = "." + basename(destination) + ".safetensors-mojo-"
    var count = 0
    for name in listdir(parent):
        if name.startswith(prefix):
            count += 1
    return count


def _canonical_tensors() -> List[SafeTensorData]:
    return [
        SafeTensorData("zeta_u8", SafeDType.U8, [UInt64(2)], [UInt8(250), 7]),
        SafeTensorData(
            "omega_u16",
            SafeDType.U16,
            [UInt64(2)],
            [UInt8(0x34), 0x12, 0xCD, 0xAB],
        ),
        SafeTensorData(
            "scalar_i64",
            SafeDType.I64,
            List[UInt64](),
            [UInt8(0xD6), 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF],
        ),
        SafeTensorData("empty_i8", SafeDType.I8, [UInt64(0)], List[UInt8]()),
        SafeTensorData(
            "alpha_u8", SafeDType.U8, [UInt64(3)], [UInt8(0), 1, 0xFF]
        ),
        SafeTensorData(
            "beta_f32",
            SafeDType.F32,
            [UInt64(2)],
            [UInt8(0), 0, 0xC0, 0x3F, 0, 0, 0x10, 0xC0],
        ),
    ]


def _canonical_metadata() -> Dict[String, String]:
    var metadata = Dict[String, String]()
    metadata["unicode"] = "Zoë 😊"
    metadata['quote"and\\slash'] = "line one\nline two"
    metadata["author"] = "safetensors.mojo"
    return metadata^


def test_writer_matches_independent_canonical_fixture() raises:
    var path = _temporary_path()
    var tensors = _canonical_tensors()
    var metadata = _canonical_metadata()
    save_safetensors(path, tensors, metadata)

    assert_equal(
        Path(path).read_bytes(),
        Path("fixtures/valid/canonical_writer.safetensors").read_bytes(),
    )
    var reader = open_safetensors(path)
    assert_equal(
        reader.metadata().offset_names(),
        [
            "scalar_i64",
            "beta_f32",
            "omega_u16",
            "empty_i8",
            "alpha_u8",
            "zeta_u8",
        ],
    )
    assert_equal(reader.metadata().metadata_value("unicode").value(), "Zoë 😊")
    assert_equal(reader.load_tensor("alpha_u8"), [UInt8(0), 1, 0xFF])
    var mapped = map_safetensors(path)
    var scalar = mapped.tensor_view[DType.int64]("scalar_i64")
    assert_equal(len(scalar), 1)
    assert_equal(scalar[0], Int64(-42))
    remove(path)


def test_writer_matches_reference_dtype_shape_matrix_exactly() raises:
    var fixture = "fixtures/valid/reference_dtype_shapes.safetensors"
    var source = open_safetensors(fixture)
    var metadata = source.metadata()
    assert_equal(len(metadata), 79)
    var tensors = List[SafeTensorData]()
    for name in metadata.names():
        var info = metadata.info(name)
        var payload = source.load_tensor(name)
        tensors.append(
            SafeTensorData(
                name.copy(),
                info.dtype,
                info.shape.copy(),
                payload^,
            )
        )

    var path = _temporary_path()
    save_safetensors(path, tensors)

    assert_equal(Path(path).read_bytes(), Path(fixture).read_bytes())
    remove(path)


def test_input_permutations_produce_identical_bytes() raises:
    var first_path = _temporary_path()
    var second_path = _temporary_path()
    var first: List[SafeTensorData] = [
        SafeTensorData("z", SafeDType.U8, [UInt64(1)], [UInt8(3)]),
        SafeTensorData(
            "wide",
            SafeDType.U64,
            [UInt64(1)],
            [UInt8(1), 0, 0, 0, 0, 0, 0, 0],
        ),
        SafeTensorData("a", SafeDType.U8, [UInt64(1)], [UInt8(2)]),
    ]
    var second: List[SafeTensorData] = [
        SafeTensorData("a", SafeDType.U8, [UInt64(1)], [UInt8(2)]),
        SafeTensorData("z", SafeDType.U8, [UInt64(1)], [UInt8(3)]),
        SafeTensorData(
            "wide",
            SafeDType.U64,
            [UInt64(1)],
            [UInt8(1), 0, 0, 0, 0, 0, 0, 0],
        ),
    ]
    var first_metadata = Dict[String, String]()
    first_metadata["z"] = "last"
    first_metadata["a"] = "first"
    var second_metadata = Dict[String, String]()
    second_metadata["a"] = "first"
    second_metadata["z"] = "last"

    save_safetensors(first_path, first, first_metadata)
    save_safetensors(second_path, second, second_metadata)

    assert_equal(Path(first_path).read_bytes(), Path(second_path).read_bytes())
    remove(first_path)
    remove(second_path)


def test_empty_and_metadata_only_archives_round_trip() raises:
    var empty_path = _temporary_path()
    save_safetensors(empty_path, List[SafeTensorData]())
    var empty = open_safetensors(empty_path)
    assert_true(empty.metadata().is_empty())

    var metadata_path = _temporary_path()
    var user_metadata = Dict[String, String]()
    user_metadata["kind"] = "metadata-only"
    save_safetensors(metadata_path, List[SafeTensorData](), user_metadata)
    var metadata_only = open_safetensors(metadata_path)
    assert_true(metadata_only.metadata().is_empty())
    assert_equal(
        metadata_only.metadata().metadata_value("kind").value(),
        "metadata-only",
    )
    remove(empty_path)
    remove(metadata_path)


def test_failed_preflight_preserves_existing_destination() raises:
    var path = _temporary_path()
    var original: List[UInt8] = [UInt8(0xAA), 0xBB, 0xCC]
    Path(path).write_bytes(original)
    var invalid: List[SafeTensorData] = [
        SafeTensorData("short", SafeDType.U16, [UInt64(1)], [UInt8(0)])
    ]

    var raised = False
    try:
        save_safetensors(path, invalid)
    except error:
        raised = True
        assert_equal(error.kind, SafeTensorErrorKind.INVALID_TENSOR_SIZE)
    assert_true(raised)
    assert_equal(Path(path).read_bytes(), original)
    remove(path)


def test_atomic_replacement_keeps_open_reader_on_previous_inode() raises:
    var path = _temporary_path()
    Path(path).write_bytes(
        Path("fixtures/valid/reference_f32.safetensors").read_bytes()
    )
    var previous = open_safetensors(path)
    var replacement: List[SafeTensorData] = [
        SafeTensorData("new", SafeDType.U8, [UInt64(2)], [UInt8(4), 5])
    ]

    save_safetensors(path, replacement)

    assert_equal(previous.metadata().names(), ["weights"])
    assert_equal(len(previous.load_tensor("weights")), 16)
    var current = open_safetensors(path)
    assert_equal(current.metadata().names(), ["new"])
    assert_equal(current.load_tensor("new"), [UInt8(4), 5])
    remove(path)


def test_symlink_entry_is_replaced_without_following_target() raises:
    var target = _temporary_path()
    var target_contents: List[UInt8] = [UInt8(0xA1), 0xB2]
    Path(target).write_bytes(target_contents)
    var destination = _temporary_path()
    remove(destination)
    symlink(target, destination)
    assert_true(islink(destination))
    var tensors: List[SafeTensorData] = [
        SafeTensorData("new", SafeDType.U8, [UInt64(1)], [UInt8(9)])
    ]

    save_safetensors(destination, tensors)

    assert_true(not islink(destination))
    assert_equal(Path(target).read_bytes(), target_contents)
    var replacement = open_safetensors(destination)
    assert_equal(replacement.load_tensor("new"), [UInt8(9)])
    assert_equal(stat(destination).st_mode & 0o777, 0o600)
    remove(destination)
    remove(target)


def test_missing_parent_is_typed_io_error() raises:
    var parent_file = _temporary_path()
    var destination = String(Path(parent_file).joinpath("archive.safetensors"))
    var raised = False
    try:
        save_safetensors(destination, List[SafeTensorData]())
    except error:
        raised = True
        assert_equal(error.kind, SafeTensorErrorKind.IO_ERROR)
    assert_true(raised)
    remove(parent_file)


def test_failed_rename_cleans_temporary_sibling() raises:
    var destination = _temporary_path()
    remove(destination)
    mkdir(destination)
    assert_true(isdir(destination))
    assert_equal(_writer_sibling_count(destination), 0)
    var raised = False
    try:
        save_safetensors(destination, List[SafeTensorData]())
    except error:
        raised = True
        assert_equal(error.kind, SafeTensorErrorKind.IO_ERROR)
    assert_true(raised)
    assert_true(isdir(destination))
    assert_equal(_writer_sibling_count(destination), 0)
    rmdir(destination)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
