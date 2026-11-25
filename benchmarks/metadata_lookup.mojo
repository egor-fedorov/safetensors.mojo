"""Compare two metadata access patterns on the same already-mapped archive."""

from std.builtin.sort import sort
from std.tempfile import NamedTemporaryFile
from std.time import perf_counter_ns

from safetensors import (
    MappedSafeTensorFile,
    SafeDType,
    SafeTensorData,
    map_safetensors,
    save_safetensors,
)


comptime _TENSOR_COUNT = 2048
comptime _SAMPLES = 9


def _walk_ns(
    archive: MappedSafeTensorFile, names: List[String], *, reuse_snapshot: Bool
) raises -> Int:
    var found = 0
    var started = perf_counter_ns()
    if reuse_snapshot:
        var metadata = archive.metadata()
        for name in names:
            found += Int(metadata.info(name).byte_length == 1)
    else:
        for name in names:
            found += Int(archive.metadata().info(name).byte_length == 1)
    var elapsed = perf_counter_ns() - started
    if found != _TENSOR_COUNT:
        raise Error("metadata lookup returned unexpected tensor lengths")
    return elapsed


def _report(label: String, var samples: List[Int]) raises:
    print(label, "samples (ns):", samples)
    sort(samples)
    print(label, "median (ms):", Float64(samples[len(samples) // 2]) / 1e6)


def main() raises:
    var fixture = NamedTemporaryFile()
    var tensors = List[SafeTensorData]()
    for i in range(_TENSOR_COUNT):
        tensors.append(
            SafeTensorData(
                "tensor_" + String(i), SafeDType.U8, [UInt64(1)], [UInt8(7)]
            )
        )
    save_safetensors(fixture.name, tensors)
    var archive = map_safetensors(fixture.name)
    var names = archive.metadata().names()
    _ = _walk_ns(archive, names, reuse_snapshot=False)
    _ = _walk_ns(archive, names, reuse_snapshot=True)

    var repeated = List[Int]()
    var reused = List[Int]()
    for i in range(_SAMPLES):
        if i % 2 == 0:
            repeated.append(_walk_ns(archive, names, reuse_snapshot=False))
            reused.append(_walk_ns(archive, names, reuse_snapshot=True))
        else:
            reused.append(_walk_ns(archive, names, reuse_snapshot=True))
            repeated.append(_walk_ns(archive, names, reuse_snapshot=False))
    print("tensors per walk:", len(names))
    _report("snapshot per lookup", repeated^)
    _report("snapshot per walk", reused^)
    fixture.close()
