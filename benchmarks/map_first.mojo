"""Measure repeated native map, validation, and first-tensor access."""

from std.sys import argv, exit
from std.time import perf_counter_ns

from safetensors import map_safetensors


comptime _FIRST_TENSOR = "tensor_000"


def _touch_first(path: String) raises -> Int:
    var archive = map_safetensors(path)
    var values = archive.tensor_view[DType.float32](_FIRST_TENSOR)
    if len(values) != 1:
        raise Error("the first benchmark tensor must contain one F32 value")
    return Int(values[0] != 0.0)


def main() raises:
    var arguments = argv()
    if len(arguments) == 2:
        print(_touch_first(String(arguments[1])))
        return
    if len(arguments) != 4:
        print("usage: map_first <archive> [<warmups> <samples>]")
        exit(2)

    var path = String(arguments[1])
    var warmups = Int(String(arguments[2]))
    var sample_count = Int(String(arguments[3]))
    if warmups < 0 or sample_count <= 0:
        print(
            "error: warmups must be non-negative and samples must be positive"
        )
        exit(2)

    var checksum = 0
    for _ in range(warmups):
        checksum += _touch_first(path)

    var samples = List[Int]()
    for _ in range(sample_count):
        var started = perf_counter_ns()
        checksum += _touch_first(path)
        var finished = perf_counter_ns()
        if finished < started:
            raise Error("monotonic clock moved backwards")
        samples.append(finished - started)

    for sample in samples:
        print("sample_ns", sample)
    print("checksum", checksum)
