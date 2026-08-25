"""Survival harness for the hostile Safetensors corpus.

Every case is driven through all three public entry points, and every byte of
every accepted tensor is dereferenced so that a bounds error in the mapped
path would fault instead of passing silently:

* ``parse_metadata`` over a caller-owned buffer;
* ``map_safetensors`` plus ``tensor_bytes`` and representative 32- and 64-bit
  ``tensor_view`` calls on the mapping, reading each returned value; and
* ``open_safetensors`` plus ``load_tensor``, reading each byte.

The pass criterion is survival, not a verdict: the harness asserts only that
the process reaches the end without a panic, a fault, or a hang. Accepted and
rejected counts are reported for corpus triage.

Usage:
    mojo run -I src tests/fuzz/fuzz_harness.mojo <corpus-directory>
"""

from std.pathlib import Path
from std.sys import argv, exit

from safetensors import map_safetensors, open_safetensors, parse_metadata


def _case_name(index: Int) -> String:
    var digits = String(index)
    while digits.byte_length() < 6:
        digits = "0" + digits
    return "case" + digits + ".bin"


def _exercise_mapped(path: String) raises -> Int:
    """Maps one case and consumes every byte and element it hands out."""
    var checksum = 0
    var mapped = map_safetensors(path)
    var metadata = mapped.metadata()

    for name in metadata.names():
        var raw = mapped.tensor_bytes(name)
        for index in range(len(raw)):
            checksum += Int(raw[index])

        try:
            var floats = mapped.tensor_view[DType.float32](name)
            for index in range(len(floats)):
                checksum += Int(floats[index] != 0)
        except:
            pass

        try:
            var integers = mapped.tensor_view[DType.int64](name)
            for index in range(len(integers)):
                checksum += Int(integers[index] != 0)
        except:
            pass

    return checksum


def _exercise_reader(path: String) raises -> Int:
    """Opens one case through the buffered reader and loads every tensor."""
    var checksum = 0
    var reader = open_safetensors(path)
    var metadata = reader.metadata()

    for name in metadata.names():
        var loaded = reader.load_tensor(name)
        for index in range(len(loaded)):
            checksum += Int(loaded[index])

    return checksum


def main() raises:
    var arguments = argv()
    if len(arguments) != 2:
        print("usage: fuzz_harness <corpus-directory>")
        exit(2)

    var corpus = Path(String(arguments[1]))
    var count = Int(corpus.joinpath("count.txt").read_text().strip())
    if count <= 0:
        print("error: fuzz corpus must contain at least one case")
        exit(2)

    var parsed = 0
    var rejected = 0
    var mapped_ok = 0
    var read_ok = 0
    var checksum = 0

    for index in range(count):
        var path = String(corpus.joinpath(_case_name(index)))
        var contents = Path(path).read_bytes()

        try:
            _ = parse_metadata(contents)
            parsed += 1
        except:
            rejected += 1

        try:
            checksum += _exercise_mapped(path)
            mapped_ok += 1
        except:
            pass

        try:
            checksum += _exercise_reader(path)
            read_ok += 1
        except:
            pass

    print("cases:            ", count)
    print("parse accepted:   ", parsed, " rejected:", rejected)
    print("mapped accepted:  ", mapped_ok)
    print("reader accepted:  ", read_ok)
    print("touch checksum:   ", checksum)
    print("survived every case without a panic or fault")
