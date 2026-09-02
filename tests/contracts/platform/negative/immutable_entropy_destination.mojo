"""Must fail because the OS entropy destination must be mutable."""

from safetensors.io._platform import _fill_os_random


def overwrite(immutable: String):
    _ = _fill_os_random(immutable.as_bytes())


def main():
    overwrite("immutable bytes")
