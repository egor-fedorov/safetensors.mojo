"""Validated local-file access for Safetensors payload bytes."""

from safetensors.io.mapped_reader import MappedSafeTensorFile, map_safetensors
from safetensors.io.reader import SafeTensorReader, open_safetensors
