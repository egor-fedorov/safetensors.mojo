"""Deterministic atomic local-file Safetensors writer."""

from safetensors.errors import SafeTensorError
from safetensors.format.checked import checked_add_u64
from safetensors.format.model import SafeTensorData
from safetensors.format.serialization import _plan_serialization
from safetensors.io._atomic_file import _create_atomic_file


def _encode_header_length(header_length: UInt64) -> List[UInt8]:
    """Encodes the declared header length as eight little-endian bytes."""
    var prefix = List[UInt8](length=8, fill=0)
    var remaining = header_length
    for index in range(8):
        prefix[index] = UInt8(remaining & 0xFF)
        remaining >>= 8
    return prefix^


def save_safetensors(
    path: String,
    tensors: List[SafeTensorData],
    user_metadata: Dict[String, String] = Dict[String, String](),
) raises SafeTensorError:
    """Validates and atomically writes one canonical Safetensors archive.

    Tensor payloads must already contain packed C-order little-endian wire
    bytes for their declared dtype and shape.
    """
    var plan = _plan_serialization(tensors, user_metadata)

    # Complete layout validation happens before the temporary file is created.
    var data_start = checked_add_u64(8, plan.header_length)
    _ = checked_add_u64(data_start, plan.data_length)

    var prefix = _encode_header_length(plan.header_length)
    var transaction = _create_atomic_file(path)
    transaction.write_all(prefix)
    transaction.write_all(plan.header)
    for index in range(len(plan.tensor_indices)):
        var source_index = plan.tensor_indices[index]
        transaction.write_all(tensors[source_index].data)
    transaction.commit()
