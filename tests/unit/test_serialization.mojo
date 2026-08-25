"""Deterministic checked Safetensors serialization planning tests."""

from std.testing import TestSuite, assert_equal, assert_true

from safetensors import SafeDType, SafeTensorErrorKind, parse_metadata
from safetensors.format import SafeTensorData
from safetensors.format.serialization import _plan_serialization


def _metadata() -> Dict[String, String]:
    return Dict[String, String]()


def _complete_file(
    plan_header: List[UInt8],
    tensor_indices: List[Int],
    tensors: List[SafeTensorData],
) -> List[UInt8]:
    var result = List[UInt8]()
    var header_length = UInt64(len(plan_header))
    for index in range(8):
        result.append(
            UInt8((header_length >> UInt64(index * 8)) & UInt64(0xFF))
        )
    for byte in plan_header:
        result.append(byte)
    for canonical_index in tensor_indices:
        for byte in tensors[canonical_index].data:
            result.append(byte)
    return result^


def _assert_plan_error(
    tensors: List[SafeTensorData], expected: SafeTensorErrorKind
) raises:
    var raised = False
    try:
        _ = _plan_serialization(tensors, _metadata())
    except error:
        raised = True
        assert_equal(error.kind, expected)
    assert_true(raised)


def test_canonical_header_layout_and_round_trip() raises:
    var tensors: List[SafeTensorData] = [
        SafeTensorData("z", SafeDType.U8, [UInt64(1)], [UInt8(9)]),
        SafeTensorData(
            "β",
            SafeDType.U64,
            List[UInt64](),
            [UInt8(1), 2, 3, 4, 5, 6, 7, 8],
        ),
        SafeTensorData("a", SafeDType.U8, [UInt64(2)], [UInt8(10), 11]),
    ]
    var metadata = Dict[String, String]()
    metadata["z"] = "line\n"
    metadata["a"] = 'quote" slash\\ Zoë\u0001'

    var plan = _plan_serialization(tensors, metadata)
    var expected = (
        '{"__metadata__":{"a":"quote\\" slash\\\\ Zoë\\u0001",'
        '"z":"line\\n"},"β":{"dtype":"U64","shape":[],"data_offsets":[0,8]},'
        '"a":{"dtype":"U8","shape":[2],"data_offsets":[8,10]},'
        '"z":{"dtype":"U8","shape":[1],"data_offsets":[10,11]}}'
    )
    while expected.byte_length() % 8 != 0:
        expected += " "

    assert_equal(String(from_utf8=Span(plan.header)), expected)
    assert_equal(plan.header_length, UInt64(expected.byte_length()))
    assert_equal(plan.header_length % 8, UInt64(0))
    assert_equal(plan.data_length, UInt64(11))
    assert_equal(plan.tensor_indices, [Int(1), 2, 0])
    assert_equal(plan.tensors[0].name, "β")
    assert_equal(plan.tensors[1].begin, UInt64(8))
    assert_equal(plan.tensors[2].end, UInt64(11))

    var complete = _complete_file(plan.header, plan.tensor_indices, tensors)
    var parsed = parse_metadata(complete)
    assert_equal(parsed.offset_names(), ["β", "a", "z"])
    assert_equal(parsed.metadata_value("a").value(), 'quote" slash\\ Zoë\u0001')


def test_empty_and_zero_dimension_plans() raises:
    var empty = _plan_serialization(
        List[SafeTensorData](), Dict[String, String]()
    )
    assert_equal(String(from_utf8=Span(empty.header)), "{}      ")
    assert_equal(empty.header_length, UInt64(8))
    assert_equal(empty.data_length, UInt64(0))
    assert_equal(len(empty.tensor_indices), 0)

    var zero = _plan_serialization(
        [
            SafeTensorData(
                "zero",
                SafeDType.U64,
                [UInt64.MAX, 0, UInt64.MAX],
                List[UInt8](),
            )
        ],
        Dict[String, String](),
    )
    assert_equal(zero.tensors[0].element_count, UInt64(0))
    assert_equal(zero.tensors[0].byte_length, UInt64(0))
    assert_equal(zero.data_length, UInt64(0))


def test_reference_dtype_ordinals_define_canonical_order() raises:
    var dtypes: List[SafeDType] = [
        SafeDType.BOOL,
        SafeDType.F4,
        SafeDType.F6_E2M3,
        SafeDType.F6_E3M2,
        SafeDType.U8,
        SafeDType.I8,
        SafeDType.F8_E5M2,
        SafeDType.F8_E4M3,
        SafeDType.F8_E8M0,
        SafeDType.F8_E4M3FNUZ,
        SafeDType.F8_E5M2FNUZ,
        SafeDType.I16,
        SafeDType.U16,
        SafeDType.F16,
        SafeDType.BF16,
        SafeDType.I32,
        SafeDType.U32,
        SafeDType.F32,
        SafeDType.C64,
        SafeDType.F64,
        SafeDType.I64,
        SafeDType.U64,
    ]
    var tensors = List[SafeTensorData]()
    for index in range(len(dtypes)):
        tensors.append(
            SafeTensorData(
                "tensor_" + String(index),
                dtypes[index],
                [UInt64(0)],
                List[UInt8](),
            )
        )

    var plan = _plan_serialization(tensors, _metadata())
    for index in range(len(plan.tensor_indices)):
        assert_equal(plan.tensor_indices[index], len(dtypes) - index - 1)


def test_byte_addressable_subbyte_tensors_are_planned_exactly() raises:
    var tensors: List[SafeTensorData] = [
        SafeTensorData("f4", SafeDType.F4, [UInt64(2)], [UInt8(0x21)]),
        SafeTensorData(
            "f6",
            SafeDType.F6_E3M2,
            [UInt64(4)],
            [UInt8(0x01), 0x23, 0x45],
        ),
    ]
    var plan = _plan_serialization(tensors, _metadata())

    assert_equal(plan.tensor_indices, [Int(1), 0])
    assert_equal(plan.data_length, UInt64(4))
    assert_equal(plan.tensors[0].bit_length, UInt64(24))
    assert_equal(plan.tensors[0].begin, UInt64(0))
    assert_equal(plan.tensors[0].end, UInt64(3))
    assert_equal(plan.tensors[1].bit_length, UInt64(8))
    assert_equal(plan.tensors[1].begin, UInt64(3))
    assert_equal(plan.tensors[1].end, UInt64(4))

    var complete = _complete_file(plan.header, plan.tensor_indices, tensors)
    var parsed = parse_metadata(complete)
    assert_equal(parsed.offset_names(), ["f6", "f4"])


def test_input_validation_errors_are_typed() raises:
    _assert_plan_error(
        [
            SafeTensorData("a", SafeDType.U8, [UInt64(0)], List[UInt8]()),
            SafeTensorData("a", SafeDType.U8, [UInt64(0)], List[UInt8]()),
        ],
        SafeTensorErrorKind.DUPLICATE_KEY,
    )
    _assert_plan_error(
        [
            SafeTensorData(
                "__metadata__",
                SafeDType.U8,
                [UInt64(0)],
                List[UInt8](),
            )
        ],
        SafeTensorErrorKind.INVALID_METADATA,
    )
    _assert_plan_error(
        [
            SafeTensorData(
                "invalid",
                SafeDType(255),
                [UInt64(0)],
                List[UInt8](),
            )
        ],
        SafeTensorErrorKind.UNSUPPORTED_DTYPE,
    )
    _assert_plan_error(
        [
            SafeTensorData(
                "shape_overflow",
                SafeDType.U8,
                [UInt64.MAX, 2],
                List[UInt8](),
            )
        ],
        SafeTensorErrorKind.VALIDATION_OVERFLOW,
    )
    _assert_plan_error(
        [
            SafeTensorData(
                "bit_overflow",
                SafeDType.U64,
                [UInt64.MAX],
                List[UInt8](),
            )
        ],
        SafeTensorErrorKind.VALIDATION_OVERFLOW,
    )
    _assert_plan_error(
        [
            SafeTensorData(
                "packed",
                SafeDType.F4,
                [UInt64(1)],
                List[UInt8](),
            )
        ],
        SafeTensorErrorKind.MISALIGNED_SLICE,
    )
    _assert_plan_error(
        [SafeTensorData("short", SafeDType.U16, [UInt64(1)], [UInt8(0)])],
        SafeTensorErrorKind.INVALID_TENSOR_SIZE,
    )


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
