"""Unit tests for exact, duplicate-aware Safetensors index parsing."""

from std.testing import TestSuite, assert_equal, assert_false, assert_true

from safetensors import SafeTensorErrorKind
from safetensors.sharding.index_parser import _parse_index


def test_parses_required_map_and_exact_total_size() raises:
    var document = String(
        '{"metadata":{"total_size":9,"total_parameters":3,'
        '"future":{"nested":[true,null,-1.25e2]}},'
        '"weight_map":{"a":"part-1.safetensors",'
        '"b":"part-2.safetensors"},"future":[1,2,3]}'
    )
    var parsed = _parse_index(document.as_bytes())
    assert_equal(len(parsed.weight_map), 2)
    assert_equal(parsed.weight_map.get("a").value(), "part-1.safetensors")
    assert_equal(parsed.weight_map.get("b").value(), "part-2.safetensors")
    assert_true(parsed.declared_total_size)
    assert_equal(parsed.declared_total_size.value(), UInt64(9))


def test_optional_metadata_and_json_whitespace() raises:
    var document = String(' \n\t{"weight_map":{"x":"one.safetensors"}}\r ')
    var parsed = _parse_index(document.as_bytes())
    assert_false(parsed.declared_total_size)


def test_rejects_empty_or_missing_weight_map() raises:
    var raised = False
    var empty = String('{"weight_map":{}}')
    try:
        _ = _parse_index(empty.as_bytes())
    except error:
        raised = True
        assert_equal(error.kind, SafeTensorErrorKind.INVALID_INDEX)
    assert_true(raised)

    raised = False
    var missing = String('{"metadata":{}}')
    try:
        _ = _parse_index(missing.as_bytes())
    except error:
        raised = True
        assert_equal(error.kind, SafeTensorErrorKind.MISSING_FIELD)
    assert_true(raised)


def test_detects_decoded_duplicates_at_each_retained_level() raises:
    var cases = List[String]()
    cases.append(
        '{"weight_map":{"a":"one.safetensors"},'
        '"weight\\u005fmap":{"b":"two.safetensors"}}'
    )
    cases.append(
        '{"metadata":{"total_size":1,"total\\u005fsize":1},'
        '"weight_map":{"a":"one.safetensors"}}'
    )
    cases.append(
        '{"weight_map":{"a":"one.safetensors","\\u0061":"two.safetensors"}}'
    )

    for document in cases:
        var raised = False
        try:
            _ = _parse_index(document.as_bytes())
        except error:
            raised = True
            assert_equal(error.kind, SafeTensorErrorKind.DUPLICATE_KEY)
        assert_true(raised)


def test_total_size_never_uses_floating_point() raises:
    var exact = String(
        '{"metadata":{"total_size":9007199254740993},'
        '"weight_map":{"a":"one.safetensors"}}'
    )
    var parsed = _parse_index(exact.as_bytes())
    assert_equal(parsed.declared_total_size.value(), UInt64(9007199254740993))

    var fraction = String(
        '{"metadata":{"total_size":1.0},"weight_map":{"a":"one.safetensors"}}'
    )
    var raised = False
    try:
        _ = _parse_index(fraction.as_bytes())
    except error:
        raised = True
        assert_equal(error.kind, SafeTensorErrorKind.INVALID_FIELD_TYPE)
    assert_true(raised)

    var overflow = String(
        '{"metadata":{"total_size":18446744073709551616},'
        '"weight_map":{"a":"one.safetensors"}}'
    )
    raised = False
    try:
        _ = _parse_index(overflow.as_bytes())
    except error:
        raised = True
        assert_equal(error.kind, SafeTensorErrorKind.VALIDATION_OVERFLOW)
    assert_true(raised)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
