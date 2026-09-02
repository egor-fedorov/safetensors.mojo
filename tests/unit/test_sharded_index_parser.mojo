"""Unit tests for exact, duplicate-aware Safetensors index parsing."""

from std.testing import TestSuite, assert_equal, assert_false, assert_true

from safetensors import SafeTensorErrorKind
from safetensors.sharding.archive import _index_shard_names
from safetensors.sharding.index_parser import (
    DEFAULT_MAX_INDEX_ENTRIES,
    _parse_index,
)


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


def test_weight_map_entry_limit_is_exact_and_typed() raises:
    assert_equal(DEFAULT_MAX_INDEX_ENTRIES, UInt64(1_000_000))
    var document = String(
        '{"weight_map":{"a":"one.safetensors","b":"two.safetensors"}}'
    )
    var parsed = _parse_index(document.as_bytes(), max_index_entries=2)
    assert_equal(len(parsed.weight_map), 2)

    var raised = False
    try:
        _ = _parse_index(document.as_bytes(), max_index_entries=1)
    except error:
        raised = True
        assert_equal(error.kind, SafeTensorErrorKind.INDEX_ENTRY_LIMIT_EXCEEDED)
    assert_true(raised)


def test_entry_limit_precedes_decoding_the_excess_key() raises:
    var document = String(
        '{"weight_map":{"a":"one.safetensors",not-a-json-key}}'
    )
    var raised = False
    try:
        _ = _parse_index(document.as_bytes(), max_index_entries=1)
    except error:
        raised = True
        assert_equal(error.kind, SafeTensorErrorKind.INDEX_ENTRY_LIMIT_EXCEEDED)
    assert_true(raised)


def test_entry_and_unique_shard_limits_have_independent_precedence() raises:
    var repeated = String(
        '{"weight_map":{"a":"one.safetensors","b":"one.safetensors",'
        '"c":"one.safetensors"}}'
    )
    var raised = False
    try:
        _ = _parse_index(repeated.as_bytes(), max_index_entries=2, max_shards=1)
    except error:
        raised = True
        assert_equal(error.kind, SafeTensorErrorKind.INDEX_ENTRY_LIMIT_EXCEEDED)
    assert_true(raised)

    var unique = String(
        '{"weight_map":{"a":"one.safetensors","b":"two.safetensors",'
        '"c":"three.safetensors"}}'
    )
    raised = False
    try:
        _ = _parse_index(unique.as_bytes(), max_index_entries=3, max_shards=2)
    except error:
        raised = True
        assert_equal(error.kind, SafeTensorErrorKind.SHARD_LIMIT_EXCEEDED)
    assert_true(raised)


def test_unique_shard_limit_precedes_any_file_open() raises:
    var document = String(
        '{"weight_map":{"a":"missing-a.safetensors",'
        '"b":"missing-b.safetensors"}}'
    )
    var parsed = _parse_index(document.as_bytes())
    assert_equal(len(_index_shard_names(parsed, 2)), 2)

    var raised = False
    try:
        _ = _parse_index(document.as_bytes(), max_shards=1)
    except error:
        raised = True
        assert_equal(error.kind, SafeTensorErrorKind.SHARD_LIMIT_EXCEEDED)
    assert_true(raised)

    raised = False
    try:
        _ = _index_shard_names(parsed, 1)
    except error:
        raised = True
        assert_equal(error.kind, SafeTensorErrorKind.SHARD_LIMIT_EXCEEDED)
    assert_true(raised)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
