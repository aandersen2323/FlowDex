"""Unit tests for small utility helpers in :mod:`server.app`."""

from __future__ import annotations

import math

import pytest

from server.app import (
    _safe_json_dumps,
    _tokenize,
    build_tool_context,
    json_patch,
    truncate_to_budget,
)


@pytest.mark.parametrize(
    ("old", "new", "expected"),
    (
        ("", "abc", {"common_prefix": 0, "added": "abc"}),
        ("abc", "abcd", {"common_prefix": 3, "added": "d"}),
        ("abcdef", "abcxyz", {"common_prefix": 3, "added": "xyz"}),
    ),
)
def test_json_patch(old: str, new: str, expected: dict[str, object]) -> None:
    assert json_patch(old, new) == expected


def test_truncate_to_budget() -> None:
    assert truncate_to_budget("abcdef", 10) == "abcdef"
    assert truncate_to_budget("abcdef", 3) == "def"


@pytest.mark.parametrize(
    ("text", "expected"),
    (
        ("Hello, world!", ["hello", "world"]),
        ("Numbers 123 should be kept", ["numbers", "123", "should", "be", "kept"]),
        ("MixedCASE", ["mixedcase"]),
    ),
)
def test_tokenize(text: str, expected: list[str]) -> None:
    assert _tokenize(text) == expected


def test_build_tool_context_truncates_and_serializes() -> None:
    complex_schema = {"callable": math.sqrt}
    tool_specs = [
        {"name": "alpha", "description": "first", "cost": 10, "schema": {"x": "int"}},
        {"name": "beta", "description": "second", "cost": "low", "schema": complex_schema},
    ]

    context = build_tool_context(tool_specs, budget_chars=500)
    assert "Tool: alpha" in context
    assert "Schema: {\"x\": \"int\"}" in context
    # Non-serializable value should fallback to repr
    assert "callable" in context


def test_safe_json_dumps_repr_fallback() -> None:
    class Foo:
        pass

    value = Foo()
    assert _safe_json_dumps(value).startswith("<")
