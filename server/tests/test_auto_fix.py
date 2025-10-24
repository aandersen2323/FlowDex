"""Unit tests for the automated remediation helpers."""

from __future__ import annotations

from server.app import (
    _extract_auto_fix_json,
    _format_prior_auto_fix,
    _parse_auto_fix_completion,
)


def test_extract_auto_fix_json_from_code_fence() -> None:
    payload = """```json\n{\n  \"status\": \"resolved\",\n  \"summary\": \"All good\"\n}\n```"""
    parsed = _extract_auto_fix_json(payload)
    assert parsed["status"] == "resolved"
    assert parsed["summary"] == "All good"


def test_parse_auto_fix_completion_normalizes_fields() -> None:
    completion = """```json\n{\n  \"status\": \"apply_fix\",\n  \"summary\": \"Missing credential\",\n  \"actions\": [\"Rotate key\"],\n  \"fix_instructions\": \"Regenerate API key\",\n  \"metrics\": {\"confidence\": 0.8}\n}\n```"""
    parsed = _parse_auto_fix_completion(completion)
    assert parsed["status"] == "apply_fix"
    assert parsed["summary"] == "Missing credential"
    assert parsed["actions"] == ["Rotate key"]
    assert parsed["fix_instructions"] == "Regenerate API key"
    assert parsed["metrics"] == {"confidence": 0.8}


def test_parse_auto_fix_completion_handles_unstructured_text() -> None:
    completion = "The workflow timed out before hitting the callback"
    parsed = _parse_auto_fix_completion(completion)
    assert parsed["status"] == "unparsed"
    assert "workflow timed out" in parsed["summary"].lower()
    assert parsed["actions"] == []


def test_format_prior_auto_fix_includes_history() -> None:
    history = [
        {
            "status": "resolved",
            "summary": "Added missing queue binding",
            "actions": ["Create binding", "Deploy"],
            "parsed": {
                "status": "resolved",
                "summary": "Added missing queue binding",
                "actions": ["Create binding", "Deploy"],
            },
        }
    ]
    formatted = _format_prior_auto_fix(history)
    assert "Attempt 1 status: resolved" in formatted
    assert "Added missing queue binding" in formatted
