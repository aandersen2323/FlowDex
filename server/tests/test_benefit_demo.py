"""Integration-style regression demonstrating FlowDex prompt optimisation."""

from __future__ import annotations

import json
from typing import Any, Dict, List

import importlib

import pytest

flowdex_app = importlib.import_module("server.app")


class _FakeRedis:
    """Very small in-memory stand-in for the Redis client used in tests."""

    def __init__(self) -> None:
        self._hashes: Dict[str, Dict[str, str]] = {}
        self._lists: Dict[str, List[str]] = {}
        self._sets: Dict[str, set[str]] = {}

    # Hash helpers -----------------------------------------------------
    def hset(self, key: str, field: str, value: str) -> None:
        self._hashes.setdefault(key, {})[field] = value

    def hget(self, key: str, field: str) -> str | None:
        return self._hashes.get(key, {}).get(field)

    def hlen(self, key: str) -> int:
        return len(self._hashes.get(key, {}))

    # List helpers -----------------------------------------------------
    def rpush(self, key: str, value: str) -> None:
        self._lists.setdefault(key, []).append(value)

    def lindex(self, key: str, index: int) -> str | None:
        items = self._lists.get(key, [])
        if not items:
            return None
        if index < 0:
            index = len(items) + index
        if 0 <= index < len(items):
            return items[index]
        return None

    # Set helpers ------------------------------------------------------
    def sadd(self, key: str, member: str) -> None:
        self._sets.setdefault(key, set()).add(member)

    def smembers(self, key: str) -> set[str]:
        return set(self._sets.get(key, set()))

    # Misc -------------------------------------------------------------
    def ping(self) -> bool:  # pragma: no cover - defensive parity with redis
        return True


def test_flowdex_prompt_budgeting_cuts_tokens(monkeypatch: pytest.MonkeyPatch, tmp_path):
    """FlowDex trims inputs to configured token budgets unlike a naive baseline."""

    fake_redis = _FakeRedis()
    monkeypatch.setattr(flowdex_app, "get_redis", lambda: fake_redis)
    monkeypatch.setattr(flowdex_app.settings, "anthropic_api_key", "test-key", raising=False)
    monkeypatch.setattr(flowdex_app, "CACHE_DIR", tmp_path)

    budget = flowdex_app.Budget(system=12, context=12, user=10, tools=20)

    context_text = "alpha beta gamma delta epsilon " * 20
    memory_payload = json.dumps({"data": context_text, "ts": 1700000000})
    fake_redis.rpush(f"{flowdex_app.MEMORY_KEY_PREFIX}ctx-1", memory_payload)

    tool_spec = {
        "name": "demo-tool",
        "description": "Summarises error queues",
        "cost": 5,
        "schema": {"type": "object", "properties": {"query": {"type": "string"}}},
    }
    fake_redis.hset(flowdex_app.TOOLS_KEY, "demo-tool", json.dumps(tool_spec))

    captured: Dict[str, Any] = {}

    def _fake_call_llm(prompt: Dict[str, Any], request: flowdex_app.InferRequest, tool_specs: List[Dict[str, Any]], retrieved: List[Dict[str, Any]]) -> Dict[str, Any]:
        captured["prompt"] = prompt
        captured["tool_specs"] = tool_specs
        captured["retrieved"] = retrieved
        return {"completion": "stubbed response", "tokens": prompt.get("tokens_used", {})}

    monkeypatch.setattr(flowdex_app, "call_llm", _fake_call_llm)

    req = flowdex_app.InferRequest(
        task="Diagnose failing workflow",
        system_prompt="System instruction " * 20,
        user_input="User provided failure details " * 20,
        context_ids=["ctx-1"],
        tool_candidates=["demo-tool"],
        budget=budget,
    )

    result = flowdex_app.run_inference(req)

    naive_counts = {
        "system": flowdex_app.count_tokens(req.system_prompt),
        "context": flowdex_app.count_tokens(context_text),
        "user": flowdex_app.count_tokens(req.user_input),
    }

    assert captured["tool_specs"], "FlowDex should pass registered tool specs to the model call."
    assert result["tools_considered"] == ["demo-tool"]

    used_tokens = result["used_tokens"]
    assert used_tokens["system"] <= budget.system
    assert used_tokens["context"] <= budget.context
    assert used_tokens["user"] <= budget.user

    assert naive_counts["system"] > used_tokens["system"]
    assert naive_counts["context"] > used_tokens["context"]
    assert naive_counts["user"] > used_tokens["user"]

    original_context_blob = f"\n<ctx id='ctx-1'>\n{context_text}\n</ctx>\n"
    assert captured["prompt"]["context"] != original_context_blob
