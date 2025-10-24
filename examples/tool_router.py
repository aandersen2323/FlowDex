"""Register a FlowDex tool and run an inference that prefers it."""
from __future__ import annotations

import os
import sys
from typing import Any, Dict

import requests

BASE_URL = os.environ.get("FLOWDEX_BASE", "http://localhost:8787")
API_KEY = os.environ.get("FLOWDEX_API_KEY")


def _headers() -> Dict[str, str]:
    headers = {"Content-Type": "application/json"}
    if API_KEY:
        headers["X-API-Key"] = API_KEY
    return headers


def register_tool() -> None:
    payload: Dict[str, Any] = {
        "name": "post_incident_report",
        "description": "Generate an executive-friendly incident summary",
        "input_schema": {
            "type": "object",
            "properties": {
                "incident_id": {"type": "string"},
                "audience": {"type": "string", "enum": ["internal", "customer"]},
            },
            "required": ["incident_id"],
        },
        "cost": {"latency_ms": 2500, "tokens": 800},
    }
    res = requests.post(f"{BASE_URL}/tools/register", json=payload, headers=_headers(), timeout=30)
    res.raise_for_status()
    print("Registered tool:", res.json())


def run_infer() -> None:
    payload: Dict[str, Any] = {
        "task": "incident_update",
        "model": "anthropic/claude-3-5-sonnet",
        "inputs": {
            "user": "Draft the customer update for incident INC-2045.",
            "context": ["runbook.postgres"],
            "tool_hints": ["post_incident_report"],
            "tool_inputs": {
                "post_incident_report": {
                    "incident_id": "INC-2045",
                    "audience": "customer",
                }
            },
        },
    }
    res = requests.post(f"{BASE_URL}/infer", json=payload, headers=_headers(), timeout=60)
    res.raise_for_status()
    data = res.json()
    print("FlowDex response:\n", data)


def main() -> int:
    try:
        register_tool()
    except requests.HTTPError as exc:
        if exc.response is not None and exc.response.status_code == 409:
            print("Tool already registered; continuing")
        else:
            raise
    run_infer()
    return 0


if __name__ == "__main__":
    sys.exit(main())
