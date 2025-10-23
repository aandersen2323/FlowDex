#!/usr/bin/env python3
import json
import os
import sys
import time
from typing import Any, Dict

import requests

BASE_URL = os.environ.get("FLOWDEX_API_URL", "http://localhost:8787")
DEFAULT_TIMEOUT = float(os.environ.get("FLOWDEX_MCP_TIMEOUT", 30))


def respond(id_value, result: Any = None, error: str = None):
    message: Dict[str, Any] = {"jsonrpc": "2.0", "id": id_value}
    if error is not None:
        message["error"] = {"code": -32000, "message": error}
    else:
        message["result"] = result
    sys.stdout.write(json.dumps(message) + "\n")
    sys.stdout.flush()


def call_api(method: str, endpoint: str, *, json_body: Dict[str, Any] = None, params: Dict[str, Any] = None) -> Dict[str, Any]:
    url = f"{BASE_URL.rstrip('/')}{endpoint}"
    try:
        if method.upper() == "POST":
            resp = requests.post(url, json=json_body or {}, timeout=DEFAULT_TIMEOUT)
        else:
            resp = requests.get(url, params=params, timeout=DEFAULT_TIMEOUT)
        resp.raise_for_status()
    except requests.RequestException as exc:
        raise RuntimeError(f"FlowDex API request failed: {exc}") from exc
    try:
        return resp.json()
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"Invalid JSON response from FlowDex API: {exc}") from exc


def main():
    for line in sys.stdin:
        if not line.strip():
            continue
        try:
            req = json.loads(line.strip())
        except json.JSONDecodeError:
            continue
        method = req.get("method")
        id_value = req.get("id")
        params = req.get("params", {})

        try:
            if method == "ping":
                respond(id_value, {"ok": True, "ts": time.time()})
            elif method == "flowdex.infer":
                result = call_api("POST", "/infer", json_body=params)
                respond(id_value, result)
            elif method == "flowdex.memory.get":
                result = call_api("GET", "/memory/get", params=params)
                respond(id_value, result)
            else:
                respond(id_value, error=f"Unknown method: {method}")
        except RuntimeError as exc:
            respond(id_value, error=str(exc))


if __name__ == "__main__":
    main()
