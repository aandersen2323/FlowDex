#!/usr/bin/env python3
import json
import os
import sys
import time
from typing import Any, Dict

import requests

FLOWDEX_BASE_URL = os.environ.get("FLOWDEX_BASE_URL", "http://localhost:8787")
FLOWDEX_TIMEOUT = float(os.environ.get("FLOWDEX_HTTP_TIMEOUT", "30"))

def respond(id, result=None, error=None):
    msg = {"jsonrpc":"2.0","id":id}
    if error is not None:
        msg["error"] = {"code": -32000, "message": error}
    else:
        msg["result"] = result
    sys.stdout.write(json.dumps(msg)+"\n")
    sys.stdout.flush()


def _post(path: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    try:
        res = requests.post(
            f"{FLOWDEX_BASE_URL}{path}",
            json=payload,
            timeout=FLOWDEX_TIMEOUT,
        )
        res.raise_for_status()
    except requests.RequestException as exc:
        raise RuntimeError(str(exc))
    return res.json()


def _get(path: str, params: Dict[str, Any]) -> Dict[str, Any]:
    try:
        res = requests.get(
            f"{FLOWDEX_BASE_URL}{path}",
            params=params,
            timeout=FLOWDEX_TIMEOUT,
        )
        res.raise_for_status()
    except requests.RequestException as exc:
        raise RuntimeError(str(exc))
    return res.json()


def main():
    for line in sys.stdin:
        try:
            req = json.loads(line.strip())
        except Exception:
            continue
        method = req.get("method")
        id_ = req.get("id")
        params = req.get("params", {})

        if method == "ping":
            respond(id_, {"ok": True, "ts": time.time()})
        elif method == "flowdex.infer":
            try:
                respond(id_, _post("/infer", params))
            except RuntimeError as exc:
                respond(id_, error=str(exc))
        elif method == "flowdex.memory.get":
            try:
                respond(id_, _get("/memory/get", params))
            except RuntimeError as exc:
                respond(id_, error=str(exc))
        elif method == "flowdex.memory.put":
            try:
                respond(id_, _post("/memory/put", params))
            except RuntimeError as exc:
                respond(id_, error=str(exc))
        else:
            respond(id_, error=f"Unknown method: {method}")

if __name__ == "__main__":
    main()
