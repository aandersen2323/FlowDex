#!/usr/bin/env python3
"""FlowDex MCP-over-HTTP Bridge server."""
from __future__ import annotations

import json
import os
import time
from typing import Any, Dict, Optional

import requests
import uvicorn
from fastapi import FastAPI, HTTPException, Request

FLOWDEX_API_URL = os.environ.get("FLOWDEX_API_URL", "http://api:8787")
DEFAULT_TIMEOUT = float(os.environ.get("FLOWDEX_MCP_TIMEOUT", 30))
BRIDGE_PORT = int(os.environ.get("FLOWDEX_MCP_PORT", 8788))
BRIDGE_HOST = os.environ.get("FLOWDEX_MCP_HOST", "0.0.0.0")
API_KEY = os.environ.get("FLOWDEX_API_KEY")

app = FastAPI(title="FlowDex MCP Bridge", version="0.1.0")


class BridgeError(RuntimeError):
    """Raised when the bridge fails to call the FlowDex API."""


def call_api(
    method: str,
    endpoint: str,
    *,
    json_body: Optional[Dict[str, Any]] = None,
    params: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Forward a request to the FlowDex API and return the parsed JSON response."""

    url = f"{FLOWDEX_API_URL.rstrip('/')}{endpoint}"
    headers = {}
    if API_KEY:
        headers["x-api-key"] = API_KEY

    try:
        if method.upper() == "POST":
            response = requests.post(
                url,
                json=json_body or {},
                headers=headers,
                timeout=DEFAULT_TIMEOUT,
            )
        else:
            response = requests.get(
                url,
                params=params,
                headers=headers,
                timeout=DEFAULT_TIMEOUT,
            )
        response.raise_for_status()
    except requests.RequestException as exc:  # pragma: no cover - network failure
        raise BridgeError(f"FlowDex API request failed: {exc}") from exc

    try:
        return response.json()
    except json.JSONDecodeError as exc:  # pragma: no cover - invalid response
        raise BridgeError(f"Invalid JSON response from FlowDex API: {exc}") from exc


def mcp_respond(id_value: Any, *, result: Any = None, error: Optional[str] = None) -> Dict[str, Any]:
    """Format a JSON-RPC response payload."""

    message: Dict[str, Any] = {"jsonrpc": "2.0", "id": id_value}
    if error is not None:
        message["error"] = {"code": -32000, "message": error}
    else:
        message["result"] = result
    return message


@app.post("/mcp")
async def handle_mcp_request(request: Request) -> Dict[str, Any]:
    """Handle a JSON-RPC request from a Claude-Code MCP client."""

    try:
        payload = await request.json()
    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="Invalid JSON body")

    method = payload.get("method")
    request_id = payload.get("id")
    params = payload.get("params") or {}

    try:
        if method == "ping":
            return mcp_respond(request_id, result={"ok": True, "ts": time.time()})
        elif method == "flowdex.infer":
            return mcp_respond(
                request_id,
                result=call_api("POST", "/infer", json_body=params),
            )
        elif method == "flowdex.infer.retry":
            run_id = params.get("run_id")
            if not run_id:
                raise BridgeError("run_id is required for flowdex.infer.retry")
            retry_payload = {
                k: v
                for k, v in params.items()
                if k in {"error_context", "system_prompt_override", "user_input_override"}
            }
            return mcp_respond(
                request_id,
                result=call_api(
                    "POST",
                    f"/infer/{run_id}/retry",
                    json_body=retry_payload,
                ),
            )
        elif method == "flowdex.memory.get":
            return mcp_respond(
                request_id,
                result=call_api("GET", "/memory/get", params=params),
            )
        elif method == "flowdex.infer.auto_fix":
            return mcp_respond(
                request_id,
                result=call_api("POST", "/infer/auto-fix", json_body=params),
            )
        elif method == "flowdex.health":
            return mcp_respond(
                request_id,
                result=call_api("GET", "/health", params=params),
            )
        else:
            raise BridgeError(f"Unknown method: {method}")
    except BridgeError as exc:
        return mcp_respond(request_id, error=str(exc))
    except Exception as exc:  # pragma: no cover - defensive programming
        return mcp_respond(request_id, error=f"Unhandled bridge error: {exc}")


if __name__ == "__main__":
    print(
        f"Starting FlowDex MCP Bridge on {BRIDGE_HOST}:{BRIDGE_PORT}, forwarding to {FLOWDEX_API_URL}",
        flush=True,
    )
    uvicorn.run(app, host=BRIDGE_HOST, port=BRIDGE_PORT)
