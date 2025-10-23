#!/usr/bin/env python3
"""Small helper CLI to submit inference requests to a FlowDex server."""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any, Dict

import requests


DEFAULT_BASE = os.environ.get("FLOWDEX_BASE", "http://localhost:8787")
DEFAULT_MODEL = os.environ.get("FLOWDEX_MODEL", "anthropic/claude-3-5-sonnet")
DEFAULT_TIMEOUT = int(os.environ.get("FLOWDEX_TIMEOUT", "30"))


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Submit a FlowDex inference request")
    parser.add_argument("--base", default=DEFAULT_BASE, help="FlowDex server base URL")
    parser.add_argument("--task", default="general", help="Task label for the run")
    parser.add_argument("--user", default="", help="User input to send to the model")
    parser.add_argument(
        "--system",
        default="",
        help="System prompt that primes the model before the user message",
    )
    parser.add_argument(
        "--ctx",
        nargs="*",
        default=[],
        metavar="CONTEXT_ID",
        help="Existing memory context identifiers to include",
    )
    parser.add_argument(
        "--tool",
        dest="tools",
        nargs="*",
        default=[],
        metavar="TOOL_NAME",
        help="Tool identifiers to consider for the run",
    )
    parser.add_argument(
        "--retrieval-query",
        default=None,
        help="Optional free-text query used for semantic recall",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help="Model identifier to use for the run",
    )
    parser.add_argument(
        "--api-key",
        default=os.environ.get("FLOWDEX_API_KEY"),
        help="API key to send via the X-API-Key header",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=DEFAULT_TIMEOUT,
        help="HTTP timeout in seconds",
    )
    return parser


def _build_payload(args: argparse.Namespace) -> Dict[str, Any]:
    payload: Dict[str, Any] = {
        "task": args.task,
        "user_input": args.user,
        "system_prompt": args.system,
        "context_ids": args.ctx,
        "tool_candidates": args.tools,
        "retrieval_query": args.retrieval_query,
        "model": args.model,
    }
    return payload


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    payload = _build_payload(args)
    headers = {}
    if args.api_key:
        headers["X-API-Key"] = args.api_key

    try:
        response = requests.post(
            f"{args.base}/infer",
            json=payload,
            timeout=args.timeout,
            headers=headers or None,
        )
        response.raise_for_status()
    except requests.RequestException as exc:  # pragma: no cover - CLI convenience
        parser.error(str(exc))

    print(json.dumps(response.json(), indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
