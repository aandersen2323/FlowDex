import hashlib
import json
import logging
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from redis import Redis
from redis.exceptions import RedisError

CACHE_DIR = Path(os.environ.get("FLOWDEX_CACHE_DIR", ".flowdex_cache"))
CACHE_DIR.mkdir(parents=True, exist_ok=True)

logger = logging.getLogger("flowdex")
if not logger.handlers:
    logging.basicConfig(level=logging.INFO)

app = FastAPI(title="FlowDex API", version="0.1.0")

MEMORY: Dict[str, List[Dict[str, Any]]] = {}
TOOLS: Dict[str, Dict[str, Any]] = {}
RUNS: Dict[str, Dict[str, Any]] = {}
LAST_CONTEXT: Dict[str, str] = {}

REDIS_URL = os.environ.get("FLOWDEX_REDIS_URL", "redis://localhost:6379/0")
REDIS: Optional[Redis]
try:
    REDIS = Redis.from_url(REDIS_URL, decode_responses=True)
    REDIS.ping()
    logger.info("Connected to Redis at %s", REDIS_URL)
except RedisError as exc:
    REDIS = None
    logger.warning("Redis unavailable (%s); falling back to in-memory storage", exc)


class Budget(BaseModel):
    """Per-channel character budgets for prompt construction."""

    system: int = Field(default=int(os.environ.get("FLOWDEX_BUDGET_SYSTEM", 1000)))
    context: int = Field(default=int(os.environ.get("FLOWDEX_BUDGET_CONTEXT", 2500)))
    user: int = Field(default=int(os.environ.get("FLOWDEX_BUDGET_USER", 1500)))
    tools: int = Field(default=int(os.environ.get("FLOWDEX_BUDGET_TOOLS", 1000)))


class InferRequest(BaseModel):
    """Schema for `/infer` requests."""

    task: str
    user_input: str = ""
    system_prompt: str = ""
    context_ids: List[str] = []
    tool_candidates: List[str] = []
    retrieval_query: Optional[str] = None
    model: str = os.environ.get("FLOWDEX_MODEL", "anthropic/claude-3-5-sonnet")
    max_tokens: int = int(os.environ.get("FLOWDEX_MAX_TOKENS", 6000))
    budget: Budget = Budget()


def _hash(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()[:16]


def _context_cache_key(context_ids: List[str]) -> str:
    if not context_ids:
        return "context_none"
    joined = "|".join(sorted(context_ids))
    return f"context_{_hash(joined)}"


def _mem_key(identifier: str) -> str:
    return f"flowdex:mem:{identifier}"


def _last_context_key(cache_key: str) -> str:
    return f"flowdex:last_context:{cache_key}"


def _run_key(run_id: str) -> str:
    return f"flowdex:run:{run_id}"


def save_run(manifest: Dict[str, Any]) -> str:
    run_id = _hash(str(time.time()) + json.dumps(manifest, sort_keys=True))
    if REDIS is not None:
        REDIS.set(_run_key(run_id), json.dumps(manifest))
    else:
        RUNS[run_id] = manifest
    (CACHE_DIR / f"run_{run_id}.json").write_text(json.dumps(manifest, indent=2))
    return run_id


def json_patch(previous: str, current: str) -> Dict[str, Any]:
    prefix_len = 0
    max_prefix = min(len(previous), len(current))
    while prefix_len < max_prefix and previous[prefix_len] == current[prefix_len]:
        prefix_len += 1
    removed = previous[prefix_len:]
    added = current[prefix_len:]
    return {"common_prefix": prefix_len, "removed": removed, "added": added}


def truncate_to_budget(text: str, budget_chars: int) -> str:
    if budget_chars <= 0:
        return ""
    if len(text) <= budget_chars:
        return text
    return text[-budget_chars:]


def _call_anthropic(
    model: str,
    system_prompt: str,
    context_text: str,
    user_text: str,
    tools: List[Dict[str, Any]],
    max_tokens: int,
) -> Dict[str, Any]:
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise HTTPException(
            500,
            (
                "ANTHROPIC_API_KEY environment variable is required for Anthropic "
                "message completion. Set it before calling /infer."
            ),
        )

    timeout = float(os.environ.get("FLOWDEX_LLM_TIMEOUT", "60"))
    system_prompt = system_prompt or "You are FlowDex, a deterministic router prioritising tools before language output."

    user_sections: List[str] = []
    if context_text:
        user_sections.append(f"<context>\n{context_text}\n</context>")
    if user_text:
        user_sections.append(user_text)
    if not user_sections:
        user_sections.append("Proceed with the task using available tools and context.")

    payload: Dict[str, Any] = {
        "model": model,
        "max_tokens": max_tokens,
        "system": system_prompt,
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "\n\n".join(user_sections),
                    }
                ],
            }
        ],
    }

    if tools:
        payload["tools"] = tools
        payload["tool_choice"] = {"type": "auto"}

    try:
        response = requests.post(
            "https://api.anthropic.com/v1/messages",
            headers={
                "x-api-key": api_key,
                "anthropic-version": "2023-06-01",
                "content-type": "application/json",
            },
            json=payload,
            timeout=timeout,
        )
        response.raise_for_status()
    except requests.RequestException as exc:
        logger.error("Anthropic request failed: %s", exc)
        raise HTTPException(502, f"Anthropic request failed: {exc}")

    body = response.json()
    text_blocks = [
        block.get("text", "")
        for block in body.get("content", [])
        if isinstance(block, dict) and block.get("type") == "text"
    ]
    recommendation = "".join(text_blocks).strip()

    return {
        "recommendation": recommendation,
        "explanation": f"anthropic:{body.get('stop_reason', 'completed')}",
        "model": body.get("model", model),
        "usage": body.get("usage", {}),
    }


def call_llm(
    req: InferRequest,
    system_prompt: str,
    context_text: str,
    user_text: str,
    tools: List[Dict[str, Any]],
) -> Dict[str, Any]:
    parts = req.model.split("/", 1)
    if len(parts) == 1:
        provider = "anthropic"
        model = parts[0]
    else:
        provider, model = parts

    if provider == "anthropic":
        return _call_anthropic(model, system_prompt, context_text, user_text, tools, req.max_tokens)

    raise HTTPException(501, f"No LLM provider adapter configured for '{provider}'")

def _persist_last_context(cache_key: str, context_blob: str) -> None:
    if REDIS is not None:
        REDIS.set(_last_context_key(cache_key), context_blob)
    else:
        LAST_CONTEXT[cache_key] = context_blob


def _load_last_context(cache_key: str) -> str:
    if REDIS is not None:
        cached = REDIS.get(_last_context_key(cache_key))
        if cached is not None:
            return cached
    return LAST_CONTEXT.get(cache_key, "")


def _memory_versions(identifier: str) -> List[Dict[str, Any]]:
    if REDIS is not None:
        key = _mem_key(identifier)
        raw_versions = REDIS.lrange(key, 0, -1)
        versions: List[Dict[str, Any]] = []
        for raw in raw_versions:
            try:
                versions.append(json.loads(raw))
            except json.JSONDecodeError:
                continue
        return list(reversed(versions))
    return MEMORY.get(identifier, [])


def _memory_latest(identifier: str) -> Optional[Dict[str, Any]]:
    if REDIS is not None:
        raw = REDIS.lindex(_mem_key(identifier), 0)
        if raw is None:
            return None
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            return None
    versions = MEMORY.get(identifier, [])
    if not versions:
        return None
    return versions[-1]


def _store_memory_version(identifier: str, data: str) -> int:
    record = {"ts": time.time(), "data": data}
    if REDIS is not None:
        REDIS.lpush(_mem_key(identifier), json.dumps(record))
        return REDIS.llen(_mem_key(identifier))
    versions = MEMORY.setdefault(identifier, [])
    versions.append(record)
    return len(versions)


def _load_tool(name: str) -> Optional[Dict[str, Any]]:
    if REDIS is not None:
        raw = REDIS.hget("flowdex:tools", name)
        if raw is None:
            return None
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            return None
    return TOOLS.get(name)


def _persist_tool(name: str, spec: Dict[str, Any]) -> int:
    if REDIS is not None:
        REDIS.hset("flowdex:tools", name, json.dumps(spec, sort_keys=True))
        return REDIS.hlen("flowdex:tools")
    TOOLS[name] = spec
    return len(TOOLS)


def _load_run(run_id: str) -> Optional[Dict[str, Any]]:
    if REDIS is not None:
        raw = REDIS.get(_run_key(run_id))
        if raw is None:
            return None
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            return None
    return RUNS.get(run_id)


@app.post("/memory/put")
def memory_put(item: Dict[str, Any]):
    id_ = item.get("id")
    if not id_:
        raise HTTPException(400, "id is required")
    versions = _store_memory_version(id_, item.get("data", ""))
    return {"ok": True, "versions": versions}

@app.get("/memory/get")
def memory_get(id: str, version: Optional[int] = None):
    versions = _memory_versions(id)
    if not versions:
        raise HTTPException(404, "not found")
    if version is None or version >= len(versions):
        return versions[-1]
    return versions[version]

@app.post("/tools/register")
def register_tool(spec: Dict[str, Any]):
    name = spec.get("name")
    if not name:
        raise HTTPException(400, "tool name required")
    count = _persist_tool(name, spec)
    return {"ok": True, "count": count}

@app.get("/runs/{run_id}")
def get_run(run_id: str):
    r = _load_run(run_id)
    if not r:
        raise HTTPException(404, "run not found")
    return r

@app.post("/infer")
def infer(req: InferRequest):
    context_blob_parts: List[str] = []
    for cid in req.context_ids:
        latest = _memory_latest(cid)
        if latest:
            context_blob_parts.append(f"<ctx id=\"{cid}\">\n{latest['data']}\n</ctx>")

    retrieved_parts: List[str] = []
    if req.retrieval_query:
        query = req.retrieval_query.lower()
        if REDIS is not None:
            for key in REDIS.scan_iter("flowdex:mem:*"):
                raw = REDIS.lindex(key, 0)
                if raw is None:
                    continue
                try:
                    payload = json.loads(raw)
                except json.JSONDecodeError:
                    continue
                body = payload.get("data", "")
                if query in body.lower():
                    ctx_id = key.split(":", 2)[-1]
                    retrieved_parts.append(f"<ctx id=\"retrieved:{ctx_id}\">\n{body}\n</ctx>")
        else:
            for cid, versions in MEMORY.items():
                if not versions:
                    continue
                body = versions[-1]["data"]
                if query in body.lower():
                    retrieved_parts.append(f"<ctx id=\"retrieved:{cid}\">\n{body}\n</ctx>")

    if retrieved_parts:
        context_blob_parts.extend(retrieved_parts)

    context_blob = "\n".join(context_blob_parts)
    cache_key = _context_cache_key(req.context_ids)
    cache_path = CACHE_DIR / f"{cache_key}.txt"

    cached_context = _load_last_context(cache_key)
    if cached_context:
        prior_text = cached_context
    elif cache_path.exists():
        prior_text = cache_path.read_text()
    else:
        prior_text = ""

    patch = json_patch(prior_text, context_blob)
    _persist_last_context(cache_key, context_blob)
    cache_path.write_text(context_blob)

    sys_b = truncate_to_budget(req.system_prompt, req.budget.system)
    ctx_b = truncate_to_budget(context_blob, req.budget.context)
    usr_b = truncate_to_budget(req.user_input, req.budget.user)

    chosen_tools: List[Dict[str, Any]] = []
    for name in req.tool_candidates:
        spec = _load_tool(name)
        if spec:
            chosen_tools.append(spec)
    trimmed_tools: List[Dict[str, Any]] = []
    remaining_tool_budget = req.budget.tools
    for tool_spec in chosen_tools:
        serialized = json.dumps(tool_spec, sort_keys=True)
        if len(serialized) > remaining_tool_budget and trimmed_tools:
            break
        trimmed_tools.append(tool_spec)
        remaining_tool_budget = max(0, remaining_tool_budget - len(serialized))

    prompt = {
        "system": sys_b,
        "context": ctx_b,
        "user": usr_b,
        "delta": patch,
        "tools": trimmed_tools,
        "model": req.model,
        "max_tokens": req.max_tokens,
    }

    output = call_llm(req, sys_b, ctx_b, usr_b, trimmed_tools)

    manifest = {
        "request": json.loads(req.json()),
        "prompt": prompt,
        "output": output,
        "ts": time.time(),
    }
    run_id = save_run(manifest)
    return {
        "run_id": run_id,
        "output": output,
        "budgets": req.budget.dict(),
        "used_context_chars": len(ctx_b),
        "delta": patch,
        "tools_considered": [tool.get("name") for tool in trimmed_tools],
    }
