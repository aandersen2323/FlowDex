import hashlib
import json
import logging
import re
import time
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional, Tuple

import redis
import requests
from fastapi import Depends, FastAPI, Header, HTTPException, Query, Request
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field
from redis.exceptions import RedisError
from pydantic_settings import BaseSettings, SettingsConfigDict

try:  # pragma: no cover - optional dependency
    import jinja2  # type: ignore
except ImportError:  # pragma: no cover - degrade gracefully when templates unavailable
    jinja2 = None  # type: ignore

from .components import (
    BaseSemanticRecall,
    BaseTokenizer,
    DefaultTokenizer,
    RedisTFIDFSemanticRecall,
    instantiate_component,
)


logger = logging.getLogger(__name__)

class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )

    flowdex_port: int = 8787
    flowdex_model: str = "anthropic/claude-3-5-sonnet"
    flowdex_cache_dir: Path = Path(".flowdex_cache")
    flowdex_max_tokens: int = 6000
    flowdex_budget_system: int = 1000
    flowdex_budget_context: int = 2500
    flowdex_budget_user: int = 1500
    flowdex_budget_tools: int = 1000
    flowdex_api_key: Optional[str] = None
    flowdex_anthropic_timeout: int = 60
    flowdex_redis_url: Optional[str] = None
    flowdex_redis_host: str = "localhost"
    flowdex_redis_port: int = 6379
    flowdex_redis_db: int = 0
    flowdex_tokenizer_impl: Optional[str] = None
    flowdex_semantic_recall_impl: Optional[str] = None
    anthropic_api_key: Optional[str] = None
    flowdex_anthropic_url: str = "https://api.anthropic.com/v1/messages"
    flowdex_anthropic_version: str = "2023-06-01"


settings = Settings()

CACHE_DIR = settings.flowdex_cache_dir
CACHE_DIR.mkdir(parents=True, exist_ok=True)

app = FastAPI(title="FlowDex API", version="0.1.0")
if jinja2 is not None:
    templates: Optional[Jinja2Templates] = Jinja2Templates(
        directory=str((Path(__file__).parent / "templates").resolve())
    )
else:  # pragma: no cover - executed only when optional dependency missing
    templates = None

_REDIS_CLIENT: Optional[redis.Redis] = None
TOKENIZER: BaseTokenizer = instantiate_component(
    settings.flowdex_tokenizer_impl, DefaultTokenizer
)
_SEMANTIC_ENGINE: Optional[BaseSemanticRecall] = None

MEMORY_KEY_PREFIX = "flowdex:memory:"
MEMORY_INDEX_KEY = "flowdex:memory:index"
TOOLS_KEY = "flowdex:tools"
RUNS_KEY = "flowdex:runs"
RUN_TAGS_KEY = "flowdex:run_tags"
RUN_TAG_INDEX_PREFIX = "flowdex:run_tag:"
RUN_METADATA_KEY = "flowdex:run_meta"
USAGE_KEY_PREFIX = "flowdex:usage:"
API_KEY = settings.flowdex_api_key
ANTHROPIC_TIMEOUT = settings.flowdex_anthropic_timeout

AUTO_FIX_SYSTEM_PROMPT = """You are FlowDex AutoFix, an autonomous incident responder for automation workflows.\n" \
    "You triage failing orchestrations by summarizing errors, identifying likely root causes, " \
    "and prescribing concrete remediation steps that can be executed without human input when possible.\n" \
    "Always respond with a single JSON object containing the following keys:\n" \
    "  - status: one of ['resolved', 'apply_fix', 'unrecoverable', 'needs_human_review'].\n" \
    "  - summary: short natural-language description of what happened.\n" \
    "  - actions: array of actionable steps to remediate the failure.\n" \
    "            When recommending automation tools, include objects such as\n" \
    "            {\"type\": \"tool\", \"tool\": \"<registered_tool_name>\", \"input\": {...}}.\n" \
    "  - fix_instructions: detailed instructions or code patches to apply.\n" \
    "  - metrics: object with optional numeric metadata such as confidence (0-1) and\n" \
    "             estimated_minutes.\n" \
    "If additional context or manual approval is required, return status 'needs_human_review' and describe\n" \
    "what is needed. If information is insufficient, set status to 'unrecoverable' and explain\n" \
    "what extra data is required. Keep responses concise but information-dense."""

_VALID_AUTO_FIX_STATUSES = {"resolved", "apply_fix", "unrecoverable", "needs_human_review"}
DEFAULT_TOOL_TIMEOUT = 30


def get_redis() -> redis.Redis:
    global _REDIS_CLIENT
    if _REDIS_CLIENT is not None:
        return _REDIS_CLIENT

    redis_url = settings.flowdex_redis_url
    try:
        if redis_url:
            client = redis.Redis.from_url(redis_url, decode_responses=True)
        else:
            client = redis.Redis(
                host=settings.flowdex_redis_host,
                port=settings.flowdex_redis_port,
                db=settings.flowdex_redis_db,
                decode_responses=True,
            )
        client.ping()
    except RedisError as exc:
        raise HTTPException(500, detail=f"Unable to connect to Redis: {exc}") from exc

    _REDIS_CLIENT = client
    return client


def get_semantic_engine() -> BaseSemanticRecall:
    global _SEMANTIC_ENGINE
    if _SEMANTIC_ENGINE is not None:
        return _SEMANTIC_ENGINE

    try:
        _SEMANTIC_ENGINE = instantiate_component(
            settings.flowdex_semantic_recall_impl,
            RedisTFIDFSemanticRecall,
            redis_getter=get_redis,
            tokenizer=TOKENIZER,
            memory_index_key=MEMORY_INDEX_KEY,
            memory_key_prefix=MEMORY_KEY_PREFIX,
        )
    except Exception as exc:  # pragma: no cover - defensive guard
        logger.error("Failed to initialize semantic recall engine", exc_info=True)
        raise HTTPException(500, detail=f"Unable to initialize semantic recall engine: {exc}") from exc
    return _SEMANTIC_ENGINE


@app.get("/health")
def health_check():
    """Return the health status of the API service and backing Redis."""

    try:
        client = get_redis()
        client.ping()
        redis_ok = True
    except Exception:  # pragma: no cover - defensive
        redis_ok = False

    return {
        "status": "ok" if redis_ok else "degraded",
        "redis_connected": redis_ok,
        "version": app.version,
    }


class Budget(BaseModel):
    system: int = Field(default=settings.flowdex_budget_system)
    context: int = Field(default=settings.flowdex_budget_context)
    user: int = Field(default=settings.flowdex_budget_user)
    tools: int = Field(default=settings.flowdex_budget_tools)


class InferRequest(BaseModel):
    task: str
    user_input: str = ""
    system_prompt: str = ""
    context_ids: List[str] = Field(default_factory=list)
    tool_candidates: List[str] = Field(default_factory=list)
    retrieval_query: Optional[str] = None
    model: str = settings.flowdex_model
    max_tokens: int = settings.flowdex_max_tokens
    budget: Budget = Field(default_factory=Budget)
    tags: List[str] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    stream: bool = False
    usage_identifier: Optional[str] = None


class RetryRequest(BaseModel):
    """Payload used to retry a previous inference run."""

    error_context: str = Field(
        ...,
        description="The error message or log output from the failed attempt.",
    )
    system_prompt_override: Optional[str] = None
    user_input_override: Optional[str] = None


class AutoFixRequest(BaseModel):
    """Request payload for the automated troubleshooting workflow."""

    error_log: str = Field(
        ...,
        min_length=1,
        description="Raw error output from the failing workflow run.",
    )
    run_id: Optional[str] = Field(
        default=None,
        description="Optional run identifier to reuse a prior /infer invocation as the base request.",
    )
    request: Optional[InferRequest] = Field(
        default=None,
        description="Explicit request payload to use when no run_id is supplied.",
    )
    max_attempts: int = Field(
        default=2,
        ge=1,
        le=5,
        description="Number of automated remediation attempts to perform.",
    )
    task_override: Optional[str] = Field(
        default=None,
        description="Optional replacement task description for the remediation attempts.",
    )
    system_prompt_override: Optional[str] = Field(
        default=None,
        description="Override the system prompt used during remediation analysis and fix runs.",
    )
    user_input_override: Optional[str] = Field(
        default=None,
        description="Override the user input used for fix attempts.",
    )
    usage_identifier: Optional[str] = Field(
        default=None,
        description="Optional usage identity for tracking token consumption.",
    )


def require_api_key(x_api_key: Optional[str] = Header(default=None)) -> str:
    """Validate the provided API key if one is configured and return the identifier used."""

    if API_KEY and x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API key")
    return x_api_key or "anonymous"


def _hash(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()[:10]


def save_run(manifest: Dict[str, Any]) -> str:
    run_id = _hash(str(time.time()) + json.dumps(manifest, sort_keys=True))
    client = get_redis()
    tags = _normalize_tags(manifest.get("tags", []))
    metadata = manifest.get("metadata") or {}
    try:
        client.hset(RUNS_KEY, run_id, json.dumps(manifest, default=str))
        client.hset(RUN_TAGS_KEY, run_id, json.dumps(tags, default=str))
        client.hset(RUN_METADATA_KEY, run_id, json.dumps(metadata, default=str))
        for tag in tags:
            client.sadd(f"{RUN_TAG_INDEX_PREFIX}{tag}", run_id)
    except RedisError as exc:
        raise HTTPException(500, detail=f"Failed to persist run: {exc}") from exc
    (CACHE_DIR / f"run_{run_id}.json").write_text(json.dumps(manifest, indent=2, default=str))
    return run_id


def json_patch(old: str, new: str) -> Dict[str, Any]:
    prefix_len = 0
    while prefix_len < min(len(old), len(new)) and old[prefix_len] == new[prefix_len]:
        prefix_len += 1
    return {"common_prefix": prefix_len, "added": new[prefix_len:]}


def _normalize_tags(tags: Iterable[str]) -> List[str]:
    seen = set()
    normalized: List[str] = []
    for tag in tags or []:
        if not isinstance(tag, str):
            continue
        cleaned = tag.strip()
        if not cleaned:
            continue
        key = cleaned.lower()
        if key in seen:
            continue
        seen.add(key)
        normalized.append(cleaned)
    return normalized


def _record_usage(
    identity: Optional[str], tokens_used: Dict[str, int], output: Dict[str, Any], run_id: str
) -> None:
    if not identity:
        identity = "anonymous"

    try:
        prompt_tokens = int(sum(int(v) for v in tokens_used.values()))
    except Exception:  # pragma: no cover - defensive conversion
        prompt_tokens = 0

    usage_data = output.get("usage") if isinstance(output, dict) else {}
    completion_tokens = 0
    if isinstance(usage_data, dict):
        for key in ("output_tokens", "completion_tokens", "completion"):
            value = usage_data.get(key)
            if value is not None:
                try:
                    completion_tokens = int(value)
                    break
                except (TypeError, ValueError):
                    continue
    if completion_tokens <= 0:
        completion_tokens = count_tokens(output.get("completion", ""))

    key = f"{USAGE_KEY_PREFIX}{identity}"
    client = get_redis()
    if not hasattr(client, "hincrby"):
        return
    try:
        client.hincrby(key, "requests", 1)
        client.hincrby(key, "prompt_tokens", max(prompt_tokens, 0))
        client.hincrby(key, "completion_tokens", max(completion_tokens, 0))
        client.hset(key, mapping={"last_run_id": run_id, "updated_at": str(time.time())})
    except RedisError as exc:  # pragma: no cover - non-critical tracking failure
        logger.warning(
            "Failed to record usage statistics", extra={"identity": identity, "run_id": run_id}
        )


def count_tokens(text: str) -> int:
    """Delegate token counting to the configured tokenizer."""

    return TOKENIZER.count_tokens(text)


def truncate_to_token_budget(text: str, budget_tokens: int) -> Tuple[str, int]:
    """Truncate ``text`` to fit the supplied token budget using the configured tokenizer."""

    return TOKENIZER.truncate(text, budget_tokens)


def _safe_json_dumps(value: Any) -> str:
    """Serialize a value to JSON, falling back to ``repr`` when necessary."""

    try:
        return json.dumps(value, sort_keys=True)
    except TypeError:
        # ``repr`` keeps debugging information without raising an exception that would
        # otherwise abort the inference run when a tool registers a complex schema.
        return repr(value)


def build_tool_context(tool_specs: List[Dict[str, Any]]) -> str:
    if not tool_specs:
        return ""
    serialized = []
    for spec in tool_specs:
        name = spec.get("name", "unknown")
        description = spec.get("description", "")
        cost = spec.get("cost", "")
        schema = spec.get("schema", {})
        serialized.append(
            "\n".join(
                (
                    f"Tool: {name}",
                    f"Description: {description}",
                    f"Cost: {cost}",
                    f"Schema: {_safe_json_dumps(schema)}",
                )
            )
        )
    blob = "\n\n".join(serialized)
    return blob


def _tokenize(text: str) -> List[str]:
    return list(TOKENIZER.tokenize_semantic(text))


def _latest_memory_version(client: redis.Redis, memory_id: str) -> Optional[Dict[str, Any]]:
    try:
        payload = client.lindex(f"{MEMORY_KEY_PREFIX}{memory_id}", -1)
    except RedisError as exc:
        raise HTTPException(500, detail=f"Failed to load memory {memory_id}: {exc}") from exc
    if not payload:
        return None
    try:
        return json.loads(payload)
    except json.JSONDecodeError:
        return {"data": payload, "ts": None}


def semantic_recall(query: str, exclude_ids: List[str], limit: int = 3) -> List[Dict[str, Any]]:
    if not query or not query.strip():
        return []

    engine = get_semantic_engine()
    try:
        return engine.recall(query, exclude_ids, limit)
    except RuntimeError as exc:
        raise HTTPException(500, detail=str(exc)) from exc


def call_llm(
    prompt: Dict[str, Any],
    request: InferRequest,
    tool_specs: List[Dict[str, Any]],
    retrieved: List[Dict[str, Any]],
    stream: bool = False,
) -> Dict[str, Any]:
    api_key = settings.anthropic_api_key
    if not api_key:
        raise HTTPException(500, detail="ANTHROPIC_API_KEY environment variable is required for inference")

    base_url = settings.flowdex_anthropic_url

    segments: List[str] = []
    if request.task:
        segments.append(f"Task:\n{request.task}")
    if prompt.get("context"):
        segments.append(f"<context>\n{prompt['context']}\n</context>")
    if retrieved:
        for r in retrieved:
            segments.append(
                f"<retrieval id=\"{r['id']}\" score=\"{r['score']}\">\n{r['data']}\n</retrieval>"
            )
    delta = prompt.get("delta", {})
    if delta.get("added"):
        segments.append(
            f"<delta common_prefix={delta.get('common_prefix', 0)}>\n{delta['added']}\n</delta>"
        )
    if prompt.get("user"):
        segments.append(f"<user_input>\n{prompt['user']}\n</user_input>")

    tool_context_blob = build_tool_context(tool_specs)
    tool_context, tool_tokens = truncate_to_token_budget(
        tool_context_blob, request.budget.tools
    )
    if tool_context:
        segments.append(f"<tools>\n{tool_context}\n</tools>")
    if "tokens_used" in prompt:
        prompt["tokens_used"]["tools"] = tool_tokens

    user_message = "\n\n".join(seg for seg in segments if seg)
    if not user_message.strip():
        user_message = "Respond to the provided task."

    payload = {
        "model": prompt["model"],
        "max_tokens": prompt["max_tokens"],
        "system": prompt.get("system", ""),
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": user_message,
                    }
                ],
            }
        ],
    }

    headers = {
        "x-api-key": api_key,
        "anthropic-version": settings.flowdex_anthropic_version,
        "content-type": "application/json",
    }

    try:
        response = requests.post(
            base_url, headers=headers, json=payload, timeout=ANTHROPIC_TIMEOUT
        )
        response.raise_for_status()
    except requests.RequestException as exc:
        status = getattr(getattr(exc, "response", None), "status_code", None)
        response_excerpt: Optional[str]
        if getattr(exc, "response", None) is not None:
            try:
                response_excerpt = exc.response.text[:500]
            except Exception:  # pragma: no cover - defensive
                response_excerpt = None
        else:
            response_excerpt = None
        logger.error(
            "Anthropic API request failed",
            extra={
                "url": base_url,
                "status": status,
                "payload_model": payload.get("model"),
                "payload_max_tokens": payload.get("max_tokens"),
                "response_excerpt": response_excerpt,
            },
            exc_info=True,
        )
        raise HTTPException(502, detail=f"Anthropic API request failed: {exc}") from exc

    try:
        data = response.json()
    except ValueError as exc:
        logger.error(
            "Anthropic API response was not valid JSON",
            extra={"url": base_url, "status": response.status_code},
            exc_info=True,
        )
        raise HTTPException(502, detail=f"Anthropic API response was not valid JSON: {exc}") from exc
    content = data.get("content", [])
    text_parts = [part.get("text", "") for part in content if part.get("type") == "text"]
    completion = "\n".join(part for part in text_parts if part)

    completion_tokens = list(TOKENIZER.iter_tokens(completion))

    return {
        "completion": completion,
        "completion_tokens": completion_tokens,
        "stop_reason": data.get("stop_reason"),
        "usage": data.get("usage", {}),
        "model": data.get("model", prompt["model"]),
        "raw": data,
    }


@app.post("/memory/put")
def memory_put(item: Dict[str, Any], authorized: str = Depends(require_api_key)):
    id_ = item.get("id")
    if not id_:
        raise HTTPException(400, "id is required")
    payload = {
        "ts": time.time(),
        "data": item.get("data", ""),
        "meta": {k: v for k, v in item.items() if k not in {"id", "data"}},
    }
    client = get_redis()
    key = f"{MEMORY_KEY_PREFIX}{id_}"
    try:
        client.rpush(key, json.dumps(payload))
        client.sadd(MEMORY_INDEX_KEY, id_)
        versions = client.llen(key)
    except RedisError as exc:
        raise HTTPException(500, detail=f"Failed to persist memory: {exc}") from exc
    return {"ok": True, "versions": versions}


@app.get("/memory/get")
def memory_get(id: str, version: Optional[int] = None, authorized: str = Depends(require_api_key)):
    client = get_redis()
    key = f"{MEMORY_KEY_PREFIX}{id}"
    try:
        total = client.llen(key)
    except RedisError as exc:
        raise HTTPException(500, detail=f"Failed to read memory index: {exc}") from exc
    if total == 0:
        raise HTTPException(404, "not found")

    if version is None or version >= total:
        index = -1
    else:
        index = version

    try:
        payload = client.lindex(key, index)
    except RedisError as exc:
        raise HTTPException(500, detail=f"Failed to read memory: {exc}") from exc

    if payload is None:
        raise HTTPException(404, "version not found")
    try:
        return json.loads(payload)
    except json.JSONDecodeError:
        return {"ts": None, "data": payload}


def _list_memory_entries(limit: int = 100) -> List[Dict[str, Any]]:
    client = get_redis()
    try:
        memory_ids = sorted(client.smembers(MEMORY_INDEX_KEY))
    except RedisError as exc:
        raise HTTPException(500, detail=f"Failed to enumerate memory: {exc}") from exc

    entries: List[Dict[str, Any]] = []
    for memory_id in memory_ids[:limit]:
        latest = _latest_memory_version(client, memory_id)
        if not latest:
            continue
        data = latest.get("data", "")
        preview = str(data)[:200]
        entries.append({"id": memory_id, "ts": latest.get("ts"), "preview": preview})
    return entries


@app.post("/tools/register")
def register_tool(spec: Dict[str, Any], authorized: str = Depends(require_api_key)):
    name = spec.get("name")
    if not name:
        raise HTTPException(400, "tool name required")
    client = get_redis()
    try:
        client.hset(TOOLS_KEY, name, json.dumps(spec))
        count = client.hlen(TOOLS_KEY)
    except RedisError as exc:
        raise HTTPException(500, detail=f"Failed to register tool: {exc}") from exc
    return {"ok": True, "count": count}


def _list_runs(tag: Optional[str], limit: int) -> List[Dict[str, Any]]:
    client = get_redis()
    try:
        if tag:
            normalized_tag = _normalize_tags([tag])
            tag_key = normalized_tag[0] if normalized_tag else tag
            run_ids = list(client.smembers(f"{RUN_TAG_INDEX_PREFIX}{tag_key}"))
        else:
            run_ids = client.hkeys(RUNS_KEY)
    except RedisError as exc:
        raise HTTPException(500, detail=f"Failed to enumerate runs: {exc}") from exc

    runs: List[Dict[str, Any]] = []
    for run_id in run_ids:
        try:
            payload = client.hget(RUNS_KEY, run_id)
        except RedisError as exc:
            raise HTTPException(500, detail=f"Failed to load run {run_id}: {exc}") from exc
        if not payload:
            continue
        try:
            manifest = json.loads(payload)
        except json.JSONDecodeError:
            continue
        request_payload = manifest.get("request") or {}
        runs.append(
            {
                "run_id": run_id,
                "task": request_payload.get("task"),
                "tags": manifest.get("tags", []),
                "metadata": manifest.get("metadata", {}),
                "ts": manifest.get("ts"),
                "model": request_payload.get("model"),
            }
        )

    runs.sort(key=lambda item: item.get("ts") or 0, reverse=True)
    if tag:
        normalized_tag = _normalize_tags([tag])
        tag_key = normalized_tag[0] if normalized_tag else tag
        runs = [run for run in runs if tag_key in _normalize_tags(run.get("tags", []))]
    return runs[:limit]


@app.get("/runs/{run_id}")
def get_run(run_id: str, authorized: str = Depends(require_api_key)):
    client = get_redis()
    try:
        payload = client.hget(RUNS_KEY, run_id)
    except RedisError as exc:
        raise HTTPException(500, detail=f"Failed to load run: {exc}") from exc
    if not payload:
        raise HTTPException(404, "run not found")
    try:
        return json.loads(payload)
    except json.JSONDecodeError:
        raise HTTPException(500, detail="Stored run data is corrupted")


@app.get("/usage/{identity}")
def get_usage(identity: str, authorized: str = Depends(require_api_key)):
    client = get_redis()
    try:
        data = client.hgetall(f"{USAGE_KEY_PREFIX}{identity}")
    except RedisError as exc:
        raise HTTPException(500, detail=f"Failed to load usage: {exc}") from exc
    if not data:
        raise HTTPException(404, detail="usage not found")

    usage: Dict[str, Any] = {}
    for key, value in data.items():
        if key in {"requests", "prompt_tokens", "completion_tokens"}:
            try:
                usage[key] = int(value)
            except (TypeError, ValueError):
                usage[key] = value
        else:
            usage[key] = value
    return {"identity": identity, "usage": usage}


@app.get("/runs")
def list_runs(
    tag: Optional[str] = Query(default=None),
    limit: int = Query(default=20, ge=1, le=200),
    authorized: str = Depends(require_api_key),
):
    runs = _list_runs(tag, limit)
    return {"runs": runs, "count": len(runs), "tag": tag, "limit": limit}


_CODE_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL)


def _extract_auto_fix_json(text: str) -> Dict[str, Any]:
    text = (text or "").strip()
    if not text:
        return {}

    try:
        return json.loads(text, strict=False)
    except json.JSONDecodeError:
        pass

    candidates: List[str] = []

    fence_match = _CODE_FENCE_RE.search(text)
    if fence_match:
        fence_payload = fence_match.group(1).strip()
        if fence_payload:
            candidates.append(fence_payload)

    first_brace = text.find("{")
    last_brace = text.rfind("}")
    if first_brace != -1 and last_brace != -1 and last_brace > first_brace:
        candidates.append(text[first_brace : last_brace + 1].strip())

    candidates.append(text)

    for candidate in candidates:
        candidate = candidate.strip()
        if not candidate:
            continue
        try:
            return json.loads(candidate, strict=False)
        except json.JSONDecodeError:
            continue

    logger.warning(
        "Could not parse JSON from auto-fix completion", extra={"preview": text[:100]}
    )
    return {}


def _parse_auto_fix_completion(text: str) -> Dict[str, Any]:
    parsed = _extract_auto_fix_json(text)
    if not parsed:
        return {
            "status": "unparsed",
            "summary": (text or "").strip(),
            "actions": [],
            "fix_instructions": "",
            "metrics": {},
            "parsed": None,
            "raw": text,
        }

    status = str(parsed.get("status", "")).lower().strip()
    if status not in _VALID_AUTO_FIX_STATUSES:
        status = "unparsed"

    actions_raw = parsed.get("actions") or parsed.get("action_items") or []
    action_objects: List[Dict[str, Any]] = []
    if isinstance(actions_raw, str):
        actions = [actions_raw.strip()]
    else:
        actions = []
        for item in actions_raw:
            if item is None:
                continue
            if isinstance(item, dict):
                action_objects.append(item)
                summary = item.get("description") or item.get("summary")
                if summary:
                    actions.append(str(summary))
                    continue
            actions.append(str(item))

    fix_instructions = parsed.get("fix_instructions") or parsed.get("resolution") or ""
    summary = parsed.get("summary") or parsed.get("analysis") or ""
    metrics = parsed.get("metrics") or {}

    return {
        "status": status,
        "summary": summary,
        "actions": actions,
        "action_objects": action_objects,
        "fix_instructions": fix_instructions,
        "metrics": metrics,
        "parsed": parsed,
        "raw": text,
    }


def _format_prior_auto_fix(prior: List[Dict[str, Any]]) -> str:
    if not prior:
        return "None. This is the first automated remediation attempt."

    blocks: List[str] = []
    for idx, item in enumerate(prior, start=1):
        status = item.get("status", "unknown")
        summary = item.get("summary", "")
        structured = item.get("parsed") or {
            "status": status,
            "summary": summary,
            "actions": item.get("actions", []),
        }
        blocks.append(
            "\n".join(
                (
                    f"Attempt {idx} status: {status}",
                    f"Summary: {summary}",
                    "Structured response:",
                    json.dumps(structured, indent=2, sort_keys=True),
                )
            )
        )
    return "\n\n".join(blocks)


def _format_actions(actions: List[str]) -> str:
    if not actions:
        return "None provided."
    return "\n".join(f"- {step}" for step in actions)


def _resolve_auto_fix_base_request(auto_req: AutoFixRequest) -> InferRequest:
    if auto_req.run_id:
        client = get_redis()
        try:
            payload = client.hget(RUNS_KEY, auto_req.run_id)
        except RedisError as exc:
            raise HTTPException(500, detail=f"Failed to load original run: {exc}") from exc
        if not payload:
            raise HTTPException(404, detail="Original run not found")
        try:
            manifest = json.loads(payload)
        except json.JSONDecodeError as exc:
            raise HTTPException(500, detail="Stored run data is corrupted") from exc
        original_req_dict = manifest.get("request") or {}
        try:
            base_req = InferRequest(**original_req_dict)
        except Exception as exc:
            raise HTTPException(400, detail=f"Stored request is invalid: {exc}") from exc
    elif auto_req.request is not None:
        base_req = auto_req.request.copy(deep=True)
    else:
        raise HTTPException(
            400,
            detail="Either run_id or request must be supplied for automated remediation.",
        )

    base_req = base_req.copy(deep=True)

    if auto_req.task_override is not None:
        base_req.task = auto_req.task_override
    if auto_req.system_prompt_override is not None:
        base_req.system_prompt = auto_req.system_prompt_override
    if auto_req.user_input_override is not None:
        base_req.user_input = auto_req.user_input_override

    return base_req


def _build_auto_fix_analysis_request(
    base_req: InferRequest,
    auto_req: AutoFixRequest,
    attempt: int,
    prior: List[Dict[str, Any]],
) -> InferRequest:
    analysis_req = base_req.copy(deep=True)
    analysis_req.task = f"Diagnose workflow failure for: {base_req.task}"
    analysis_req.system_prompt = auto_req.system_prompt_override or AUTO_FIX_SYSTEM_PROMPT

    history_blob = _format_prior_auto_fix(prior)
    original_user_input = base_req.user_input or "(empty)"
    user_sections = [
        f"Automated remediation attempt {attempt} of {auto_req.max_attempts}.",
        "Summarize the failure, determine root cause, and produce actionable remediation steps.",
        "--- Latest Error Log ---",
        auto_req.error_log,
        "--- Original User Input ---",
        original_user_input,
        "--- Prior Analyses ---",
        history_blob,
    ]
    analysis_req.user_input = "\n\n".join(section for section in user_sections if section)

    return analysis_req


def _build_auto_fix_fix_request(
    base_req: InferRequest,
    auto_req: AutoFixRequest,
    analysis: Dict[str, Any],
) -> InferRequest:
    fix_req = base_req.copy(deep=True)
    if auto_req.system_prompt_override:
        fix_req.system_prompt = auto_req.system_prompt_override

    base_input = fix_req.user_input or ""
    actions_blob = _format_actions([str(step) for step in analysis.get("actions", [])])
    plan_blob = _safe_json_dumps(analysis.get("parsed") or {})
    remediation = analysis.get("fix_instructions") or analysis.get("summary") or ""
    tool_results_blob = _safe_json_dumps(analysis.get("tool_results") or [])

    fix_req.user_input = "\n\n".join(
        section
        for section in (
            base_input,
            "<auto_fix_context>",
            f"Latest error log:\n{auto_req.error_log}",
            f"Analysis summary:\n{analysis.get('summary', '')}",
            f"Action steps:\n{actions_blob}",
            f"Tool execution results:\n{tool_results_blob}",
            f"Remediation plan:\n{remediation}",
            f"Structured response:\n{plan_blob}",
            "</auto_fix_context>",
            "Implement the remediation plan above and provide the corrected output.",
        )
        if section
    )

    return fix_req


def _execute_tool_action(tool_name: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    client = get_redis()
    try:
        stored = client.hget(TOOLS_KEY, tool_name)
    except RedisError as exc:
        return {"status": "error", "error": f"Failed to load tool: {exc}"}

    if not stored:
        return {"status": "error", "error": "Tool not registered"}

    try:
        spec = json.loads(stored)
    except json.JSONDecodeError:
        spec = {"name": tool_name}

    runner = spec.get("runner") or {}
    if not runner and spec.get("endpoint"):
        runner = {
            "type": "http",
            "url": spec.get("endpoint"),
            "method": spec.get("method", "POST"),
            "headers": spec.get("headers"),
            "timeout": spec.get("timeout"),
        }

    if isinstance(runner, dict) and str(runner.get("type", "http")).lower() == "http":
        method = str(runner.get("method", "POST")).upper()
        url = runner.get("url") or runner.get("endpoint")
        if not url:
            return {"status": "error", "error": "HTTP runner missing URL"}
        headers = runner.get("headers") or {}
        timeout = runner.get("timeout") or spec.get("timeout") or DEFAULT_TOOL_TIMEOUT
        try:
            response = requests.request(
                method,
                url,
                json=payload,
                headers=headers,
                timeout=float(timeout),
            )
        except requests.RequestException as exc:
            return {"status": "error", "error": str(exc), "tool": tool_name}

        content_type = response.headers.get("content-type", "")
        body: Any
        if "application/json" in content_type.lower():
            try:
                body = response.json()
            except ValueError:
                body = response.text
        else:
            try:
                body = response.json()
            except ValueError:
                body = response.text

        return {
            "status": "ok" if response.ok else "error",
            "http_status": response.status_code,
            "body": body,
            "tool": tool_name,
        }

    return {"status": "skipped", "error": "Unsupported tool runner", "tool": tool_name}


def run_auto_fix(auto_req: AutoFixRequest, usage_identity: Optional[str] = None) -> Dict[str, Any]:
    base_req = _resolve_auto_fix_base_request(auto_req)

    attempts: List[Dict[str, Any]] = []
    prior_analyses: List[Dict[str, Any]] = []
    final_status = "unrecoverable"

    for attempt in range(1, auto_req.max_attempts + 1):
        print(f"--- Auto-Fix Attempt {attempt}/{auto_req.max_attempts} ---")
        analysis_req = _build_auto_fix_analysis_request(
            base_req, auto_req, attempt, prior_analyses
        )

        print(f"Running analysis infer (task: {analysis_req.task})...")
        analysis_result = run_inference(analysis_req, usage_identity=usage_identity)
        parsed_analysis = _parse_auto_fix_completion(
            analysis_result["output"].get("completion", "")
        )
        summary_preview = parsed_analysis.get("summary", "")
        print(
            "Analysis complete. Status: "
            f"{parsed_analysis.get('status', 'unknown')}, Summary: {summary_preview[:100]}..."
        )

        tool_results: List[Dict[str, Any]] = []
        for action in parsed_analysis.get("action_objects", []) or []:
            if not isinstance(action, dict):
                continue
            if str(action.get("type", "")).lower() != "tool":
                continue
            tool_name = action.get("tool") or action.get("name")
            if not tool_name:
                continue
            tool_input = action.get("input") or action.get("payload") or {}
            result = _execute_tool_action(str(tool_name), tool_input)
            tool_results.append({"tool": tool_name, "input": tool_input, "result": result})
        parsed_analysis["tool_results"] = tool_results

        attempt_record: Dict[str, Any] = {
            "attempt": attempt,
            "analysis_run_id": analysis_result.get("run_id"),
            "analysis": parsed_analysis,
            "fix_run_id": None,
            "fix_output": None,
            "tool_results": tool_results,
        }
        attempts.append(attempt_record)
        prior_analyses.append(parsed_analysis)

        final_status = parsed_analysis.get("status", "unparsed")

        if final_status == "unrecoverable":
            print("Analysis determined issue is unrecoverable. Stopping.")
            break

        if final_status == "needs_human_review":
            print("Analysis requested human review. Pausing automation loop.")
            break

        has_actions = bool(parsed_analysis.get("fix_instructions") or parsed_analysis.get("actions"))
        if not has_actions and final_status in {"unparsed", "apply_fix"}:
            print("Analysis provided no actionable fix instructions. Stopping.")
            final_status = "unrecoverable"
            break

        if final_status == "apply_fix":
            fix_req = _build_auto_fix_fix_request(base_req, auto_req, parsed_analysis)
            print(f"Running fix infer (task: {fix_req.task})...")
            fix_result = run_inference(fix_req, usage_identity=usage_identity)
            attempt_record["fix_run_id"] = fix_result.get("run_id")
            attempt_record["fix_output"] = fix_result.get("output")
            print(f"Fix attempt complete. Run ID: {fix_result.get('run_id')}")

        if final_status == "resolved":
            print("Analysis determined issue is resolved. Stopping.")
            break

    return {
        "base_request": base_req.dict(),
        "final_status": final_status,
        "attempts": attempts,
    }


def run_inference(
    req: InferRequest, usage_identity: Optional[str] = None, stream: bool = False
) -> Dict[str, Any]:
    client = get_redis()
    normalized_tags = _normalize_tags(req.tags)
    context_blob = ""
    for cid in req.context_ids:
        latest = _latest_memory_version(client, cid)
        if latest and latest.get("data"):
            context_blob += f"\n<ctx id='{cid}'>\n{latest['data']}\n</ctx>\n"

    retrieved_contexts: List[Dict[str, Any]] = []
    if req.retrieval_query:
        retrieved_contexts = semantic_recall(req.retrieval_query, req.context_ids)
        for item in retrieved_contexts:
            context_blob += (
                f"\n<ctx id='{item['id']}' source='retrieval' score='{item['score']}'>\n"
                f"{item['data']}\n</ctx>\n"
            )

    context_hash = _hash(context_blob)
    prior_cache_path = CACHE_DIR / f"context_{context_hash}.txt"
    prior_text = prior_cache_path.read_text() if prior_cache_path.exists() else ""
    patch = json_patch(prior_text, context_blob)
    prior_cache_path.write_text(context_blob)

    sys_b, sys_tokens = truncate_to_token_budget(req.system_prompt, req.budget.system)
    ctx_b, ctx_tokens = truncate_to_token_budget(context_blob, req.budget.context)
    usr_b, usr_tokens = truncate_to_token_budget(req.user_input, req.budget.user)

    tool_names: List[str] = []
    tool_specs: List[Dict[str, Any]] = []
    for t in req.tool_candidates:
        try:
            stored = client.hget(TOOLS_KEY, t)
        except RedisError as exc:
            raise HTTPException(500, detail=f"Failed to load tool {t}: {exc}") from exc
        if stored:
            try:
                spec = json.loads(stored)
            except json.JSONDecodeError:
                spec = {"name": t, "raw": stored}
            tool_names.append(spec.get("name", t))
            tool_specs.append(spec)

    prompt = {
        "system": sys_b,
        "context": ctx_b,
        "user": usr_b,
        "delta": patch,
        "tools": tool_specs,
        "model": req.model,
        "max_tokens": req.max_tokens,
        "retrievals": retrieved_contexts,
        "tokens_used": {
            "system": sys_tokens,
            "context": ctx_tokens,
            "user": usr_tokens,
        },
    }

    try:
        output = call_llm(prompt, req, tool_specs, retrieved_contexts, stream=stream)
    except TypeError:
        output = call_llm(prompt, req, tool_specs, retrieved_contexts)
        if isinstance(output, dict) and "completion_tokens" not in output:
            output["completion_tokens"] = list(
                TOKENIZER.iter_tokens(output.get("completion", ""))
            )

    manifest = {
        "request": req.model_dump(),
        "prompt": prompt,
        "output": output,
        "tools_used": tool_names,
        "retrievals": retrieved_contexts,
        "ts": time.time(),
        "tags": normalized_tags,
        "metadata": req.metadata,
    }
    run_id = save_run(manifest)
    identity = usage_identity or req.usage_identifier
    _record_usage(identity, prompt["tokens_used"], output, run_id)
    return {
        "run_id": run_id,
        "output": output,
        "budgets": req.budget.model_dump(),
        "used_context_chars": len(ctx_b),
        "used_tokens": prompt["tokens_used"],
        "retrievals": retrieved_contexts,
        "tools_considered": tool_names,
        "tags": normalized_tags,
        "metadata": req.metadata,
    }


@app.post("/infer")
def infer(
    req: InferRequest,
    authorized: str = Depends(require_api_key),
    stream: Optional[bool] = Query(default=None, description="Stream the completion token-by-token"),
):
    use_stream = req.stream or bool(stream)
    result = run_inference(req, usage_identity=req.usage_identifier or authorized, stream=use_stream)
    if not use_stream:
        return result

    tokens = result["output"].get("completion_tokens") or []

    def _stream() -> Iterator[str]:
        for token in tokens:
            yield json.dumps({"token": token}) + "\n"
        yield json.dumps({"event": "completed", "run_id": result["run_id"], "output": result["output"]}) + "\n"

    return StreamingResponse(_stream(), media_type="application/json")


@app.post("/infer/{run_id}/retry")
def infer_retry(
    run_id: str,
    retry_req: RetryRequest,
    authorized: str = Depends(require_api_key),
):
    client = get_redis()

    try:
        payload = client.hget(RUNS_KEY, run_id)
    except RedisError as exc:
        raise HTTPException(500, detail=f"Failed to load original run: {exc}") from exc

    if not payload:
        raise HTTPException(404, detail="Original run not found")

    try:
        original_manifest = json.loads(payload)
    except json.JSONDecodeError as exc:
        raise HTTPException(500, detail="Original run data is corrupted") from exc

    original_req_dict = original_manifest.get("request") or {}

    try:
        original_req = InferRequest(**original_req_dict)
    except Exception as exc:
        raise HTTPException(400, detail=f"Failed to parse original request: {exc}") from exc

    original_req.system_prompt = (
        retry_req.system_prompt_override or original_req.system_prompt
    )

    original_user_input = retry_req.user_input_override or original_req.user_input
    error_context_block = (
        "--- Context: Previous Attempt Failed ---\n"
        "The previous attempt to solve the task failed. Analyze the following error "
        "and provide a corrected solution based on the original request.\n"
        f"<error_details>\n{retry_req.error_context}\n</error_details>\n"
        "--- Original User Input ---\n"
    )
    original_req.user_input = error_context_block + original_user_input

    print(f"Retrying run {run_id}. Added error context to user input.")
    return run_inference(original_req, usage_identity=authorized)


@app.post("/infer/auto-fix")
def infer_auto_fix(auto_req: AutoFixRequest, authorized: str = Depends(require_api_key)):
    return run_auto_fix(auto_req, usage_identity=auto_req.usage_identifier or authorized)


@app.get("/ui", response_class=HTMLResponse)
def ui_index(request: Request, authorized: str = Depends(require_api_key)):
    tmpl = _get_templates()
    return tmpl.TemplateResponse("index.html", {"request": request})


@app.get("/ui/runs", response_class=HTMLResponse)
def ui_runs(
    request: Request,
    tag: Optional[str] = Query(default=None),
    limit: int = Query(default=20, ge=1, le=200),
    authorized: str = Depends(require_api_key),
):
    runs = _list_runs(tag, limit)
    tmpl = _get_templates()
    return tmpl.TemplateResponse(
        "runs.html",
        {"request": request, "runs": runs, "tag": tag, "limit": limit},
    )


@app.get("/ui/runs/{run_id}", response_class=HTMLResponse)
def ui_run_detail(run_id: str, request: Request, authorized: str = Depends(require_api_key)):
    client = get_redis()
    try:
        payload = client.hget(RUNS_KEY, run_id)
    except RedisError as exc:
        raise HTTPException(500, detail=f"Failed to load run: {exc}") from exc
    if not payload:
        raise HTTPException(404, "run not found")
    try:
        manifest = json.loads(payload)
    except json.JSONDecodeError:
        manifest = {"error": "Stored manifest is not valid JSON", "raw": payload}

    manifest_pretty = json.dumps(manifest, indent=2, default=str)
    tmpl = _get_templates()
    return tmpl.TemplateResponse(
        "run_detail.html",
        {"request": request, "run_id": run_id, "manifest": manifest, "manifest_json": manifest_pretty},
    )


@app.get("/ui/memory", response_class=HTMLResponse)
def ui_memory(
    request: Request,
    limit: int = Query(default=50, ge=1, le=200),
    authorized: str = Depends(require_api_key),
):
    entries = _list_memory_entries(limit)
    tmpl = _get_templates()
    return tmpl.TemplateResponse(
        "memory.html", {"request": request, "entries": entries, "limit": limit}
    )


@app.get("/ui/memory/{memory_id}", response_class=HTMLResponse)
def ui_memory_detail(memory_id: str, request: Request, authorized: str = Depends(require_api_key)):
    client = get_redis()
    key = f"{MEMORY_KEY_PREFIX}{memory_id}"
    try:
        history = client.lrange(key, 0, -1)
    except RedisError as exc:
        raise HTTPException(500, detail=f"Failed to load memory: {exc}") from exc

    versions: List[Dict[str, Any]] = []
    for idx, payload in enumerate(history):
        try:
            data = json.loads(payload)
        except json.JSONDecodeError:
            data = {"data": payload, "ts": None}
        versions.append({"version": idx, "payload": data})

    tmpl = _get_templates()
    return tmpl.TemplateResponse(
        "memory_detail.html",
        {"request": request, "memory_id": memory_id, "versions": list(reversed(versions))},
    )


@app.get("/ui/infer", response_class=HTMLResponse)
def ui_infer_form(request: Request, authorized: str = Depends(require_api_key)):
    sample_payload = {
        "task": "summarize_incident",
        "user_input": "Summarize the latest incident log entry.",
        "tags": ["demo"],
    }
    tmpl = _get_templates()
    return tmpl.TemplateResponse(
        "infer_form.html",
        {
            "request": request,
            "sample_payload": json.dumps(sample_payload, indent=2),
        },
    )


@app.get("/ui/auto-fix", response_class=HTMLResponse)
def ui_auto_fix_form(request: Request, authorized: str = Depends(require_api_key)):
    sample_payload = {
        "error_log": "Traceback (most recent call last): ...",
        "request": {
            "task": "repair_pipeline",
            "user_input": "Original request body",
        },
    }
    tmpl = _get_templates()
    return tmpl.TemplateResponse(
        "auto_fix_form.html",
        {
            "request": request,
            "sample_payload": json.dumps(sample_payload, indent=2),
        },
    )


@app.get("/ui/usage", response_class=HTMLResponse)
def ui_usage(request: Request, authorized: str = Depends(require_api_key)):
    tmpl = _get_templates()
    return tmpl.TemplateResponse("usage.html", {"request": request})
def _get_templates() -> Jinja2Templates:
    if templates is None:
        raise HTTPException(
            503,
            detail="Template rendering requires the optional 'jinja2' dependency.",
        )
    return templates
