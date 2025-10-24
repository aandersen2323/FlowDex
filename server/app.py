import hashlib
import json
import math
import re
import time
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import redis
import requests
from fastapi import Depends, FastAPI, Header, HTTPException
from pydantic import BaseModel, Field
from redis.exceptions import RedisError
from pydantic_settings import BaseSettings, SettingsConfigDict

try:
    import tiktoken
    try:
        _TOKEN_ENCODER = tiktoken.get_encoding("cl100k_base")
        _HAS_TIKTOKEN = True
    except Exception:  # pragma: no cover - optional dependency retrieval failure
        _TOKEN_ENCODER = None
        _HAS_TIKTOKEN = False
except ImportError:  # pragma: no cover - optional dependency
    _TOKEN_ENCODER = None
    _HAS_TIKTOKEN = False

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
    anthropic_api_key: Optional[str] = None
    flowdex_anthropic_url: str = "https://api.anthropic.com/v1/messages"
    flowdex_anthropic_version: str = "2023-06-01"


settings = Settings()

CACHE_DIR = settings.flowdex_cache_dir
CACHE_DIR.mkdir(parents=True, exist_ok=True)

app = FastAPI(title="FlowDex API", version="0.1.0")

_REDIS_CLIENT: Optional[redis.Redis] = None
MEMORY_KEY_PREFIX = "flowdex:memory:"
MEMORY_INDEX_KEY = "flowdex:memory:index"
TOOLS_KEY = "flowdex:tools"
RUNS_KEY = "flowdex:runs"
API_KEY = settings.flowdex_api_key
ANTHROPIC_TIMEOUT = settings.flowdex_anthropic_timeout

AUTO_FIX_SYSTEM_PROMPT = """You are FlowDex AutoFix, an autonomous incident responder for automation workflows.\n" \
    "You triage failing orchestrations by summarizing errors, identifying likely root causes, " \
    "and prescribing concrete remediation steps that can be executed without human input.\n" \
    "Always respond with a single JSON object containing the following keys:\n" \
    "  - status: one of ['resolved', 'apply_fix', 'unrecoverable'].\n" \
    "  - summary: short natural-language description of what happened.\n" \
    "  - actions: array of actionable steps to remediate the failure.\n" \
    "  - fix_instructions: detailed instructions or code patches to apply.\n" \
    "  - metrics: object with optional numeric metadata such as confidence (0-1) and\n" \
    "             estimated_minutes.\n" \
    "If the provided information is insufficient, set status to 'unrecoverable' and explain\n" \
    "what extra data is required. Keep responses concise but information-dense."""

_VALID_AUTO_FIX_STATUSES = {"resolved", "apply_fix", "unrecoverable"}


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


def require_api_key(x_api_key: Optional[str] = Header(default=None)) -> bool:
    """Validate the provided API key if one is configured."""

    if API_KEY and x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API key")
    return True


def _hash(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()[:10]


def save_run(manifest: Dict[str, Any]) -> str:
    run_id = _hash(str(time.time()) + json.dumps(manifest, sort_keys=True))
    client = get_redis()
    try:
        client.hset(RUNS_KEY, run_id, json.dumps(manifest))
    except RedisError as exc:
        raise HTTPException(500, detail=f"Failed to persist run: {exc}") from exc
    (CACHE_DIR / f"run_{run_id}.json").write_text(json.dumps(manifest, indent=2))
    return run_id


def json_patch(old: str, new: str) -> Dict[str, Any]:
    prefix_len = 0
    while prefix_len < min(len(old), len(new)) and old[prefix_len] == new[prefix_len]:
        prefix_len += 1
    return {"common_prefix": prefix_len, "added": new[prefix_len:]}


def count_tokens(text: str) -> int:
    """Count tokens using ``tiktoken`` when available, otherwise estimate."""

    if not text:
        return 0
    if _HAS_TIKTOKEN and _TOKEN_ENCODER is not None:
        return len(_TOKEN_ENCODER.encode(text))
    return max(1, len(text) // 4)


def truncate_to_token_budget(text: str, budget_tokens: int) -> Tuple[str, int]:
    """Truncate ``text`` to fit the supplied token budget.

    Returns a tuple of the truncated text and the number of tokens consumed.
    """

    if not text or budget_tokens <= 0:
        return "", 0

    if not (_HAS_TIKTOKEN and _TOKEN_ENCODER is not None):
        char_budget = max(budget_tokens * 4, 0)
        truncated_text = text[-char_budget:] if char_budget else ""
        return truncated_text, count_tokens(truncated_text)

    tokens = _TOKEN_ENCODER.encode(text)
    if len(tokens) <= budget_tokens:
        return text, len(tokens)

    truncated_tokens = tokens[-budget_tokens:]
    try:
        truncated_text = _TOKEN_ENCODER.decode(truncated_tokens)
    except Exception:  # pragma: no cover - defensive decoding fallback
        start_char = max(
            0,
            len(text)
            - int(len(text) * (budget_tokens / max(len(tokens), 1)) * 1.1),
        )
        truncated_text = text[start_char:]
        truncated_tokens = _TOKEN_ENCODER.encode(truncated_text)[-budget_tokens:]
        truncated_text = _TOKEN_ENCODER.decode(truncated_tokens)

    return truncated_text, len(truncated_tokens)


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


TOKEN_PATTERN = re.compile(r"\b\w+\b", re.UNICODE)


def _tokenize(text: str) -> List[str]:
    return [tok for tok in TOKEN_PATTERN.findall(text.lower()) if tok]


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
    tokens_query = _tokenize(query)
    if not tokens_query:
        return []

    client = get_redis()
    try:
        memory_ids = client.smembers(MEMORY_INDEX_KEY)
    except RedisError as exc:
        raise HTTPException(500, detail=f"Failed to enumerate memory: {exc}") from exc

    query_counter = Counter(tokens_query)
    query_norm = math.sqrt(sum(v * v for v in query_counter.values())) or 1.0

    scored: List[Dict[str, Any]] = []
    for mem_id in memory_ids:
        latest = _latest_memory_version(client, mem_id)
        if not latest:
            continue
        doc_text = latest.get("data", "")
        doc_tokens = _tokenize(doc_text)
        if not doc_tokens:
            continue
        doc_counter = Counter(doc_tokens)
        doc_norm = math.sqrt(sum(v * v for v in doc_counter.values())) or 1.0
        dot = sum(query_counter[token] * doc_counter.get(token, 0) for token in query_counter)
        score = dot / (query_norm * doc_norm)
        if score <= 0:
            continue
        scored.append({"id": mem_id, "score": round(score, 6), "data": doc_text, "ts": latest.get("ts")})

    scored.sort(key=lambda item: item["score"], reverse=True)
    results: List[Dict[str, Any]] = []
    for entry in scored:
        if entry["id"] in exclude_ids:
            continue
        results.append(entry)
        if len(results) >= limit:
            break
    return results


def call_llm(prompt: Dict[str, Any], request: InferRequest, tool_specs: List[Dict[str, Any]], retrieved: List[Dict[str, Any]]) -> Dict[str, Any]:
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
        raise HTTPException(502, detail=f"Anthropic API request failed: {exc}") from exc

    try:
        data = response.json()
    except ValueError as exc:
        raise HTTPException(502, detail=f"Anthropic API response was not valid JSON: {exc}") from exc
    content = data.get("content", [])
    text_parts = [part.get("text", "") for part in content if part.get("type") == "text"]
    completion = "\n".join(part for part in text_parts if part)

    return {
        "completion": completion,
        "stop_reason": data.get("stop_reason"),
        "usage": data.get("usage", {}),
        "model": data.get("model", prompt["model"]),
        "raw": data,
    }


@app.post("/memory/put")
def memory_put(item: Dict[str, Any], authorized: bool = Depends(require_api_key)):
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
def memory_get(id: str, version: Optional[int] = None, authorized: bool = Depends(require_api_key)):
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


@app.post("/tools/register")
def register_tool(spec: Dict[str, Any], authorized: bool = Depends(require_api_key)):
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


@app.get("/runs/{run_id}")
def get_run(run_id: str, authorized: bool = Depends(require_api_key)):
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


_CODE_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL)


def _extract_auto_fix_json(text: str) -> Dict[str, Any]:
    text = (text or "").strip()
    if not text:
        return {}

    candidates: List[str] = []

    fence_match = _CODE_FENCE_RE.search(text)
    if fence_match:
        candidates.append(fence_match.group(1).strip())

    brace_match = re.search(r"\{.*\}", text, re.DOTALL)
    if brace_match:
        candidates.append(brace_match.group(0).strip())

    candidates.append(text)

    for candidate in candidates:
        candidate = candidate.strip()
        if not candidate:
            continue
        try:
            return json.loads(candidate, strict=False)
        except json.JSONDecodeError:
            continue

    print(f"Warning: Could not parse JSON from auto-fix completion: {text[:100]}...")
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
    if isinstance(actions_raw, str):
        actions = [actions_raw.strip()]
    else:
        actions = [str(item) for item in actions_raw if item is not None]

    fix_instructions = parsed.get("fix_instructions") or parsed.get("resolution") or ""
    summary = parsed.get("summary") or parsed.get("analysis") or ""
    metrics = parsed.get("metrics") or {}

    return {
        "status": status,
        "summary": summary,
        "actions": actions,
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

    fix_req.user_input = "\n\n".join(
        section
        for section in (
            base_input,
            "<auto_fix_context>",
            f"Latest error log:\n{auto_req.error_log}",
            f"Analysis summary:\n{analysis.get('summary', '')}",
            f"Action steps:\n{actions_blob}",
            f"Remediation plan:\n{remediation}",
            f"Structured response:\n{plan_blob}",
            "</auto_fix_context>",
            "Implement the remediation plan above and provide the corrected output.",
        )
        if section
    )

    return fix_req


def run_auto_fix(auto_req: AutoFixRequest) -> Dict[str, Any]:
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
        analysis_result = run_inference(analysis_req)
        parsed_analysis = _parse_auto_fix_completion(
            analysis_result["output"].get("completion", "")
        )
        summary_preview = parsed_analysis.get("summary", "")
        print(
            "Analysis complete. Status: "
            f"{parsed_analysis.get('status', 'unknown')}, Summary: {summary_preview[:100]}..."
        )

        attempt_record: Dict[str, Any] = {
            "attempt": attempt,
            "analysis_run_id": analysis_result.get("run_id"),
            "analysis": parsed_analysis,
            "fix_run_id": None,
            "fix_output": None,
        }
        attempts.append(attempt_record)
        prior_analyses.append(parsed_analysis)

        final_status = parsed_analysis.get("status", "unparsed")

        if final_status == "unrecoverable":
            print("Analysis determined issue is unrecoverable. Stopping.")
            break

        has_actions = bool(parsed_analysis.get("fix_instructions") or parsed_analysis.get("actions"))
        if not has_actions and final_status in {"unparsed", "apply_fix"}:
            print("Analysis provided no actionable fix instructions. Stopping.")
            final_status = "unrecoverable"
            break

        if final_status == "apply_fix":
            fix_req = _build_auto_fix_fix_request(base_req, auto_req, parsed_analysis)
            print(f"Running fix infer (task: {fix_req.task})...")
            fix_result = run_inference(fix_req)
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


def run_inference(req: InferRequest) -> Dict[str, Any]:
    client = get_redis()
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

    output = call_llm(prompt, req, tool_specs, retrieved_contexts)

    manifest = {
        "request": req.dict(),
        "prompt": prompt,
        "output": output,
        "tools_used": tool_names,
        "retrievals": retrieved_contexts,
        "ts": time.time(),
    }
    run_id = save_run(manifest)
    return {
        "run_id": run_id,
        "output": output,
        "budgets": req.budget.dict(),
        "used_context_chars": len(ctx_b),
        "used_tokens": prompt["tokens_used"],
        "retrievals": retrieved_contexts,
        "tools_considered": tool_names,
    }


@app.post("/infer")
def infer(req: InferRequest, authorized: bool = Depends(require_api_key)):
    return run_inference(req)


@app.post("/infer/{run_id}/retry")
def infer_retry(
    run_id: str,
    retry_req: RetryRequest,
    authorized: bool = Depends(require_api_key),
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
    return run_inference(original_req)


@app.post("/infer/auto-fix")
def infer_auto_fix(auto_req: AutoFixRequest, authorized: bool = Depends(require_api_key)):
    return run_auto_fix(auto_req)
