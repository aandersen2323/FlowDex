import os, hashlib, json, time
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Any, Dict, List, Optional
from pathlib import Path

CACHE_DIR = Path(os.environ.get("FLOWDEX_CACHE_DIR", ".flowdex_cache"))
CACHE_DIR.mkdir(parents=True, exist_ok=True)

app = FastAPI(title="FlowDex API", version="0.1.0")

MEMORY: Dict[str, List[Dict[str, Any]]] = {}
TOOLS: Dict[str, Dict[str, Any]] = {}
RUNS: Dict[str, Dict[str, Any]] = {}

class Budget(BaseModel):
    system: int = Field(default=int(os.environ.get("FLOWDEX_BUDGET_SYSTEM", 1000)))
    context: int = Field(default=int(os.environ.get("FLOWDEX_BUDGET_CONTEXT", 2500)))
    user: int = Field(default=int(os.environ.get("FLOWDEX_BUDGET_USER", 1500)))
    tools: int = Field(default=int(os.environ.get("FLOWDEX_BUDGET_TOOLS", 1000)))

class InferRequest(BaseModel):
    task: str
    user_input: str = ""
    system_prompt: str = ""
    context_ids: List[str] = []
    tool_candidates: List[str] = []
    retrieval_query: Optional[str] = None
    model: str = os.environ.get("FLOWDEX_MODEL", "anthropic/claude-3-5-sonnet")
    max_tokens: int = int(os.environ.get("FLOWDEX_MAX_TOKENS", 6000))
    budget: Budget = Budget()

def _hash(s: str) -> str:
    import hashlib
    return hashlib.sha256(s.encode("utf-8")).hexdigest()[:10]

def save_run(manifest: Dict[str, Any]) -> str:
    run_id = _hash(str(time.time()) + json.dumps(manifest, sort_keys=True))
    RUNS[run_id] = manifest
    (CACHE_DIR / f"run_{run_id}.json").write_text(json.dumps(manifest, indent=2))
    return run_id

def json_patch(old: str, new: str) -> Dict[str, Any]:
    prefix_len = 0
    while prefix_len < min(len(old), len(new)) and old[prefix_len] == new[prefix_len]:
        prefix_len += 1
    return {"common_prefix": prefix_len, "added": new[prefix_len:]}

def truncate_to_budget(text: str, budget_chars: int) -> str:
    if len(text) <= budget_chars:
        return text
    return text[-budget_chars:]

@app.post("/memory/put")
def memory_put(item: Dict[str, Any]):
    id_ = item.get("id")
    if not id_:
        raise HTTPException(400, "id is required")
    versions = MEMORY.setdefault(id_, [])
    versions.append({"ts": time.time(), "data": item.get("data", "")})
    return {"ok": True, "versions": len(versions)}

@app.get("/memory/get")
def memory_get(id: str, version: Optional[int] = None):
    versions = MEMORY.get(id, [])
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
    TOOLS[name] = spec
    return {"ok": True, "count": len(TOOLS)}

@app.get("/runs/{run_id}")
def get_run(run_id: str):
    r = RUNS.get(run_id)
    if not r:
        raise HTTPException(404, "run not found")
    return r

@app.post("/infer")
def infer(req: InferRequest):
    context_blob = ""
    for cid in req.context_ids:
        ver = MEMORY.get(cid, [])
        if ver:
            context_blob += f"\n<ctx id='{cid}'>\n{ver[-1]['data']}\n</ctx>\n"

    context_hash = _hash(context_blob)
    prior_cache_path = CACHE_DIR / f"context_{context_hash}.txt"
    prior_text = prior_cache_path.read_text() if prior_cache_path.exists() else ""
    patch = json_patch(prior_text, context_blob)
    prior_cache_path.write_text(context_blob)

    sys_b = truncate_to_budget(req.system_prompt, req.budget.system)
    ctx_b = truncate_to_budget(context_blob, req.budget.context)
    usr_b = truncate_to_budget(req.user_input, req.budget.user)

    chosen_tools = []
    for t in req.tool_candidates:
        if t in TOOLS:
            chosen_tools.append(t)

    prompt = {
        "system": sys_b,
        "context": ctx_b,
        "user": usr_b,
        "delta": patch,
        "tools": chosen_tools,
        "model": req.model,
        "max_tokens": req.max_tokens
    }

    output = {
        "explanation": "Stubbed completion for offline skeleton. Wire your LLM here.",
        "recommendation": "Use tool first if it resolves the task cheaply; otherwise minimal prompt with delta-only context."
    }

    manifest = {
        "request": req.dict(),
        "prompt": prompt,
        "output": output,
        "ts": time.time()
    }
    run_id = save_run(manifest)
    return {"run_id": run_id, "output": output, "budgets": req.budget.dict(), "used_context_chars": len(ctx_b)}
