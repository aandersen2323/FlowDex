# FlowDex

Token‑efficient LLM router + context manager for **n8n** and **Claude Code (MCP)**.

> Goal: keep massive *working* context without massive token use. Replace fragile, token‑hungry prompts with **diffed context**, **semantic retrieval**, **function routing**, and **strict token budgets**.

- **API**: Python FastAPI with caching, prompt macros, delta-context, routing, and live Anthropic completions.
- **MCP server**: Tools for Claude Code to query FlowDex context, fetch diffs, run functions, and log decisions.
- **n8n node**: Minimal custom node that hits FlowDex `/infer` with **few parameters** (model, task, inputs). Avoids native nodes explosion.
- **CLI**: Inspect budgets, cache, and reproduce runs.
- **Local stack**: `docker-compose.yml` (API + Redis). Context and run manifests persist to the filesystem; swap in your preferred store later.

## Key Ideas

1. **Delta Context (Patch Prompting)**: send only what changed. We compute a content hash keyed by context IDs and emit a minimal JSON diff between turns.
2. **Token Budgeter**: hard caps for **system**, **context**, **tools**, **user** with graceful degradation and logs.
3. **Semantic Recall (Optional)**: substring search across stored contexts (Redis-backed) so you can surface related notes even if they were not explicitly referenced.
4. **Function Registry**: strongly-typed tools with cost hints; router chooses *tool > text* when cheaper.
5. **Determinism & Repro**: Run manifests saved as JSON; one‑click replay.
6. **n8n First‑Class**: one compact node → FlowDex API. No sprawl of native nodes.
7. **Claude Code via MCP**: surface FlowDex as tools inside your editor, not another chat tab.

## Quick Start

```bash
# 1) Docker (recommended)
docker compose up --build

# 2) Or local (Python 3.10+)
python -m venv .venv && source .venv/bin/activate
pip install -r server/requirements.txt
uvicorn server.app:app --reload --port 8787
```

### Env

Create `.env` (or use Docker envs):
```
FLOWDEX_PORT=8787
FLOWDEX_MODEL=anthropic/claude-3-5-sonnet
FLOWDEX_CACHE_DIR=.flowdex_cache
FLOWDEX_MAX_TOKENS=6000
FLOWDEX_BUDGET_SYSTEM=1000
FLOWDEX_BUDGET_CONTEXT=2500
FLOWDEX_BUDGET_USER=1500
FLOWDEX_BUDGET_TOOLS=1000
FLOWDEX_REDIS_URL=redis://localhost:6379/0
FLOWDEX_LLM_TIMEOUT=60
FLOWDEX_HTTP_TIMEOUT=30
ANTHROPIC_API_KEY=replace-with-your-key
```

`ANTHROPIC_API_KEY` must be set for `/infer` to call Anthropic. Without Redis the API falls back to in-memory storage, but running the bundled Redis service ensures context, tools, and run manifests persist across restarts.

### Context Delta Cache

FlowDex keeps a cache per set of context IDs. Every `/infer` request emits a `delta` that includes the shared prefix length, removed tail, and newly added text so you can replay or inspect what actually changed. Cached context snapshots are also written to Redis (when available) so the delta survives restarts.

#### Example run

```bash
curl -s -X POST http://localhost:8787/infer \
  -H 'content-type: application/json' \
  -d '{
    "task":"fix_connection",
    "user_input":"WP REST is flaky; propose cheapest checks before LLM.",
    "system_prompt":"Be concise, tool-first.",
    "context_ids":["proeye_guides"],
    "tool_candidates":["http_check","wp_restore"]
  }' | jq .
```

The response includes the `delta` object, Anthropic completion metadata, and a record of the tools that fit within the configured budget.

## Endpoints

- `POST /infer` – run a task with budgets, diffed context, semantic retrieval, and live Anthropic completions.
- `POST /memory/put` – store or update named context blobs (versioned).
- `GET /memory/get?id=...` – retrieve latest or a specific version.
- `POST /tools/register` – declare a tool with a schema and cost hints.
- `GET /runs/{id}` – view a prior run manifest for repro.

## Claude Code (MCP)

Run `python mcp/server.py`. In Claude Code settings, add a new MCP server pointing to that script. The bridge forwards `flowdex.infer`, `flowdex.memory.get`, and `flowdex.memory.put` calls to the FastAPI service using the `FLOWDEX_BASE_URL` you configure.

## n8n Node

Install from `n8n-node/` into your n8n custom nodes folder. The node calls `/infer` with minimal configuration.

---

© 2025 FlowDex. Generated 2025-10-23T00:31:05.007257Z
