# FlowDex

Token‑efficient LLM router + context manager for **n8n** and **Claude Code (MCP)**—deployable locally or via remote HTTP bridges.

> Goal: keep massive *working* context without massive token use. Replace fragile, token‑hungry prompts with **diffed context**, **semantic retrieval**, **function routing**, and **strict token budgets**.

- **API**: Python FastAPI with caching, prompt macros, delta-context, routing, retry support, and health checks.
- **MCP bridge**: FastAPI server that exposes FlowDex MCP tools over HTTP for remote VS Code / Claude Code setups.
- **n8n node**: Minimal custom node that hits FlowDex `/infer` with **few parameters** (model, task, inputs). Avoids native nodes explosion.
- **CLI**: Inspect budgets, cache, and reproduce runs.
- **Local stack**: `docker-compose.yml` (API + Redis + MCP bridge). SQLite used for persistence by default. Swap in real vector DB later.

## Why FlowDex Exists

FlowDex grew out of shipping real automations where “just throw more tokens at the prompt” stopped scaling. Product teams and
operators needed to preserve weeks of context, tool transcripts, and troubleshooting notes without paying for massive context
windows or rewriting flows every time the prompt changed. FlowDex packages those lessons into a single, opinionated service that:

- Preserves the **working memory** of an automation across turns without duplicating the full transcript.
- Routes expensive LLM calls only when they add value, preferring cached results or cheaper tools.
- Gives operators an auditable trail of what the assistant saw, which tool it picked, and why decisions were made.

## What FlowDex Does

At its core FlowDex is a FastAPI service that tracks every interaction in an LLM-powered workflow. It stores diffs between turns,
enforces token budgets, records retries, and makes every run reproducible. The same backend also powers:

- An MCP bridge so Claude Code (VS Code) can call the exact API used in production.
- An n8n node that lets low-code builders orchestrate FlowDex without custom HTTP glue.
- A CLI for inspecting cache hits, budgets, manifests, and reproducing runs locally.

## How FlowDex Helps You

By centralizing prompting logic and run history, FlowDex lets you:

1. **Ship faster** – Build against one API whether the assistant runs in n8n, Claude Code, or bespoke scripts.
2. **Spend less** – Delta-context, semantic recall, and strict budgets keep token usage predictable and low.
3. **Operate confidently** – Persisted run manifests, diffed context, and replay tooling make debugging straightforward.
4. **Stay flexible** – Swap models, retrieval engines, or tool definitions without rewriting every workflow node.

## Key Ideas

1. **Delta Context (Patch Prompting)**: send only what changed. We compute a content hash and minimal JSON Patch between turns.
2. **Token Budgeter**: hard caps for **system**, **context**, **tools**, **user** with graceful degradation and logs.
3. **Semantic Recall (Optional)**: simple bag-of-words + sqlite index today; plug your own vectors later.
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
FLOWDEX_API_KEY=change-me
```

Expose `FLOWDEX_API_KEY` wherever the API is reachable from untrusted networks. The bundled MCP bridge forwards this header automatically.

> Real token counting & embeddings are pluggable. Stubs are provided so it runs offline now.

## Endpoints

- `GET /health` – lightweight readiness + Redis connectivity check.
- `POST /infer` – run a task with budgets, diffed context, and optional retrieval.
- `POST /infer/{run_id}/retry` – re-run a previous task with new error context for automated repair loops.
- `POST /memory/put` – store or update named context blobs (versioned).
- `GET /memory/get?id=...` – retrieve latest or a specific version.
- `POST /tools/register` – declare a tool with a schema and cost hints.
- `GET /runs/{id}` – view a prior run manifest for repro.

## Claude Code (MCP)

Two options depending on where VS Code / Claude Code runs:

1. **Local desktop** – run the original stdio bridge:
   ```bash
   python mcp/server.py
   ```
   Then configure Claude Code to spawn that script.
2. **Remote / browser (Unraid, Codespaces, etc.)** – run the HTTP bridge:
   ```bash
   python mcp/mcp_http_server.py  # or let docker compose manage it
   ```
   Point Claude Code to `https://your-host/mcp` (behind Cloudflare, etc.). The bridge proxies `flowdex.infer`, `flowdex.infer.retry`, `flowdex.memory.get`, and `flowdex.health` over HTTP with API-key auth support.

The default `docker-compose.yml` now builds and runs the bridge alongside the API, exposing port `8788`.

## n8n Node

Install from `n8n-node/` into your n8n custom nodes folder. The node calls `/infer` with minimal configuration.

---

© 2025 FlowDex. Generated 2025-10-23T00:31:05.007257Z
