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

## Step-by-Step Setup

### 0. Prerequisites

| Component | Requirement | Notes |
| --- | --- | --- |
| Docker (optional) | `docker` & `docker compose` | Recommended for the quickest path with Redis + MCP bridge bundled. |
| Python | 3.10+ | Needed for local installs, CLI tooling, and development. |
| Redis (optional) | 7+ | Docker compose spins this up automatically. Local installs can point to an existing instance or fall back to the in-memory cache. |

Have an Anthropic-compatible API key (or whichever backend model you configure) ready before deploying to remote environments.

### 1. Clone the Repository

```bash
git clone https://github.com/your-org/FlowDex.git
cd FlowDex
```

### 2. Choose a Deployment Path

#### Option A: Docker Compose (recommended)

1. Copy the sample environment file or create a new `.env` file at the project root.
2. Update secrets such as `FLOWDEX_API_KEY` and the model you plan to call.
3. Launch the stack:
   ```bash
   docker compose up --build
   ```
4. The API will be available at `http://localhost:8787`, and the MCP HTTP bridge at `http://localhost:8788`.
5. Stop the stack with <kbd>Ctrl</kbd>+<kbd>C</kbd> or `docker compose down` when you are done.

#### Option B: Local Python Environment

1. Create and activate a virtual environment:
   ```bash
   python -m venv .venv
   source .venv/bin/activate
   ```
2. Install dependencies:
   ```bash
   pip install -r server/requirements.txt
   ```
3. (Optional) Start Redis locally if you want persistence/caching beyond the process lifetime.
4. Launch the API server:
   ```bash
   uvicorn server.app:app --reload --port 8787
   ```
5. In a second terminal, start the MCP HTTP bridge if you need Claude Code connectivity:
   ```bash
   python mcp/mcp_http_server.py
   ```

### 3. Configure Environment Variables

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
FLOWDEX_REDIS_URL=redis://localhost:6379/0
```

- `FLOWDEX_MODEL` can be any provider/model name supported by your downstream LLM proxy.
- `FLOWDEX_REDIS_URL` is optional; omit it to run in in-memory mode (good for quick trials).
- Expose `FLOWDEX_API_KEY` whenever the API is reachable from untrusted networks. The MCP bridge forwards this header automatically.

> Real token counting & embeddings are pluggable. Stubs are provided so it runs offline now.

### 4. Verify the Installation

With the API running, hit the health check:

```bash
curl http://localhost:8787/health
```

You should see a JSON response similar to `{"status": "ok"}`. If Redis is unavailable, the response will note degraded caching.

### 5. Seed Memory (Optional)

You can preload durable snippets that will be referenced by workflows:

```bash
curl -X POST http://localhost:8787/memory/put \
  -H "Content-Type: application/json" \
  -H "x-flowdex-api-key: $FLOWDEX_API_KEY" \
  -d '{
        "id": "runbook.postgres",
        "title": "Postgres On-Call Runbook",
        "body": "Check pg_stat_activity; restart read replicas if backlog > 500",
        "tags": ["db", "incident"]
      }'
```

The memory store keeps previous versions, so you can roll back or diff updates.

### 6. Make an Inference Request

```bash
curl -X POST http://localhost:8787/infer \
  -H "Content-Type: application/json" \
  -H "x-flowdex-api-key: $FLOWDEX_API_KEY" \
  -d '{
        "task": "triage",
        "model": "anthropic/claude-3-5-sonnet",
        "inputs": {
          "user": "Summarize the last incident and draft an update.",
          "context": ["runbook.postgres"],
          "tool_hints": ["post_incident_report"]
        }
      }'
```

The response contains a `run_id`, token usage, tool decisions, and the generated text. Store the `run_id` to retry or reproduce later.

### 7. Replay or Retry a Run

```bash
curl -X POST http://localhost:8787/infer/<run_id>/retry \
  -H "Content-Type: application/json" \
  -H "x-flowdex-api-key: $FLOWDEX_API_KEY" \
  -d '{
        "error": "Tool failed: post_incident_report",
        "patch": {"path": "/inputs/tool_hints", "op": "add", "value": ["post_incident_report", "status_page"]}
      }'
```

Retries automatically reuse context diffs to keep token usage minimal.

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

### Claude Code Configuration Walkthrough

1. Install the Claude Code extension in VS Code.
2. Open the extension settings and add a new MCP server.
3. For local setups select **Process** and point to `python mcp/server.py`. For remote setups select **HTTP** and supply the URL `http://<your-host>:8788/mcp`.
4. Set the environment variable `FLOWDEX_API_KEY` within the MCP configuration if your API requires it.
5. Reload the Claude Code extension. You should now see FlowDex tools (`flowdex.infer`, `flowdex.memory.get`, etc.) available in the tool palette.

## n8n Node

Install from `n8n-node/` into your n8n custom nodes folder. The node calls `/infer` with minimal configuration.

### n8n Usage Example

1. Copy `n8n-node/flowdex` into your n8n custom nodes directory and restart n8n.
2. In your workflow, add the **FlowDex** node.
3. Configure the node parameters:
   - **API Base URL**: `http://flowdex:8787` (if running via docker compose) or `http://localhost:8787` for local runs.
   - **Task**: e.g., `triage` or `summarize`.
   - **User Input**: the user prompt or payload from upstream nodes.
   - **System Prompt**: optional guardrails to preface the run.
   - **Context IDs**: comma-separated memory IDs or contextual notes.
   - **Tool Candidates**: comma-separated tool identifiers FlowDex should consider.
   - **Model**: override the default model if needed.
4. Set the `X-API-Key` credential globally in n8n or append a `Header Auth` node if your server requires authentication.
5. Trigger the workflow manually or via webhook to confirm the integration.

## Command-Line Interface

The CLI under `cli/` offers a lightweight way to send inference requests and experiment with different context/tool combinations from the terminal.

### Install the CLI

```bash
pip install -r cli/requirements.txt
python cli/flowdex_cli.py --help
```

### Example Commands

- Send a quick inference:
  ```bash
  python cli/flowdex_cli.py --task summarize --user "List recent changes"
  ```
- Include context IDs for richer memory:
  ```bash
  python cli/flowdex_cli.py --task incident_review --user "Draft a customer update" --ctx runbook.postgres comms.template
  ```
- Provide tool candidates and a custom model:
  ```bash
  python cli/flowdex_cli.py --task repair --user "Regenerate with tool assistance" --tool post_incident_report status_page --model anthropic/claude-3-5-sonnet
  ```

All CLI commands respect the same `.env` file for API URL and auth credentials.

## Examples Directory

The `examples/` folder contains end-to-end scenarios you can run as templates:

- `examples/incident_triage.ipynb` – notebook illustrating memory seeding, inference, and retries.
- `examples/tool_router.py` – Python script registering tools and demonstrating the router choosing them over free-form LLM output.
- `examples/n8n_flow.json` – importable n8n workflow connecting FlowDex to Slack + PagerDuty.

Each example is annotated with the commands needed to reproduce the run, making it easy to adapt to your own workflows.

## Contributing

Interested in improving FlowDex? Read the [contribution guide](CONTRIBUTING.md) for setup instructions and the recommended debugging workflow (automated tests ➝ Playwright verification ➝ manual checks only when unavoidable). The guide also includes a pull-request checklist to keep changes ship-ready.

---

© 2025 FlowDex. Generated 2025-10-23T00:31:05.007257Z
