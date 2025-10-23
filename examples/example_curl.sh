#!/usr/bin/env bash
curl -s -X POST http://localhost:8787/infer   -H 'content-type: application/json'   -d '{
    "task": "fix_connection",
    "user_input": "WordPress REST 403 intermittently; suggest cheaper tool path.",
    "system_prompt": "Be concise.",
    "context_ids": ["proeye_guides"],
    "tool_candidates": ["http_check","wp_restore"]
  }' | jq .
