#!/usr/bin/env python3
import argparse, requests, json, os

def main():
  p = argparse.ArgumentParser()
  p.add_argument("--base", default=os.environ.get("FLOWDEX_BASE","http://localhost:8787"))
  p.add_argument("--task", default="general")
  p.add_argument("--user", default="")
  p.add_argument("--system", default="")
  p.add_argument("--ctx", nargs="*", default=[])
  p.add_argument("--model", default="anthropic/claude-3-5-sonnet")
  args = p.parse_args()

  payload = {
    "task": args.task,
    "user_input": args.user,
    "system_prompt": args.system,
    "context_ids": args.ctx,
    "model": args.model
  }
  r = requests.post(f"{args.base}/infer", json=payload, timeout=30)
  r.raise_for_status()
  print(json.dumps(r.json(), indent=2))

if __name__ == "__main__":
  main()
