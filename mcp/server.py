#!/usr/bin/env python3
import json, sys, time

def respond(id, result=None, error=None):
    msg = {"jsonrpc":"2.0","id":id}
    if error is not None:
        msg["error"] = {"code": -32000, "message": error}
    else:
        msg["result"] = result
    sys.stdout.write(json.dumps(msg)+"\n")
    sys.stdout.flush()

def main():
    for line in sys.stdin:
        try:
            req = json.loads(line.strip())
        except Exception:
            continue
        method = req.get("method")
        id_ = req.get("id")
        params = req.get("params", {})

        if method == "ping":
            respond(id_, {"ok": True, "ts": time.time()})
        elif method == "flowdex.infer":
            respond(id_, {"message":"call FastAPI /infer from here with your params", "echo": params})
        elif method == "flowdex.memory.get":
            respond(id_, {"message":"call FastAPI /memory/get", "echo": params})
        else:
            respond(id_, error=f"Unknown method: {method}")

if __name__ == "__main__":
    main()
