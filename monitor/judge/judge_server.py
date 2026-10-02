#!/usr/bin/env python3
"""
judge_server.py  (monitor/judge/)

Loads the judge model once and serves it over localhost, so the agent process
(a different conda env, a different GPU) can call it live during a run.

Run in the gpt-oss env, pinned to the second GPU:

"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

# ---- GPU pinning + caches must precede any torch import ----
_pre = argparse.ArgumentParser(add_help=False)
_pre.add_argument("--gpu", default="1")
_args, _ = _pre.parse_known_args()
os.environ["CUDA_VISIBLE_DEVICES"] = str(_args.gpu)

_user = os.environ.get("USER") or "unknown"
_scratch = Path(f"/scratch/{_user}")
if _scratch.parent.exists():
    (_scratch / "hf_cache" / "hub").mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("HF_HOME", str(_scratch / "hf_cache"))
    os.environ.setdefault("HF_HUB_CACHE", str(_scratch / "hf_cache" / "hub"))

import json                                    
from http.server import BaseHTTPRequestHandler, HTTPServer   

BASE_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(BASE_DIR / "src"))

from inference_engine import build_inference_engine   
from chat_adapter import ChatAdapter                 


def make_handler(adapter, model_key):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *a):         
            pass

        def _send(self, code, payload):
            body = json.dumps(payload).encode("utf-8")
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            if self.path == "/health":
                self._send(200, {"ok": True, "model": model_key})
            else:
                self._send(404, {"error": "not found"})

        def do_POST(self):
            if self.path != "/chat":
                return self._send(404, {"error": "not found"})
            n = int(self.headers.get("Content-Length", 0))
            try:
                req = json.loads(self.rfile.read(n) or b"{}")
                messages = []
                if req.get("system"):
                    messages.append({"role": "system", "content": req["system"]})
                messages.append({"role": "user", "content": req.get("user", "")})
                text, _ = adapter.chat(messages)
                self._send(200, {"text": text})
            except Exception as exc:                      
                self._send(500, {"error": f"{type(exc).__name__}: {exc}"})

    return Handler


def main() -> None:
    ap = argparse.ArgumentParser(description="Serve the judge model over localhost.")
    ap.add_argument("--model", default="gptoss_120b", help="Judge model key in model_config.yaml")
    ap.add_argument("--gpu", default="1", help="GPU index for the judge")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--models-config", default=str(BASE_DIR / "configs" / "model_config.yaml"))
    args = ap.parse_args()

    print(f"Loading judge model {args.model} on GPU {args.gpu} ...")
    engine = build_inference_engine(args.model, config_path=args.models_config)
    adapter = ChatAdapter(engine)
    print("Judge ready.")

    srv = HTTPServer((args.host, args.port), make_handler(adapter, args.model))
    print(f"Listening on http://{args.host}:{args.port}  (Ctrl-C to stop)")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down.")


if __name__ == "__main__":
    main()