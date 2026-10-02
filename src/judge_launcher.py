"""
judge_launcher.py

Starts the judge server in its own conda env / GPU as a subprocess, waits until
the model is loaded, and shuts it down at exit. 
"""
from __future__ import annotations

import atexit
import json
import socket
import subprocess
import time
import urllib.request
from pathlib import Path


def _pick_port(preferred: int) -> int:
    """Return `preferred` if free, otherwise any free port."""
    with socket.socket() as s:
        try:
            s.bind(("127.0.0.1", preferred))
            return preferred
        except OSError:
            s.bind(("127.0.0.1", 0))
            return s.getsockname()[1]


def launch_judge(
    *, model: str, env: str, gpu: str, port: int,
    base_dir: Path, models_config: str, wait_seconds: int = 2400,
) -> tuple[subprocess.Popen, str]:
    """Spawn the judge server and block until /health responds. Returns (proc, url)."""
    port = _pick_port(port)
    url = f"http://127.0.0.1:{port}"
    server = Path(base_dir) / "monitor" / "judge" / "judge_server.py"
    if not server.exists():
        raise FileNotFoundError(f"judge server not found: {server}")

    cmd = ["conda", "run", "-n", env, "--no-capture-output", "python", str(server),
           "--model", model, "--gpu", str(gpu), "--port", str(port),
           "--models-config", str(models_config)]

    print(f"Starting judge '{model}' in env '{env}' on GPU {gpu} (port {port}) ...")
    proc = subprocess.Popen(cmd)

    def _stop() -> None:
        if proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=30)
            except subprocess.TimeoutExpired:
                proc.kill()
    atexit.register(_stop)

    deadline = time.time() + wait_seconds
    while time.time() < deadline:
        if proc.poll() is not None:
            raise RuntimeError(f"judge server exited during startup (code {proc.returncode})")
        try:
            with urllib.request.urlopen(f"{url}/health", timeout=5) as r:
                info = json.loads(r.read().decode("utf-8"))
            print(f"Judge ready: {info.get('model')} at {url}\n")
            return proc, url
        except Exception:                     # noqa: BLE001 - not up yet
            time.sleep(5)

    _stop()
    raise TimeoutError(f"judge server not ready after {wait_seconds}s")