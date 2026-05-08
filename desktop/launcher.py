"""Desktop launcher.

Bundled with `flet build {windows,macos,linux}` as the entry point. Boot order:

    1. Bootstrap Ollama (start daemon, pull the fine-tuned GGUF if missing).
    2. Spawn uvicorn on 127.0.0.1:8000 with LLM_BACKEND=ollama.
    3. Wait for /api/health.
    4. Launch the Flet UI in the foreground.
    5. On exit, terminate the background processes.

Web build does NOT use this file — it talks to Fly.io over HTTPS instead.
"""

from __future__ import annotations

import atexit
import logging
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import httpx

from desktop import ollama_bootstrap

log = logging.getLogger("finnav.desktop.launcher")
logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))

API_HOST = "127.0.0.1"
API_PORT = int(os.getenv("FINNAV_DESKTOP_API_PORT", "8000"))
API_BASE = f"http://{API_HOST}:{API_PORT}"


def _wait_for_api(timeout_s: float = 60.0) -> None:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        try:
            r = httpx.get(f"{API_BASE}/api/health", timeout=2.0)
            if r.status_code == 200 and r.json().get("ok"):
                return
        except Exception:
            pass
        time.sleep(0.5)
    raise RuntimeError("FastAPI server did not become ready.")


def _spawn_api() -> subprocess.Popen:
    env = os.environ.copy()
    env.setdefault("LLM_BACKEND", "ollama")
    env.setdefault("OLLAMA_HOST", "http://127.0.0.1:11434")
    env.setdefault("OLLAMA_MODEL", ollama_bootstrap.DEFAULT_MODEL)
    log.info("Spawning uvicorn on %s …", API_BASE)
    return subprocess.Popen(
        [
            sys.executable, "-m", "uvicorn",
            "api.server:app",
            "--host", API_HOST,
            "--port", str(API_PORT),
            "--log-level", "warning",
        ],
        cwd=str(Path(__file__).resolve().parent.parent),
        env=env,
    )


def main() -> None:
    procs: list[subprocess.Popen] = []

    def _cleanup() -> None:
        for p in procs:
            if p.poll() is None:
                try:
                    p.terminate()
                    p.wait(timeout=5)
                except Exception:
                    p.kill()

    atexit.register(_cleanup)
    signal.signal(signal.SIGINT, lambda *_: (_cleanup(), sys.exit(0)))
    signal.signal(signal.SIGTERM, lambda *_: (_cleanup(), sys.exit(0)))

    try:
        ollama_proc = ollama_bootstrap.bootstrap()
        if ollama_proc:
            procs.append(ollama_proc)

        api_proc = _spawn_api()
        procs.append(api_proc)
        _wait_for_api()

        # Point the Flet client at our local API
        os.environ["FINNAV_API_URL"] = API_BASE

        # Hand off to Flet — runs in foreground until window closes
        import flet as ft

        from ui.app import main as ui_main

        ft.app(target=ui_main)
    finally:
        _cleanup()


if __name__ == "__main__":
    main()
