"""Ollama runtime + model bootstrap for the FinNavigator desktop build.

On first launch:
  1. Verify `ollama` is installed; if not, surface an install link.
  2. Spawn `ollama serve` in the background if no daemon is running.
  3. `ollama pull` the fine-tuned GGUF from HuggingFace if not already present.

Ollama supports HF GGUF pulls directly:
    ollama pull hf.co/MOH749/finnav-qwen3-VL-4b-gguf:Q4_K_M
"""

from __future__ import annotations

import logging
import shutil
import subprocess
import time
from typing import Optional

import httpx

log = logging.getLogger("finnav.desktop.ollama")

OLLAMA_HOST = "http://127.0.0.1:11434"
DEFAULT_MODEL = "hf.co/MOH749/finnav-qwen3-VL-4b-gguf:Q4_K_M"
INSTALL_URL = "https://ollama.com/download"


class OllamaError(RuntimeError):
    """Raised when the desktop launcher can't satisfy an Ollama precondition."""


def _is_running() -> bool:
    try:
        r = httpx.get(f"{OLLAMA_HOST}/api/tags", timeout=2.0)
        return r.status_code == 200
    except Exception:
        return False


def ensure_installed() -> str:
    """Return the path to the ollama binary, or raise with an install link."""
    path = shutil.which("ollama")
    if not path:
        raise OllamaError(
            f"Ollama is not installed. Download from {INSTALL_URL} and re-run FinNavigator."
        )
    return path


def ensure_serving(timeout_s: float = 30.0) -> Optional[subprocess.Popen]:
    """Start `ollama serve` if no daemon is running. Returns the spawned process or None."""
    if _is_running():
        log.info("Ollama already serving on %s", OLLAMA_HOST)
        return None
    binary = ensure_installed()
    log.info("Starting `ollama serve` …")
    proc = subprocess.Popen(
        [binary, "serve"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        if _is_running():
            log.info("Ollama daemon up after %.1fs", timeout_s - (deadline - time.time()))
            return proc
        time.sleep(0.5)
    proc.terminate()
    raise OllamaError("`ollama serve` did not become ready within timeout.")


def model_present(model: str) -> bool:
    binary = ensure_installed()
    out = subprocess.run([binary, "list"], capture_output=True, text=True, check=False)
    return model in out.stdout


def ensure_model(model: str = DEFAULT_MODEL) -> None:
    """Pull the GGUF if not already present locally. ~2.5 GB on first run."""
    if model_present(model):
        log.info("Model %s already present.", model)
        return
    binary = ensure_installed()
    log.info("Pulling %s — this may take a few minutes on first launch …", model)
    rc = subprocess.run([binary, "pull", model]).returncode
    if rc != 0:
        raise OllamaError(f"`ollama pull {model}` failed (exit {rc}).")


def bootstrap(model: str = DEFAULT_MODEL) -> Optional[subprocess.Popen]:
    """One-call helper used by `desktop/launcher.py`."""
    proc = ensure_serving()
    ensure_model(model)
    return proc
