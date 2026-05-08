"""Append-only JSONL store for chat turns.

One file shared across all sessions for v1. Each line is a single turn:
    {"role": "user" | "assistant", "content": "...", "ts": "<iso>", "backend": "ollama"}

Read paths slice the tail. Clear truncates the file.
"""

from __future__ import annotations

import json
import os
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

_LOCK = threading.Lock()


def _store_path() -> Path:
    base = os.getenv("FINNAV_CHAT_HISTORY_PATH", "data/chat_history.jsonl")
    p = Path(base)
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def append(role: str, content: str, backend: str = "", reasoning: List[Dict[str, Any]] | None = None) -> Dict[str, Any]:
    entry = {
        "role": role,
        "content": content,
        "ts": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "backend": backend,
    }
    if reasoning:
        entry["reasoning"] = reasoning
    with _LOCK, _store_path().open("a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    return entry


def history(limit: int = 100) -> List[Dict[str, Any]]:
    """Return the most recent `limit` turns, oldest-first."""
    p = _store_path()
    if not p.exists():
        return []
    with _LOCK:
        lines = p.read_text(encoding="utf-8").splitlines()
    out: List[Dict[str, Any]] = []
    for line in lines[-limit:]:
        line = line.strip()
        if not line:
            continue
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return out


def clear() -> int:
    """Truncate the store. Returns number of turns removed."""
    p = _store_path()
    if not p.exists():
        return 0
    with _LOCK:
        n = sum(1 for _ in p.open(encoding="utf-8"))
        p.write_text("", encoding="utf-8")
    return n
