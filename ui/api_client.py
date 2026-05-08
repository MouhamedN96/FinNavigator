"""Async HTTP client to the FinNavigator FastAPI backend.

Web build:    talks to FINNAV_API_URL (Cloudflare → Fly).
Desktop:      talks to http://127.0.0.1:8000 (local uvicorn spawned by launcher).
"""

from __future__ import annotations

import json
import os
from typing import Any, AsyncIterator, Dict, List, Optional

import httpx

DEFAULT_TIMEOUT = httpx.Timeout(60.0, connect=10.0)


def base_url() -> str:
    return os.getenv("FINNAV_API_URL", "http://127.0.0.1:8000").rstrip("/")


async def health() -> Dict[str, Any]:
    async with httpx.AsyncClient(timeout=DEFAULT_TIMEOUT) as c:
        r = await c.get(f"{base_url()}/api/health")
        r.raise_for_status()
        return r.json()


async def agent_run(prompt: str, image_b64: Optional[str] = None) -> Dict[str, Any]:
    async with httpx.AsyncClient(timeout=DEFAULT_TIMEOUT) as c:
        r = await c.post(
            f"{base_url()}/api/agent/run",
            json={"prompt": prompt, "image_b64": image_b64, "show_reasoning": True},
        )
        r.raise_for_status()
        return r.json()


async def research(ticker: str, focus: str, depth: str) -> Dict[str, Any]:
    async with httpx.AsyncClient(timeout=DEFAULT_TIMEOUT) as c:
        r = await c.post(
            f"{base_url()}/api/research",
            json={"ticker": ticker, "focus": focus, "depth": depth},
        )
        r.raise_for_status()
        return r.json()


async def portfolio_analyze(positions: List[Dict[str, Any]], analysis_type: str) -> Dict[str, Any]:
    async with httpx.AsyncClient(timeout=DEFAULT_TIMEOUT) as c:
        r = await c.post(
            f"{base_url()}/api/portfolio/analyze",
            json={"positions": positions, "analysis_type": analysis_type},
        )
        r.raise_for_status()
        return r.json()


async def team_status() -> Dict[str, Any]:
    async with httpx.AsyncClient(timeout=DEFAULT_TIMEOUT) as c:
        r = await c.get(f"{base_url()}/api/team/status")
        r.raise_for_status()
        return r.json()


async def get_config() -> Dict[str, Any]:
    async with httpx.AsyncClient(timeout=DEFAULT_TIMEOUT) as c:
        r = await c.get(f"{base_url()}/api/config")
        r.raise_for_status()
        return r.json()


async def set_backend(name: str) -> Dict[str, Any]:
    async with httpx.AsyncClient(timeout=DEFAULT_TIMEOUT) as c:
        r = await c.post(f"{base_url()}/api/config/backend", json={"name": name})
        r.raise_for_status()
        return r.json()


# ---- tools ----

async def list_tools() -> Dict[str, Any]:
    async with httpx.AsyncClient(timeout=DEFAULT_TIMEOUT) as c:
        r = await c.get(f"{base_url()}/api/tools")
        r.raise_for_status()
        return r.json()


async def run_tool(name: str, args: Dict[str, Any]) -> Dict[str, Any]:
    async with httpx.AsyncClient(timeout=DEFAULT_TIMEOUT) as c:
        r = await c.post(f"{base_url()}/api/tools/{name}/run", json={"args": args})
        r.raise_for_status()
        return r.json()


# ---- channels ----

async def list_channels() -> Dict[str, Any]:
    async with httpx.AsyncClient(timeout=DEFAULT_TIMEOUT) as c:
        r = await c.get(f"{base_url()}/api/channels")
        r.raise_for_status()
        return r.json()


async def test_channel(user_id: str = "finnav_test_user", message: str = "FinNavigator test ping.") -> Dict[str, Any]:
    async with httpx.AsyncClient(timeout=DEFAULT_TIMEOUT) as c:
        r = await c.post(f"{base_url()}/api/channels/test", json={"user_id": user_id, "message": message})
        r.raise_for_status()
        return r.json()


# ---- alerts ----

async def list_alerts() -> Dict[str, Any]:
    async with httpx.AsyncClient(timeout=DEFAULT_TIMEOUT) as c:
        r = await c.get(f"{base_url()}/api/alerts")
        r.raise_for_status()
        return r.json()


async def create_alert(ticker: str, condition: str, threshold: float, user_id: str = "finnav_test_user", channel: str = "voiceflow") -> Dict[str, Any]:
    async with httpx.AsyncClient(timeout=DEFAULT_TIMEOUT) as c:
        r = await c.post(f"{base_url()}/api/alerts", json={
            "ticker": ticker, "condition": condition, "threshold": threshold,
            "user_id": user_id, "channel": channel,
        })
        r.raise_for_status()
        return r.json()


async def delete_alert(alert_id: int) -> Dict[str, Any]:
    async with httpx.AsyncClient(timeout=DEFAULT_TIMEOUT) as c:
        r = await c.delete(f"{base_url()}/api/alerts/{alert_id}")
        r.raise_for_status()
        return r.json()


# ---- chat history ----

async def get_history(limit: int = 100) -> Dict[str, Any]:
    async with httpx.AsyncClient(timeout=DEFAULT_TIMEOUT) as c:
        r = await c.get(f"{base_url()}/api/memory/history", params={"limit": limit})
        r.raise_for_status()
        return r.json()


async def clear_history() -> Dict[str, Any]:
    async with httpx.AsyncClient(timeout=DEFAULT_TIMEOUT) as c:
        r = await c.delete(f"{base_url()}/api/memory/clear")
        r.raise_for_status()
        return r.json()


# ---- tool toggles ----

async def get_enabled_tools() -> Dict[str, Any]:
    async with httpx.AsyncClient(timeout=DEFAULT_TIMEOUT) as c:
        r = await c.get(f"{base_url()}/api/tools/enabled")
        r.raise_for_status()
        return r.json()


async def set_enabled_tools(enabled: list[str]) -> Dict[str, Any]:
    async with httpx.AsyncClient(timeout=DEFAULT_TIMEOUT) as c:
        r = await c.post(f"{base_url()}/api/tools/enabled", json={"enabled": enabled})
        r.raise_for_status()
        return r.json()


# ---- file uploads ----

async def upload_file(filename: str, data: bytes) -> Dict[str, Any]:
    async with httpx.AsyncClient(timeout=httpx.Timeout(120.0, connect=10.0)) as c:
        r = await c.post(
            f"{base_url()}/api/upload",
            files={"file": (filename, data, "application/octet-stream")},
        )
        r.raise_for_status()
        return r.json()


async def list_uploads() -> Dict[str, Any]:
    async with httpx.AsyncClient(timeout=DEFAULT_TIMEOUT) as c:
        r = await c.get(f"{base_url()}/api/uploads")
        r.raise_for_status()
        return r.json()


# ---- schedules ----

async def list_schedules() -> Dict[str, Any]:
    async with httpx.AsyncClient(timeout=DEFAULT_TIMEOUT) as c:
        r = await c.get(f"{base_url()}/api/schedules")
        r.raise_for_status()
        return r.json()


async def create_schedule(name: str, cron: str, prompt: str, active: bool = True) -> Dict[str, Any]:
    async with httpx.AsyncClient(timeout=DEFAULT_TIMEOUT) as c:
        r = await c.post(
            f"{base_url()}/api/schedules",
            json={"name": name, "cron": cron, "prompt": prompt, "active": active},
        )
        r.raise_for_status()
        return r.json()


async def delete_schedule(job_id: str) -> Dict[str, Any]:
    async with httpx.AsyncClient(timeout=DEFAULT_TIMEOUT) as c:
        r = await c.delete(f"{base_url()}/api/schedules/{job_id}")
        r.raise_for_status()
        return r.json()


async def run_schedule_now(job_id: str) -> Dict[str, Any]:
    async with httpx.AsyncClient(timeout=httpx.Timeout(120.0, connect=10.0)) as c:
        r = await c.post(f"{base_url()}/api/schedules/{job_id}/run")
        r.raise_for_status()
        return r.json()


# ---- streaming agent ----

async def agent_stream(prompt: str) -> AsyncIterator[Dict[str, Any]]:
    """Async generator yielding SSE events from /api/agent/stream.

    Each yielded value is the parsed JSON dict from a `data: ...` line.
    The generator returns after yielding an event with type == "done".
    """
    async with httpx.AsyncClient(timeout=DEFAULT_TIMEOUT) as c:
        async with c.stream(
            "POST",
            f"{base_url()}/api/agent/stream",
            json={"prompt": prompt, "show_reasoning": False},
        ) as r:
            async for line in r.aiter_lines():
                if not line or not line.startswith("data: "):
                    continue
                payload = line[len("data: "):]
                try:
                    ev = json.loads(payload)
                except json.JSONDecodeError:
                    continue
                yield ev
                if isinstance(ev, dict) and ev.get("type") == "done":
                    return
