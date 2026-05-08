"""APScheduler-based recurring research jobs.

Each job runs `state.supervisor.process(prompt)` on a cron schedule and writes
the answer to chat_history.

Persistence: jobs live in `data/schedules.json`. Load on startup, write on every
mutation. State is global (one job list for the whole app).
"""

from __future__ import annotations

import json
import logging
import os
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Awaitable, Callable, Dict, List, Optional
from uuid import uuid4

from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger

log = logging.getLogger("finnav.scheduler")

_LOCK = threading.Lock()


def _store_path() -> Path:
    p = Path(os.getenv("FINNAV_SCHEDULES_PATH", "data/schedules.json"))
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def _load_jobs() -> List[Dict[str, Any]]:
    p = _store_path()
    if not p.exists():
        return []
    try:
        with _LOCK:
            return json.loads(p.read_text(encoding="utf-8") or "[]")
    except Exception:
        return []


def _save_jobs(jobs: List[Dict[str, Any]]) -> None:
    p = _store_path()
    with _LOCK:
        p.write_text(json.dumps(jobs, indent=2), encoding="utf-8")


class ScheduleService:
    """Singleton owning the AsyncIOScheduler and the JSON-backed job list.

    Pass `runner` at start_async() — a callable that takes a prompt string and
    returns an awaitable result string. We don't import the supervisor here to
    keep this module decoupled from the agent layer.
    """

    def __init__(self) -> None:
        self.scheduler = AsyncIOScheduler()
        self.jobs: List[Dict[str, Any]] = []
        self.runner: Optional[Callable[[str], Awaitable[str]]] = None

    # ---- lifecycle ----

    def start(self, runner: Callable[[str], Awaitable[str]]) -> None:
        self.runner = runner
        self.jobs = _load_jobs()
        if not self.scheduler.running:
            self.scheduler.start()
        for job in self.jobs:
            if job.get("active", True):
                self._register(job)
        log.info("ScheduleService started with %d jobs", len(self.jobs))

    def shutdown(self) -> None:
        try:
            if self.scheduler.running:
                self.scheduler.shutdown(wait=False)
        except Exception:
            pass

    # ---- job mgmt ----

    def list(self) -> List[Dict[str, Any]]:
        return list(self.jobs)

    def add(self, name: str, cron: str, prompt: str, active: bool = True) -> Dict[str, Any]:
        job = {
            "id": uuid4().hex[:12],
            "name": name,
            "cron": cron,
            "prompt": prompt,
            "active": active,
            "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "last_run": None,
            "last_result": None,
        }
        self.jobs.append(job)
        _save_jobs(self.jobs)
        if active:
            self._register(job)
        return job

    def remove(self, job_id: str) -> bool:
        before = len(self.jobs)
        self.jobs = [j for j in self.jobs if j["id"] != job_id]
        _save_jobs(self.jobs)
        try:
            self.scheduler.remove_job(job_id)
        except Exception:
            pass
        return len(self.jobs) < before

    async def run_now(self, job_id: str) -> Dict[str, Any]:
        job = next((j for j in self.jobs if j["id"] == job_id), None)
        if not job:
            raise KeyError(f"job {job_id} not found")
        return await self._tick(job)

    # ---- internals ----

    def _register(self, job: Dict[str, Any]) -> None:
        try:
            trigger = CronTrigger.from_crontab(job["cron"])
        except Exception as e:
            log.warning("Invalid cron %r for job %s: %s", job["cron"], job["id"], e)
            return
        self.scheduler.add_job(
            self._tick,
            trigger=trigger,
            args=[job],
            id=job["id"],
            replace_existing=True,
            misfire_grace_time=60,
        )

    async def _tick(self, job: Dict[str, Any]) -> Dict[str, Any]:
        if not self.runner:
            log.warning("ScheduleService tick fired but no runner registered")
            return {"ok": False, "error": "no runner"}
        log.info("Schedule tick: %s — %s", job["name"], job["prompt"][:80])
        try:
            result = await self.runner(job["prompt"])
            job["last_run"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
            job["last_result"] = (result or "")[:2000]
            _save_jobs(self.jobs)
            return {"ok": True, "result": job["last_result"]}
        except Exception as e:
            job["last_run"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
            job["last_result"] = f"ERROR: {type(e).__name__}: {e}"
            _save_jobs(self.jobs)
            log.exception("Schedule tick failed for %s", job["name"])
            return {"ok": False, "error": str(e)}


# Module-level singleton
service = ScheduleService()
