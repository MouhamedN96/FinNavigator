"""
Cron Jobs Integration - Agent Task Automation
==============================================

Comprehensive cron-like scheduling for FinNavigator agents including:
- Built-in scheduled tasks (data refresh, reports, alerts)
- Agent task automation
- Webhook triggers
- Integration with existing cron systems

Author: MiniMax Agent
"""

from typing import Dict, List, Optional, Any, Callable
from datetime import datetime, timedelta
from dataclasses import dataclass, field
import json
import os
import asyncio
import threading

from .scheduler import TaskScheduler, TaskTrigger, TaskStatus, cron_validate, CRON_PATTERNS
from .report_scheduler import ReportScheduler, ReportType
from .alert_monitor import AlertMonitor, AlertType, AlertPriority, AlertCondition, create_alert_from_template


@dataclass
class CronJobConfig:
    """Configuration for a cron job"""
    job_id: str
    name: str
    description: str = ""

    # Schedule (cron expression)
    schedule: str

    # Task to execute
    task_type: str  # "sec_refresh", "report", "alert_check", "custom"
    task_config: Dict = field(default_factory=dict)

    # Execution settings
    enabled: bool = True
    timeout_seconds: int = 300
    retry_on_failure: bool = True
    max_retries: int = 3

    # Notification
    notify_on: List[str] = field(default_factory=list)  # ["success", "failure"]

    # Metadata
    created_at: datetime = field(default_factory=datetime.now)
    last_run: Optional[datetime] = None
    next_run: Optional[datetime] = None
    run_count: int = 0


class AgentCronManager:
    """
    Comprehensive cron job management for agents.

    Features:
    - Pre-built agent tasks (SEC refresh, reports, alerts)
    - Custom task registration
    - Cron expression validation
    - Execution history and statistics
    - Webhook integration

    Example:
        manager = AgentCronManager(agent_team=team)

        # Add SEC data refresh (every market open)
        manager.add_sec_refresh_job(
            name="Daily SEC Refresh",
            schedule="0 10 * * 1-5",  # 10 AM weekdays
            tickers=["NVDA", "AAPL", "TSLA"]
        )

        # Add portfolio report
        manager.add_report_job(
            name="Weekly Portfolio",
            schedule="0 18 * * 5",  # 6 PM Friday
            report_type="weekly_portfolio_review"
        )

        manager.start()
    """

    def __init__(self, agent_team: Optional[Any] = None):
        self.agent_team = agent_team
        self.scheduler = TaskScheduler()
        self.report_scheduler = ReportScheduler(agent_team)
        self.alert_monitor = AlertMonitor()

        self.jobs: Dict[str, CronJobConfig] = {}
        self.custom_tasks: Dict[str, Callable] = {}

        self.running = False
        self._setup_default_callbacks()

    def _setup_default_callbacks(self):
        """Setup default notification callbacks"""
        def on_task_failure(task, result):
            logger.info(f"Task {task.config.task_id} failed: {result.error}")
            # Could send notification here

        self.scheduler.on_task_failure = on_task_failure

    def validate_schedule(self, cron_expression: str) -> bool:
        """Validate a cron expression"""
        return cron_validate(cron_expression)

    def get_schedule_preview(self, cron_expression: str, count: int = 5) -> List[datetime]:
        """Preview next N run times for a cron expression"""
        from scheduler import croniter

        try:
            cron = croniter(cron_expression, datetime.now())
            return [cron.get_next(datetime) for _ in range(count)]
        except Exception:
            return []

    # =========================================================================
    # Pre-built Job Types
    # =========================================================================

    def add_sec_refresh_job(
        self,
        name: str,
        schedule: str,
        tickers: List[str],
        form_types: List[str] = None,
        description: str = ""
    ) -> CronJobConfig:
        """Add SEC data refresh job"""
        if form_types is None:
            form_types = ["10-Q"]

        job_id = f"sec_refresh_{len(self.jobs)}"

        config = CronJobConfig(
            job_id=job_id,
            name=name,
            description=description or f"Refresh SEC filings for {tickers}",
            schedule=schedule,
            task_type="sec_refresh",
            task_config={
                "tickers": tickers,
                "form_types": form_types
            }
        )

        self.jobs[job_id] = config

        # Add to scheduler
        self.scheduler.add_task(
            task_id=job_id,
            name=name,
            trigger_type=TaskTrigger.CRON,
            cron_expression=schedule,
            task_function=self._exec_sec_refresh,
            notify_on_failure=True
        )

        return config

    def _exec_sec_refresh(self, job_id: str):
        """Execute SEC refresh task"""
        job = self.jobs.get(job_id)
        if not job:
            return {"error": "Job not found"}

        tickers = job.task_config.get("tickers", [])
        form_types = job.task_config.get("form_types", ["10-Q"])

        results = []
        for ticker in tickers:
            # In production, this would call the actual SEC API
            results.append({
                "ticker": ticker,
                "status": "success",
                "filings_updated": len(form_types)
            })

        job.last_run = datetime.now()
        job.run_count += 1

        return {"results": results}

    def add_report_job(
        self,
        name: str,
        schedule: str,
        report_type: str,
        recipients: List[str],
        platforms: List[str] = None,
        description: str = ""
    ) -> CronJobConfig:
        """Add scheduled report job"""
        job_id = f"report_{len(self.jobs)}"

        config = CronJobConfig(
            job_id=job_id,
            name=name,
            description=description or f"{report_type} report",
            schedule=schedule,
            task_type="report",
            task_config={
                "report_type": report_type,
                "recipients": recipients,
                "platforms": platforms or ["telegram"]
            }
        )

        self.jobs[job_id] = config

        # Add to report scheduler
        try:
            report_type_enum = ReportType(report_type)
        except ValueError:
            report_type_enum = ReportType.CUSTOM

        self.report_scheduler.add_report(
            report_type=report_type_enum,
            schedule=schedule,
            name=name,
            recipients=recipients,
            delivery_platforms=platforms
        )

        return config

    def add_alert_check_job(
        self,
        name: str,
        schedule: str,
        alert_template: str,
        recipients: List[str],
        **alert_params
    ) -> CronJobConfig:
        """Add alert monitoring job"""
        job_id = f"alert_{len(self.jobs)}"

        config = CronJobConfig(
            job_id=job_id,
            name=name,
            description=f"Alert check: {alert_template}",
            schedule=schedule,
            task_type="alert_check",
            task_config={
                "template": alert_template,
                "recipients": recipients,
                "params": alert_params
            }
        )

        self.jobs[job_id] = config

        # Add alert rule
        alert_params["recipients"] = recipients
        create_alert_from_template(self.alert_monitor, alert_template, **alert_params)

        return config

    def add_custom_job(
        self,
        name: str,
        schedule: str,
        task_function: Callable,
        task_config: Optional[Dict] = None,
        description: str = ""
    ) -> CronJobConfig:
        """Add custom task job"""
        job_id = f"custom_{len(self.jobs)}"

        config = CronJobConfig(
            job_id=job_id,
            name=name,
            description=description,
            schedule=schedule,
            task_type="custom",
            task_config=task_config or {}
        )

        self.jobs[job_id] = config
        self.custom_tasks[job_id] = task_function

        # Add to scheduler
        self.scheduler.add_task(
            task_id=job_id,
            name=name,
            trigger_type=TaskTrigger.CRON,
            cron_expression=schedule,
            task_function=lambda jid=job_id: self._exec_custom_task(jid),
            notify_on_failure=True
        )

        return config

    def _exec_custom_task(self, job_id: str):
        """Execute custom task"""
        job = self.jobs.get(job_id)
        if not job:
            return {"error": "Job not found"}

        task_func = self.custom_tasks.get(job_id)
        if not task_func:
            return {"error": "Task function not found"}

        try:
            result = task_func(**job.task_config)
            job.last_run = datetime.now()
            job.run_count += 1
            return result
        except Exception as e:
            return {"error": str(e)}

    # =========================================================================
    # Webhook Integration
    # =========================================================================

    def add_webhook_trigger(
        self,
        name: str,
        endpoint: str,
        task_function: Callable,
        auth_token: Optional[str] = None
    ) -> Dict:
        """Add webhook trigger for on-demand task execution"""
        webhook_id = f"webhook_{len(self.jobs)}"

        # Store webhook configuration
        webhook_config = {
            "webhook_id": webhook_id,
            "name": name,
            "endpoint": endpoint,
            "auth_token": auth_token,
            "enabled": True
        }

        self.jobs[webhook_id] = CronJobConfig(
            job_id=webhook_id,
            name=name,
            description=f"Webhook trigger: {endpoint}",
            schedule="* * * * *",  # Not scheduled, triggered externally
            task_type="webhook"
        )

        # In production, this would setup the actual webhook endpoint
        return webhook_config

    # =========================================================================
    # Job Management
    # =========================================================================

    def get_job(self, job_id: str) -> Optional[CronJobConfig]:
        """Get job configuration"""
        return self.jobs.get(job_id)

    def list_jobs(self, include_disabled: bool = True) -> List[CronJobConfig]:
        """List all jobs"""
        jobs = list(self.jobs.values())
        if not include_disabled:
            jobs = [j for j in jobs if j.enabled]
        return jobs

    def enable_job(self, job_id: str) -> bool:
        """Enable a job"""
        job = self.jobs.get(job_id)
        if job:
            job.enabled = True
            self.scheduler.enable_task(job_id)
            return True
        return False

    def disable_job(self, job_id: str) -> bool:
        """Disable a job"""
        job = self.jobs.get(job_id)
        if job:
            job.enabled = False
            self.scheduler.disable_task(job_id)
            return True
        return False

    def remove_job(self, job_id: str) -> bool:
        """Remove a job"""
        if job_id in self.jobs:
            del self.jobs[job_id]
            self.scheduler.remove_task(job_id)
            self.report_scheduler.remove_report(job_id)
            return True
        return False

    def trigger_job(self, job_id: str) -> Dict:
        """Manually trigger a job"""
        job = self.jobs.get(job_id)
        if not job:
            return {"error": "Job not found"}

        result = self.scheduler.trigger_task(job_id)
        return result.to_dict()

    # =========================================================================
    # Statistics and Monitoring
    # =========================================================================

    def get_next_scheduled_jobs(self, limit: int = 10) -> List[Dict]:
        """Get upcoming scheduled jobs"""
        upcoming = self.scheduler.get_next_scheduled_tasks(limit)

        results = []
        for task in upcoming:
            job_id = task["task_id"]
            job = self.jobs.get(job_id)
            if job:
                results.append({
                    "job_id": job_id,
                    "name": job.name,
                    "description": job.description,
                    "task_type": job.task_type,
                    "next_run": task["next_run"]
                })

        return results

    def get_execution_history(self, job_id: Optional[str] = None, limit: int = 50) -> List[Dict]:
        """Get execution history"""
        history = self.scheduler.get_execution_history(task_id=job_id, limit=limit)
        return [r.to_dict() for r in history]

    def get_stats(self) -> Dict:
        """Get overall statistics"""
        scheduler_stats = self.scheduler.get_task_stats()
        report_stats = {
            "total_reports": len(self.report_scheduler.reports),
            "reports_generated": len(self.report_scheduler.generated_reports)
        }
        alert_stats = self.alert_monitor.get_stats()

        return {
            "scheduler": scheduler_stats,
            "reports": report_stats,
            "alerts": alert_stats,
            "total_jobs": len(self.jobs),
            "enabled_jobs": sum(1 for j in self.jobs.values() if j.enabled)
        }

    # =========================================================================
    # Lifecycle Management
    # =========================================================================

    def start(self):
        """Start all scheduled services"""
        self.running = True
        self.scheduler.start()
        self.report_scheduler.start()
        self.alert_monitor.start()

    def stop(self):
        """Stop all scheduled services"""
        self.running = False
        self.scheduler.stop()
        self.report_scheduler.stop()
        self.alert_monitor.stop()

    def export_config(self) -> Dict:
        """Export all job configurations"""
        return {
            "exported_at": datetime.now().isoformat(),
            "jobs": [
                {
                    "job_id": j.job_id,
                    "name": j.name,
                    "description": j.description,
                    "schedule": j.schedule,
                    "task_type": j.task_type,
                    "task_config": j.task_config,
                    "enabled": j.enabled,
                    "run_count": j.run_count
                }
                for j in self.jobs.values()
            ],
            "stats": self.get_stats()
        }

    def import_config(self, config: Dict):
        """Import job configurations"""
        for job_data in config.get("jobs", []):
            task_type = job_data.get("task_type")

            if task_type == "sec_refresh":
                self.add_sec_refresh_job(
                    name=job_data["name"],
                    schedule=job_data["schedule"],
                    tickers=job_data["task_config"].get("tickers", []),
                    description=job_data.get("description")
                )
            elif task_type == "report":
                self.add_report_job(
                    name=job_data["name"],
                    schedule=job_data["schedule"],
                    report_type=job_data["task_config"].get("report_type"),
                    recipients=job_data["task_config"].get("recipients", [])
                )


# Import logger
import logging
logger = logging.getLogger(__name__)