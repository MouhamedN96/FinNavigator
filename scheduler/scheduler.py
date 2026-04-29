"""
Task Scheduler - Cron-like Scheduling for Agents
==================================================

Provides cron-like scheduling capabilities for automated agent tasks.

Features:
- Cron expression parsing
- Interval-based scheduling
- One-time and recurring tasks
- Task dependencies
- Execution history
- Error handling and retries

Author: MiniMax Agent
"""

from typing import Callable, Dict, List, Optional, Any, Union
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
import asyncio
import threading
import time
import json
import re
from croniter import croniter
import logging

logger = logging.getLogger(__name__)


class TaskStatus(Enum):
    """Task execution status"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    SKIPPED = "skipped"


class TaskTrigger(Enum):
    """Task trigger types"""
    CRON = "cron"           # Cron expression based
    INTERVAL = "interval"   # Fixed interval
    ONCE = "once"           # One-time execution
    MANUAL = "manual"       # Manual trigger
    EVENT = "event"         # Event-based trigger


@dataclass
class TaskConfig:
    """Configuration for a scheduled task"""
    task_id: str
    name: str
    description: Optional[str] = None
    trigger_type: TaskTrigger = TaskTrigger.MANUAL

    # Cron expression (for CRON trigger)
    cron_expression: Optional[str] = None

    # Interval (for INTERVAL trigger)
    interval_seconds: Optional[int] = None

    # One-time execution (for ONCE trigger)
    run_at: Optional[datetime] = None

    # Execution settings
    enabled: bool = True
    max_retries: int = 3
    retry_delay_seconds: int = 60
    timeout_seconds: int = 300

    # Notification settings
    notify_on_success: bool = False
    notify_on_failure: bool = True
    notification_channels: List[str] = field(default_factory=list)


@dataclass
class TaskResult:
    """Result of task execution"""
    task_id: str
    status: TaskStatus
    started_at: datetime
    completed_at: Optional[datetime] = None
    duration_seconds: Optional[float] = None
    output: Optional[Any] = None
    error: Optional[str] = None
    retry_count: int = 0

    def to_dict(self) -> Dict:
        return {
            "task_id": self.task_id,
            "status": self.status.value,
            "started_at": self.started_at.isoformat(),
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "duration_seconds": self.duration_seconds,
            "output": str(self.output) if self.output else None,
            "error": self.error,
            "retry_count": self.retry_count
        }


@dataclass
class ScheduledTask:
    """Represents a scheduled task"""
    config: TaskConfig
    task_function: Optional[Callable] = None
    args: tuple = field(default_factory=tuple)
    kwargs: Dict = field(default_factory=dict)

    last_run: Optional[datetime] = None
    next_run: Optional[datetime] = None
    run_count: int = 0
    consecutive_failures: int = 0

    status: TaskStatus = TaskStatus.PENDING
    results: List[TaskResult] = field(default_factory=list)

    def get_next_run_time(self) -> Optional[datetime]:
        """Calculate next run time based on trigger"""
        now = datetime.now()

        if self.config.trigger_type == TaskTrigger.CRON:
            if self.config.cron_expression:
                try:
                    cron = croniter(self.config.cron_expression, now)
                    return cron.get_next(datetime)
                except Exception:
                    return None

        elif self.config.trigger_type == TaskTrigger.INTERVAL:
            if self.config.interval_seconds:
                if self.last_run:
                    return self.last_run + timedelta(seconds=self.config.interval_seconds)
                return now

        elif self.config.trigger_type == TaskTrigger.ONCE:
            return self.config.run_at

        return None


class TaskScheduler:
    """
    Main scheduler for managing and executing tasks.

    Features:
    - Cron-based scheduling
    - Interval-based scheduling
    - Task dependencies
    - Retry logic
    - Execution history
    - Thread-safe operation

    Example:
        scheduler = TaskScheduler()

        # Add a daily task
        scheduler.add_task(
            task_id="daily_report",
            name="Daily Report",
            trigger_type=TaskTrigger.CRON,
            cron_expression="0 8 * * *",  # 8 AM daily
            task_function=generate_daily_report
        )

        # Start scheduler
        scheduler.start()
    """

    def __init__(self, enable_webhooks: bool = True):
        self.tasks: Dict[str, ScheduledTask] = {}
        self.running = False
        self.scheduler_thread: Optional[threading.Thread] = None
        self.lock = threading.RLock()
        self.enable_webhooks = enable_webhooks

        # Execution history
        self.execution_history: List[TaskResult] = []
        self.max_history_size = 1000

        # Callbacks
        self.on_task_start: Optional[Callable] = None
        self.on_task_complete: Optional[Callable] = None
        self.on_task_failure: Optional[Callable] = None

        logger.info("TaskScheduler initialized")

    def add_task(
        self,
        task_id: str,
        name: str,
        task_function: Optional[Callable] = None,
        trigger_type: TaskTrigger = TaskTrigger.MANUAL,
        cron_expression: Optional[str] = None,
        interval_seconds: Optional[int] = None,
        run_at: Optional[datetime] = None,
        description: Optional[str] = None,
        **kwargs
    ) -> ScheduledTask:
        """Add a new task to the scheduler"""
        with self.lock:
            config = TaskConfig(
                task_id=task_id,
                name=name,
                description=description,
                trigger_type=trigger_type,
                cron_expression=cron_expression,
                interval_seconds=interval_seconds,
                run_at=run_at,
                **kwargs
            )

            task = ScheduledTask(
                config=config,
                task_function=task_function,
                args=kwargs.get("args", ()),
                kwargs=kwargs.get("kwargs", {})
            )

            task.next_run = task.get_next_run_time()
            self.tasks[task_id] = task

            logger.info(f"Added task: {task_id} ({trigger_type.value})")
            return task

    def remove_task(self, task_id: str) -> bool:
        """Remove a task from the scheduler"""
        with self.lock:
            if task_id in self.tasks:
                del self.tasks[task_id]
                logger.info(f"Removed task: {task_id}")
                return True
            return False

    def get_task(self, task_id: str) -> Optional[ScheduledTask]:
        """Get a task by ID"""
        return self.tasks.get(task_id)

    def list_tasks(self, status_filter: Optional[TaskStatus] = None) -> List[ScheduledTask]:
        """List all tasks, optionally filtered by status"""
        with self.lock:
            tasks = list(self.tasks.values())
            if status_filter:
                tasks = [t for t in tasks if t.status == status_filter]
            return tasks

    def trigger_task(self, task_id: str, **kwargs) -> TaskResult:
        """Manually trigger a task"""
        task = self.get_task(task_id)
        if not task:
            return TaskResult(
                task_id=task_id,
                status=TaskStatus.FAILED,
                started_at=datetime.now(),
                error=f"Task {task_id} not found"
            )

        return self._execute_task(task, **kwargs)

    def enable_task(self, task_id: str) -> bool:
        """Enable a task"""
        task = self.get_task(task_id)
        if task:
            task.config.enabled = True
            task.next_run = task.get_next_run_time()
            return True
        return False

    def disable_task(self, task_id: str) -> bool:
        """Disable a task"""
        task = self.get_task(task_id)
        if task:
            task.config.enabled = False
            return True
        return False

    def _execute_task(self, task: ScheduledTask, **override_kwargs) -> TaskResult:
        """Execute a task with retry logic"""
        start_time = datetime.now()
        result = TaskResult(
            task_id=task.config.task_id,
            status=TaskStatus.RUNNING,
            started_at=start_time
        )

        task.status = TaskStatus.RUNNING

        # Call start callback
        if self.on_task_start:
            try:
                self.on_task_start(task)
            except Exception as e:
                logger.warning(f"Task start callback error: {e}")

        retry_count = 0
        max_retries = task.config.max_retries

        while retry_count <= max_retries:
            try:
                # Execute task function
                if task.task_function:
                    args = override_kwargs.get("args", task.args)
                    kwargs = override_kwargs.get("kwargs", task.kwargs)

                    # Handle async functions
                    if asyncio.iscoroutinefunction(task.task_function):
                        loop = asyncio.new_event_loop()
                        output = loop.run_until_complete(task.task_function(*args, **kwargs))
                        loop.close()
                    else:
                        output = task.task_function(*args, **kwargs)

                    result.output = output
                    result.status = TaskStatus.COMPLETED
                else:
                    # No task function - just mark as completed
                    result.status = TaskStatus.COMPLETED

                break  # Exit retry loop on success

            except Exception as e:
                logger.error(f"Task {task.config.task_id} failed: {e}")
                retry_count += 1
                result.retry_count = retry_count

                if retry_count <= max_retries:
                    logger.info(f"Retrying task {task.config.task_id} in {task.config.retry_delay_seconds}s")
                    time.sleep(task.config.retry_delay_seconds)
                else:
                    result.status = TaskStatus.FAILED
                    result.error = str(e)

        # Calculate duration
        end_time = datetime.now()
        result.completed_at = end_time
        result.duration_seconds = (end_time - start_time).total_seconds()

        # Update task state
        task.status = result.status
        task.last_run = start_time
        task.run_count += 1

        if result.status == TaskStatus.FAILED:
            task.consecutive_failures += 1
        else:
            task.consecutive_failures = 0

        # Calculate next run
        task.next_run = task.get_next_run_time()

        # Store result
        self.execution_history.append(result)
        if len(self.execution_history) > self.max_history_size:
            self.execution_history = self.execution_history[-self.max_history_size:]

        # Call completion callbacks
        if result.status == TaskStatus.COMPLETED:
            if self.on_task_complete and task.config.notify_on_success:
                try:
                    self.on_task_complete(task, result)
                except Exception as e:
                    logger.warning(f"Task complete callback error: {e}")
        elif result.status == TaskStatus.FAILED:
            if self.on_task_failure and task.config.notify_on_failure:
                try:
                    self.on_task_failure(task, result)
                except Exception as e:
                    logger.warning(f"Task failure callback error: {e}")

        logger.info(f"Task {task.config.task_id} {result.status.value} in {result.duration_seconds:.2f}s")
        return result

    def _scheduler_loop(self):
        """Main scheduler loop - runs in background thread"""
        logger.info("Scheduler loop started")

        while self.running:
            try:
                now = datetime.now()

                with self.lock:
                    tasks_to_run = []

                    for task in self.tasks.values():
                        if not task.config.enabled:
                            continue

                        if task.next_run and task.next_run <= now:
                            tasks_to_run.append(task)

                # Execute tasks (can run in parallel if needed)
                for task in tasks_to_run:
                    # Run in thread to avoid blocking scheduler
                    thread = threading.Thread(
                        target=self._execute_task,
                        args=(task,),
                        daemon=True
                    )
                    thread.start()

                # Sleep for a short interval
                time.sleep(1)

            except Exception as e:
                logger.error(f"Scheduler loop error: {e}")
                time.sleep(5)  # Longer sleep on error

    def start(self):
        """Start the scheduler"""
        if self.running:
            logger.warning("Scheduler already running")
            return

        self.running = True
        self.scheduler_thread = threading.Thread(target=self._scheduler_loop, daemon=True)
        self.scheduler_thread.start()
        logger.info("Scheduler started")

    def stop(self):
        """Stop the scheduler"""
        self.running = False
        if self.scheduler_thread:
            self.scheduler_thread.join(timeout=5)
        logger.info("Scheduler stopped")

    def get_next_scheduled_tasks(self, limit: int = 10) -> List[Dict]:
        """Get upcoming scheduled tasks"""
        with self.lock:
            upcoming = []
            for task in self.tasks.values():
                if task.next_run and task.config.enabled:
                    upcoming.append({
                        "task_id": task.config.task_id,
                        "name": task.config.name,
                        "next_run": task.next_run.isoformat(),
                        "trigger_type": task.config.trigger_type.value,
                        "cron_expression": task.config.cron_expression
                    })

            upcoming.sort(key=lambda x: x["next_run"])
            return upcoming[:limit]

    def get_execution_history(
        self,
        task_id: Optional[str] = None,
        status: Optional[TaskStatus] = None,
        limit: int = 100
    ) -> List[TaskResult]:
        """Get execution history, optionally filtered"""
        history = self.execution_history

        if task_id:
            history = [r for r in history if r.task_id == task_id]

        if status:
            history = [r for r in history if r.status == status]

        return history[-limit:]

    def get_task_stats(self) -> Dict:
        """Get scheduler statistics"""
        with self.lock:
            total = len(self.tasks)
            enabled = sum(1 for t in self.tasks.values() if t.config.enabled)
            disabled = total - enabled

            running = sum(1 for t in self.tasks.values() if t.status == TaskStatus.RUNNING)
            completed = sum(1 for t in self.tasks.values() if t.status == TaskStatus.COMPLETED)
            failed = sum(1 for t in self.tasks.values() if t.status == TaskStatus.FAILED)

            return {
                "total_tasks": total,
                "enabled": enabled,
                "disabled": disabled,
                "running": running,
                "completed": completed,
                "failed": failed,
                "total_executions": len(self.execution_history),
                "is_running": self.running
            }

    def export_schedule(self) -> Dict:
        """Export all tasks and schedule for backup/migration"""
        with self.lock:
            tasks_data = []
            for task in self.tasks.values():
                tasks_data.append({
                    "config": {
                        "task_id": task.config.task_id,
                        "name": task.config.name,
                        "description": task.config.description,
                        "trigger_type": task.config.trigger_type.value,
                        "cron_expression": task.config.cron_expression,
                        "interval_seconds": task.config.interval_seconds,
                        "run_at": task.config.run_at.isoformat() if task.config.run_at else None,
                        "enabled": task.config.enabled,
                        "max_retries": task.config.max_retries,
                        "notify_on_success": task.config.notify_on_success,
                        "notify_on_failure": task.config.notify_on_failure
                    },
                    "stats": {
                        "run_count": task.run_count,
                        "consecutive_failures": task.consecutive_failures,
                        "last_run": task.last_run.isoformat() if task.last_run else None,
                        "next_run": task.next_run.isoformat() if task.next_run else None
                    }
                })

            return {
                "exported_at": datetime.now().isoformat(),
                "scheduler_stats": self.get_task_stats(),
                "tasks": tasks_data
            }


def cron_validate(expression: str) -> bool:
    """Validate a cron expression"""
    try:
        croniter(expression)
        return True
    except Exception:
        return False


def cron_next_run(expression: str, from_time: Optional[datetime] = None) -> Optional[datetime]:
    """Get next run time for a cron expression"""
    try:
        base = from_time or datetime.now()
        cron = croniter(expression, base)
        return cron.get_next(datetime)
    except Exception:
        return None


def cron_prev_run(expression: str, from_time: Optional[datetime] = None) -> Optional[datetime]:
    """Get previous run time for a cron expression"""
    try:
        base = from_time or datetime.now()
        cron = croniter(expression, base)
        return cron.get_prev(datetime)
    except Exception:
        return None


# Common cron patterns
CRON_PATTERNS = {
    "every_minute": "* * * * *",
    "every_5_minutes": "*/5 * * * *",
    "every_15_minutes": "*/15 * * * *",
    "every_30_minutes": "*/30 * * * *",
    "every_hour": "0 * * * *",
    "every_day_midnight": "0 0 * * *",
    "every_day_8am": "0 8 * * *",
    "every_day_6pm": "0 18 * * *",
    "every_week_monday": "0 9 * * 1",
    "every_month_first": "0 0 1 * *",
    "every_quarter": "0 0 1 */3 *",
    "market_open_weekdays": "30 9 * * 1-5",  # 9:30 AM weekdays
    "market_close_weekdays": "0 16 * * 1-5",  # 4:00 PM weekdays
}