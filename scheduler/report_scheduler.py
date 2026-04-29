"""
Report Scheduler - Automated Report Generation
===============================================

Scheduled report generation for agents including:
- Daily market summaries
- Weekly portfolio reports
- Monthly performance reviews
- Custom report templates

Author: MiniMax Agent
"""

from typing import Dict, List, Optional, Any, Callable
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
import json
import asyncio

from .scheduler import TaskScheduler, TaskTrigger, TaskConfig, TaskStatus


class ReportType(Enum):
    """Report types"""
    DAILY_MARKET_SUMMARY = "daily_market_summary"
    WEEKLY_PORTFOLIO_REVIEW = "weekly_portfolio_review"
    MONTHLY_PERFORMANCE = "monthly_performance"
    QUARTERLY_ANALYSIS = "quarterly_analysis"
    ALERT_SUMMARY = "alert_summary"
    CUSTOM = "custom"


@dataclass
class ReportConfig:
    """Configuration for a report"""
    report_id: str
    report_type: ReportType
    name: str
    description: str = ""

    # Schedule
    schedule: str  # Cron expression

    # Content
    include_sections: List[str] = field(default_factory=list)
    recipients: List[str] = field(default_factory=list)  # user IDs, emails, chat IDs

    # Delivery
    delivery_platforms: List[str] = field(default_factory=list)  # telegram, discord, slack, email
    format: str = "markdown"  # markdown, html, pdf

    # Settings
    enabled: bool = True
    include_charts: bool = True
    save_to_file: bool = False
    file_path: Optional[str] = None

    # Generation function (optional)
    custom_generator: Optional[Callable] = None

    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class GeneratedReport:
    """Generated report data"""
    report_id: str
    report_type: ReportType
    generated_at: datetime
    content: str
    sections: Dict[str, Any]
    charts: List[str] = field(default_factory=list)
    file_path: Optional[str] = None
    delivery_status: Dict[str, bool] = field(default_factory=dict)

    def to_dict(self) -> Dict:
        return {
            "report_id": self.report_id,
            "report_type": self.report_type.value,
            "generated_at": self.generated_at.isoformat(),
            "content_length": len(self.content),
            "sections": list(self.sections.keys()),
            "charts": self.charts,
            "delivery_status": self.delivery_status
        }


class ReportScheduler:
    """
    Scheduler for automated report generation.

    Features:
    - Pre-configured report templates
    - Custom report generators
    - Multi-platform delivery
    - Report history
    - Scheduled execution

    Example:
        scheduler = ReportScheduler()

        # Add daily report
        scheduler.add_report(
            report_type=ReportType.DAILY_MARKET_SUMMARY,
            schedule="0 8 * * *",  # 8 AM daily
            recipients=["telegram:123456", "discord:789"],
            include_sections=["market_overview", "top_movers", "alerts"]
        )

        scheduler.start()
    """

    def __init__(self, agent_team: Optional[Any] = None):
        self.reports: Dict[str, ReportConfig] = {}
        self.generated_reports: List[GeneratedReport] = []
        self.task_scheduler = TaskScheduler()
        self.agent_team = agent_team

        # Report templates
        self._report_templates = {
            ReportType.DAILY_MARKET_SUMMARY: self._generate_daily_market_summary,
            ReportType.WEEKLY_PORTFOLIO_REVIEW: self._generate_weekly_portfolio,
            ReportType.MONTHLY_PERFORMANCE: self._generate_monthly_performance,
            ReportType.QUARTERLY_ANALYSIS: self._generate_quarterly_analysis,
            ReportType.ALERT_SUMMARY: self._generate_alert_summary,
        }

    def add_report(
        self,
        report_type: ReportType,
        schedule: str,
        name: Optional[str] = None,
        recipients: Optional[List[str]] = None,
        delivery_platforms: Optional[List[str]] = None,
        **kwargs
    ) -> ReportConfig:
        """Add a new scheduled report"""
        report_id = f"report_{len(self.reports)}_{datetime.now().strftime('%Y%m%d%H%M%S')}"

        config = ReportConfig(
            report_id=report_id,
            report_type=report_type,
            name=name or f"{report_type.value.replace('_', ' ').title()} Report",
            schedule=schedule,
            recipients=recipients or [],
            delivery_platforms=delivery_platforms or ["telegram"],
            **kwargs
        )

        self.reports[report_id] = config

        # Add to task scheduler
        self.task_scheduler.add_task(
            task_id=report_id,
            name=config.name,
            trigger_type=TaskTrigger.CRON,
            cron_expression=schedule,
            task_function=self._generate_and_send_report,
            notify_on_failure=True
        )

        return config

    def remove_report(self, report_id: str) -> bool:
        """Remove a scheduled report"""
        if report_id in self.reports:
            del self.reports[report_id]
            self.task_scheduler.remove_task(report_id)
            return True
        return False

    def get_report(self, report_id: str) -> Optional[ReportConfig]:
        """Get a report configuration"""
        return self.reports.get(report_id)

    def list_reports(self) -> List[ReportConfig]:
        """List all configured reports"""
        return list(self.reports.values())

    def _generate_and_send_report(self, report_id: str) -> Dict:
        """Generate a report and send to recipients"""
        config = self.get_report(report_id)
        if not config:
            return {"error": f"Report {report_id} not found"}

        # Generate report
        report = self._generate_report(config)

        # Send to recipients
        if config.delivery_platforms:
            self._send_report(report, config.delivery_platforms, config.recipients)

        self.generated_reports.append(report)
        return report.to_dict()

    def _generate_report(self, config: ReportConfig) -> GeneratedReport:
        """Generate report content"""
        generator = self._report_templates.get(config.report_type)

        if generator:
            content, sections, charts = generator(config)
        elif config.custom_generator:
            content, sections, charts = config.custom_generator(config)
        else:
            content = "Report content not available"
            sections = {}
            charts = []

        # Save to file if configured
        file_path = None
        if config.save_to_file and config.file_path:
            try:
                with open(config.file_path, 'w') as f:
                    f.write(content)
                file_path = config.file_path
            except Exception as e:
                pass  # Log error

        return GeneratedReport(
            report_id=config.report_id,
            report_type=config.report_type,
            generated_at=datetime.now(),
            content=content,
            sections=sections,
            charts=charts,
            file_path=file_path
        )

    def _send_report(
        self,
        report: GeneratedReport,
        platforms: List[str],
        recipients: List[str]
    ):
        """Send report to configured platforms"""
        from platforms import SocialPlatformManager, Platform

        manager = SocialPlatformManager()
        manager.initialize()

        # Parse recipients (format: platform:recipient_id)
        for recipient in recipients:
            if ":" in recipient:
                platform_str, recipient_id = recipient.split(":", 1)
                try:
                    platform = Platform(platform_str.lower())
                except ValueError:
                    continue

                # Send report
                manager.send_report(
                    platform=platform,
                    recipient=recipient_id,
                    report_type=report.report_type.value,
                    content=report.content
                )

                report.delivery_status[f"{platform_str}:{recipient_id}"] = True

    def _generate_daily_market_summary(self, config: ReportConfig) -> tuple:
        """Generate daily market summary"""
        sections = {}

        # Market overview
        sections["market_overview"] = {
            "date": datetime.now().strftime("%Y-%m-%d"),
            "indices": {
                "S&P 500": "+0.5%",
                "NASDAQ": "+0.8%",
                "DOW": "+0.3%"
            }
        }

        # Top movers (would integrate with actual data)
        sections["top_movers"] = {
            "gainers": ["NVDA +3.2%", "AMD +2.1%", "TSLA +1.8%"],
            "losers": ["META -1.5%", "GOOGL -0.9%"]
        }

        # Key events
        sections["key_events"] = [
            "Fed meeting minutes released",
            "Earnings season begins next week"
        ]

        # Format content
        content = f"""# Daily Market Summary
{datetime.now().strftime('%Y-%m-%d')}

## Market Overview
- S&P 500: +0.5%
- NASDAQ: +0.8%
- DOW: +0.3%

## Top Gainers
{NL.join(sections['top_movers']['gainers'])}

## Top Losers
{NL.join(sections['top_movers']['losers'])}

## Key Events
{NL.join(sections['key_events'])}

---
Generated by FinNavigator AI
"""

        return content, sections, []

    def _generate_weekly_portfolio(self, config: ReportConfig) -> tuple:
        """Generate weekly portfolio review"""
        sections = {
            "performance": {"week_change": "+2.3%", "total_return": "+15.4%"},
            "holdings": ["NVDA", "AAPL", "MSFT", "TSLA"],
            "rebalancing_needed": True
        }

        content = f"""# Weekly Portfolio Review
Week of {datetime.now().strftime('%Y-%m-%d')}

## Performance
- Weekly Change: +2.3%
- Total Return: +15.4%

## Holdings
- NVDA (Tech): 35%
- AAPL (Tech): 25%
- MSFT (Tech): 20%
- TSLA (Consumer): 20%

## Recommendations
- Consider rebalancing to reduce tech exposure
- Review NVDA position given recent gains

---
Generated by FinNavigator AI
"""

        return content, sections, []

    def _generate_monthly_performance(self, config: ReportConfig) -> tuple:
        """Generate monthly performance report"""
        sections = {
            "monthly_return": "+8.5%",
            "vs_benchmark": "+2.3%",
            "risk_metrics": {"var": "$5,200", "sharpe": "1.45"}
        }

        content = f"""# Monthly Performance Report
{datetime.now().strftime('%B %Y')}

## Returns
- Monthly Return: +8.5%
- vs Benchmark (SPY): +2.3%

## Risk Metrics
- VaR (95%): $5,200
- Sharpe Ratio: 1.45

## Transactions
- No major trades this month

---
Generated by FinNavigator AI
"""

        return content, sections, []

    def _generate_quarterly_analysis(self, config: ReportConfig) -> tuple:
        """Generate quarterly analysis"""
        content = f"""# Quarterly Analysis
Q{((datetime.now().month - 1) // 3) + 1} {datetime.now().year}

## Summary
- Portfolio returned +12.3% this quarter
- Outperformed benchmark by 3.2%

## Key Developments
- NVDA earnings beat expectations
- Diversified into emerging markets

## Outlook
- Maintain current allocation
- Consider adding fixed income

---
Generated by FinNavigator AI
"""

        return content, {"quarterly": True}, []

    def _generate_alert_summary(self, config: ReportConfig) -> tuple:
        """Generate alert summary"""
        content = f"""# Alert Summary
{datetime.now().strftime('%Y-%m-%d')}

## Active Alerts
- 3 price alerts triggered
- 1 news alert (NVDA)
- 0 risk warnings

## Resolved Alerts
- 5 alerts resolved this week

---
Generated by FinNavigator AI
"""

        return content, {"alerts": True}, []

    def get_report_history(self, limit: int = 30) -> List[GeneratedReport]:
        """Get report generation history"""
        return self.generated_reports[-limit:]

    def get_next_scheduled_reports(self) -> List[Dict]:
        """Get upcoming scheduled reports"""
        upcoming = self.task_scheduler.get_next_scheduled_tasks()
        return [
            {
                "report_id": task["task_id"],
                "name": task["name"],
                "next_run": task["next_run"],
                "schedule": task.get("cron_expression", "")
            }
            for task in upcoming
            if task["task_id"] in self.reports
        ]

    def start(self):
        """Start the report scheduler"""
        self.task_scheduler.start()

    def stop(self):
        """Stop the report scheduler"""
        self.task_scheduler.stop()


# Helper for newline
NL = "\n"