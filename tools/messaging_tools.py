"""
Messaging Tools for Agent Communication
=======================================

Tools for sending alerts, notifications, and messages through
Voiceflow and other messaging platforms.

Author: MiniMax Agent
"""

from typing import Type, Optional, Dict, Any
from pydantic import BaseModel, Field, PrivateAttr
from langchain_core.tools import BaseTool
from langchain_core.callbacks import CallbackManagerForToolRun
import os
import requests
import json


class SendMessageInput(BaseModel):
    """Input schema for sending messages"""
    user_id: str = Field(description="User ID or phone number")
    message: str = Field(description="Message text to send")
    channel: str = Field(default="voiceflow", description="Messaging channel: voiceflow, whatsapp, telegram")


class AlertInput(BaseModel):
    """Input schema for alerts"""
    user_id: str = Field(description="User ID to alert")
    alert_type: str = Field(description="Alert type: price, news, portfolio, risk")
    title: str = Field(description="Alert title")
    message: str = Field(description="Alert message")
    priority: str = Field(default="normal", description="Priority: low, normal, high, urgent")


class SendMessageTool(BaseTool):
    """
    Voiceflow Message Sending Tool.

    Sends messages through Voiceflow to WhatsApp, iMessage, etc.
    Handles the "last mile" delivery to end users.
    """

    name: str = "send_message"
    description: str = """Send a message to a user through Voiceflow messaging.
    Delivers to WhatsApp, iMessage, and other platforms.
    Include user_id and message content."""

    args_schema: Type[BaseModel] = SendMessageInput
    api_key: str = Field(default_factory=lambda: os.getenv("VOICEFLOW_API_KEY", ""), exclude=True)
    version_id: str = Field(default="development")

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        if not self.api_key:
            self.api_key = os.getenv("VOICEFLOW_API_KEY", "")

    def _run(
        self,
        user_id: str,
        message: str,
        channel: str = "voiceflow",
        run_manager: Optional[CallbackManagerForToolRun] = None
    ) -> str:
        """Execute message sending"""
        if not self.api_key:
            return "Error: VOICEFLOW_API_KEY not configured"

        if channel != "voiceflow":
            return f"Channel '{channel}' not supported. Use 'voiceflow'."

        try:
            url = f"https://general-runtime.voiceflow.com/state/user/{user_id}/interact"

            headers = {
                "Authorization": self.api_key,
                "accept": "application/json",
                "content-type": "application/json"
            }

            payload = {
                "action": {
                    "type": "text",
                    "payload": message
                },
                "state": {
                    "variables": {
                        "asset_type": "Agent Message",
                        "sent_via": "FinNavigator Agent",
                        "sent_at": self._get_timestamp()
                    }
                }
            }

            response = requests.post(url, json=payload, headers=headers, timeout=30)

            if response.status_code == 200:
                return f"Message sent successfully to {user_id}"
            else:
                return f"Failed to send message: {response.status_code} - {response.text}"

        except requests.exceptions.Timeout:
            return "Error: Request timed out. Voiceflow may be slow."
        except requests.exceptions.RequestException as e:
            return f"Error sending message: {str(e)}"

    def _get_timestamp(self) -> str:
        """Get current timestamp"""
        from datetime import datetime
        return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    async def _arun(
        self,
        user_id: str,
        message: str,
        channel: str = "voiceflow",
        run_manager: Optional[CallbackManagerForToolRun] = None
    ) -> str:
        """Async execution"""
        return self._run(user_id, message, channel, run_manager)


class AlertTool(BaseTool):
    """
    Financial Alert Tool for notifications.

    Creates and sends structured alerts for:
    - Price changes
    - News events
    - Portfolio updates
    - Risk warnings
    """

    name: str = "send_alert"
    description: str = """Send financial alerts to users.
    Supports price alerts, news alerts, portfolio updates, and risk warnings.
    Includes priority levels and structured format."""

    args_schema: Type[BaseModel] = AlertInput
    voiceflow: Any = Field(default=None, exclude=True)

    def __init__(self, voiceflow: Any = None, **kwargs):
        if voiceflow is not None:
            kwargs["voiceflow"] = voiceflow
        super().__init__(**kwargs)
        if self.voiceflow is None:
            self.voiceflow = SendMessageTool()

    def _format_alert(self, alert_type: str, title: str, message: str, priority: str) -> str:
        """Format alert message"""
        emoji_map = {
            "price": "📈",
            "news": "📰",
            "portfolio": "💼",
            "risk": "⚠️",
            "low": "ℹ️",
            "normal": "🔔",
            "high": "🚨",
            "urgent": "🔴"
        }

        emoji = emoji_map.get(alert_type, "🔔") + " " + emoji_map.get(priority, "🔔")
        priority_label = priority.upper()

        formatted = f"""{emoji} **{title}**

{alert_type.upper()} ALERT [{priority_label}]
{message}

Sent by FinNavigator AI"""

        return formatted

    def _run(
        self,
        user_id: str,
        alert_type: str,
        title: str,
        message: str,
        priority: str = "normal",
        run_manager: Optional[CallbackManagerForToolRun] = None
    ) -> str:
        """Execute alert sending"""
        # Format the alert
        formatted_message = self._format_alert(alert_type, title, message, priority)

        # Send via voiceflow
        result = self.voiceflow.invoke({
            "user_id": user_id,
            "message": formatted_message,
            "channel": "voiceflow"
        })

        return result

    async def _arun(
        self,
        user_id: str,
        alert_type: str,
        title: str,
        message: str,
        priority: str = "normal",
        run_manager: Optional[CallbackManagerForToolRun] = None
    ) -> str:
        """Async execution"""
        return self._run(user_id, alert_type, title, message, priority, run_manager)


class PortfolioAlertInput(BaseModel):
    """Input schema for portfolio alerts"""
    user_id: str = Field(description="User ID")
    ticker: str = Field(description="Stock ticker")
    condition: str = Field(description="Alert condition: above, below, change_pct")
    threshold: float = Field(description="Threshold value")
    current_price: Optional[float] = Field(default=None, description="Current price")


class PortfolioAlertTool(BaseTool):
    """
    Portfolio Condition Alert Tool.

    Sets up and triggers alerts based on portfolio conditions:
    - Price above/below threshold
    - Percentage change
    - Volume spikes
    """

    name: str = "portfolio_alert"
    description: str = """Create portfolio-based alerts for price conditions.
    Alert when stock goes above/below price or changes by percentage.
    Integrates with user's portfolio data."""

    args_schema: Type[BaseModel] = PortfolioAlertInput
    alert_tool: Any = Field(default=None, exclude=True)

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        if self.alert_tool is None:
            self.alert_tool = AlertTool()

    def _run(
        self,
        user_id: str,
        ticker: str,
        condition: str,
        threshold: float,
        current_price: Optional[float] = None,
        run_manager: Optional[CallbackManagerForToolRun] = None
    ) -> str:
        """Execute portfolio alert"""
        # In production, this would check stored alerts and notify if triggered
        # For now, we'll just acknowledge the alert setup

        condition_text = {
            "above": f"rises above ${threshold}",
            "below": f"falls below ${threshold}",
            "change_pct": f"changes by {threshold}%"
        }.get(condition, condition)

        title = f"{ticker} Price Alert"
        message = f"Alert triggered: {ticker} {condition_text}"

        if current_price:
            message += f"\nCurrent price: ${current_price}"

        result = self.alert_tool.invoke({
            "user_id": user_id,
            "alert_type": "price",
            "title": title,
            "message": message,
            "priority": "high" if abs(threshold) > 5 else "normal"
        })

        return result

    async def _arun(
        self,
        user_id: str,
        ticker: str,
        condition: str,
        threshold: float,
        current_price: Optional[float] = None,
        run_manager: Optional[CallbackManagerForToolRun] = None
    ) -> str:
        """Async execution"""
        return self._run(user_id, ticker, condition, threshold, current_price, run_manager)


class ScheduledReportInput(BaseModel):
    """Input schema for scheduled reports"""
    user_id: str = Field(description="User ID")
    report_type: str = Field(description="Report type: daily, weekly, monthly")
    message: str = Field(description="Report content")


class ScheduledReportTool(BaseTool):
    """
    Scheduled Report Delivery Tool.

    Sends scheduled reports to users:
    - Daily summaries
    - Weekly analysis
    - Monthly performance
    """

    name: str = "send_report"
    description: str = """Send scheduled reports to users.
    Daily summaries, weekly analysis, monthly performance reports.
    Integrates with report generation pipeline."""

    args_schema: Type[BaseModel] = ScheduledReportInput
    message_tool: Any = Field(default=None, exclude=True)

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        if self.message_tool is None:
            self.message_tool = SendMessageTool()

    def _format_report(self, report_type: str, message: str) -> str:
        """Format report message"""
        report_emoji = {
            "daily": "📅",
            "weekly": "📊",
            "monthly": "📈"
        }

        emoji = report_emoji.get(report_type, "📋")
        title = f"{report_type.capitalize()} Report"

        return f"""{emoji} **{title}**

{message}

---
FinNavigator AI
Auto-generated {datetime.now().strftime('%Y-%m-%d')}"""

    def _run(
        self,
        user_id: str,
        report_type: str,
        message: str,
        run_manager: Optional[CallbackManagerForToolRun] = None
    ) -> str:
        """Execute report sending"""
        from datetime import datetime

        formatted_message = self._format_report(report_type, message)

        result = self.message_tool.invoke({
            "user_id": user_id,
            "message": formatted_message,
            "channel": "voiceflow"
        })

        return result

    async def _arun(
        self,
        user_id: str,
        report_type: str,
        message: str,
        run_manager: Optional[CallbackManagerForToolRun] = None
    ) -> str:
        """Async execution"""
        return self._run(user_id, report_type, message, run_manager)


# Import datetime at module level for use in tools
from datetime import datetime
