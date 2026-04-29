"""
Base Platform Tool for Social Integrations
==========================================

Base class for all messaging platform integrations.

Author: MiniMax Agent
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional, Any
from pydantic import BaseModel, Field
from langchain_core.tools import BaseTool
from langchain_core.callbacks import CallbackManagerForToolRun
import requests
import json


@dataclass
class PlatformMessage:
    """Standardized message format for all platforms"""
    platform: str
    user_id: str
    username: Optional[str]
    content: str
    timestamp: datetime
    message_id: Optional[str] = None
    metadata: Dict[str, Any] = None

    def to_dict(self) -> Dict:
        return {
            "platform": self.platform,
            "user_id": self.user_id,
            "username": self.username,
            "content": self.content,
            "timestamp": self.timestamp.isoformat(),
            "message_id": self.message_id,
            "metadata": self.metadata or {}
        }


class BasePlatformTool(BaseTool, ABC):
    """
    Abstract base class for messaging platform tools.

    Provides common functionality for all platform integrations:
    - Message sending
    - User management
    - Group/channel handling
    - Webhook configuration
    """

    platform_name: str = "base"

    def __init__(
        self,
        api_token: Optional[str] = None,
        bot_name: str = "FinNavigator Bot",
        **kwargs
    ):
        super().__init__(**kwargs)
        self.api_token = api_token
        self.bot_name = bot_name
        self.webhook_url: Optional[str] = None

    @property
    @abstractmethod
    def api_base_url(self) -> str:
        """Base URL for platform API"""
        pass

    @abstractmethod
    def _send_message(self, chat_id: str, text: str, **kwargs) -> Dict:
        """Send message to a chat/user"""
        pass

    @abstractmethod
    def _get_user_info(self, user_id: str) -> Optional[Dict]:
        """Get user information"""
        pass

    def _make_request(
        self,
        method: str,
        endpoint: str,
        data: Optional[Dict] = None,
        params: Optional[Dict] = None
    ) -> Dict:
        """Make HTTP request to platform API"""
        url = f"{self.api_base_url}{endpoint}"
        headers = self._get_headers()

        try:
            if method.upper() == "GET":
                response = requests.get(url, headers=headers, params=params, timeout=30)
            elif method.upper() == "POST":
                response = requests.post(url, headers=headers, json=data, timeout=30)
            else:
                return {"error": f"Unsupported method: {method}"}

            if response.status_code == 200:
                return response.json()
            else:
                return {"error": f"API error: {response.status_code}", "details": response.text}

        except requests.exceptions.Timeout:
            return {"error": "Request timed out"}
        except requests.exceptions.RequestException as e:
            return {"error": str(e)}

    def _get_headers(self) -> Dict[str, str]:
        """Get headers for API requests"""
        return {
            "Authorization": f"Bearer {self.api_token}",
            "Content-Type": "application/json"
        }

    def _format_alert_message(self, title: str, content: str, priority: str = "normal") -> str:
        """Format alert message for the platform"""
        emoji = {
            "urgent": "🚨",
            "high": "🚨",
            "normal": "🔔",
            "low": "ℹ️"
        }.get(priority, "🔔")

        return f"""{emoji} **{title}**

{content}

🤖 FinNavigator AI"""

    def _format_report_message(self, report_type: str, content: str) -> str:
        """Format report message for the platform"""
        emoji = {
            "daily": "📅",
            "weekly": "📊",
            "monthly": "📈"
        }.get(report_type, "📋")

        return f"""{emoji} **{report_type.title()} Report**

{content}

---
🤖 FinNavigator AI
{datetime.now().strftime('%Y-%m-%d %H:%M')}"""


class SendMessageInput(BaseModel):
    """Input schema for sending messages"""
    user_id: str = Field(description="User ID or chat ID to send message to")
    message: str = Field(description="Message content to send")
    parse_mode: Optional[str] = Field(default="markdown", description="Parse mode: markdown, html")
    reply_to: Optional[str] = Field(default=None, description="Message ID to reply to")


class GetUserInput(BaseModel):
    """Input schema for getting user info"""
    user_id: str = Field(description="User ID to look up")


class SendAlertInput(BaseModel):
    """Input schema for sending alerts"""
    user_id: str = Field(description="User ID to send alert to")
    title: str = Field(description="Alert title")
    message: str = Field(description="Alert message content")
    priority: str = Field(default="normal", description="Alert priority: low, normal, high, urgent")


class SendReportInput(BaseModel):
    """Input schema for sending reports"""
    user_id: str = Field(description="User ID or channel ID")
    report_type: str = Field(description="Report type: daily, weekly, monthly")
    content: str = Field(description="Report content to send")


class GetMessagesInput(BaseModel):
    """Input schema for getting messages"""
    limit: int = Field(default=10, description="Number of messages to retrieve")
    offset: Optional[int] = Field(default=None, description="Offset for pagination")


class BaseSocialTool(BasePlatformTool):
    """Base class for social platform tools"""

    @abstractmethod
    def get_updates(self, offset: Optional[int] = None) -> List[Dict]:
        """Get new messages/updates"""
        pass

    @abstractmethod
    def set_webhook(self, webhook_url: str) -> bool:
        """Set webhook URL for receiving updates"""
        pass

    @abstractmethod
    def remove_webhook(self) -> bool:
        """Remove webhook"""
        pass

    def _run(
        self,
        operation: str,
        **kwargs
    ) -> str:
        """Execute platform operation"""
        if operation == "send":
            return json.dumps(self._send_message(kwargs.get("user_id"), kwargs.get("message")))
        elif operation == "alert":
            formatted = self._format_alert_message(
                kwargs.get("title"),
                kwargs.get("message"),
                kwargs.get("priority", "normal")
            )
            return json.dumps(self._send_message(kwargs.get("user_id"), formatted))
        elif operation == "report":
            formatted = self._format_report_message(
                kwargs.get("report_type", "daily"),
                kwargs.get("content")
            )
            return json.dumps(self._send_message(kwargs.get("user_id"), formatted))
        else:
            return json.dumps({"error": f"Unknown operation: {operation}"})

    async def _arun(self, operation: str, **kwargs) -> str:
        """Async execution"""
        return self._run(operation, **kwargs)