"""
Telegram Bot Integration for FinNavigator
=========================================

LangChain tool for Telegram bot operations.

Features:
- Send messages to users and groups
- Get user information
- Send formatted alerts and reports
- Handle incoming updates
- Webhook configuration for real-time updates

Author: MiniMax Agent
"""

from typing import Type, Optional, List, Dict, Any
from datetime import datetime
from pydantic import BaseModel, Field
from langchain_core.tools import BaseTool
from langchain_core.callbacks import CallbackManagerForToolRun
import requests
import json
import os

from .base_platform import BasePlatformTool, PlatformMessage


class TelegramMessageInput(BaseModel):
    """Input schema for Telegram messages"""
    chat_id: str = Field(description="Telegram chat ID (user or group)")
    text: str = Field(description="Message text to send")
    parse_mode: str = Field(default="Markdown", description="Parse mode: Markdown or HTML")
    reply_to_message_id: Optional[str] = Field(default=None, description="Message ID to reply to")
    disable_web_page_preview: bool = Field(default=True, description="Disable link preview")


class TelegramUserInput(BaseModel):
    """Input schema for Telegram user lookup"""
    user_id: str = Field(description="Telegram user ID or username")


class TelegramAlertInput(BaseModel):
    """Input schema for Telegram alerts"""
    chat_id: str = Field(description="Telegram chat ID")
    alert_type: str = Field(description="Alert type: price, news, portfolio, risk")
    title: str = Field(description="Alert title")
    message: str = Field(description="Alert message")
    priority: str = Field(default="normal", description="Priority: low, normal, high, urgent")


class TelegramReportInput(BaseModel):
    """Input schema for Telegram reports"""
    chat_id: str = Field(description="Telegram chat ID or channel")
    report_type: str = Field(description="Report type: daily, weekly, monthly")
    content: str = Field(description="Report content")
    schedule_time: Optional[str] = Field(default=None, description="Schedule time (ISO format)")


class TelegramInlineButton(BaseModel):
    """Telegram inline button"""
    text: str
    url: Optional[str] = None
    callback_data: Optional[str] = None


class TelegramKeyboard(BaseModel):
    """Telegram inline keyboard"""
    buttons: List[List[TelegramInlineButton]]


class TelegramBotTool(BaseTool):
    """
    Telegram Bot Tool for FinNavigator.

    Enables agents to send messages, alerts, and reports through Telegram.
    Integrates with existing alerting system for real-time notifications.

    Setup:
    1. Create a bot via @BotFather
    2. Get the bot token
    3. Set TELEGRAM_BOT_TOKEN environment variable

    Usage:
        - Send messages to users/groups
        - Send formatted alerts with priority
        - Send scheduled reports
        - Create interactive keyboards
    """

    name: str = "telegram_bot"
    description: str = """Send messages and alerts via Telegram. Use for real-time
    notifications to users. Supports markdown formatting, alerts with priority levels,
    and scheduled reports. Chat ID can be a user ID, group ID, or channel username."""

    args_schema: Type[BaseModel] = TelegramMessageInput

    def __init__(
        self,
        bot_token: Optional[str] = None,
        default_parse_mode: str = "Markdown",
        **kwargs
    ):
        super().__init__(**kwargs)
        self.bot_token = bot_token or os.getenv("TELEGRAM_BOT_TOKEN", "")
        self.api_base_url = f"https://api.telegram.org/bot{self.bot_token}"
        self.default_parse_mode = default_parse_mode

    def _get_me(self) -> Dict:
        """Get bot information"""
        response = requests.get(f"{self.api_base_url}/getMe", timeout=10)
        return response.json()

    def _send_message(
        self,
        chat_id: str,
        text: str,
        parse_mode: Optional[str] = None,
        reply_to_message_id: Optional[str] = None,
        disable_web_page_preview: bool = True,
        reply_markup: Optional[Dict] = None
    ) -> Dict:
        """Send message to Telegram chat"""
        if not self.bot_token:
            return {"success": False, "error": "Bot token not configured"}

        url = f"{self.api_base_url}/sendMessage"
        data = {
            "chat_id": chat_id,
            "text": text,
            "parse_mode": parse_mode or self.default_parse_mode,
            "disable_web_page_preview": disable_web_page_preview
        }

        if reply_to_message_id:
            data["reply_to_message_id"] = reply_to_message_id

        if reply_markup:
            data["reply_markup"] = json.dumps(reply_markup)

        try:
            response = requests.post(url, json=data, timeout=30)
            result = response.json()

            if result.get("ok"):
                return {
                    "success": True,
                    "message_id": result["result"]["message_id"],
                    "chat_id": chat_id
                }
            else:
                return {"success": False, "error": result.get("description", "Unknown error")}

        except requests.exceptions.RequestException as e:
            return {"success": False, "error": str(e)}

    def _get_chat(self, chat_id: str) -> Dict:
        """Get chat information"""
        url = f"{self.api_base_url}/getChat"
        response = requests.get(url, params={"chat_id": chat_id}, timeout=10)
        return response.json()

    def _get_updates(self, offset: Optional[int] = None, limit: int = 100) -> List[Dict]:
        """Get recent updates"""
        url = f"{self.api_base_url}/getUpdates"
        params = {"timeout": 0, "limit": limit}
        if offset:
            params["offset"] = offset

        response = requests.get(url, params=params, timeout=30)
        result = response.json()

        if result.get("ok"):
            return result.get("result", [])
        return []

    def _format_alert(self, alert_type: str, title: str, message: str, priority: str) -> str:
        """Format alert message with emojis and formatting"""
        emoji_map = {
            "price": "📈",
            "news": "📰",
            "portfolio": "💼",
            "risk": "⚠️",
            "urgent": "🚨",
            "high": "🚨",
            "normal": "🔔",
            "low": "ℹ️"
        }

        emoji = emoji_map.get(priority, "🔔")
        priority_badge = f"[{priority.upper()}]"

        formatted = f"""{emoji} *{title}* {priority_badge}

{message}

🤖 *FinNavigator AI*"""

        return formatted

    def _format_report(self, report_type: str, content: str) -> str:
        """Format report message"""
        emoji_map = {
            "daily": "📅",
            "weekly": "📊",
            "monthly": "📈"
        }

        emoji = emoji_map.get(report_type, "📋")
        title = f"{report_type.title()} Report"

        formatted = f"""{emoji} *{title}*

{content}

─────────────────────
🤖 FinNavigator AI
{datetime.now().strftime('%Y-%m-%d %H:%M')}"""

        return formatted

    def _create_inline_keyboard(self, buttons: List[List[Dict]]) -> Dict:
        """Create inline keyboard markup"""
        return {
            "inline_keyboard": buttons
        }

    def _run(
        self,
        chat_id: str,
        text: str,
        parse_mode: Optional[str] = None,
        reply_to_message_id: Optional[str] = None,
        disable_web_page_preview: bool = True,
        run_manager: Optional[CallbackManagerForToolRun] = None
    ) -> str:
        """Execute send message operation"""
        result = self._send_message(
            chat_id=chat_id,
            text=text,
            parse_mode=parse_mode,
            reply_to_message_id=reply_to_message_id,
            disable_web_page_preview=disable_web_page_preview
        )
        return json.dumps(result, indent=2)

    async def _arun(
        self,
        chat_id: str,
        text: str,
        parse_mode: Optional[str] = None,
        reply_to_message_id: Optional[str] = None,
        disable_web_page_preview: bool = True,
        run_manager: Optional[CallbackManagerForToolRun] = None
    ) -> str:
        """Async execution"""
        return self._run(
            chat_id, text, parse_mode, reply_to_message_id,
            disable_web_page_preview, run_manager
        )

    def send_alert(
        self,
        chat_id: str,
        alert_type: str,
        title: str,
        message: str,
        priority: str = "normal"
    ) -> Dict:
        """Send formatted alert to chat"""
        formatted_text = self._format_alert(alert_type, title, message, priority)
        return self._send_message(chat_id, formatted_text)

    def send_report(
        self,
        chat_id: str,
        report_type: str,
        content: str
    ) -> Dict:
        """Send formatted report to chat"""
        formatted_text = self._format_report(report_type, content)
        return self._send_message(chat_id, formatted_text)

    def send_with_buttons(
        self,
        chat_id: str,
        text: str,
        buttons: List[List[Dict]]
    ) -> Dict:
        """Send message with inline keyboard buttons"""
        keyboard = self._create_inline_keyboard(buttons)
        return self._send_message(chat_id, text, reply_markup=keyboard)

    def get_user_info(self, user_id: str) -> Dict:
        """Get user information by ID or username"""
        try:
            # Try as numeric ID first
            chat = self._get_chat(user_id)
            if chat.get("ok"):
                return {
                    "success": True,
                    "user": {
                        "id": chat["result"].get("id"),
                        "username": chat["result"].get("username"),
                        "first_name": chat["result"].get("first_name"),
                        "last_name": chat["result"].get("last_name"),
                        "type": chat["result"].get("type", "private")
                    }
                }
            return {"success": False, "error": chat.get("description", "User not found")}
        except Exception as e:
            return {"success": False, "error": str(e)}

    def set_webhook(self, webhook_url: str, max_connections: int = 40) -> Dict:
        """Set webhook for incoming updates"""
        url = f"{self.api_base_url}/setWebhook"
        data = {
            "url": webhook_url,
            "max_connections": max_connections
        }

        response = requests.post(url, json=data, timeout=10)
        return response.json()

    def delete_webhook(self) -> Dict:
        """Delete webhook"""
        url = f"{self.api_base_url}/deleteWebhook"
        response = requests.get(url, timeout=10)
        return response.json()

    def get_webhook_info(self) -> Dict:
        """Get webhook info"""
        url = f"{self.api_base_url}/getWebhookInfo"
        response = requests.get(url, timeout=10)
        return response.json()

    def parse_update(self, update: Dict) -> Optional[PlatformMessage]:
        """Parse Telegram update into standardized message"""
        if "message" in update:
            msg = update["message"]
            return PlatformMessage(
                platform="telegram",
                user_id=str(msg["chat"]["id"]),
                username=msg["from"].get("username"),
                content=msg.get("text", ""),
                timestamp=datetime.fromtimestamp(msg["date"]),
                message_id=str(msg["message_id"]),
                metadata={
                    "first_name": msg["from"].get("first_name"),
                    "chat_type": msg["chat"].get("type")
                }
            )
        elif "edited_message" in update:
            msg = update["edited_message"]
            return PlatformMessage(
                platform="telegram",
                user_id=str(msg["chat"]["id"]),
                username=msg["from"].get("username"),
                content=f"[Edited] {msg.get('text', '')}",
                timestamp=datetime.fromtimestamp(msg["edit_date"]),
                metadata={"edited": True}
            )
        elif "callback_query" in update:
            query = update["callback_query"]
            return PlatformMessage(
                platform="telegram",
                user_id=str(query["message"]["chat"]["id"]),
                username=query["from"].get("username"),
                content=query.get("data", ""),
                timestamp=datetime.fromtimestamp(query["message"]["date"]),
                message_id=str(query["message"]["message_id"]),
                metadata={
                    "callback_id": query["id"],
                    "button_data": query.get("data")
                }
            )
        return None


class TelegramGroupTool(TelegramBotTool):
    """
    Telegram Group Bot Tool

    Extended tool for group management:
    - Monitor group messages
    - Filter content
    - Manage group settings
    """

    name: str = "telegram_group"
    description: str = """Manage Telegram groups and channels. Send announcements,
    monitor activity, and manage group settings."""

    def __init__(self, admin_chat_id: Optional[str] = None, **kwargs):
        super().__init__(**kwargs)
        self.admin_chat_id = admin_chat_id

    def get_group_members(self, chat_id: str, limit: int = 100) -> Dict:
        """Get group member count"""
        url = f"{self.api_base_url}/getChatMemberCount"
        response = requests.get(url, params={"chat_id": chat_id}, timeout=10)
        return response.json()

    def send_announcement(
        self,
        chat_ids: List[str],
        message: str
    ) -> List[Dict]:
        """Send message to multiple chats"""
        results = []
        for chat_id in chat_ids:
            result = self._send_message(chat_id, message)
            results.append({"chat_id": chat_id, "result": result})
        return results

    def report_to_admin(self, alert_message: str) -> Dict:
        """Report activity to admin"""
        if not self.admin_chat_id:
            return {"success": False, "error": "Admin chat ID not configured"}

        return self._send_message(
            self.admin_chat_id,
            f"⚠️ *Group Alert*\n\n{alert_message}",
            parse_mode="Markdown"
        )