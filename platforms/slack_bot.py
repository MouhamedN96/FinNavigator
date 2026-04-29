"""
Slack Bot Integration for FinNavigator
======================================

LangChain tool for Slack bot operations and API integration.

Features:
- Send messages to channels and users via Slack API
- Send formatted Block Kit messages
- Interactive message components (buttons, select menus)
- Schedule messages
- Search messages
- User management

Author: MiniMax Agent
"""

from typing import Type, Optional, List, Dict, Any
from datetime import datetime, timedelta
from pydantic import BaseModel, Field
from langchain_core.tools import BaseTool
from langchain_core.callbacks import CallbackManagerForToolRun
import requests
import json
import os

from .base_platform import BasePlatformTool


class SlackMessageInput(BaseModel):
    """Input schema for Slack messages"""
    channel: str = Field(description="Slack channel ID or name (e.g., C12345 or #general)")
    text: str = Field(description="Message text (supports Slack markdown)")
    thread_ts: Optional[str] = Field(default=None, description="Thread timestamp to reply in")
    unfurl_links: bool = Field(default=False, description="Unfurl URLs in message")


class SlackBlocksInput(BaseModel):
    """Input schema for Slack Block Kit messages"""
    channel: str = Field(description="Slack channel ID or name")
    blocks: List[Dict] = Field(description="Block Kit JSON blocks")
    text: str = Field(description="Fallback text for notifications")


class SlackAlertInput(BaseModel):
    """Input schema for Slack alerts"""
    channel: str = Field(description="Slack channel ID or name")
    alert_type: str = Field(description="Alert type: price, news, portfolio, risk")
    title: str = Field(description="Alert title")
    message: str = Field(description="Alert message")
    priority: str = Field(default="normal", description="Priority: low, normal, high, urgent")


class SlackEphemeralInput(BaseModel):
    """Input schema for ephemeral messages (visible only to user)"""
    channel: str = Field(description="Slack channel ID")
    user_id: str = Field(description="User ID to show message to")
    text: str = Field(description="Message text")


class SlackBotTool(BaseTool):
    """
    Slack Bot Tool for FinNavigator.

    Enables agents to send messages, interactive blocks, and alerts to Slack.
    Supports Block Kit for rich formatting and interactive components.

    Setup:
    1. Create a Slack App at https://api.slack.com/apps
    2. Enable necessary scopes (chat:write, channels:read, etc.)
    3. Install to your workspace
    4. Set SLACK_BOT_TOKEN environment variable

    Scopes needed:
    - chat:write (send messages)
    - channels:read (list channels)
    - users:read (get user info)
    - chat:write.public (optional, for public channels)
    """

    name: str = "slack_bot"
    description: str = """Send messages and alerts via Slack. Supports rich Block Kit
    formatting, interactive components, and thread replies. Use channel ID or name."""

    args_schema: Type[BaseModel] = SlackMessageInput

    def __init__(
        self,
        bot_token: Optional[str] = None,
        default_channel: Optional[str] = None,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.bot_token = bot_token or os.getenv("SLACK_BOT_TOKEN", "")
        self.api_base_url = "https://slack.com/api"
        self.default_channel = default_channel

    def _get_headers(self) -> Dict[str, str]:
        """Get headers for Slack API requests"""
        return {
            "Authorization": f"Bearer {self.bot_token}",
            "Content-Type": "application/json"
        }

    def _post(self, endpoint: str, data: Dict) -> Dict:
        """Make POST request to Slack API"""
        url = f"{self.api_base_url}/{endpoint}"

        try:
            response = requests.post(url, headers=self._get_headers(), json=data, timeout=30)
            result = response.json()

            if result.get("ok"):
                return {"success": True, **result}
            else:
                return {"success": False, "error": result.get("error", "Unknown error")}

        except requests.exceptions.RequestException as e:
            return {"success": False, "error": str(e)}

    def _get(self, endpoint: str, params: Optional[Dict] = None) -> Dict:
        """Make GET request to Slack API"""
        url = f"{self.api_base_url}/{endpoint}"

        try:
            response = requests.get(url, headers=self._get_headers(), params=params, timeout=30)
            result = response.json()

            if result.get("ok"):
                return {"success": True, **result}
            else:
                return {"success": False, "error": result.get("error", "Unknown error")}

        except requests.exceptions.RequestException as e:
            return {"success": False, "error": str(e)}

    def _send_message(
        self,
        channel: str,
        text: str,
        thread_ts: Optional[str] = None,
        blocks: Optional[List[Dict]] = None,
        unfurl_links: bool = False
    ) -> Dict:
        """Send message to Slack channel"""
        if not self.bot_token:
            return {"success": False, "error": "Bot token not configured"}

        data = {
            "channel": channel,
            "text": text,
            "unfurl_links": unfurl_links
        }

        if thread_ts:
            data["thread_ts"] = thread_ts

        if blocks:
            data["blocks"] = blocks

        return self._post("chat.postMessage", data)

    def _format_alert_blocks(
        self,
        alert_type: str,
        title: str,
        message: str,
        priority: str
    ) -> List[Dict]:
        """Format alert as Block Kit blocks"""
        emoji_map = {
            "price": ":chart_with_upwards_trend:",
            "news": ":newspaper:",
            "portfolio": ":briefcase:",
            "risk": ":warning:"
        }

        priority_emoji = {
            "urgent": ":fire:",
            "high": ":rotating_light:",
            "normal": ":bell:",
            "low": ":information_source:"
        }

        emoji = emoji_map.get(alert_type, ":bell:")
        priority_icon = priority_emoji.get(priority, ":bell:")

        header_emoji = emoji if priority == "normal" else priority_icon

        blocks = [
            {
                "type": "header",
                "text": {
                    "type": "plain_text",
                    "text": f"{header_emoji} {title}",
                    "emoji": True
                }
            },
            {
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": message
                }
            },
            {
                "type": "context",
                "elements": [
                    {
                        "type": "mrkdwn",
                        "text": f"*Priority:* {priority.upper()} | *FinNavigator AI* | {datetime.now().strftime('%Y-%m-%d %H:%M')}"
                    }
                ]
            }
        ]

        return blocks

    def _create_section_block(
        self,
        text: str,
        fields: Optional[List[str]] = None,
        accessory: Optional[Dict] = None
    ) -> Dict:
        """Create a section block"""
        block = {"type": "section"}

        if text:
            block["text"] = {"type": "mrkdwn", "text": text}

        if fields:
            block["fields"] = [{"type": "mrkdwn", "text": f} for f in fields]

        if accessory:
            block["accessory"] = accessory

        return block

    def _create_button_element(
        self,
        text: str,
        action_id: str,
        url: Optional[str] = None,
        value: Optional[str] = None
    ) -> Dict:
        """Create a button element"""
        element = {
            "type": "button",
            "text": {"type": "plain_text", "text": text, "emoji": True},
            "action_id": action_id
        }

        if url:
            element["url"] = url

        if value:
            element["value"] = value

        return element

    def _run(
        self,
        channel: str,
        text: str,
        thread_ts: Optional[str] = None,
        unfurl_links: bool = False,
        run_manager: Optional[CallbackManagerForToolRun] = None
    ) -> str:
        """Execute send message operation"""
        result = self._send_message(channel, text, thread_ts, unfurl_links=unfurl_links)
        return json.dumps(result, indent=2)

    async def _arun(
        self,
        channel: str,
        text: str,
        thread_ts: Optional[str] = None,
        unfurl_links: bool = False,
        run_manager: Optional[CallbackManagerForToolRun] = None
    ) -> str:
        """Async execution"""
        return self._run(channel, text, thread_ts, unfurl_links, run_manager)

    def send_alert(
        self,
        channel: str,
        alert_type: str,
        title: str,
        message: str,
        priority: str = "normal"
    ) -> Dict:
        """Send formatted alert as Block Kit"""
        blocks = self._format_alert_blocks(alert_type, title, message, priority)
        fallback = f"{title}: {message}"
        return self._send_message(channel, fallback, blocks=blocks)

    def send_blocks(
        self,
        channel: str,
        blocks: List[Dict],
        thread_ts: Optional[str] = None
    ) -> Dict:
        """Send Block Kit message"""
        fallback = "FinNavigator Alert"
        return self._send_message(channel, fallback, thread_ts, blocks=blocks)

    def send_with_buttons(
        self,
        channel: str,
        text: str,
        buttons: List[Dict],
        thread_ts: Optional[str] = None
    ) -> Dict:
        """Send message with action buttons"""
        # Create actions block
        actions_block = {
            "type": "actions",
            "elements": buttons
        }

        # Create section with text
        section_block = self._create_section_block(text)

        blocks = [section_block, actions_block]
        return self._send_message(channel, text, thread_ts, blocks=blocks)

    def send_ephemeral(
        self,
        channel: str,
        user_id: str,
        text: str
    ) -> Dict:
        """Send ephemeral message (visible only to user)"""
        data = {
            "channel": channel,
            "user": user_id,
            "text": text
        }
        return self._post("chat.postEphemeral", data)

    def send_report(
        self,
        channel: str,
        report_type: str,
        content: str,
        fields: Optional[List[Dict]] = None
    ) -> Dict:
        """Send formatted report"""
        emoji_map = {
            "daily": ":calendar:",
            "weekly": ":bar_chart:",
            "monthly": ":chart_with_upwards_trend:"
        }

        emoji = emoji_map.get(report_type, ":clipboard:")
        title = f"{emoji} {report_type.title()} Report"

        blocks = [
            {
                "type": "header",
                "text": {
                    "type": "plain_text",
                    "text": title,
                    "emoji": True
                }
            },
            {
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": content
                }
            }
        ]

        if fields:
            fields_text = [f"{f['label']}: {f['value']}" for f in fields]
            blocks.append(self._create_section_block(fields=fields_text))

        blocks.append({
            "type": "context",
            "elements": [
                {"type": "mrkdwn", "text": f"*FinNavigator AI* | {datetime.now().strftime('%Y-%m-%d %H:%M')}"}
            ]
        })

        return self._send_message(channel, title, blocks=blocks)

    def schedule_message(
        self,
        channel: str,
        text: str,
        post_at: datetime
    ) -> Dict:
        """Schedule message for future delivery"""
        if not self.bot_token:
            return {"success": False, "error": "Bot token not configured"}

        # Unix timestamp
        scheduled_time = int(post_at.timestamp())

        data = {
            "channel": channel,
            "text": text,
            "post_at": str(scheduled_time)
        }

        return self._post("chat.scheduleMessage", data)

    def get_channel_info(self, channel: str) -> Dict:
        """Get channel information"""
        return self._get("conversations.info", {"channel": channel})

    def list_channels(self, limit: int = 100) -> Dict:
        """List available channels"""
        return self._get("conversations.list", {"limit": limit})

    def get_user_info(self, user_id: str) -> Dict:
        """Get user information"""
        return self._get("users.info", {"user": user_id})

    def find_user_by_email(self, email: str) -> Optional[str]:
        """Find user by email and return user ID"""
        result = self._get("users.lookupByEmail", {"email": email})

        if result.get("success") and result.get("user"):
            return result["user"]["id"]
        return None


class SlackWebhookTool(BaseTool):
    """
    Slack Webhook Tool for simple channel integration.

    Allows sending messages via Slack incoming webhooks without bot authentication.
    Useful for:
    - Simple notifications
    - Easy channel setup
    - External integrations
    """

    name: str = "slack_webhook"
    description: str = """Send messages via Slack incoming webhooks. Simple setup -
    just a webhook URL. Good for notifications without bot authentication."""

    args_schema: Type[BaseModel] = SlackMessageInput

    def __init__(self, webhook_url: Optional[str] = None, **kwargs):
        super().__init__(**kwargs)
        self.webhook_url = webhook_url or os.getenv("SLACK_WEBHOOK_URL", "")

    def _send_webhook(self, text: str, blocks: Optional[List[Dict]] = None) -> Dict:
        """Send message via webhook"""
        data = {"text": text}

        if blocks:
            data["blocks"] = blocks

        try:
            response = requests.post(self.webhook_url, json=data, timeout=30)

            if response.status_code == 200:
                return {"success": True}
            else:
                return {"success": False, "error": f"HTTP {response.status_code}"}

        except requests.exceptions.RequestException as e:
            return {"success": False, "error": str(e)}

    def _run(
        self,
        channel: str,
        text: str,
        thread_ts: Optional[str] = None,
        unfurl_links: bool = False,
        run_manager: Optional[CallbackManagerForToolRun] = None
    ) -> str:
        """Execute webhook send"""
        result = self._send_webhook(text)
        return json.dumps(result, indent=2)

    async def _arun(
        self,
        channel: str,
        text: str,
        thread_ts: Optional[str] = None,
        unfurl_links: bool = False,
        run_manager: Optional[CallbackManagerForToolRun] = None
    ) -> str:
        """Async execution"""
        return self._run(channel, text, thread_ts, unfurl_links, run_manager)

    def send_alert(
        self,
        channel: str,
        title: str,
        message: str,
        priority: str = "normal"
    ) -> Dict:
        """Send alert via webhook"""
        emoji = ":bell:" if priority == "normal" else ":fire:"
        text = f"{emoji} *{title}*\n>{message}"
        return self._send_webhook(text)


class SlackMessageParser:
    """Parse incoming Slack events"""

    @staticmethod
    def parse_event(event: Dict) -> Optional[PlatformMessage]:
        """Parse Slack event into standardized message"""
        if event.get("type") == "message" and "subtype" not in event:
            return PlatformMessage(
                platform="slack",
                user_id=event.get("user", ""),
                username=None,  # Would need additional API call
                content=event.get("text", ""),
                timestamp=datetime.fromtimestamp(float(event.get("ts", 0))),
                message_id=event.get("ts"),
                metadata={
                    "channel": event.get("channel"),
                    "thread_ts": event.get("thread_ts"),
                    "team": event.get("team")
                }
            )
        elif event.get("type") == "app_mention":
            return PlatformMessage(
                platform="slack",
                user_id=event.get("user", ""),
                username=None,
                content=event.get("text", ""),
                timestamp=datetime.fromtimestamp(float(event.get("ts", 0))),
                metadata={
                    "channel": event.get("channel"),
                    "mentioned": True
                }
            )
        return None