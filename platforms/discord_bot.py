"""
Discord Bot Integration for FinNavigator
=======================================

LangChain tool for Discord bot operations.

Features:
- Send messages to channels and users
- Send formatted embeds with rich content
- Create Discord webhooks for alerts
- Handle slash commands (future)
- Channel management

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

from .base_platform import BasePlatformTool


class DiscordMessageInput(BaseModel):
    """Input schema for Discord messages"""
    channel_id: str = Field(description="Discord channel ID to send message to")
    content: str = Field(description="Message content (max 2000 chars)")
    username: Optional[str] = Field(default=None, description="Override bot username")
    tts: bool = Field(default=False, description="Text-to-speech message")
    embed: Optional[Dict] = Field(default=None, description="Discord embed object")


class DiscordEmbedInput(BaseModel):
    """Input schema for Discord embeds"""
    title: str = Field(description="Embed title")
    description: Optional[str] = Field(default=None, description="Embed description")
    color: Optional[int] = Field(default=None, description="Embed color (hex as int)")
    url: Optional[str] = Field(default=None, description="URL for title link")
    fields: Optional[List[Dict]] = Field(default=None, description="Embed fields")
    footer: Optional[str] = Field(default=None, description="Footer text")


class DiscordAlertInput(BaseModel):
    """Input schema for Discord alerts"""
    channel_id: str = Field(description="Discord channel ID")
    alert_type: str = Field(description="Alert type: price, news, portfolio, risk")
    title: str = Field(description="Alert title")
    message: str = Field(description="Alert message")
    priority: str = Field(default="normal", description="Priority: low, normal, high, urgent")


class DiscordWebhookInput(BaseModel):
    """Input schema for Discord webhooks"""
    webhook_url: str = Field(description="Discord webhook URL")
    content: str = Field(description="Message content")
    username: Optional[str] = Field(default=None, description="Override username")
    embeds: Optional[List[Dict]] = Field(default=None, description="Embed objects")


class DiscordBotTool(BaseTool):
    """
    Discord Bot Tool for FinNavigator.

    Enables agents to send messages and rich embeds to Discord channels.
    Supports webhook integration for easy setup and rich formatting.

    Setup:
    1. Create a Discord application at https://discord.com/developers/applications
    2. Create a bot and get the token
    3. Add bot to your server with required permissions
    4. Set DISCORD_BOT_TOKEN environment variable

    Usage:
        - Send messages to channels
        - Send rich embeds with colors and fields
        - Create webhook-based alerts
        - Monitor channel activity (future)
    """

    name: str = "discord_bot"
    description: str = """Send messages and alerts via Discord. Supports rich embeds
    with colors, fields, and images. Use channel ID for direct messages or
    create webhooks for easy channel integration."""

    args_schema: Type[BaseModel] = DiscordMessageInput

    def __init__(
        self,
        bot_token: Optional[str] = None,
        default_channel_id: Optional[str] = None,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.bot_token = bot_token or os.getenv("DISCORD_BOT_TOKEN", "")
        self.api_base_url = "https://discord.com/api/v10"
        self.default_channel_id = default_channel_id

    def _get_headers(self) -> Dict[str, str]:
        """Get headers for Discord API requests"""
        return {
            "Authorization": f"Bot {self.bot_token}",
            "Content-Type": "application/json"
        }

    def _send_message(
        self,
        channel_id: str,
        content: str,
        username: Optional[str] = None,
        tts: bool = False,
        embeds: Optional[List[Dict]] = None
    ) -> Dict:
        """Send message to Discord channel"""
        if not self.bot_token:
            return {"success": False, "error": "Bot token not configured"}

        url = f"{self.api_base_url}/channels/{channel_id}/messages"
        data = {
            "content": content[:2000],  # Discord limit
            "tts": tts
        }

        if username:
            data["username"] = username

        if embeds:
            data["embeds"] = embeds

        try:
            response = requests.post(url, headers=self._get_headers(), json=data, timeout=30)
            result = response.json()

            if response.status_code == 200 or response.status_code == 201:
                return {
                    "success": True,
                    "message_id": result.get("id"),
                    "channel_id": channel_id
                }
            else:
                return {"success": False, "error": result.get("message", f"HTTP {response.status_code}")}

        except requests.exceptions.RequestException as e:
            return {"success": False, "error": str(e)}

    def _create_embed(
        self,
        title: str,
        description: Optional[str] = None,
        color: Optional[int] = None,
        url: Optional[str] = None,
        fields: Optional[List[Dict]] = None,
        footer: Optional[str] = None,
        timestamp: bool = True
    ) -> Dict:
        """Create Discord embed object"""
        embed = {
            "title": title,
            "color": color or self._get_color_for_priority("normal")
        }

        if description:
            embed["description"] = description

        if url:
            embed["url"] = url

        if fields:
            embed["fields"] = [
                {"name": f.get("name", ""), "value": f.get("value", ""), "inline": f.get("inline", True)}
                for f in fields
            ]

        if footer:
            embed["footer"] = {"text": footer}

        if timestamp:
            embed["timestamp"] = datetime.utcnow().isoformat()

        return embed

    def _get_color_for_priority(self, priority: str) -> int:
        """Get embed color for priority level"""
        colors = {
            "urgent": 0xFF0000,   # Red
            "high": 0xFF8C00,    # Orange
            "normal": 0x00FF00,   # Green
            "low": 0x808080      # Gray
        }
        return colors.get(priority, 0x00FF00)

    def _format_alert(self, alert_type: str, title: str, message: str, priority: str) -> List[Dict]:
        """Format alert as Discord embed"""
        emoji_map = {
            "price": "📈",
            "news": "📰",
            "portfolio": "💼",
            "risk": "⚠️"
        }

        emoji = emoji_map.get(alert_type, "🔔")

        embed = self._create_embed(
            title=f"{emoji} {title}",
            description=message,
            color=self._get_color_for_priority(priority),
            footer=f"FinNavigator AI | {priority.upper()} Priority"
        )

        return [embed]

    def _run(
        self,
        channel_id: str,
        content: str,
        username: Optional[str] = None,
        tts: bool = False,
        embed: Optional[Dict] = None,
        run_manager: Optional[CallbackManagerForToolRun] = None
    ) -> str:
        """Execute send message operation"""
        embeds = [embed] if embed else None
        result = self._send_message(channel_id, content, username, tts, embeds)
        return json.dumps(result, indent=2)

    async def _arun(
        self,
        channel_id: str,
        content: str,
        username: Optional[str] = None,
        tts: bool = False,
        embed: Optional[Dict] = None,
        run_manager: Optional[CallbackManagerForToolRun] = None
    ) -> str:
        """Async execution"""
        return self._run(channel_id, content, username, tts, embed, run_manager)

    def send_alert(
        self,
        channel_id: str,
        alert_type: str,
        title: str,
        message: str,
        priority: str = "normal"
    ) -> Dict:
        """Send formatted alert as embed"""
        embeds = self._format_alert(alert_type, title, message, priority)
        return self._send_message(channel_id, "", embeds=embeds)

    def send_embed_message(
        self,
        channel_id: str,
        title: str,
        description: str,
        fields: Optional[List[Dict]] = None,
        color: Optional[int] = None,
        footer: Optional[str] = None
    ) -> Dict:
        """Send rich embed message"""
        embed = self._create_embed(
            title=title,
            description=description,
            color=color,
            fields=fields,
            footer=footer
        )
        return self._send_message(channel_id, "", embeds=[embed])

    def send_report(self, channel_id: str, report_type: str, content: str) -> Dict:
        """Send formatted report"""
        emoji_map = {
            "daily": "📅",
            "weekly": "📊",
            "monthly": "📈"
        }

        emoji = emoji_map.get(report_type, "📋")
        title = f"{emoji} {report_type.title()} Report"

        embed = self._create_embed(
            title=title,
            description=content,
            color=0x0099FF,  # Blue
            footer=f"FinNavigator AI | {datetime.now().strftime('%Y-%m-%d %H:%M')}"
        )

        return self._send_message(channel_id, "", embeds=[embed])


class DiscordWebhookTool(BaseTool):
    """
    Discord Webhook Tool for simple channel integration.

    Allows sending messages via Discord webhooks without bot authentication.
    Useful for:
    - Channel-specific notifications
    - Integration with existing Discord servers
    - Simple alert channels
    """

    name: str = "discord_webhook"
    description: str = """Send messages via Discord webhooks. No bot token needed -
    just a webhook URL. Good for simple notifications and alerts to channels."""

    args_schema: Type[BaseModel] = DiscordWebhookInput

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _send_webhook(
        self,
        webhook_url: str,
        content: str,
        username: Optional[str] = None,
        embeds: Optional[List[Dict]] = None,
        avatar_url: Optional[str] = None
    ) -> Dict:
        """Send message via webhook"""
        data = {"content": content[:2000]}

        if username:
            data["username"] = username

        if avatar_url:
            data["avatar_url"] = avatar_url

        if embeds:
            data["embeds"] = embeds

        try:
            response = requests.post(webhook_url, json=data, timeout=30)

            if response.status_code in [200, 204]:
                return {"success": True, "webhook_sent": True}
            else:
                return {"success": False, "error": f"HTTP {response.status_code}"}

        except requests.exceptions.RequestException as e:
            return {"success": False, "error": str(e)}

    def _run(
        self,
        webhook_url: str,
        content: str,
        username: Optional[str] = None,
        embeds: Optional[List[Dict]] = None,
        run_manager: Optional[CallbackManagerForToolRun] = None
    ) -> str:
        """Execute webhook send"""
        result = self._send_webhook(webhook_url, content, username, embeds)
        return json.dumps(result, indent=2)

    async def _arun(
        self,
        webhook_url: str,
        content: str,
        username: Optional[str] = None,
        embeds: Optional[List[Dict]] = None,
        run_manager: Optional[CallbackManagerForToolRun] = None
    ) -> str:
        """Async execution"""
        return self._run(webhook_url, content, username, embeds, run_manager)

    def send_alert(self, webhook_url: str, title: str, message: str, priority: str = "normal") -> Dict:
        """Send alert via webhook"""
        emoji = "🔔" if priority == "normal" else "🚨"
        content = f"{emoji} **{title}**\n\n{message}"
        return self._send_webhook(webhook_url, content, username="FinNavigator")


class DiscordServerMonitor:
    """
    Discord Server Monitoring Tool

    Monitors Discord server for:
    - New messages
    - Member joins/leaves
    - Channel updates
    - Role changes

    Note: Requires bot with appropriate intents enabled
    """

    def __init__(self, bot_tool: DiscordBotTool):
        self.bot = bot_tool

    def get_member_count(self, guild_id: str) -> Optional[int]:
        """Get server member count"""
        url = f"{self.bot.api_base_url}/guilds/{guild_id}"
        response = requests.get(url, headers=self.bot._get_headers(), timeout=10)

        if response.status_code == 200:
            return response.json().get("member_count")
        return None

    def get_channel_list(self, guild_id: str) -> List[Dict]:
        """Get list of channels in server"""
        url = f"{self.bot.api_base_url}/guilds/{guild_id}/channels"
        response = requests.get(url, headers=self.bot._get_headers(), timeout=10)

        if response.status_code == 200:
            return response.json()
        return []

    def send_to_all_channels(self, guild_id: str, message: str) -> List[Dict]:
        """Send message to all text channels in server"""
        channels = self.get_channel_list(guild_id)
        text_channels = [c for c in channels if c.get("type") == 0]  # 0 = text channel

        results = []
        for channel in text_channels:
            result = self.bot._send_message(channel["id"], message)
            results.append({
                "channel_id": channel["id"],
                "channel_name": channel["name"],
                "result": result
            })

        return results