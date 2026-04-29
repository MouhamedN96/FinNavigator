"""
Social Platform Manager - Unified Interface
============================================

Centralized management of all social/messaging platform integrations.
Provides agents with unified access to Telegram, Discord, and Slack.

Usage:
    manager = SocialPlatformManager()
    manager.send_alert("telegram", chat_id="123", title="Price Alert", message="...")
    manager.send_report("discord", channel_id="456", report_type="daily", content="...")

Author: MiniMax Agent
"""

from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import json

from .base_platform import PlatformMessage, BasePlatformTool
from .telegram_bot import TelegramBotTool, TelegramGroupTool
from .discord_bot import DiscordBotTool, DiscordWebhookTool
from .slack_bot import SlackBotTool, SlackWebhookTool


class Platform(Enum):
    """Supported messaging platforms"""
    TELEGRAM = "telegram"
    DISCORD = "discord"
    SLACK = "slack"
    VOICEFLOW = "voiceflow"


@dataclass
class AlertConfig:
    """Configuration for an alert"""
    platform: Platform
    recipient: str  # chat_id, channel_id, user_id
    alert_type: str  # price, news, portfolio, risk
    priority: str = "normal"  # low, normal, high, urgent
    title: str = ""
    message: str = ""
    scheduled_time: Optional[datetime] = None
    repeat: Optional[str] = None  # daily, weekly, monthly


@dataclass
class MessageResult:
    """Result of a message send operation"""
    success: bool
    platform: str
    message_id: Optional[str] = None
    error: Optional[str] = None
    timestamp: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict:
        return {
            "success": self.success,
            "platform": self.platform,
            "message_id": self.message_id,
            "error": self.error,
            "timestamp": self.timestamp.isoformat()
        }


class SocialPlatformManager:
    """
    Centralized social platform management.

    Provides:
    - Unified messaging interface
    - Platform-specific formatting
    - Alert broadcasting
    - Multi-platform coordination
    """

    def __init__(self):
        self.platforms: Dict[Platform, BasePlatformTool] = {}
        self._initialized = False

    def initialize(
        self,
        telegram_token: Optional[str] = None,
        discord_token: Optional[str] = None,
        slack_token: Optional[str] = None,
        default_channels: Optional[Dict[Platform, str]] = None
    ) -> None:
        """Initialize all platform tools"""
        import os

        # Telegram
        if telegram_token or os.getenv("TELEGRAM_BOT_TOKEN"):
            self.platforms[Platform.TELEGRAM] = TelegramBotTool(
                bot_token=telegram_token
            )

        # Discord
        if discord_token or os.getenv("DISCORD_BOT_TOKEN"):
            self.platforms[Platform.DISCORD] = DiscordBotTool(
                bot_token=discord_token
            )

        # Slack
        if slack_token or os.getenv("SLACK_BOT_TOKEN"):
            self.platforms[Platform.SLACK] = SlackBotTool(
                bot_token=slack_token
            )

        self.default_channels = default_channels or {}
        self._initialized = True

    def send_message(
        self,
        platform: Union[Platform, str],
        recipient: str,
        message: str,
        **kwargs
    ) -> MessageResult:
        """Send message to a platform"""
        if isinstance(platform, str):
            platform = Platform(platform.lower())

        if platform not in self.platforms:
            return MessageResult(
                success=False,
                platform=platform.value,
                error=f"Platform {platform.value} not initialized"
            )

        tool = self.platforms[platform]

        try:
            if platform == Platform.TELEGRAM:
                result = tool._send_message(recipient, message, **kwargs)
            elif platform == Platform.DISCORD:
                result = tool._send_message(recipient, message, **kwargs)
            elif platform == Platform.SLACK:
                result = tool._send_message(recipient, message, **kwargs)
            else:
                return MessageResult(success=False, platform=platform.value, error="Unknown platform")

            if result.get("success"):
                return MessageResult(
                    success=True,
                    platform=platform.value,
                    message_id=result.get("message_id")
                )
            else:
                return MessageResult(
                    success=False,
                    platform=platform.value,
                    error=result.get("error")
                )

        except Exception as e:
            return MessageResult(
                success=False,
                platform=platform.value,
                error=str(e)
            )

    def send_alert(
        self,
        platform: Union[Platform, str],
        recipient: str,
        alert_type: str,
        title: str,
        message: str,
        priority: str = "normal"
    ) -> MessageResult:
        """Send formatted alert to a platform"""
        if isinstance(platform, str):
            platform = Platform(platform.lower())

        if platform not in self.platforms:
            return MessageResult(
                success=False,
                platform=platform.value,
                error=f"Platform {platform.value} not initialized"
            )

        tool = self.platforms[platform]

        try:
            if platform == Platform.TELEGRAM:
                result = tool.send_alert(recipient, alert_type, title, message, priority)
            elif platform == Platform.DISCORD:
                result = tool.send_alert(recipient, alert_type, title, message, priority)
            elif platform == Platform.SLACK:
                result = tool.send_alert(recipient, alert_type, title, message, priority)
            else:
                return MessageResult(success=False, platform=platform.value, error="Unknown platform")

            return MessageResult(
                success=result.get("success", False),
                platform=platform.value,
                error=result.get("error")
            )

        except Exception as e:
            return MessageResult(
                success=False,
                platform=platform.value,
                error=str(e)
            )

    def send_report(
        self,
        platform: Union[Platform, str],
        recipient: str,
        report_type: str,
        content: str,
        **kwargs
    ) -> MessageResult:
        """Send formatted report to a platform"""
        if isinstance(platform, str):
            platform = Platform(platform.lower())

        if platform not in self.platforms:
            return MessageResult(
                success=False,
                platform=platform.value,
                error=f"Platform {platform.value} not initialized"
            )

        tool = self.platforms[platform]

        try:
            if platform == Platform.TELEGRAM:
                result = tool.send_report(recipient, report_type, content)
            elif platform == Platform.DISCORD:
                result = tool.send_report(recipient, report_type, content)
            elif platform == Platform.SLACK:
                result = tool.send_report(recipient, report_type, content)
            else:
                return MessageResult(success=False, platform=platform.value, error="Unknown platform")

            return MessageResult(
                success=result.get("success", False),
                platform=platform.value,
                error=result.get("error")
            )

        except Exception as e:
            return MessageResult(
                success=False,
                platform=platform.value,
                error=str(e)
            )

    def broadcast_alert(
        self,
        alert_config: AlertConfig,
        platforms: Optional[List[Platform]] = None
    ) -> List[MessageResult]:
        """Broadcast alert to multiple platforms"""
        if platforms is None:
            platforms = list(self.platforms.keys())

        results = []
        for platform in platforms:
            result = self.send_alert(
                platform=platform,
                recipient=alert_config.recipient,
                alert_type=alert_config.alert_type,
                title=alert_config.title,
                message=alert_config.message,
                priority=alert_config.priority
            )
            results.append(result)

        return results

    def get_platform_status(self) -> Dict[str, Any]:
        """Get status of all platforms"""
        status = {}
        for platform, tool in self.platforms.items():
            status[platform.value] = {
                "initialized": True,
                "name": type(tool).__name__
            }
        return status

    def format_multi_platform_alert(
        self,
        title: str,
        message: str,
        priority: str = "normal"
    ) -> Dict[str, str]:
        """Format alert message for each platform"""
        formatted = {}

        # Telegram markdown
        formatted["telegram"] = f"🔔 **{title}**\n\n{message}"

        # Discord embed
        formatted["discord"] = f"**{title}**\n\n{message}"

        # Slack
        formatted["slack"] = f"*{title}*\n>{message}"

        return formatted


class SocialToolsFactory:
    """Factory for creating social platform tools as LangChain tools"""

    @staticmethod
    def create_all_tools() -> List[BasePlatformTool]:
        """Create all available social platform tools"""
        return [
            TelegramBotTool(),
            TelegramGroupTool(),
            DiscordBotTool(),
            DiscordWebhookTool(),
            SlackBotTool(),
            SlackWebhookTool(),
        ]

    @staticmethod
    def create_telegram_tools() -> List[TelegramBotTool]:
        """Create Telegram-specific tools"""
        return [
            TelegramBotTool(),
            TelegramGroupTool(),
        ]

    @staticmethod
    def create_discord_tools() -> List[Union[DiscordBotTool, DiscordWebhookTool]]:
        """Create Discord-specific tools"""
        return [
            DiscordBotTool(),
            DiscordWebhookTool(),
        ]

    @staticmethod
    def create_slack_tools() -> List[Union[SlackBotTool, SlackWebhookTool]]:
        """Create Slack-specific tools"""
        return [
            SlackBotTool(),
            SlackWebhookTool(),
        ]

    @staticmethod
    def create_social_tools_for_agent(platforms: List[str]) -> List[BasePlatformTool]:
        """Create tools for specific platforms"""
        tools = []
        platform_list = [p.lower() for p in platforms]

        if "telegram" in platform_list:
            tools.extend(SocialToolsFactory.create_telegram_tools())

        if "discord" in platform_list:
            tools.extend(SocialToolsFactory.create_discord_tools())

        if "slack" in platform_list:
            tools.extend(SocialToolsFactory.create_slack_tools())

        return tools


# Quick access singleton
_manager: Optional[SocialPlatformManager] = None


def get_social_manager() -> SocialPlatformManager:
    """Get singleton social manager instance"""
    global _manager
    if _manager is None:
        _manager = SocialPlatformManager()
        _manager.initialize()
    return _manager


def send_telegram_alert(chat_id: str, title: str, message: str, priority: str = "normal") -> Dict:
    """Quick helper for Telegram alerts"""
    manager = get_social_manager()
    result = manager.send_alert(Platform.TELEGRAM, chat_id, "alert", title, message, priority)
    return result.to_dict()


def send_discord_alert(channel_id: str, title: str, message: str, priority: str = "normal") -> Dict:
    """Quick helper for Discord alerts"""
    manager = get_social_manager()
    result = manager.send_alert(Platform.DISCORD, channel_id, "alert", title, message, priority)
    return result.to_dict()


def send_slack_alert(channel: str, title: str, message: str, priority: str = "normal") -> Dict:
    """Quick helper for Slack alerts"""
    manager = get_social_manager()
    result = manager.send_alert(Platform.SLACK, channel, "alert", title, message, priority)
    return result.to_dict()