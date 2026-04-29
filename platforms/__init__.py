"""
Social/Messaging Platform Integrations for FinNavigator
=======================================================

Provides bot integrations for:
- Telegram
- Discord
- Slack

As LangChain tools that agents can use for communication.

Author: MiniMax Agent
"""

from .telegram_bot import TelegramBotTool
from .discord_bot import DiscordBotTool
from .slack_bot import SlackBotTool
from .base_platform import BasePlatformTool, PlatformMessage

__all__ = [
    "TelegramBotTool",
    "DiscordBotTool",
    "SlackBotTool",
    "BasePlatformTool",
    "PlatformMessage",
]