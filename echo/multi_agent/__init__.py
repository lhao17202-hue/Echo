"""Multi-agent — sub-agent, teammate, message bus, global task manager."""

from echo.multi_agent.message_bus import MessageItem
from echo.multi_agent.task_manager import GlobalTask

__all__ = ["MessageItem", "GlobalTask"]
