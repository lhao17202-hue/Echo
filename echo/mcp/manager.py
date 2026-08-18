"""Manager for configured MCP stdio servers."""

from __future__ import annotations

import logging
from pathlib import Path

from echo.mcp.adapter import McpToolAdapter
from echo.mcp.client import McpClientSession, McpToolDefinition
from echo.mcp.config import McpConfig, load_mcp_config, normalize_mcp_name
from echo.tools.registry import ToolRegistry

logger = logging.getLogger("echo.mcp")


class McpManager:
    """Owns MCP sessions and registers remote tools into Echo."""

    def __init__(self, config: McpConfig):
        self.config = config
        self._sessions: list[McpClientSession] = []
        self._registered_tools: dict[str, list[str]] = {}
        self._last_errors: dict[str, str] = {}

    @classmethod
    def from_file(cls, path: Path) -> "McpManager":
        """Create a manager from a workspace .echo/mcp.json path."""
        return cls(load_mcp_config(path))

    def register_tools(self, registry: ToolRegistry) -> None:
        """Start configured servers and register their tools.

        A failed server is logged and skipped. Other servers and built-in tools remain available.
        """
        for server in self.config.servers:
            session = McpClientSession(server)
            try:
                session.initialize()
                tools = session.list_tools()
                adapters = self._build_adapters(server.name, session, tools)
                registered = []
                for adapter in adapters:
                    registry.register(adapter)
                    registered.append(adapter.name)
                self._registered_tools[server.name] = registered
                self._last_errors.pop(server.name, None)
                self._sessions.append(session)
            except Exception as exc:
                logger.warning("Skipping MCP server %s: %s", server.name, exc)
                self._last_errors[server.name] = str(exc)
                session.close()

    def close(self) -> None:
        """Close all live MCP sessions."""
        while self._sessions:
            session = self._sessions.pop()
            try:
                session.close()
            except Exception as exc:
                logger.warning("Error closing MCP server %s: %s", session.config.name, exc)

    def snapshot(self) -> dict:
        """Return concise status for configured MCP servers."""
        live_names = {session.config.name for session in self._sessions}
        servers = []
        for server in self.config.servers:
            servers.append({
                "name": server.name,
                "status": "running" if server.name in live_names else "configured",
                "registered_tools": list(self._registered_tools.get(server.name, [])),
                "last_error": self._last_errors.get(server.name, ""),
            })
        return {"servers": servers}

    def _build_adapters(
        self,
        server_name: str,
        session: McpClientSession,
        tools: list[McpToolDefinition],
    ) -> list[McpToolAdapter]:
        server_segment = normalize_mcp_name(server_name)
        adapters: list[McpToolAdapter] = []
        seen: set[str] = set()

        for tool in tools:
            tool_segment = normalize_mcp_name(tool.name)
            echo_name = f"mcp_{server_segment}_{tool_segment}"
            if echo_name in seen:
                raise ValueError(f"Duplicate MCP tool name for server {server_name}: {echo_name}")
            seen.add(echo_name)
            adapters.append(McpToolAdapter(echo_name, server_name, tool, session))

        return adapters
