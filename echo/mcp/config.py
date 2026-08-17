"""Configuration loading for workspace MCP stdio servers."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path


class McpConfigError(ValueError):
    """Raised when .echo/mcp.json is malformed or unsafe to use."""


@dataclass(frozen=True)
class McpServerConfig:
    """One Claude Desktop-compatible stdio MCP server entry."""

    name: str
    command: str
    args: list[str]
    env: dict[str, str]


@dataclass(frozen=True)
class McpConfig:
    """All MCP servers configured for a workspace."""

    servers: list[McpServerConfig]


def normalize_mcp_name(name: str) -> str:
    """Normalize a server or tool name into a safe Echo tool-name segment."""
    normalized = re.sub(r"[^a-z0-9]+", "_", str(name).strip().lower()).strip("_")
    if not normalized:
        raise McpConfigError(f"MCP name {name!r} normalizes to an empty name")
    return normalized


def load_mcp_config(path: Path) -> McpConfig:
    """Load Claude Desktop-compatible .echo/mcp.json config."""
    path = Path(path)
    if not path.exists():
        return McpConfig(servers=[])

    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise McpConfigError(f"Invalid MCP config JSON in {path}: {exc}") from exc

    if not isinstance(raw, dict):
        raise McpConfigError("MCP config must be a JSON object")

    servers_raw = raw.get("mcpServers", {})
    if not isinstance(servers_raw, dict):
        raise McpConfigError("mcpServers must be an object")

    servers: list[McpServerConfig] = []
    for name, server_raw in servers_raw.items():
        if not isinstance(name, str) or not name.strip():
            raise McpConfigError("MCP server names must be non-empty strings")
        normalize_mcp_name(name)

        if not isinstance(server_raw, dict):
            raise McpConfigError(f"{name} must be an object")

        command = server_raw.get("command")
        if not isinstance(command, str) or not command.strip():
            raise McpConfigError(f"{name}.command must be a non-empty string")

        args = server_raw.get("args", [])
        if not isinstance(args, list) or not all(isinstance(arg, str) for arg in args):
            raise McpConfigError(f"{name}.args must be a list of strings")

        env = server_raw.get("env", {})
        if (
            not isinstance(env, dict)
            or not all(isinstance(key, str) and isinstance(value, str) for key, value in env.items())
        ):
            raise McpConfigError(f"{name}.env must be an object with string keys and values")

        servers.append(McpServerConfig(
            name=name,
            command=command,
            args=list(args),
            env=dict(env),
        ))

    return McpConfig(servers=servers)
