"""Echo BaseTool adapter for remote MCP tools."""

from __future__ import annotations

import json
from typing import Any

from echo.tools.base import BaseTool, ToolContext, ToolResult


class McpToolAdapter(BaseTool):
    """Expose one remote MCP tool as a normal Echo BaseTool."""

    risk_level = "danger"
    is_read_only = False
    params_model = None

    def __init__(self, echo_name: str, server_name: str, tool: Any, session: Any):
        self.name = echo_name
        self.server_name = server_name
        self.remote_tool_name = tool.name
        self.description = _format_description(server_name, tool.name, tool.description)
        self.input_schema = _schema_or_empty(tool.input_schema)
        self.session = session

    def to_schema(self) -> dict[str, Any]:
        """Return the remote MCP schema under the Echo adapter name."""
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": self.input_schema,
        }

    def execute(self, ctx: ToolContext, params: dict[str, Any]) -> ToolResult:
        """Call the original MCP tool and normalize its result."""
        del ctx
        try:
            raw_result = self.session.call_tool(self.remote_tool_name, params)
            normalized = normalize_mcp_result(raw_result)
            if _is_error_result(raw_result):
                return ToolResult.fail(f"MCP tool error: {normalized}")
            return ToolResult.ok(normalized)
        except Exception as exc:
            return ToolResult.fail(f"MCP tool error: {exc}")


def _format_description(server_name: str, tool_name: str, remote_description: str) -> str:
    prefix = f"[MCP: {server_name}] Original tool: {tool_name}"
    if remote_description:
        return f"{prefix}\n{remote_description}"
    return prefix


def _schema_or_empty(schema: Any) -> dict[str, Any]:
    if isinstance(schema, dict) and schema:
        return schema
    return {"type": "object", "properties": {}}


def _is_error_result(result: Any) -> bool:
    if isinstance(result, dict):
        return bool(result.get("isError"))
    return bool(getattr(result, "isError", False) or getattr(result, "is_error", False))


def normalize_mcp_result(result: Any) -> str:
    """Normalize MCP SDK or dict tool results into stable text."""
    if result is None:
        return ""

    if isinstance(result, str):
        return result

    structured = _get_attr_or_key(result, "structuredContent")
    if structured is None:
        structured = _get_attr_or_key(result, "structured_content")
    if structured is not None:
        return _json_text(structured)

    content = _get_attr_or_key(result, "content")
    if content is not None:
        blocks = [_normalize_content_block(block) for block in content]
        return "\n\n".join(block for block in blocks if block is not None)

    return _json_text(result)


def _normalize_content_block(block: Any) -> str:
    block_type = _get_attr_or_key(block, "type")
    if block_type == "text":
        text = _get_attr_or_key(block, "text")
        return "" if text is None else str(text)
    if isinstance(block, (dict, list)):
        if block_type:
            return f"[Unsupported MCP content block: {block_type}]"
        return _json_text(block)
    if block_type:
        return f"[Unsupported MCP content block: {block_type}]"
    return str(block)


def _get_attr_or_key(value: Any, name: str) -> Any:
    if isinstance(value, dict):
        return value.get(name)
    return getattr(value, name, None)


def _json_text(value: Any) -> str:
    try:
        return json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True)
    except TypeError:
        return str(value)
