"""Ollama provider — ollama SDK wrapper (local + remote).

Ollama 的 chat API 是 OpenAI 兼容格式的。消息格式与 OpenAI 类似，
但 tool calling 有一些细节差异（message.tool_calls 在 assistant 消息中）。

使用方式：
  client = OllamaClient(model="qwen3:4b", host="http://localhost:11434")
  response = client.chat(messages, tools, system)
"""

import json
import logging
from typing import Iterator
from echo.providers.base import (
    BaseLLMClient, ModelResponse, TokenUsage,
    TextBlock, ToolUseBlock, StreamEvent,
)

logger = logging.getLogger("echo.ollama")


class OllamaClient(BaseLLMClient):
    """封装 ollama SDK。

    Ollama 默认在 localhost:11434 运行，无需 API key。
    工具调用通过 message.tool_calls 返回（OpenAI 兼容格式）。
    """

    def __init__(self, model: str = "qwen3:4b",
                 api_key: str = "", base_url: str | None = None):
        # base_url for Ollama = host (e.g. "http://localhost:11434")
        super().__init__(model, api_key or "ollama", base_url)
        import ollama
        self._host = base_url or "http://localhost:11434"
        # ollama 模块级函数直接使用，也可以创建 Client
        try:
            self._client = ollama.Client(host=self._host)
        except Exception:
            self._client = None

    @property
    def host(self) -> str:
        """Ollama 服务地址。"""
        return self._host

    # ── chat (non-streaming) ──────────────────────

    def chat(self, messages: list[dict], tools: list[dict] | None = None,
             system: str = "", max_tokens: int = 8000,
             temperature: float = 0.0) -> ModelResponse:
        import ollama

        api_messages = self._build_messages(messages, system)
        api_tools = self._convert_tools(tools) if tools else None

        kwargs = {
            "model": self.model,
            "messages": api_messages,
            "options": {"temperature": temperature, "num_predict": max_tokens},
        }
        if api_tools:
            kwargs["tools"] = api_tools

        response = ollama.chat(**kwargs)
        return self._parse_response(response)

    # ── chat_stream ───────────────────────────────

    def chat_stream(self, messages: list[dict], tools: list[dict] | None = None,
                    system: str = "", max_tokens: int = 8000,
                    temperature: float = 0.0) -> Iterator[StreamEvent]:
        import ollama

        api_messages = self._build_messages(messages, system)
        api_tools = self._convert_tools(tools) if tools else None

        kwargs = {
            "model": self.model,
            "messages": api_messages,
            "options": {"temperature": temperature, "num_predict": max_tokens},
            "stream": True,
        }
        if api_tools:
            kwargs["tools"] = api_tools

        for chunk in ollama.chat(**kwargs):
            yield self._parse_stream_chunk(chunk)

    # ── 内部格式 → Ollama 消息 ────────────────────

    def _build_messages(self, messages: list[dict], system: str = "") -> list[dict]:
        """构建 Ollama 消息列表（OpenAI 兼容格式）。

        Ollama 格式：
          {"role": "system", "content": "..."}
          {"role": "user", "content": "..."}
          {"role": "assistant", "content": "..."}  (tool_calls 可选)
          {"role": "tool", "content": "..."}
        """
        result = []
        if system:
            result.append({"role": "system", "content": system})

        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", [])

            if role == "tool":
                # 工具结果 → role=tool 消息（兜底兼容旧格式）
                for block in (content if isinstance(content, list) else [content]):
                    if isinstance(block, dict) and block.get("type") == "tool_result":
                        result.append({
                            "role": "tool",
                            "content": block.get("content", ""),
                        })
            elif self._is_tool_results(content):
                # role="user" + tool_result blocks → role=tool（当前主格式）
                for block in content:
                    if isinstance(block, dict) and block.get("type") == "tool_result":
                        result.append({
                            "role": "tool",
                            "content": block.get("content", ""),
                        })
            elif role == "assistant":
                text = self._blocks_to_text(content)
                result.append({"role": "assistant", "content": text})
            else:
                text = self._blocks_to_text(content)
                result.append({"role": "user", "content": text})

        return result

    # ── 工具 schema 转换 ──────────────────────────

    @staticmethod
    def _convert_tools(tools: list[dict]) -> list[dict]:
        """内部 input_schema 格式 → Ollama function 格式。

        Ollama 支持两种格式：
          1. Python 函数（自动生成 schema）— 不使用
          2. dict schema（与 OpenAI 兼容）
        """
        converted = []
        for tool in tools:
            converted.append({
                "type": "function",
                "function": {
                    "name": tool["name"],
                    "description": tool.get("description", ""),
                    "parameters": tool.get("input_schema", {}),
                },
            })
        return converted

    # ── 响应解析 ──────────────────────────────────

    @staticmethod
    def _parse_response(response) -> ModelResponse:
        """Ollama chat response → ModelResponse。"""
        content = []
        msg = getattr(response, "message", response)

        # 文本内容
        msg_content = getattr(msg, "content", "")
        if msg_content:
            content.append(TextBlock(text=str(msg_content)))

        # 工具调用
        tool_calls = getattr(msg, "tool_calls", None) or []
        for tc in tool_calls:
            func = getattr(tc, "function", None)
            if func:
                try:
                    args = json.loads(getattr(func, "arguments", "{}"))
                except json.JSONDecodeError:
                    args = {}
                content.append(ToolUseBlock(
                    id=getattr(tc, "id", getattr(func, "name", "call_00")),
                    name=getattr(func, "name", ""),
                    input=args,
                ))

        return ModelResponse(
            content=content,
            stop_reason="tool_use" if tool_calls else "end_turn",
            usage=TokenUsage(),  # Ollama 不返回 token 数
            model=getattr(response, "model", ""),
        )

    @staticmethod
    def _parse_stream_chunk(chunk) -> StreamEvent:
        """解析 Ollama stream chunk。"""
        msg = getattr(chunk, "message", chunk)
        done = getattr(chunk, "done", False)

        if done:
            return StreamEvent(type="done")

        content = getattr(msg, "content", None)
        tool_calls = getattr(msg, "tool_calls", None)

        if tool_calls:
            # Ollama 流式中 tool_calls 可能完整返回
            return StreamEvent(
                type="tool_use_start",
                tool_name=getattr(tool_calls[0].function, "name", "") if tool_calls else "",
            )

        if content:
            return StreamEvent(type="text_delta", text=str(content))

        return StreamEvent(type="text_delta", text="")

    # ── 辅助 ──────────────────────────────────────

    @staticmethod
    def _is_tool_results(content: list) -> bool:
        if not content:
            return False
        return any(
            isinstance(b, dict) and b.get("type") == "tool_result"
            for b in content
        )

    @staticmethod
    def _blocks_to_text(content: list) -> str:
        parts = []
        for block in (content or []):
            if isinstance(block, TextBlock):
                parts.append(block.text)
            elif isinstance(block, dict):
                parts.append(block.get("text", block.get("content", "")))
        return "\n".join(parts)
