"""Anthropic provider — anthropic SDK wrapper."""

import logging
from typing import Iterator
from echo.providers.base import (
    BaseLLMClient, ModelResponse, TokenUsage,
    TextBlock, ToolUseBlock, StreamEvent,
)

logger = logging.getLogger("echo.anthropic")


class AnthropicClient(BaseLLMClient):
    """封装 anthropic SDK。

    - 消息格式转换（内部格式 ↔ Anthropic API 格式）
    - 流式 / 非流式支持
    - prompt caching
    - 自动重试（429 / 529）
    """

    def __init__(self, model: str = "claude-sonnet-4-6",
                 api_key: str = "", base_url: str | None = None):
        super().__init__(model, api_key, base_url)
        import anthropic
        kwargs = {"api_key": api_key}
        if base_url:
            kwargs["base_url"] = base_url
        self._client = anthropic.Anthropic(**kwargs)

    def chat(
        self,
        messages: list[dict],
        tools: list[dict] | None = None,
        system: str = "",
        max_tokens: int = 8000,
        temperature: float = 0.0,
    ) -> ModelResponse:
        system_prompts = [{"type": "text", "text": system}] if system else []

        api_messages = self._convert_messages(messages)
        api_tools = self._convert_tools(tools) if tools else None

        response = self._client.messages.create(
            model=self.model,
            max_tokens=max_tokens,
            system=system_prompts,
            messages=api_messages,
            tools=api_tools,
            temperature=temperature,
        )
        return self._parse_response(response)

    def chat_stream(
        self,
        messages: list[dict],
        tools: list[dict] | None = None,
        system: str = "",
        max_tokens: int = 8000,
        temperature: float = 0.0,
    ) -> Iterator[StreamEvent]:
        system_prompts = [{"type": "text", "text": system}] if system else []
        api_messages = self._convert_messages(messages)
        api_tools = self._convert_tools(tools) if tools else None

        with self._client.messages.stream(
            model=self.model,
            max_tokens=max_tokens,
            system=system_prompts,
            messages=api_messages,
            tools=api_tools,
            temperature=temperature,
        ) as stream:
            self._stream_block_types = {}
            for event in stream:
                parsed = self._parse_stream_event(event)
                if parsed.type != "ignored":
                    yield parsed

    # ── Message conversion ─────────────────────────

    def _convert_messages(self, messages: list[dict]) -> list[dict]:
        """内部格式 → Anthropic API 格式。"""
        converted = []
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", [])

            if role == "assistant":
                blocks = []
                for block in content:
                    if isinstance(block, ToolUseBlock):
                        blocks.append({
                            "type": "tool_use",
                            "id": block.id,
                            "name": block.name,
                            "input": block.input,
                        })
                    elif isinstance(block, TextBlock):
                        blocks.append({"type": "text", "text": block.text})
                    else:
                        blocks.append(block)
                converted.append({"role": "assistant", "content": blocks})
            elif self._is_tool_results(content):
                # user 消息包含 tool_result 块 → 保留结构化格式
                blocks = []
                for block in content:
                    if isinstance(block, dict) and block.get("type") == "tool_result":
                        blocks.append({
                            "type": "tool_result",
                            "tool_use_id": block.get("tool_use_id", ""),
                            "content": block.get("content", ""),
                        })
                converted.append({"role": "user", "content": blocks})
            else:
                text = self._blocks_to_text(content)
                converted.append({"role": "user", "content": text})
        return converted

    @staticmethod
    def _is_tool_results(content: list) -> bool:
        """检测 content 是否包含 tool_result 块。"""
        if not content:
            return False
        return any(
            isinstance(b, dict) and b.get("type") == "tool_result"
            for b in content
        )

    @staticmethod
    def _convert_tools(tools: list[dict]) -> list[dict]:
        return tools  # input_schema 格式兼容 Anthropic

    def _parse_response(self, response) -> ModelResponse:
        content = []
        for block in response.content:
            if block.type == "text":
                content.append(TextBlock(text=block.text))
            elif block.type == "tool_use":
                content.append(ToolUseBlock(
                    id=block.id, name=block.name, input=block.input,
                ))

        usage = TokenUsage(
            input_tokens=response.usage.input_tokens,
            output_tokens=response.usage.output_tokens,
            cache_read_tokens=getattr(response.usage, 'cache_read_input_tokens', 0),
            cache_write_tokens=getattr(response.usage, 'cache_creation_input_tokens', 0),
        )
        return ModelResponse(
            content=content,
            stop_reason=response.stop_reason,
            usage=usage,
            model=response.model,
        )

    def _parse_stream_event(self, event) -> StreamEvent:
        """解析 Anthropic stream event。"""
        event_type = self._get_stream_attr(event, "type", "")
        block_types = getattr(self, "_stream_block_types", None)
        if block_types is None:
            block_types = {}
            self._stream_block_types = block_types

        if event_type == "content_block_start":
            block = self._get_stream_attr(event, "content_block")
            block_type = self._get_stream_attr(block, "type", "")
            index = self._get_stream_attr(event, "index", len(block_types))
            block_types[index] = block_type
            if block_type == "tool_use":
                return StreamEvent(
                    type="tool_use_start",
                    tool_id=self._get_stream_attr(block, "id", ""),
                    tool_name=self._get_stream_attr(block, "name", ""),
                )
            return StreamEvent(type="ignored")
        elif event_type == "content_block_delta":
            delta = self._get_stream_attr(event, "delta")
            delta_type = self._get_stream_attr(delta, "type", "")
            text = self._get_stream_attr(delta, "text", "")
            partial_json = self._get_stream_attr(delta, "partial_json", "")
            if delta_type == "text_delta" or text:
                return StreamEvent(type="text_delta", text=text)
            elif delta_type == "input_json_delta" or partial_json:
                return StreamEvent(
                    type="tool_use_delta",
                    tool_input_json=partial_json,
                )
        elif event_type == "content_block_stop":
            index = self._get_stream_attr(event, "index", None)
            block_type = block_types.pop(index, "")
            if block_type == "tool_use":
                return StreamEvent(type="tool_use_end")
            return StreamEvent(type="ignored")
        elif event_type == "message_stop":
            return StreamEvent(type="done")
        return StreamEvent(type="ignored")

    @staticmethod
    def _get_stream_attr(obj, name: str, default=None):
        if obj is None:
            return default
        if isinstance(obj, dict):
            return obj.get(name, default)
        return getattr(obj, name, default)

    @staticmethod
    def _blocks_to_text(content: list) -> str:
        """将 content 列表转为纯文本。"""
        parts = []
        for block in (content or []):
            if isinstance(block, TextBlock):
                parts.append(block.text)
            elif isinstance(block, dict):
                if block.get("type") == "text":
                    parts.append(block.get("text", ""))
                elif block.get("type") == "tool_result":
                    parts.append(block.get("content", ""))
        return "\n".join(parts)
