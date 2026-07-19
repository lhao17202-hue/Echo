"""OpenAI provider — openai SDK Responses API wrapper.

使用最新的 Responses API（不推荐 Chat Completions），原生支持 tool calling。
消息格式转换：内部格式 ↔ OpenAI Responses API 格式。
"""

import json
import logging
from typing import Iterator
from echo.providers.base import (
    BaseLLMClient, ModelResponse, TokenUsage,
    TextBlock, ToolUseBlock, StreamEvent,
)

logger = logging.getLogger("echo.openai")


class OpenAIClient(BaseLLMClient):
    """封装 openai SDK 的 Responses API。

    工具调用流程（与 Anthropic 不同）：
      1. 构建 input 数组（对话历史）
      2. 调用 client.responses.create()
      3. response.output 包含 message items + function_call items
      4. 工具结果以 function_call_output items 发送回 API
    """

    def __init__(self, model: str = "gpt-4o-mini",
                 api_key: str = "", base_url: str | None = None):
        super().__init__(model, api_key, base_url)
        import openai
        kwargs = {"api_key": api_key}
        if base_url:
            kwargs["base_url"] = base_url
        self._client = openai.OpenAI(**kwargs)

    # ── chat (non-streaming) ──────────────────────

    def chat(self, messages: list[dict], tools: list[dict] | None = None,
             system: str = "", max_tokens: int = 8000,
             temperature: float = 0.0) -> ModelResponse:
        input_items = self._build_input(messages, system)
        converted_tools = self._convert_tools(tools) if tools else None

        response = self._client.responses.create(
            model=self.model,
            input=input_items,
            tools=converted_tools,
            max_output_tokens=max_tokens,
            temperature=temperature,
        )
        return self._parse_response(response)

    # ── chat_stream ───────────────────────────────

    def chat_stream(self, messages: list[dict], tools: list[dict] | None = None,
                    system: str = "", max_tokens: int = 8000,
                    temperature: float = 0.0) -> Iterator[StreamEvent]:
        input_items = self._build_input(messages, system)
        converted_tools = self._convert_tools(tools) if tools else None

        stream = self._client.responses.create(
            model=self.model,
            input=input_items,
            tools=converted_tools,
            max_output_tokens=max_tokens,
            temperature=temperature,
            stream=True,
        )
        for event in stream:
            yield self._parse_stream_event(event)

    # ── 内部格式 → OpenAI input ───────────────────

    def _build_input(self, messages: list[dict], system: str = "") -> list[dict]:
        """构建 OpenAI Responses API 的 input 数组。

        OpenAI Responses API 格式与 Anthropic 不同：
          - assistant 中的 function_call 需拆分为独立的 "function_call" items
          - user 中的 tool_result 需拆分为 "function_call_output" items
        """
        input_items = []
        if system:
            input_items.append({"role": "system", "content": system})

        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", [])
            if not isinstance(content, list):
                content = [content]

            if role == "tool":
                # AgentLoop writes role="tool" for tool results (primary path)
                for block in content:
                    if isinstance(block, dict) and block.get("type") == "tool_result":
                        input_items.append({
                            "type": "function_call_output",
                            "call_id": block.get("tool_use_id", ""),
                            "output": block.get("content", ""),
                        })

            elif role == "user" and any(
                isinstance(b, dict) and b.get("type") == "tool_result"
                for b in content
            ):
                # 兜底：某些代码路径可能仍写 role="user"
                for block in content:
                    if isinstance(block, dict) and block.get("type") == "tool_result":
                        input_items.append({
                            "type": "function_call_output",
                            "call_id": block.get("tool_use_id", ""),
                            "output": block.get("content", ""),
                        })

            elif role == "assistant":
                # 分离文本块和工具调用块
                text_parts = []
                for block in content:
                    if isinstance(block, ToolUseBlock):
                        # 工具调用 → 独立 function_call item
                        input_items.append({
                            "type": "function_call",
                            "call_id": block.id,
                            "name": block.name,
                            "arguments": json.dumps(block.input),
                        })
                    elif isinstance(block, TextBlock):
                        text_parts.append(block.text)
                    elif isinstance(block, dict) and block.get("type") == "text":
                        text_parts.append(block.get("text", ""))
                if text_parts:
                    input_items.append({
                        "role": "assistant",
                        "content": "\n".join(text_parts),
                    })

            else:
                text = self._blocks_to_text(content)
                input_items.append({"role": "user", "content": text})

        return input_items

    # ── 工具 schema 转换 ──────────────────────────

    @staticmethod
    def _convert_tools(tools: list[dict]) -> list[dict]:
        """内部 input_schema 格式 → OpenAI function 格式。"""
        converted = []
        for tool in tools:
            converted.append({
                "type": "function",
                "name": tool["name"],
                "description": tool.get("description", ""),
                "parameters": tool.get("input_schema", {}),
            })
        return converted

    # ── 响应解析 ──────────────────────────────────

    @staticmethod
    def _parse_response(response) -> ModelResponse:
        """OpenAI response.output → ModelResponse。"""
        content = []
        usage = TokenUsage()

        if hasattr(response, "output"):
            for item in response.output:
                if item.type == "message":
                    for c in getattr(item, "content", []):
                        if hasattr(c, "text"):
                            content.append(TextBlock(text=c.text))
                elif item.type == "function_call":
                    content.append(ToolUseBlock(
                        id=getattr(item, "call_id", ""),
                        name=getattr(item, "name", ""),
                        input=json.loads(getattr(item, "arguments", "{}")),
                    ))

        if hasattr(response, "usage") and response.usage:
            usage = TokenUsage(
                input_tokens=getattr(response.usage, "input_tokens", 0),
                output_tokens=getattr(response.usage, "output_tokens", 0),
            )

        return ModelResponse(
            content=content,
            stop_reason="tool_use" if any(isinstance(b, ToolUseBlock) for b in content) else "end_turn",
            usage=usage,
            model=getattr(response, "model", ""),
        )

    @staticmethod
    def _parse_stream_event(event) -> StreamEvent:
        """解析 OpenAI stream event。"""
        etype = getattr(event, "type", "")
        if etype == "response.output_text.delta":
            return StreamEvent(type="text_delta", text=getattr(event, "delta", ""))
        elif etype == "response.function_call_arguments.delta":
            return StreamEvent(type="tool_use_delta", tool_input_json=getattr(event, "delta", ""))
        elif etype == "response.function_call_arguments.done":
            return StreamEvent(type="tool_use_end")
        elif etype == "response.completed":
            return StreamEvent(type="done")
        return StreamEvent(type="text_delta", text="")

    # ── 辅助 ──────────────────────────────────────

    @staticmethod
    def _blocks_to_text(content: list) -> str:
        parts = []
        for block in (content or []):
            if isinstance(block, TextBlock):
                parts.append(block.text)
            elif isinstance(block, dict):
                parts.append(block.get("text", block.get("content", "")))
        return "\n".join(parts)
