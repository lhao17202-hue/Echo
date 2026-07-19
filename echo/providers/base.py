"""LLM client — data models + BaseLLMClient abstract class."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Iterator


# ── Data Models ────────────────────────────────────

@dataclass
class TokenUsage:
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0

    @property
    def total(self) -> int:
        return self.input_tokens + self.output_tokens


@dataclass
class TextBlock:
    text: str


@dataclass
class ToolUseBlock:
    id: str
    name: str
    input: dict


@dataclass
class ModelResponse:
    content: list  # list[TextBlock | ToolUseBlock]
    stop_reason: str = ""   # "end_turn" | "tool_use" | "max_tokens" | "stop"
    usage: TokenUsage | None = None
    model: str = ""


@dataclass
class StreamEvent:
    type: str = ""           # "text_delta" | "tool_use_start" | "tool_use_delta" | "tool_use_end" | "done"
    text: str = ""
    tool_id: str = ""
    tool_name: str = ""
    tool_input_json: str = ""


# ── Abstract Base ─────────────────────────────────

class BaseLLMClient(ABC):
    """统一的 LLM 客户端接口。全同步。

    具体实现：
      - AnthropicClient: 封装 anthropic SDK
      - OpenAIClient:    封装 openai SDK
    """

    def __init__(self, model: str, api_key: str, base_url: str | None = None):
        self.model = model
        self.api_key = api_key
        self.base_url = base_url

    @abstractmethod
    def chat(
        self,
        messages: list[dict],
        tools: list[dict] | None = None,
        system: str = "",
        max_tokens: int = 8000,
        temperature: float = 0.0,
    ) -> ModelResponse:
        """同步调用，返回完整响应。"""
        ...

    @abstractmethod
    def chat_stream(
        self,
        messages: list[dict],
        tools: list[dict] | None = None,
        system: str = "",
        max_tokens: int = 8000,
        temperature: float = 0.0,
    ) -> Iterator[StreamEvent]:
        """流式调用，返回生成器。"""
        ...

    def count_tokens(self, text: str) -> int:
        """估算 token 数。默认按字符数 / 3 估算，子类可覆写。"""
        return len(text) // 3
