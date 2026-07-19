"""Provider layer — LLM client abstraction (Anthropic / OpenAI / Ollama / Fake)."""

from echo.providers.base import (
    BaseLLMClient, TokenUsage, TextBlock, ToolUseBlock, ModelResponse, StreamEvent,
)
from echo.providers.anthropic_client import AnthropicClient
from echo.providers.openai_client import OpenAIClient
from echo.providers.ollama_client import OllamaClient
from echo.providers.fake_client import FakeLLMClient

__all__ = [
    "BaseLLMClient", "TokenUsage", "TextBlock", "ToolUseBlock",
    "ModelResponse", "StreamEvent",
    "AnthropicClient", "OpenAIClient", "OllamaClient", "FakeLLMClient",
]
