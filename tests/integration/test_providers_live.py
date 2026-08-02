"""Live provider integration tests.

Run explicitly:
    ECHO_RUN_PROVIDER_LIVE=1 python -m pytest tests/integration/test_providers_live.py -q

These tests load .env and call real provider APIs. They are skipped by default.
"""

import importlib.util
import json
import os
import sys
import time
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parents[2]))

from echo.config import (
    DEFAULT_ANTHROPIC_MODEL,
    DEFAULT_OPENAI_MODEL,
    load_dotenv,
    provider_env,
)
from echo.providers.base import TextBlock, ToolUseBlock


load_dotenv(str(Path(__file__).parents[2]))

RUN_LIVE = os.environ.get("ECHO_RUN_PROVIDER_LIVE") == "1"

pytestmark = [
    pytest.mark.integration,
    pytest.mark.provider_live,
    pytest.mark.skipif(
        not RUN_LIVE,
        reason="set ECHO_RUN_PROVIDER_LIVE=1 to call real provider APIs",
    ),
]


def _require_package(package: str) -> None:
    if importlib.util.find_spec(package) is None:
        pytest.skip(f"{package} package is not installed")


def _text(response) -> str:
    return "\n".join(
        block.text for block in response.content if isinstance(block, TextBlock)
    )


def _tool_uses(response) -> list[ToolUseBlock]:
    return [block for block in response.content if isinstance(block, ToolUseBlock)]


def _probe_tool() -> dict:
    return {
        "name": "echo_probe",
        "description": "Return a probe value for integration testing.",
        "input_schema": {
            "type": "object",
            "properties": {
                "value": {
                    "type": "string",
                    "description": "The exact probe value.",
                },
            },
            "required": ["value"],
            "additionalProperties": False,
        },
    }


def _assert_probe_tool_use(response) -> None:
    tool_uses = _tool_uses(response)
    assert tool_uses, f"expected a tool call, got text: {_text(response)!r}"
    probe = tool_uses[0]
    assert probe.name == "echo_probe"
    assert probe.input.get("value") == "ping"


def _run_probe_tool(name: str, tool_input: dict) -> str:
    assert name == "echo_probe"
    assert tool_input == {"value": "ping"}
    return f"probe:{tool_input['value']}"


def _anthropic_client():
    _require_package("anthropic")
    api_key = provider_env("ANTHROPIC_API_KEY")
    if not api_key:
        pytest.skip("ANTHROPIC_API_KEY is not set")

    from echo.providers.anthropic_client import AnthropicClient

    return AnthropicClient(
        model=provider_env("ANTHROPIC_MODEL", default=DEFAULT_ANTHROPIC_MODEL),
        api_key=api_key,
        base_url=provider_env("ANTHROPIC_BASE_URL") or None,
    )


class TestAnthropicLiveProvider:
    def test_plain_chat_responds(self):
        _require_package("anthropic")
        api_key = provider_env("ANTHROPIC_API_KEY")
        if not api_key:
            pytest.skip("ANTHROPIC_API_KEY is not set")

        from echo.providers.anthropic_client import AnthropicClient

        client = AnthropicClient(
            model=provider_env("ANTHROPIC_MODEL", default=DEFAULT_ANTHROPIC_MODEL),
            api_key=api_key,
            base_url=provider_env("ANTHROPIC_BASE_URL") or None,
        )
        response = client.chat(
            [{"role": "user", "content": [TextBlock(text="Reply with exactly: pong")]}],
            max_tokens=128,
            temperature=0.0,
        )
        
        print("=" * 60)
        print("完整 ModelResponse 对象信息")
        print("=" * 60)
        print(f"response.model = {repr(response.model)}")
        print(f"response.stop_reason = {repr(response.stop_reason)}")
        print(f"response.content 完整列表 = {response.content}")
        print(f"提取纯文本 _text(response) = {repr(_text(response))}")
    
        print("\n----- TokenUsage 用量信息 -----")
        if response.usage is not None:
            print(f"input_tokens = {response.usage.input_tokens}")
            print(f"output_tokens = {response.usage.output_tokens}")
            print(f"cache_read_tokens = {response.usage.cache_read_tokens}")
            print(f"cache_write_tokens = {response.usage.cache_write_tokens}")
        else:
            print("usage = None（无token消耗数据）")

        print("\n----- 逐个解析 content 内每一块 -----")
        for idx, block in enumerate(response.content):
            print(f"块{idx} 类型: {type(block).__name__}, 完整对象: {repr(block)}")
            if hasattr(block, "text"):
                print(f"  -> text = {repr(block.text)}")
            if hasattr(block, "name"):
                print(f"  -> tool name = {repr(block.name)}")
                print(f"  -> tool input = {block.input}")
        print("=" * 60 + "\n")
        assert "pong" in _text(response).lower()

    def test_tool_call_responds(self):
        _require_package("anthropic")
        api_key = provider_env("ANTHROPIC_API_KEY")
        if not api_key:
            pytest.skip("ANTHROPIC_API_KEY is not set")

        from echo.providers.anthropic_client import AnthropicClient

        client = AnthropicClient(
            model=provider_env("ANTHROPIC_MODEL", default=DEFAULT_ANTHROPIC_MODEL),
            api_key=api_key,
            base_url=provider_env("ANTHROPIC_BASE_URL") or None,
        )
        response = client.chat(
            [{
                "role": "user",
                "content": [TextBlock(
                    text=(
                        "Call the echo_probe tool once with value set exactly "
                        "to ping. Do not answer in plain text."
                    ),
                )],
            }],
            tools=[_probe_tool()],
            max_tokens=80,
            temperature=0.0,
        )
        print("完整文本：", repr(_text(response)), "\n")
    
    # 新增：打印全部内容块、ToolUseBlock详情
        print("==== response.content 所有块 ====")
        print(response.content)

        tool_list = _tool_uses(response)
        print("\n==== 提取到的工具调用列表 ====")
        for i, tool in enumerate(tool_list):
            print(f"工具{i} 完整对象: {repr(tool)}")
            print(f"  工具名称 name = {tool.name}")
            print(f"  入参 input = {tool.input}")
        _assert_probe_tool_use(response)


class TestAnthropicLiveStreamingProvider:
    def test_plain_chat_stream_pushes_text_before_completion(self):
        client = _anthropic_client()

        started_at = time.perf_counter()
        stream = client.chat_stream(
            [{
                "role": "user",
                "content": [TextBlock(
                    text=(
                        "Reply with exactly this sentence and nothing else: "
                        "alpha beta gamma delta epsilon zeta eta theta iota kappa."
                    ),
                )],
            }],
            max_tokens=96,
            temperature=0.0,
        )
        text_deltas = []
        render_snapshots = []
        events = []
        first_text_at = None
        done_at = None

        for event in stream:
            events.append(event)
            if event.type == "done":
                done_at = time.perf_counter()
                break
            if event.type == "text_delta" and event.text:
                if first_text_at is None:
                    first_text_at = time.perf_counter()
                text_deltas.append(event.text)
                render_snapshots.append("".join(text_deltas))
                if len(text_deltas) == 2:
                    break

        assert len(text_deltas) >= 2, "expected multiple text deltas before completion"
        assert first_text_at is not None
        assert first_text_at >= started_at
        assert not any(event.type == "done" for event in events)
        assert len(render_snapshots) >= 2
        assert render_snapshots[0] != render_snapshots[-1]

        for event in stream:
            events.append(event)
            if event.type == "done":
                done_at = time.perf_counter()
                break

        streamed_text = "".join(
            event.text for event in events if event.type == "text_delta"
        )
        assert "alpha beta gamma" in streamed_text.lower()
        assert any(event.type == "done" for event in events)
        assert done_at is not None
        assert first_text_at < done_at

    def test_tool_call_stream_provides_executable_tool_call_before_completion(self):
        client = _anthropic_client()

        stream = client.chat_stream(
            [{
                "role": "user",
                "content": [TextBlock(
                    text=(
                        "Call the echo_probe tool once with value set exactly "
                        "to ping. Do not answer in plain text."
                    ),
                )],
            }],
            tools=[_probe_tool()],
            max_tokens=80,
            temperature=0.0,
        )
        events = []
        current_tool = None
        input_chunks = []
        executed_result = None
        executed_at_event_index = None
        done_event_index = None

        for event_index, event in enumerate(stream):
            events.append(event)
            if event.type == "done":
                done_event_index = event_index
                break
            if event.type == "tool_use_start":
                current_tool = event
            elif event.type == "tool_use_delta" and current_tool is not None:
                input_chunks.append(event.tool_input_json)
            elif event.type == "tool_use_end" and current_tool is not None:
                tool_input = json.loads("".join(input_chunks))
                executed_result = _run_probe_tool(current_tool.tool_name, tool_input)
                executed_at_event_index = event_index
                break

        assert executed_result == "probe:ping"
        assert executed_at_event_index is not None
        assert not any(event.type == "done" for event in events)

        for offset, event in enumerate(stream, start=len(events)):
            events.append(event)
            if event.type == "done":
                done_event_index = offset
                break

        assert any(event.type == "done" for event in events)
        assert done_event_index is not None
        assert executed_at_event_index < done_event_index


class TestOpenAILiveProvider:
    def test_plain_chat_responds(self):
        _require_package("openai")
        api_key = provider_env("OPENAI_API_KEY")
        if not api_key:
            pytest.skip("OPENAI_API_KEY is not set")

        from echo.providers.openai_client import OpenAIClient

        client = OpenAIClient(
            model=provider_env("OPENAI_MODEL", default=DEFAULT_OPENAI_MODEL),
            api_key=api_key,
            base_url=provider_env("OPENAI_API_BASE") or None,
        )
        response = client.chat(
            [{"role": "user", "content": [TextBlock(text="Reply with exactly: pong")]}],
            max_tokens=16,
            temperature=0.0,
        )
        assert "pong" in _text(response).lower()

    def test_tool_call_responds(self):
        _require_package("openai")
        api_key = provider_env("OPENAI_API_KEY")
        if not api_key:
            pytest.skip("OPENAI_API_KEY is not set")

        from echo.providers.openai_client import OpenAIClient

        client = OpenAIClient(
            model=provider_env("OPENAI_MODEL", default=DEFAULT_OPENAI_MODEL),
            api_key=api_key,
            base_url=provider_env("OPENAI_API_BASE") or None,
        )
        response = client.chat(
            [{
                "role": "user",
                "content": [TextBlock(
                    text=(
                        "Call the echo_probe tool once with value set exactly "
                        "to ping. Do not answer in plain text."
                    ),
                )],
            }],
            tools=[_probe_tool()],
            max_tokens=80,
            temperature=0.0,
        )

        _assert_probe_tool_use(response)
