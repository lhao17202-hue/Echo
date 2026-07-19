"""Provider adapter unit tests — tool result format conversion (no real API).

Run: python -m pytest tests/test_providers.py -v
"""

import sys
from pathlib import Path
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from echo.providers.base import TextBlock, ToolUseBlock, StreamEvent
from echo.providers.anthropic_client import AnthropicClient
from echo.providers.openai_client import OpenAIClient
from echo.providers.ollama_client import OllamaClient


# ── 测试用的标准内部消息 ────────────────────────────

def _make_messages(tool_result_count: int = 2):
    """构建 AgentLoop 格式的标准消息序列。

    模拟一轮完整的 LLM → 工具调用 → 工具结果。
    """
    tool_use_blocks = [
        ToolUseBlock(id=f"call_{i:02d}", name=f"tool_{i}", input={"arg": f"val_{i}"})
        for i in range(tool_result_count)
    ]
    tool_result_blocks = [
        {"type": "tool_result", "tool_use_id": f"call_{i:02d}", "content": f"result_{i}"}
        for i in range(tool_result_count)
    ]

    return [
        {"role": "user", "content": [TextBlock(text="do something")]},
        {"role": "assistant", "content": [TextBlock(text="Let me check."), *tool_use_blocks]},
        {"role": "user", "content": tool_result_blocks},
    ]


# ═══════════════════════════════════════════════════
# AnthropicClient
# ═══════════════════════════════════════════════════

class TestAnthropicAdapter:
    """Anthropic adapter: 内部消息 → Anthropic Messages API 格式。"""

    def test_user_messages_preserved(self):
        c = AnthropicClient(api_key="test-key")
        msgs = _make_messages(1)
        result = c._convert_messages(msgs)
        assert result[0]["role"] == "user"
        assert isinstance(result[0]["content"], str)

    def test_assistant_preserves_tool_use(self):
        c = AnthropicClient(api_key="test-key")
        msgs = _make_messages(2)
        result = c._convert_messages(msgs)
        ass = result[1]
        assert ass["role"] == "assistant"
        assert len(ass["content"]) == 3  # text + 2 tool_use
        assert ass["content"][1]["type"] == "tool_use"
        assert ass["content"][1]["id"] == "call_00"
        assert ass["content"][1]["name"] == "tool_0"

    def test_tool_results_preserved_with_ids(self):
        c = AnthropicClient(api_key="test-key")
        msgs = _make_messages(2)
        result = c._convert_messages(msgs)
        r = result[2]
        assert r["role"] == "user"
        assert isinstance(r["content"], list)
        assert r["content"][0]["type"] == "tool_result"
        assert r["content"][0]["tool_use_id"] == "call_00"
        assert r["content"][0]["content"] == "result_0"
        assert r["content"][1]["tool_use_id"] == "call_01"

    def test_tool_result_ids_are_not_empty(self):
        c = AnthropicClient(api_key="test-key")
        msgs = _make_messages(1)
        result = c._convert_messages(msgs)
        r = result[2]["content"][0]
        assert r["tool_use_id"] == "call_00"

    def test_single_tool_result(self):
        c = AnthropicClient(api_key="test-key")
        msgs = _make_messages(1)
        result = c._convert_messages(msgs)
        assert len(result) == 3
        assert result[2]["content"][0]["tool_use_id"] == "call_00"

    def test_multiple_tool_results(self):
        c = AnthropicClient(api_key="test-key")
        msgs = _make_messages(5)
        result = c._convert_messages(msgs)
        blocks = result[2]["content"]
        assert len(blocks) == 5
        for i, b in enumerate(blocks):
            assert b["tool_use_id"] == f"call_{i:02d}"

    def test_plain_text_user_not_mistaken_for_tool(self):
        c = AnthropicClient(api_key="test-key")
        msgs = [{"role": "user", "content": [TextBlock(text="plain text")]}]
        result = c._convert_messages(msgs)
        assert isinstance(result[0]["content"], str)
        assert "plain text" in result[0]["content"]

    def test_schema_passthrough(self):
        c = AnthropicClient(api_key="test-key")
        tools = [{"name": "t", "description": "d", "input_schema": {"type": "object"}}]
        assert c._convert_tools(tools) == tools

    def test_stream_event_types(self):
        """验证 stream 事件类型解析覆盖所有主要事件。"""
        c = AnthropicClient(api_key="test-key")
        from unittest.mock import Mock

        e = Mock(type="content_block_delta")
        e.delta = Mock(type="text_delta")
        e.delta.text = "hi"
        se = c._parse_stream_event(e)
        assert se.type == "text_delta" and se.text == "hi"

        e2 = Mock(type="content_block_start")
        e2.content_block = Mock(type="tool_use")
        e2.content_block.id = "t1"
        e2.content_block.name = "grep"
        se2 = c._parse_stream_event(e2)
        assert se2.type == "tool_use_start" and se2.tool_name == "grep"

        e3 = Mock(type="message_stop")
        se3 = c._parse_stream_event(e3)
        assert se3.type == "done"


# ═══════════════════════════════════════════════════
# OpenAIClient
# ═══════════════════════════════════════════════════

class TestOpenAIAdapter:
    """OpenAI adapter: 内部消息 → OpenAI Responses API 格式。"""

    def test_tool_results_become_function_call_outputs(self):
        oc = OpenAIClient(api_key="test-key")
        msgs = _make_messages(2)
        result = oc._build_input(msgs, system="you are helpful")
        fcos = [x for x in result if x.get("type") == "function_call_output"]
        assert len(fcos) == 2
        assert fcos[0]["call_id"] == "call_00"
        assert fcos[0]["output"] == "result_0"

    def test_assistant_tool_uses_become_function_calls(self):
        oc = OpenAIClient(api_key="test-key")
        msgs = _make_messages(2)
        result = oc._build_input(msgs)
        fcs = [x for x in result if x.get("type") == "function_call"]
        assert len(fcs) == 2
        import json
        assert fcs[0]["call_id"] == "call_00"
        assert fcs[0]["name"] == "tool_0"
        assert json.loads(fcs[0]["arguments"]) == {"arg": "val_0"}

    def test_system_appended_first(self):
        oc = OpenAIClient(api_key="test-key")
        msgs = _make_messages(1)
        result = oc._build_input(msgs, system="be helpful")
        assert result[0]["role"] == "system"

    def test_tool_schema_conversion(self):
        oc = OpenAIClient(api_key="test-key")
        tools = [{"name": "read_file", "description": "read", "input_schema": {"type": "object"}}]
        converted = oc._convert_tools(tools)
        assert converted[0]["type"] == "function"
        assert converted[0]["name"] == "read_file"
        assert converted[0]["parameters"] == {"type": "object"}

    def test_parse_response_function_call(self):
        from unittest.mock import Mock
        item = Mock(type="function_call")
        item.call_id = "c1"
        item.name = "read_file"
        item.arguments = '{"path":"x"}'
        response = Mock()
        response.output = [item]
        response.usage = Mock(input_tokens=10, output_tokens=5)
        response.model = "gpt-4o-mini"
        result = OpenAIClient._parse_response(response)
        assert len(result.content) == 1
        assert isinstance(result.content[0], ToolUseBlock)
        assert result.content[0].name == "read_file"

    def test_parse_response_text(self):
        from unittest.mock import Mock
        response = Mock(
            output=[
                Mock(type="message", content=[Mock(text="hello world")]),
            ],
            usage=Mock(input_tokens=5, output_tokens=3),
            model="gpt-4o-mini",
        )
        result = OpenAIClient._parse_response(response)
        assert len(result.content) == 1
        assert result.content[0].text == "hello world"


# ═══════════════════════════════════════════════════
# OllamaClient
# ═══════════════════════════════════════════════════

class TestOllamaAdapter:
    """Ollama adapter: 内部消息 → Ollama chat API 格式。"""

    def test_tool_results_become_role_tool(self):
        ol = OllamaClient(api_key="")
        msgs = _make_messages(2)
        result = ol._build_messages(msgs, system="be helpful")
        tools = [x for x in result if x.get("role") == "tool"]
        assert len(tools) == 2
        assert tools[0]["content"] == "result_0"
        assert tools[1]["content"] == "result_1"

    def test_user_role_tool_results_also_work(self):
        ol = OllamaClient(api_key="")
        # 直接 role="user" + tool_result blocks（当前主格式）
        result = ol._build_messages(_make_messages(1))
        tools = [x for x in result if x.get("role") == "tool"]
        assert len(tools) == 1

    def test_system_message(self):
        ol = OllamaClient(api_key="")
        result = ol._build_messages([], system="you are helpful")
        assert result[0]["role"] == "system"

    def test_tool_schema_conversion(self):
        ol = OllamaClient(api_key="")
        tools = [{"name": "read_file", "description": "read", "input_schema": {"type": "object"}}]
        converted = ol._convert_tools(tools)
        assert converted[0]["type"] == "function"
        assert converted[0]["function"]["name"] == "read_file"

    def test_parse_response_with_tool_calls(self):
        from unittest.mock import Mock
        func = Mock()
        func.name = "read_file"
        func.arguments = '{"path":"x.py"}'
        tc = Mock()
        tc.function = func
        tc.id = "tc_001"
        msg = Mock()
        msg.content = "I'll help."
        msg.tool_calls = [tc]
        response = Mock()
        response.message = msg
        response.model = "qwen3:4b"
        result = OllamaClient._parse_response(response)
        assert len(result.content) == 2
        assert isinstance(result.content[1], ToolUseBlock)
        assert result.content[1].name == "read_file"


# ═══════════════════════════════════════════════════
# 跨 Provider 统一性
# ═══════════════════════════════════════════════════

class TestCrossProviderConsistency:
    """三个 Provider 对同一内部消息的转换都保留 tool_use_id。"""

    def test_all_preserve_tool_ids(self):
        msgs = _make_messages(2)

        ac = AnthropicClient(api_key="test-key")
        anthro = ac._convert_messages(msgs)
        anthro_ids = [b["tool_use_id"] for b in anthro[2]["content"]]
        assert anthro_ids == ["call_00", "call_01"]

        oc = OpenAIClient(api_key="test-key")
        oai = oc._build_input(msgs)
        oai_ids = [x["call_id"] for x in oai if x.get("type") == "function_call_output"]
        assert oai_ids == ["call_00", "call_01"]

        ol = OllamaClient(api_key="")
        oll = ol._build_messages(msgs)
        oll_has_content = all("result_" in x["content"] for x in oll if x.get("role") == "tool")
        assert oll_has_content

    def test_ids_never_empty(self):
        """关键检查：tool_use_id 不能为空（否则 LLM 无法匹配）。"""
        msgs = _make_messages(3)

        ac = AnthropicClient(api_key="test-key")
        for b in ac._convert_messages(msgs)[2]["content"]:
            assert b["tool_use_id"] != "", f"Anthropic: empty tool_use_id"

        oc = OpenAIClient(api_key="test-key")
        for x in oc._build_input(msgs):
            if x.get("type") == "function_call_output":
                assert x["call_id"] != "", f"OpenAI: empty call_id"

    def test_tool_call_id_matches_between_assistant_and_result(self):
        """assistant 的 tool_use id 和 tool_result 的 tool_use_id 必须匹配。"""
        msgs = _make_messages(2)

        ac = AnthropicClient(api_key="test-key")
        result = ac._convert_messages(msgs)
        ass_ids = [b["id"] for b in result[1]["content"] if isinstance(b, dict) and b.get("type") == "tool_use"]
        result_ids = [b["tool_use_id"] for b in result[2]["content"]]
        assert ass_ids == result_ids
