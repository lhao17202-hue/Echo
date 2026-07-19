"""FakeLLMClient —— 确定性测试用 LLM 客户端。

Deterministic test LLM client for local tests and benchmarks.

不调用任何真实模型，消费预先编排的回复序列。每个 chat() 调用
从序列中弹出一条预设回复，解析为 ModelResponse 返回。

支持两种输出格式：
  1. XML 属性格式:  '<tool name="read_file" path="x.py" start="1" />'
  2. JSON 块格式:   '<tool>{"name":"read_file","args":{"path":"x.py"}}</tool>'
  3. 纯文本格式:    任意不含 <tool> 的字符串 → 作为最终回复

用于单元测试和标杆评估，也可支持后续扩展为 socket 回放模式。

使用方式:
    fake = FakeLLMClient(outputs=[
        '<tool name="read_file" path="README.md" />',
        "这个项目包含以下文件: ...",
    ])

    # 第一次调用 → 模型返回 tool_use
    resp1 = fake.chat(messages, tools, system)
    # resp1.content[0] → ToolUseBlock(name="read_file", input={"path": "README.md"})

    # 第二次调用 → 模型返回文本回复（最终答案前可能有多个工具轮次）
    resp2 = fake.chat(messages, tools, system)
    # resp2.content[0] → TextBlock(text="这个项目包含以下文件: ...")

    # 第三次调用（如果序列耗尽）→ 返回一个空的最终回复
    resp3 = fake.chat(messages, tools, system)
    # resp3.content[0] → TextBlock(text="")

    print(f"共调用了 {fake.call_count} 次模型")
    print(f"收集了 {len(fake.prompts)} 条 prompt")
"""

import json
import re
import logging
from dataclasses import dataclass, field
from typing import Iterator, Any
from echo.providers.base import (
    BaseLLMClient, ModelResponse, TokenUsage,
    TextBlock, ToolUseBlock, StreamEvent,
)

logger = logging.getLogger("echo.fake")


# ── 输出格式常量 ─────────────────────────────────────

# XML 属性工具调用: <tool name="X" key="value" ... />
_TOOL_XML_ATTR = re.compile(
    r'<tool\s+name="([^"]+)"\s*(.*?)\s*/?>',
    re.DOTALL,
)

# XML 子标签工具调用: <tool name="X"...><child>...</child></tool>
# 支持 XML 子标签工具调用格式: <tool name="write_file" path="x"><content>multi\nline</content></tool>
_TOOL_XML_CHILDREN = re.compile(
    r'<tool\s+name="([^"]+)"\s*(.*?)>(.*?)</tool>',
    re.DOTALL,
)

# JSON 块工具调用: <tool>{"name":"X","args":{...}}</tool>
_TOOL_JSON_BLOCK = re.compile(
    r'<tool>\s*(\{.*?\})\s*</tool>',
    re.DOTALL,
)

# 最终回复标记: <final>text</final> 或纯文本
_FINAL_TAG = re.compile(r'<final>(.*?)</final>', re.DOTALL)


# ── 辅助函数 ──────────────────────────────────────────

def _parse_xml_attrs(attrs_str: str) -> dict:
    """解析 XML 属性字符串为 dict。

    例如: 'path="x.py" start="1" end="200"'
    → {"path": "x.py", "start": "1", "end": "200"}
    """
    result = {}
    for match in re.finditer(r'(\w+)="([^"]*)"', attrs_str):
        result[match.group(1)] = match.group(2)
    return result


def _parse_output(raw: str) -> list[TextBlock | ToolUseBlock]:
    """解析单条预设输出字符串，返回 ContentBlock 列表。

    解析优先级:
      1. JSON 块格式: <tool>{"name":"X","args":{...}}</tool>
      2. XML 属性格式: <tool name="X" attr="val" />
      3. <final>text</final> → 纯文本
      4. 普通文本 → 纯文本
    """
    content: list[TextBlock | ToolUseBlock] = []

    # 尝试 JSON 块格式
    json_blocks = _TOOL_JSON_BLOCK.findall(raw)
    if json_blocks:
        remaining = raw
        for block_str in json_blocks:
            try:
                data = json.loads(block_str)
                name = data.get("name", "unknown")
                args = data.get("args", data.get("input", {}))
                content.append(ToolUseBlock(
                    id=f"fake_{name}_{len(content)}",
                    name=name,
                    input=args,
                ))
                # 移除已解析的块
                remaining = remaining.replace(
                    f'<tool>{block_str}</tool>', '', 1
                )
            except json.JSONDecodeError:
                pass
        # 剩余文本
        remaining = remaining.strip()
        if remaining:
            content.append(TextBlock(text=remaining))
        return content

    # 2. XML 子标签格式
    xml_child_matches = _TOOL_XML_CHILDREN.findall(raw)
    if xml_child_matches:
        remaining = raw
        for name, attrs, body in xml_child_matches:
            input_dict = _parse_xml_attrs(attrs)
            for child_match in re.finditer(
                r'<(\w+)>(.*?)</\1>', body, re.DOTALL
            ):
                input_dict[child_match.group(1)] = child_match.group(2).strip()
            content.append(ToolUseBlock(
                id=f"fake_{name}_{len(content)}",
                name=name,
                input=input_dict,
            ))
            tag_text = f'<tool name="{name}"{attrs}>{body}</tool>'
            remaining = remaining.replace(tag_text, '', 1)
        remaining = remaining.strip()
        if remaining:
            content.append(TextBlock(text=remaining))
        return content

    # 3. XML 属性格式
    xml_matches = _TOOL_XML_ATTR.findall(raw)
    if xml_matches:
        remaining = raw
        for name, attrs in xml_matches:
            input_dict = _parse_xml_attrs(attrs)
            content.append(ToolUseBlock(
                id=f"fake_{name}_{len(content)}",
                name=name,
                input=input_dict,
            ))
            # 移除已解析的块
            remaining = remaining.replace(
                f'<tool name="{name}"{attrs}/>', '', 1
            ).replace(
                f'<tool name="{name}" {attrs}/>', '', 1
            )
        remaining = remaining.strip()
        if remaining:
            content.append(TextBlock(text=remaining))
        return content

    # 纯文本（可能包含 <final> 标记）
    match = _FINAL_TAG.search(raw)
    if match:
        content.append(TextBlock(text=match.group(1).strip()))
        return content

    content.append(TextBlock(text=raw))
    return content


# ═══════════════════════════════════════════════════════
# FakeLLMClient
# ═══════════════════════════════════════════════════════

class FakeLLMClient(BaseLLMClient):
    """确定性测试用 LLM 客户端。

    不调用任何真实模型，消费预先编排的回复序列。
    每条预设回复在首次 chat() 调用时自动解析为 ModelResponse。

    FakeLLMClient returns ModelResponse objects compatible with BaseLLMClient.
    """

    def __init__(self, outputs: list[str] | None = None):
        """初始化假客户端。

        Args:
            outputs: 预设的模型回复序列（按调用顺序逐个消费）。
                     每个元素是一条完整回复 —— 可以是纯文本、XML 工具调用、JSON 工具调用。
                     不传或传空列表 → 每次 chat() 返回空文本回复。
        """
        # 不传 api_key，FakeClient 不需要
        super().__init__(
            model="FakeLLMClient",
            api_key="fake-key-not-used",
        )
        self._outputs: list[str] = list(outputs or [])
        self._index: int = 0

        # ── 可观测字段（测试断言用）─────────────────
        self.call_count: int = 0            # chat() 被调用次数
        self.prompts: list[dict] = []       # 每次调用的参数记录
        self.last_response: ModelResponse | None = None
        self.last_system: str | None = None  # 最近一次 chat() 收到的 system prompt 文本
        self._pending_outputs: list[str] = list(self._outputs)

    # ── BaseLLMClient 接口 ─────────────────────────

    def chat(
        self,
        messages: list[dict] | None = None,
        tools: list[dict] | None = None,
        system: str = "",
        max_tokens: int = 8000,
        temperature: float = 0.0,
    ) -> ModelResponse:
        """消费下一条预设回复，解析为 ModelResponse。

        每次调用从预设序列中弹出一条。序列耗尽时返回空的最终回复。

        Args:
            messages: 对话历史（记录但不使用）。
            tools: 工具 schema（记录但不使用）。
            system: system prompt（记录但不使用）。
            max_tokens: 忽略。
            temperature: 忽略。
        """
        self.call_count += 1
        self.last_system = system  # 记录 system prompt（测试用）
        self.prompts.append({
            "message_count": len(messages or []),
            "tool_count": len(tools or []),
            "system_len": len(system),
        })

        # 序列耗尽 → 返回空回复（模拟模型停止发言）
        if not self._pending_outputs:
            resp = ModelResponse(
                content=[TextBlock(text="")],
                stop_reason="end_turn",
                model=self.model,
            )
            self.last_response = resp
            return resp

        raw = self._pending_outputs.pop(0)
        self._index += 1

        content = _parse_output(raw)
        has_tools = any(isinstance(b, ToolUseBlock) for b in content)

        resp = ModelResponse(
            content=content,
            stop_reason="tool_use" if has_tools else "end_turn",
            usage=TokenUsage(),
            model=self.model,
        )
        self.last_response = resp
        return resp

    def chat_stream(
        self,
        messages: list[dict] | None = None,
        tools: list[dict] | None = None,
        system: str = "",
        max_tokens: int = 8000,
        temperature: float = 0.0,
    ) -> Iterator[StreamEvent]:
        """流式版本 —— 先发射文本 delta，再 done。

        先调用非流式 chat() 拿到完整响应，再拆成 stream events 返回。
        """
        resp = self.chat(messages, tools, system, max_tokens, temperature)

        for block in resp.content:
            if isinstance(block, TextBlock):
                yield StreamEvent(type="text_delta", text=block.text)
            elif isinstance(block, ToolUseBlock):
                yield StreamEvent(
                    type="tool_use_start",
                    tool_id=block.id,
                    tool_name=block.name,
                )
                yield StreamEvent(
                    type="tool_use_delta",
                    tool_input_json=json.dumps(block.input),
                )
                yield StreamEvent(type="tool_use_end")

        yield StreamEvent(type="done")

    # ── 序列管理 ──────────────────────────────────

    def reset(self) -> None:
        """重置到序列开头（所有预设回复重新可用）。"""
        self._pending_outputs = list(self._outputs)
        self._index = 0
        self.call_count = 0
        self.prompts = []
        self.last_response = None

    def feed(self, *outputs: str) -> None:
        """动态追加预设回复到序列末尾。

        用于运行时补充新的预期回复（如 Agent 需要多轮对话）。
        """
        self._outputs.extend(outputs)
        self._pending_outputs.extend(outputs)

    @property
    def remaining(self) -> int:
        """剩余未消费的预设回复数。"""
        return len(self._pending_outputs)

    @property
    def exhausted(self) -> bool:
        """是否所有预设回复已消费完毕。"""
        return self.remaining == 0

    def __repr__(self) -> str:
        return (
            f"FakeLLMClient(outputs={len(self._outputs)}, "
            f"remaining={self.remaining}, called={self.call_count})"
        )
