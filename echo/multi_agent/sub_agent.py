"""One-shot, isolated, read-only sub-agent for delegate tool."""

from __future__ import annotations
from echo.tools.base import ToolContext
from echo.providers.base import BaseLLMClient, TextBlock, ToolUseBlock

MAX_SUMMARY_CHARS = 8000


class SubAgent:
    """一次性隔离的只读子 Agent。

    - 独立 messages[] 列表（不污染主 Agent）
    - 只能用只读工具（不允许 write/patch/shell/delegate/compact）
    - 同步阻塞主 Agent
    - 最多 N 轮后返回截断摘要
    - 子 Agent 失败不影响主 Agent 继续运行
    """

    def __init__(self, llm: BaseLLMClient, tools: list,
                 ctx: ToolContext):
        self.llm = llm
        self.tools = tools   # list of BaseTool (已筛选为只读 + 无 delegate)
        self.ctx = ctx
        self.max_steps = 15

    def run(self, task: str, max_steps: int = 15) -> str:
        # 类型安全：强制 int + clamp，防止调用方传入字符串或超出范围的值
        try:
            self.max_steps = int(max_steps)
        except (ValueError, TypeError):
            self.max_steps = 15
        self.max_steps = max(1, min(self.max_steps, 30))

        system = (
            f"You are a coding sub-agent. "
            f"Complete the task and return a CONCISE final summary. "
            f"Do NOT spawn more agents. "
            f"Report key findings, file paths, and relevant code snippets."
        )

        messages = [{"role": "user", "content": [TextBlock(text=task)]}]
        steps = 0
        schemas = [t.to_schema() for t in self.tools]

        while steps < self.max_steps:
            response = self.llm.chat(
                messages=messages,
                tools=schemas,
                system=system,
                max_tokens=8000,
            )
            messages.append({"role": "assistant", "content": response.content})

            tool_blocks = [b for b in response.content
                          if isinstance(b, ToolUseBlock)]
            if not tool_blocks:
                texts = [b.text for b in response.content
                        if isinstance(b, TextBlock)]
                summary = " ".join(texts) if texts else "Sub-agent finished."
                return summary[:MAX_SUMMARY_CHARS]

            results = []
            for block in tool_blocks:
                tool = self._find_tool(block.name)
                success = True
                if tool is None:
                    output = f"Error: unknown tool '{block.name}'"
                    success = False
                else:
                    try:
                        r = tool.run(self.ctx, block.input)
                        output = r.output if hasattr(r, "output") else str(r)
                        if hasattr(r, "error") and r.error:
                            output = f"Error: {r.error}\n{output}"
                            success = False
                    except Exception as e:
                        output = f"Sub-agent tool error: {e}"
                        success = False

                # trace: 只记元数据摘要，不泄露文件内容
                if self.ctx.trace_logger:
                    output_len = len(str(output))
                    self.ctx.trace_logger.log(
                        "sub_tool_executed",
                        run_id=self.ctx.run_id,
                        depth=self.ctx.depth,
                        tool=block.name,
                        success=success,
                        input_summary=str(block.input)[:200],
                        output_chars=output_len,
                        output_truncated=output_len > 50_000,
                        error_preview=(str(output)[:200]
                                       if not success else ""),
                    )

                results.append({
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": str(output)[:50_000],
                })
                steps += 1

            messages.append({"role": "user", "content": results})

        return "Sub-agent reached step limit."

    def _find_tool(self, name: str):
        for t in self.tools:
            if t.name == name:
                return t
        return None
