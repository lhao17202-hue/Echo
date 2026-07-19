"""工具执行器 —— 薄封装，只做"取工具 → 调 run → 返回结果"。

与 v0.1 的关键变化：
  - BackgroundTaskManager 已移除（职责移到 AgentLoop / scheduler）
  - 重复调用守卫已移除（职责移到 AgentLoop）
  - execute() 直接返回 ToolResult，不再返回裸 str
  - 不再持有 HookManager（权限由 AgentLoop 在调用前通过 Hook 处理）
"""

from echo.tools.base import BaseTool, ToolContext, ToolResult
from echo.tools.registry import ToolRegistry


class ToolExecutor:
    """工具执行器 —— AgentLoop 和 BaseTool 之间的薄胶水层。

    执行链路：
      AgentLoop
        → hooks.trigger("pre_tool_use")  ← 权限唯一入口
        → executor.execute(name, input, ctx)
          → tool.run(ctx, params)        ← 生命周期：validate → execute → post_process
        → hooks.trigger("post_tool_use")
    """

    def __init__(self, registry: ToolRegistry):
        """初始化执行器。

        Args:
            registry: ToolRegistry 实例（包含所有已注册工具）。
        """
        self.registry = registry

    def execute(self, tool_name: str, params: dict,
                ctx: ToolContext) -> ToolResult:
        """同步执行工具。

        1. 从注册表查找工具
        2. 构造 ToolContext（如果没有外部 ctx 也会使用内置默认值）
        3. 调用 tool.run(ctx, params)
        4. 返回 ToolResult

        Args:
            tool_name: 工具名称（与 LLM tool_use block 的 name 匹配）。
            params: 工具参数 dict（已从 tool_use block 解析）。
            ctx: 工具执行上下文。

        Returns:
            ToolResult 实例（output 给 LLM，meta 给框架）。
        """
        tool = self.registry.get(tool_name)
        if tool is None:
            return ToolResult.fail(f"未知工具: '{tool_name}'")

        return tool.run(ctx, params)
