"""工具注册表 —— 动态注册、发现、按属性过滤。

与 v0.1 的关键变化：
  - get_read_only_tools() 返回的是同一实例的引用（不克隆）
  - filter_by_depth() 同上
  - 工具实例是无状态单例，多个 Agent 可以安全共享
  - 新增 list_by_risk() 和 get_names() 辅助方法
"""

import importlib
import inspect
import logging
from echo.tools.base import BaseTool

logger = logging.getLogger("echo.registry")


class ToolRegistry:
    """工具注册表。

    支持三种注册方式：
      1. register(ToolClass())           — 手动注册
      2. discover("echo.tools.builtin")  — 自动扫描模块中的 BaseTool 子类
      3. register_mcp(server, tools)     — MCP 工具动态挂载（预留）
    """

    def __init__(self):
        self._tools: dict[str, BaseTool] = {}
        self._mcp_tools: dict[str, dict] = {}  # {prefixed_name: schema}

    # ── 注册 / 注销 ───────────────────────────────

    def register(self, tool: BaseTool) -> "ToolRegistry":
        """注册一个工具实例。同名工具会报错。

        Returns:
            self，支持链式调用。
        """
        if not tool.name:
            raise ValueError("工具必须有 name")
        if tool.name in self._tools:
            raise ValueError(f"工具 '{tool.name}' 已注册")
        self._tools[tool.name] = tool
        tool._registry = self
        return self

    def unregister(self, name: str) -> None:
        """注销工具。"""
        tool = self._tools.pop(name, None)
        if tool is not None:
            tool._registry = None

    def discover(self, module_path: str) -> "ToolRegistry":
        """自动发现模块中所有 BaseTool 子类并注册。

        跳过抽象子类（无法实例化的）和已注册的同名工具。

        Returns:
            self，支持链式调用。
        """
        module = importlib.import_module(module_path)
        for _name, obj in inspect.getmembers(module, inspect.isclass):
            if not issubclass(obj, BaseTool) or obj is BaseTool:
                continue
            if inspect.isabstract(obj):
                continue
            try:
                self.register(obj())
            except ValueError:
                # 同名工具已注册 → 跳过
                pass
            except Exception:
                logger.warning(f"无法实例化工具 {obj.__name__}", exc_info=True)
        return self

    # ── 查询 ──────────────────────────────────────

    def get(self, name: str) -> BaseTool | None:
        """按名获取工具。不存在返回 None。"""
        return self._tools.get(name)

    def get_all(self) -> list[BaseTool]:
        """获取所有已注册工具。"""
        return list(self._tools.values())

    def get_names(self) -> list[str]:
        """获取所有已注册工具的名称。"""
        return list(self._tools.keys())

    def list_schemas(self) -> list[dict]:
        """导出所有工具的 Function Call 格式 schema（给 LLM）。

        包含内置工具和 MCP 工具的 schema。
        """
        schemas = [t.to_schema() for t in self._tools.values()]
        for mcp_schema in self._mcp_tools.values():
            schemas.append(mcp_schema)
        return schemas

    # ── 按属性过滤（返回引用，不克隆）──────────────

    def get_read_only_tools(self) -> list[BaseTool]:
        """获取所有只读工具（返回引用，不创建新实例）。

        用于 delegate / sub-agent 的工具白名单。
        """
        return [t for t in self._tools.values() if t.is_read_only]

    def filter_by_depth(self, depth: int, max_depth: int) -> list[BaseTool]:
        """按代理深度过滤工具列表。

        达到最大深度时排除 delegate 工具，防止递归委托。
        返回引用，不创建新实例。
        """
        if depth >= max_depth:
            return [t for t in self._tools.values() if t.name != "delegate"]
        return list(self._tools.values())

    def list_by_risk(self, risk_level: str) -> list[BaseTool]:
        """按风险级别获取工具列表。"""
        return [t for t in self._tools.values() if t.risk_level == risk_level]

    # ── 大小 ──────────────────────────────────────

    def __len__(self) -> int:
        return len(self._tools)

    def __contains__(self, name: str) -> bool:
        return name in self._tools
