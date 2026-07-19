"""工具系统的核心抽象层 —— ToolResult + ToolContext + BaseTool。

BaseTool 是 Echo 架构中连接「调度层（AgentLoop）」和「基建层（Sandbox / Shell /
Memory / Hooks）」的唯一抽象合约。新增工具只需继承 BaseTool 并实现 execute()，
无需修改框架任何代码。

设计原则：
  - ToolResult 结构化返回 — 不只是 text，还带文件变更、记忆笔记、执行状态
  - ToolContext 最小化注入 — 只暴露 4 个下层能力 + trace，不暴露 AgentLoop 等上层
  - 权限不在本层 — 统一由 HookManager("pre_tool_use") 处理，BaseTool 不重复
  - 工具无状态单例 — 所有环境信息通过 ToolContext 注入，多 worktree 场景不冲突
"""

import threading
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable, TYPE_CHECKING

from pydantic import BaseModel

# 模块级导入（非 lazy import，避免热路径重复 import）
from echo.security.redaction import redact_text

if TYPE_CHECKING:
    from echo.security.sandbox import Sandbox
    from echo.security.env_filter import ShellExecutor
    from echo.memory.base import MemoryManager
    from echo.persistence.trace import TraceEmitter


# ═══════════════════════════════════════════════════════
# ToolResult — 工具执行的统一返回结构
# ═══════════════════════════════════════════════════════

@dataclass
class ToolResult:
    """工具执行的统一返回结构。

    替代之前的裸 str 返回值，让工具可以向框架报告：
      - 给 LLM 看的文本（output）
      - 是否执行失败（error）
      - 结构化元数据（meta）

    框架在收到 ToolResult 后自动处理：
      - output → 写入 messages[] 作为 tool_result content
      - error 非空 → 标记工具执行状态
      - meta["files_touched"] → checkpoint 文件新鲜度追踪
      - meta["memory_notes"] → 自动调用 memory.add()

    Attributes:
        output: 给 LLM 看的文本输出（纯文本，会被截断到 max_output_chars）。
        error: 非空表示执行失败或部分失败。格式："<简短错误类型>: <描述>"。
        meta: 结构化元数据。约定键见下方 META_KEYS。
    """

    output: str = ""
    error: str | None = None
    meta: dict = field(default_factory=dict)

    # ── meta 约定键常量 ────────────────────────────
    # 使用这些常量而非裸字符串，避免拼写错误

    KEY_FILES_TOUCHED = "files_touched"        # list[str]: 受影响的文件绝对路径
    KEY_WORKSPACE_CHANGES = "workspace_changes" # list[str]: 工作区变更摘要
    KEY_MEMORY_NOTES = "memory_notes"           # list[str]: 应记入工作记忆的内容
    KEY_TOKENS_USED = "tokens_used"             # int: 工具自身消耗的 token
    KEY_PARTIAL_SUCCESS = "partial_success"     # bool: 部分成功（如命令失败但文件已修改）

    # ── 属性 ──────────────────────────────────────

    @property
    def success(self) -> bool:
        """是否成功执行（无错误）。"""
        return self.error is None

    @property
    def is_partial(self) -> bool:
        """是否部分成功。"""
        return self.meta.get(self.KEY_PARTIAL_SUCCESS, False)

    @property
    def files_touched(self) -> list[str]:
        """受影响的文件路径列表。"""
        return self.meta.get(self.KEY_FILES_TOUCHED, [])

    @property
    def memory_notes(self) -> list[str]:
        """应记入记忆的内容列表。"""
        return self.meta.get(self.KEY_MEMORY_NOTES, [])

    # ── 工厂方法 ──────────────────────────────────

    @classmethod
    def ok(cls, output: str, **meta) -> "ToolResult":
        """创建成功结果。"""
        return cls(output=output, meta=dict(meta))

    @classmethod
    def fail(cls, error: str, output: str = "", **meta) -> "ToolResult":
        """创建失败结果。"""
        return cls(output=output, error=error, meta=dict(meta))

    @classmethod
    def partial(cls, output: str, error: str, **meta) -> "ToolResult":
        """创建部分成功结果。"""
        meta[cls.KEY_PARTIAL_SUCCESS] = True
        return cls(output=output, error=error, meta=dict(meta))


# ═══════════════════════════════════════════════════════
# ToolContext — 工具执行时的最小上下文
# ═══════════════════════════════════════════════════════

@dataclass
class ToolContext:
    """工具执行时的最小上下文。

    工具的 execute() 只能通过这个对象访问外部世界。
    不直接 import os/subprocess/memory —— 沙箱才能生效。

    工具实例无状态，所有环境信息通过此对象传入，
    天然适配多 worktree、多 Agent 深度限制等场景。

    不暴露到 ToolContext 的内容（设计决策）：
      ❌ AgentLoop    — 工具不应该操控主循环
      ❌ MessageBus   — 工具不应该直接给队友发消息
      ❌ SessionStore — 持久化由框架层负责
      ✅ sandbox      — 安全路径解析 + 文件快照
      ✅ shell        — 环境隔离的 Shell 执行
      ✅ memory       — 工作记忆读写
      ✅ trace        — 工具自己的 trace 事件（可选）
    """

    workspace_root: str = ""
    sandbox: "Sandbox | None" = None       # 路径沙箱
    shell: "ShellExecutor | None" = None   # Shell 执行器
    memory: "MemoryManager | None" = None     # 工作记忆管理器
    trace: "TraceEmitter | None" = None    # Trace 事件发射器（可选）
    task_state: Any = None                  # TaskState 引用（todo_write 等工具需写回）
    llm: Any = None                         # LLM 客户端（delegate 工具创建子 Agent 用）
    tool_registry: Any = None               # ToolRegistry 引用（delegate 筛选只读工具）
    run_id: str = ""                         # 当前 run_id（trace 事件用，delegate 传递）
    trace_logger: Any = None                 # RunStore 引用（子 Agent 工具调用 trace 用）
    depth: int = 0                           # 代理深度（0=主Agent）
    max_depth: int = 1                      # 最大嵌套深度
    # 审批回调（由框架注入，工具不直接调用——权限走 Hook 层）
    approval: Callable[[str, str], bool] = field(default=lambda _n, _d: True)

    def resolve_path(self, raw: str):
        """将用户输入路径解析为安全的绝对 Path。

        委托给 Sandbox.resolve()，自动完成：
          1. 相对路径拼接 workspace_root
          2. Path.resolve() 规范化（消除 ..、跟踪符号链接）
          3. os.path.commonpath() 校验（防逃逸）

        Raises:
            RuntimeError: sandbox 未注入（框架 bug）。
            PathEscapedError: 路径试图逃逸工作区。
        """
        if self.sandbox is None:
            raise RuntimeError("ToolContext.sandbox 未注入 — 无法解析路径。请检查 AgentLoop 是否正确构建 ctx。")
        return self.sandbox.resolve(raw)


# ═══════════════════════════════════════════════════════
# BaseTool — 所有工具的基类
# ═══════════════════════════════════════════════════════

class BaseTool(ABC):
    """所有工具的基类。

    子类仅需定义 3 个类属性 + 实现 1 个方法：
      1. name: str           — 工具唯一名称（snake_case，LLM 通信用）
      2. description: str    — 工具功能描述（给 LLM 看）
      3. risk_level: str     — "safe" | "warn" | "danger"
      4. execute(ctx, params) → ToolResult — 核心业务逻辑

    框架自动处理：
      JSON Schema 生成     → to_schema() 从 pydantic 模型自动生成
      pydantic 参数校验    → pre_validate() 在 execute 前自动调用
      后置脱敏 + 截断      → post_process() 自动处理 ToolResult.output
      异常兜底             → handle_error() 格式化错误不暴露栈信息

    权限检查不在此处 —— 由 HookManager("pre_tool_use") 统一处理。
    """

    # ── 子类必须定义 ──────────────────────────────
    name: str = ""                           # 工具唯一名称（与 LLM tool_use block 匹配）
    description: str = ""                    # 工具描述（给 LLM 看）
    risk_level: str = "safe"                 # "safe" | "warn" | "danger"

    # ── 子类可选覆写 ──────────────────────────────
    params_model: type[BaseModel] | None = None  # pydantic 参数模型
    max_timeout: int = 30                       # 最大执行秒数
    max_output_chars: int = 50_000              # 输出截断阈值
    is_read_only: bool = True                   # delegate 模式下是否可用

    # ── 自动管理（不要手写）───────────────────────
    _registry: Any = None  # ToolRegistry 反向引用

    # ═══════════════════════════════════════════════
    # Schema 生成
    # ═══════════════════════════════════════════════

    def to_schema(self) -> dict:
        """生成 OpenAI/Anthropic 兼容的 tool definition。

        从 pydantic BaseModel 自动提取 JSON Schema。
        无 params_model 时返回空 properties。

        Returns:
            {"name": ..., "description": ..., "input_schema": {...}}
        """
        if self.params_model is None:
            return {
                "name": self.name,
                "description": self.description,
                "input_schema": {"type": "object", "properties": {}, "required": []},
            }
        schema = self.params_model.model_json_schema()
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": {
                "type": "object",
                "properties": schema.get("properties", {}),
                "required": schema.get("required", []),
            },
        }

    # ═══════════════════════════════════════════════
    # 生命周期方法（按调用顺序排列）
    # ═══════════════════════════════════════════════

    def pre_validate(self, params: dict) -> tuple[bool, str]:
        """参数前置校验 —— 基于 pydantic 模型自动校验。

        execute() 被调用前自动执行。子类可覆写添加自定义校验，
        但应调用 super().pre_validate(params) 保留基础校验。

        注意：此处不做权限校验，权限统一在 Hook 层处理。

        Returns:
            (是否通过, 错误描述)
        """
        if self.params_model is None:
            return True, ""
        try:
            self.params_model(**params)
            return True, ""
        except Exception as e:
            return False, f"参数校验失败: {e}"

    def pre_hook(self, ctx: ToolContext, params: dict) -> tuple[bool, str]:
        """工具自身的前置业务校验（可选的，不含权限检查）。

        在 pre_validate 之后、execute 之前调用。
        子类覆写做自定义检查：
          - 文件是否存在
          - 路径格式是否正确
          - 深度限制是否达到

        Returns:
            (是否通过, 拒绝原因)
        """
        return True, ""

    @abstractmethod
    def execute(self, ctx: ToolContext, params: dict) -> ToolResult:
        """核心执行逻辑 —— 子类必须实现的唯一方法。

        只关注业务逻辑，无需处理：
          - 参数校验（pre_validate 已做）
          - 权限检查（Hook 层已做）
          - 输出脱敏/截断（post_process 已做）
          - 异常格式化（handle_error 已做）

        Args:
            ctx: 工具执行上下文（沙箱、Shell、记忆、Trace 等）。
            params: 已通过 pre_validate 校验的参数字典。

        Returns:
            ToolResult: output 给 LLM 看, error 表示失败, meta 给框架消费。
        """
        ...

    def post_process(self, result: ToolResult) -> ToolResult:
        """后置处理 —— 脱敏 + 截断 ToolResult.output。

        子类可覆写添加自定义后处理，但应调用 super() 保留基础处理。

        Args:
            result: execute() 返回的原始 ToolResult。

        Returns:
            处理后（脱敏+截断）的 ToolResult（可能是原对象原位修改）。
        """
        result.output = redact_text(str(result.output))
        if len(result.output) > self.max_output_chars:
            result.output = result.output[:self.max_output_chars] + "\n... [truncated]"
        return result

    def handle_error(self, error: Exception) -> ToolResult:
        """异常统一处理 —— 格式化错误信息，不暴露内部栈。

        Args:
            error: execute() 抛出的异常。

        Returns:
            包含格式化错误信息的 ToolResult。
        """
        return ToolResult.fail(f"Tool error: {type(error).__name__}: {str(error)}")

    # ═══════════════════════════════════════════════
    # 统一入口（ToolExecutor 只调此方法）
    # ═══════════════════════════════════════════════

    def run(self, ctx: ToolContext, params: dict) -> ToolResult:
        """工具统一执行入口。

        封装完整生命周期：
          1. pre_validate(params)           — 参数校验
          2. pre_hook(ctx, params)          — 业务前置检查
          3. execute(ctx, params)           — 核心逻辑
          4. post_process(result)           — 脱敏 + 截断
          (异常) handle_error(error)         — 异常兜底

        ToolExecutor 只调用此方法，不直接调 execute()。
        返回的 ToolResult 由框架层消费（output 给 LLM，meta 用于记忆/检查点）。

        Args:
            ctx: 工具执行上下文。
            params: 原始参数字典（将被 pydantic 校验）。

        Returns:
            ToolResult 实例。
        """
        # 1. 参数校验
        valid, err_msg = self.pre_validate(params)
        if not valid:
            return ToolResult.fail(err_msg)

        # 2. 业务前置检查
        ok, msg = self.pre_hook(ctx, params)
        if not ok:
            return ToolResult.fail(f"Pre-check failed: {msg}")

        # 3. 核心执行 + 超时约束 + 异常兜底
        result_container: list[ToolResult] = []
        error_container: list[Exception] = []

        def _run():
            try:
                r = self.execute(ctx, params)
                if not isinstance(r, ToolResult):
                    r = ToolResult.ok(str(r))
                result_container.append(r)
            except Exception as e:
                error_container.append(e)

        t = threading.Thread(target=_run, daemon=True)
        t.start()
        t.join(timeout=self.max_timeout)

        if t.is_alive():
            # 超时 —— 线程无法强杀（Python 限制），但标记为超时失败
            return self.handle_error(
                TimeoutError(f"工具 '{self.name}' 执行超时 ({self.max_timeout}s)")
            )

        if error_container:
            return self.handle_error(error_container[0])

        result = result_container[0]

        # 4. 后置处理
        return self.post_process(result)
