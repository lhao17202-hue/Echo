"""Hook 系统 —— BaseHook 抽象 + HookManager + HookEvent 事件类型常量。

Hook 是 Echo 的横切关注点拦截框架。权限、日志、审计、告警全部以 Hook
形式挂载到 AgentLoop 的关键节点上，不侵入主循环代码。

设计原则：
  - 一个 Hook = 一个事件类型 + 一个 handle 方法
  - 短路机制：任意 Hook 返回 str 则终止后续 Hook（str = 拦截原因）
  - priority 越小越先执行（0 = 最高优先级）
  - Hook 不应执行长耗时操作、不应抛异常
"""

from abc import ABC, abstractmethod
from collections import defaultdict
from enum import Enum


# ═══════════════════════════════════════════════════════
# HookEvent — 事件类型常量（替代魔法字符串）
# ═══════════════════════════════════════════════════════

class HookEvent(str, Enum):
    """Hook 事件类型 —— AgentLoop 中的 4 个拦截点。

    每个事件携带特定的 **kwargs 参数，由 AgentLoop 调用 trigger 时传入。
    编写 Hook 时，handle(**kwargs) 可按需从中提取参数。

    Events:
        USER_PROMPT    — 用户输入被提交后
            kwargs: request: str（用户输入文本）
            用途: 输入预处理、注入额外上下文、记录用户消息
            当前状态: 预留扩展点，暂无内置 Hook 监听

        PRE_TOOL_USE   — 每个工具执行前（权限门）
            kwargs: tool: ToolUseBlock（LLM 返回的 tool_use 块）
                    tool_input: dict（已解析的工具参数）
            返回值: None=放行, str=拦截（拦截原因成为 tool_result）
            内置 Hook: PermissionHook(pri=0), LogHook(pri=100)

        POST_TOOL_USE  — 每个工具执行后
            kwargs: tool: ToolUseBlock
                    result: ToolResult（工具返回结果）
            返回值: None=继续, str=拦截（极少使用，通常只做日志）
            内置 Hook: PostLogHook(pri=100), LargeOutputHook(pri=200)

        RUN_STOP       — Agent 停止时（正常结束或异常终止）
            kwargs: state: TaskState（终止时的状态快照）
            用途: 统计输出、会话摘要、资源清理
            内置 Hook: StatsHook(pri=100)
    """

    USER_PROMPT = "user_prompt"
    PRE_TOOL_USE = "pre_tool_use"
    POST_TOOL_USE = "post_tool_use"
    RUN_STOP = "run_stop"


# ═══════════════════════════════════════════════════════
# BaseHook — 所有 Hook 的抽象基类
# ═══════════════════════════════════════════════════════

class BaseHook(ABC):
    """Hook 基类。

    子类只需实现 event 属性（事件类型）和 handle 方法（处理逻辑）。

    使用方式：
      class MyHook(BaseHook):
          event = HookEvent.PRE_TOOL_USE
          def handle(self, **kwargs) -> str | None:
              tool = kwargs.get("tool")
              if is_bad(tool):
                  return "不允许使用此工具"
              return None  # 放行

    注意:
      - handle() 返回 None = 放行，返回 str = 拦截
      - handle() 不应抛异常（异常会中断整个 Hook 链）
      - handle() 不应执行长耗时操作
    """

    @property
    @abstractmethod
    def event(self) -> str:
        """绑定的事件类型。使用 HookEvent 枚举值。"""
        ...

    @abstractmethod
    def handle(self, **kwargs) -> str | None:
        """钩子处理逻辑。

        Args:
            **kwargs: 事件携带的参数（见 HookEvent 文档）。

        Returns:
            None  → 放行，继续执行后续 Hook。
            str   → 拦截，返回值作为拦截原因（中止此事件的所有后续 Hook）。
        """
        ...


# ═══════════════════════════════════════════════════════
# HookManager — 注册、触发、注销
# ═══════════════════════════════════════════════════════

class HookManager:
    """Hook 管理器 —— 全局单例，在 AgentLoop 各节点触发对应事件。

    使用模式：
      manager = HookManager()
      manager.register(PermissionHook(), priority=0)   # 权限最先执行
      manager.register(LogHook(), priority=100)        # 日志在权限之后

      # 在 AgentLoop 中：
      result = manager.trigger(HookEvent.PRE_TOOL_USE, tool=block, tool_input=block.input)
      if result:
          # 被拦截，result 是拦截原因
          ...
    """

    def __init__(self):
        self._hooks: dict[str, list[tuple[int, BaseHook]]] = defaultdict(list)

    # ── 注册 / 注销 ───────────────────────────────

    def register(self, hook: BaseHook, priority: int = 100) -> "HookManager":
        """注册一个 Hook。

        priority 越小越先执行：
          0   = 权限类 Hook（最先执行，可提前拦截）
          100 = 日志类 Hook
          200 = 告警 / 统计类 Hook

        Args:
            hook: BaseHook 实例。
            priority: 执行优先级（默认 100）。

        Returns:
            self，支持链式调用。
        """
        self._hooks[hook.event].append((priority, hook))
        self._hooks[hook.event].sort(key=lambda x: x[0])
        return self

    def unregister(self, hook: BaseHook) -> bool:
        """注销一个 Hook。

        通过实例比对（is）找到并移除。

        Args:
            hook: 要移除的 Hook 实例。

        Returns:
            True 如果找到并移除，False 如果未找到。
        """
        event_hooks = self._hooks.get(hook.event, [])
        for i, (_, registered) in enumerate(event_hooks):
            if registered is hook:
                event_hooks.pop(i)
                return True
        return False

    def clear(self, event: str | None = None) -> None:
        """清空 Hook。

        Args:
            event: 指定事件类型则只清空该事件。None 则清空全部。
        """
        if event:
            self._hooks.pop(event, None)
        else:
            self._hooks.clear()

    # ── 触发 ──────────────────────────────────────

    def trigger(self, event: str, **kwargs) -> str | None:
        """触发事件，按 priority 顺序执行所有已注册 Hook。

        任一 Hook 返回非 None 字符串则立即短路，不再执行后续 Hook。
        短路机制保证了：
          - 权限 Hook 拦截后，日志 Hook 不会执行
          - 被拦截的工具调用不会留下"已执行"的日志

        Args:
            event: HookEvent 值（"pre_tool_use" 等）。
            **kwargs: 事件携带的参数（见 HookEvent 文档）。

        Returns:
            None = 所有 Hook 都放行。
            str  = 被某个 Hook 拦截（字符串为拦截原因）。
        """
        for _priority, hook in self._hooks.get(event, []):
            try:
                result = hook.handle(**kwargs)
            except Exception:
                # Hook 不应抛异常，兜底保护
                continue
            if result is not None:
                return result
        return None

    # ── 查询 ──────────────────────────────────────

    def count(self, event: str | None = None) -> int:
        """统计注册的 Hook 数量。

        Args:
            event: 指定事件类型则只统计该事件。None 则统计全部。
        """
        if event:
            return len(self._hooks.get(event, []))
        return sum(len(v) for v in self._hooks.values())

    def list_hooks(self, event: str | None = None) -> list[dict]:
        """列出已注册的 Hook（用于调试）。

        Returns:
            [{"event": ..., "hook": ..., "priority": ...}, ...]
        """
        result = []
        events = [event] if event else self._hooks.keys()
        for evt in events:
            for priority, hook in self._hooks.get(evt, []):
                result.append({
                    "event": evt,
                    "hook": type(hook).__name__,
                    "priority": priority,
                })
        return sorted(result, key=lambda x: (x["event"], x["priority"]))
