"""Unit tests for echo.hooks — BaseHook, HookEvent, HookManager, Builtin hooks.

Run: python -m pytest tests/test_hooks.py -v
"""

import sys, logging
from pathlib import Path
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from echo.hooks.base import BaseHook, HookManager, HookEvent
from echo.hooks.builtin import (
    PermissionHook, LogHook, PostLogHook, LargeOutputHook, StatsHook,
)
from echo.tools.base import BaseTool, ToolContext, ToolResult
from echo.security.permission import PermissionGuard


# ═══════════════════════════════════════════════════
# HookEvent
# ═══════════════════════════════════════════════════

class TestHookEvent:
    """HookEvent 枚举。"""

    def test_all_four_events_exist(self):
        assert HookEvent.USER_PROMPT == "user_prompt"
        assert HookEvent.PRE_TOOL_USE == "pre_tool_use"
        assert HookEvent.POST_TOOL_USE == "post_tool_use"
        assert HookEvent.RUN_STOP == "run_stop"

    def test_str_enum(self):
        """HookEvent 是 str Enum，可当字符串用。"""
        assert isinstance(HookEvent.PRE_TOOL_USE, str)
        assert HookEvent.PRE_TOOL_USE == "pre_tool_use"

    def test_four_events_total(self):
        assert len(HookEvent) == 4


# ═══════════════════════════════════════════════════
# BaseHook
# ═══════════════════════════════════════════════════

class TestBaseHook:
    """BaseHook 抽象性。"""

    def test_cannot_instantiate_abstract(self):
        with pytest.raises(TypeError):
            BaseHook()  # 抽象类不能实例化

    def test_concrete_subclass(self):
        class MyHook(BaseHook):
            event = HookEvent.USER_PROMPT
            def handle(self, **kwargs):
                return None

        h = MyHook()
        assert h.event == "user_prompt"
        assert h.handle() is None


# ═══════════════════════════════════════════════════
# HookManager
# ═══════════════════════════════════════════════════

class _FakeTool(BaseTool):
    name = "fake_tool"
    description = "a fake tool"
    risk_level = "safe"
    def execute(self, ctx, params):
        return ToolResult.ok("ok")


class _AlwaysBlock(BaseHook):
    event = HookEvent.PRE_TOOL_USE
    def handle(self, **kwargs):
        return "always blocked"


class _AlwaysPass(BaseHook):
    event = HookEvent.PRE_TOOL_USE
    def handle(self, **kwargs):
        return None


class TestHookManager:
    """HookManager 注册、触发、注销。"""

    def test_register_and_trigger(self):
        hm = HookManager()
        hm.register(_AlwaysBlock(), priority=0)
        result = hm.trigger(HookEvent.PRE_TOOL_USE, tool=_FakeTool())
        assert result == "always blocked"

    def test_trigger_passes_when_no_block(self):
        hm = HookManager()
        hm.register(_AlwaysPass(), priority=0)
        result = hm.trigger(HookEvent.PRE_TOOL_USE, tool=_FakeTool())
        assert result is None

    def test_priority_order(self):
        """低 priority 先执行。"""
        hm = HookManager()
        order = []

        class A(BaseHook):
            event = HookEvent.PRE_TOOL_USE
            def handle(self, **kw):
                order.append("A"); return None

        class B(BaseHook):
            event = HookEvent.PRE_TOOL_USE
            def handle(self, **kw):
                order.append("B"); return None

        hm.register(A(), priority=10)
        hm.register(B(), priority=5)  # B should run first
        hm.trigger(HookEvent.PRE_TOOL_USE)
        assert order == ["B", "A"]

    def test_short_circuit(self):
        """第一个 Hook 拦截后，后续不执行。"""
        hm = HookManager()
        calls = []

        class Blocker(BaseHook):
            event = HookEvent.PRE_TOOL_USE
            def handle(self, **kw):
                calls.append("blocker")
                return "stop"

        class NeverRun(BaseHook):
            event = HookEvent.PRE_TOOL_USE
            def handle(self, **kw):
                calls.append("never")
                return None

        hm.register(Blocker(), priority=0)
        hm.register(NeverRun(), priority=100)
        result = hm.trigger(HookEvent.PRE_TOOL_USE)
        assert result == "stop"
        assert calls == ["blocker"]  # NeverRun 从未执行

    def test_unregister(self):
        hm = HookManager()
        h = _AlwaysBlock()
        hm.register(h, priority=0)
        assert hm.count(HookEvent.PRE_TOOL_USE) == 1
        hm.unregister(h)
        assert hm.count(HookEvent.PRE_TOOL_USE) == 0

    def test_unregister_nonexistent(self):
        hm = HookManager()
        assert hm.unregister(_AlwaysBlock()) is False

    def test_clear_event(self):
        hm = HookManager()
        hm.register(_AlwaysBlock(), priority=0)
        hm.register(_AlwaysPass(), priority=100)
        hm.clear(HookEvent.PRE_TOOL_USE)
        assert hm.count(HookEvent.PRE_TOOL_USE) == 0

    def test_clear_all(self):
        hm = HookManager()
        hm.register(_AlwaysBlock(), priority=0)

        class RunStopHook(BaseHook):
            event = HookEvent.RUN_STOP
            def handle(self, **kw):
                return None

        hm.register(RunStopHook())
        hm.clear()
        assert hm.count() == 0

    def test_count(self):
        hm = HookManager()
        hm.register(_AlwaysBlock(), priority=0)
        hm.register(_AlwaysPass(), priority=100)
        assert hm.count(HookEvent.PRE_TOOL_USE) == 2
        assert hm.count(HookEvent.RUN_STOP) == 0
        assert hm.count() == 2

    def test_list_hooks(self):
        hm = HookManager()
        hm.register(_AlwaysBlock(), priority=0)
        hooks = hm.list_hooks(HookEvent.PRE_TOOL_USE)
        assert len(hooks) == 1
        assert hooks[0]["hook"] == "_AlwaysBlock"
        assert hooks[0]["priority"] == 0

    def test_trigger_hook_exception_suppressed(self):
        """Hook 抛异常时不中断，跳过该 Hook。"""
        hm = HookManager()

        class CrashHook(BaseHook):
            event = HookEvent.PRE_TOOL_USE
            def handle(self, **kw):
                raise RuntimeError("crash")

        class NextHook(BaseHook):
            event = HookEvent.PRE_TOOL_USE
            def handle(self, **kw):
                return "next called"

        hm.register(CrashHook(), priority=0)
        hm.register(NextHook(), priority=10)
        result = hm.trigger(HookEvent.PRE_TOOL_USE)
        assert result == "next called"  # CrashHook 被跳过, NextHook 正常执行

    def test_trigger_empty_event(self):
        hm = HookManager()
        result = hm.trigger(HookEvent.USER_PROMPT, request="hello")
        assert result is None  # 无 Hook 注册 = 放行


# ═══════════════════════════════════════════════════
# Builtin Hooks
# ═══════════════════════════════════════════════════

class _SafeTool(BaseTool):
    name = "safe_tool"
    description = "safe"
    risk_level = "safe"
    def execute(self, ctx, params):
        return ToolResult.ok("ok")


class _WarnTool(BaseTool):
    name = "warn_tool"
    description = "warn"
    risk_level = "warn"
    def execute(self, ctx, params):
        return ToolResult.ok("ok")


class _DangerTool(BaseTool):
    name = "danger_tool"
    description = "danger"
    risk_level = "danger"
    def execute(self, ctx, params):
        return ToolResult.ok("ok")


class TestPermissionHook:
    """PermissionHook 权限校验。"""

    def test_safe_passes(self):
        h = PermissionHook()
        r = h.handle(tool=_SafeTool(), tool_input={})
        assert r is None

    def test_warn_blocked(self):
        h = PermissionHook()
        r = h.handle(tool=_WarnTool(), tool_input={}, approval_policy="never")
        assert r is not None
        assert "拒绝" in r

    def test_danger_blocked(self):
        h = PermissionHook()
        r = h.handle(tool=_DangerTool(), tool_input={}, approval_policy="never")
        assert r is not None
        assert "拒绝" in r

    def test_none_tool_passes(self):
        h = PermissionHook()
        r = h.handle(tool=None, tool_input={})
        assert r is None

    def test_non_basetool_passes_with_warning(self):
        h = PermissionHook()
        r = h.handle(tool="not a tool", tool_input={})
        assert r is None  # 放行，但会打 warning


class TestLogHook:
    """LogHook 日志。"""

    def test_always_passes(self):
        h = LogHook()
        r = h.handle(tool=_SafeTool(), tool_input={"path": "x"})
        assert r is None  # 日志 Hook 永远不放行拦截

    def test_handles_none_tool(self):
        h = LogHook()
        r = h.handle(tool=None, tool_input={})
        assert r is None


class TestPostLogHook:
    """PostLogHook 结果日志。"""

    def test_reads_toolresult(self):
        h = PostLogHook()
        r = h.handle(result=ToolResult.ok("all good"))
        assert r is None

    def test_reads_string_result(self):
        h = PostLogHook()
        r = h.handle(result="plain string output")
        assert r is None

    def test_handles_none_result(self):
        h = PostLogHook()
        r = h.handle(result=None)
        assert r is None


class TestLargeOutputHook:
    """LargeOutputHook 大输出告警。"""

    def test_large_output(self):
        h = LargeOutputHook()
        r = h.handle(result=ToolResult.ok("x" * 200_000))
        assert r is None  # 只告警，不拦截

    def test_small_output(self):
        h = LargeOutputHook()
        r = h.handle(result=ToolResult.ok("small"))
        assert r is None

    def test_string_result(self):
        h = LargeOutputHook()
        r = h.handle(result="x" * 200_000)
        assert r is None

    def test_none_result(self):
        h = LargeOutputHook()
        r = h.handle(result=None)
        assert r is None


class TestStatsHook:
    """StatsHook 运行统计。"""

    def test_always_passes(self):
        from echo.core.task_state import TaskState
        ts = TaskState.create("test")
        ts.finish_success("done")
        h = StatsHook()
        r = h.handle(state=ts)
        assert r is None

    def test_no_state(self):
        h = StatsHook()
        r = h.handle(state=None)
        assert r is None


# ═══════════════════════════════════════════════════
# 完整管道模拟
# ═══════════════════════════════════════════════════

class TestFullPipeline:
    """模拟 AgentLoop 中的完整 Hook 管道。"""

    def test_pre_tool_pipeline(self):
        """PRE_TOOL_USE: Permission(0) → Log(100)。safe 工具：两个都执行。"""
        hm = HookManager()
        hm.register(PermissionHook(), priority=0)
        hm.register(LogHook(), priority=100)

        # safe 工具 — 权限放行, 日志执行
        result = hm.trigger(HookEvent.PRE_TOOL_USE,
                            tool=_SafeTool(), tool_input={"path": "x"})
        assert result is None

    def test_pre_tool_pipeline_blocked(self):
        """PRE_TOOL_USE: Permission 拦截后 Log 不执行。"""
        hm = HookManager()
        hm.register(PermissionHook(), priority=0)
        hm.register(LogHook(), priority=100)

        # warn 工具 — 权限拦截
        result = hm.trigger(HookEvent.PRE_TOOL_USE,
                            tool=_WarnTool(), tool_input={"path": "x"},
                            approval_policy="never")
        assert result is not None
        assert "拒绝" in result

    def test_post_tool_pipeline(self):
        """POST_TOOL_USE: PostLog + LargeOutput。"""
        hm = HookManager()
        hm.register(PostLogHook(), priority=100)
        hm.register(LargeOutputHook(), priority=200)

        result = hm.trigger(HookEvent.POST_TOOL_USE,
                            tool=_SafeTool(), result=ToolResult.ok("output"))
        assert result is None

    def test_run_stop_pipeline(self):
        """RUN_STOP: StatsHook。"""
        from echo.core.task_state import TaskState
        hm = HookManager()
        hm.register(StatsHook(), priority=100)

        ts = TaskState.create("test")
        ts.finish_success("done")
        result = hm.trigger(HookEvent.RUN_STOP, state=ts)
        assert result is None

    def test_user_prompt_reserved(self):
        """USER_PROMPT: 预留事件，无 Hook 注册但不报错。"""
        hm = HookManager()
        result = hm.trigger(HookEvent.USER_PROMPT, request="hello")
        assert result is None
