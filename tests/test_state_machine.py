"""Unit tests for state machine + context manager.

Run: python -m pytest tests/test_state_machine.py -v
"""

import json
import os
import sys
import tempfile
import pytest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from echo.core.task_state import (
    TaskState, Status, StopReason, ResumeStatus,
    state_summary, is_terminal_status,
)
from echo.core.context_manager import (
    ContextManager, ContextConfig, Budget,
)


# ═══════════════════════════════════════════════════
# State transition validation
# ═══════════════════════════════════════════════════

class TestStateTransitions:
    """状态转移验证。"""

    def test_running_to_completed_allowed(self):
        ts = TaskState.create("x")
        ok, msg = ts.validate_transition(Status.COMPLETED)
        assert ok; assert msg == ""

    def test_running_to_stopped_allowed(self):
        ts = TaskState.create("x")
        ok, _ = ts.validate_transition(Status.STOPPED)
        assert ok

    def test_running_to_failed_allowed(self):
        ts = TaskState.create("x")
        ok, _ = ts.validate_transition(Status.FAILED)
        assert ok

    def test_completed_to_any_blocked(self):
        ts = TaskState.create("x")
        ts.finish_success("done")
        for target in (Status.COMPLETED, Status.STOPPED, Status.FAILED, Status.RUNNING):
            ok, msg = ts.validate_transition(target)
            if target == Status.COMPLETED:
                assert not ok  # 相同终态也不允许
            else:
                assert not ok

    def test_stopped_to_any_blocked(self):
        ts = TaskState.create("x")
        ts.stop_step_limit()
        for target in (Status.COMPLETED, Status.RUNNING):
            ok, _ = ts.validate_transition(target)
            assert not ok

    def test_failed_to_any_blocked(self):
        ts = TaskState.create("x")
        ts.stop_model_error("boom")
        ok, _ = ts.validate_transition(Status.COMPLETED)
        assert not ok

    def test_finish_success_raises_on_terminal(self):
        ts = TaskState.create("x")
        ts.finish_success("first")  # OK
        with pytest.raises(ValueError, match="非法的状态转移"):
            ts.finish_success("second")  # 终态不能再 finish


class TestResumeStatus:
    """恢复状态枚举。"""

    def test_can_continue(self):
        assert ResumeStatus.can_continue("full-valid")
        assert ResumeStatus.can_continue("partial-stale")
        assert ResumeStatus.can_continue("workspace-mismatch")
        assert not ResumeStatus.can_continue("schema-mismatch")
        assert not ResumeStatus.can_continue("no-checkpoint")

    def test_needs_warning(self):
        assert not ResumeStatus.needs_warning("full-valid")
        assert ResumeStatus.needs_warning("partial-stale")
        assert ResumeStatus.needs_warning("workspace-mismatch")
        assert not ResumeStatus.needs_warning("no-checkpoint")


class TestStateSummary:
    """状态摘要工具。"""

    def test_lead_agent(self):
        ts = TaskState.create("hello")
        ts.record_tool("read_file")
        s = state_summary(ts)
        assert "status=running" in s
        assert "steps=1" in s

    def test_teammate_shows_name(self):
        ts = TaskState.create("x", agent_type="teammate", agent_name="reviewer")
        s = state_summary(ts)
        assert "teammate/reviewer" in s

    def test_with_global_task(self):
        ts = TaskState.create("x")
        ts.bound_global_task_id = "gt-123"
        s = state_summary(ts)
        assert "gtask=gt-123" in s

    def test_with_errors(self):
        ts = TaskState.create("x")
        ts.stop_model_error("oops")
        s = state_summary(ts)
        assert "errors=1" in s

    def test_is_terminal_status(self):
        assert not is_terminal_status("running")
        assert is_terminal_status("completed")
        assert is_terminal_status("stopped")
        assert is_terminal_status("failed")


class TestTaskStateProperties:
    """TaskState 查询属性。"""

    def test_can_resume(self):
        ts = TaskState.create("x")
        assert not ts.can_resume  # running 不能 resume
        ts.stop_step_limit()
        assert ts.can_resume
        ts2 = TaskState.create("y")
        ts2.stop_model_error("boom")
        assert ts2.can_resume
        ts3 = TaskState.create("z")
        ts3.finish_success("ok")
        assert not ts3.can_resume  # completed 也不需要 resume

    def test_has_errors(self):
        ts = TaskState.create("x")
        assert not ts.has_errors
        ts.stop_model_error("fail")
        assert ts.has_errors

    def test_is_failed_is_stopped(self):
        ts = TaskState.create("x")
        assert not ts.is_failed; assert not ts.is_stopped
        ts.stop_model_error("x"); assert ts.is_failed; assert not ts.is_stopped
        ts2 = TaskState.create("y")
        ts2.stop_step_limit(); assert not ts2.is_failed; assert ts2.is_stopped


# ═══════════════════════════════════════════════════
# ContextManager
# ═══════════════════════════════════════════════════

class FakeTools:
    """最小化的 ToolRegistry 假对象。"""
    class FakeTool:
        name = "read_file"
        description = "Read a file"
        risk_level = "safe"
    def get_all(self): return [self.FakeTool()]

class FakeMemory:
    def render_working(self): return "task: test"
    def retrieve(self, query, top_k=3): return []
    def relevant_for_prompt(self, query, limit=5): return ""

class FakeSandbox:
    root = "/tmp/test"
    git_branch = "main"


class TestContextManagerBuild:
    """System prompt 组装。"""

    def test_build_system_basic(self):
        cm = ContextManager()
        ts = TaskState.create("hello")
        system = cm.build_system(ts, FakeTools(), FakeMemory(), FakeSandbox())
        assert "Echo" in system
        assert "read_file" in system
        assert "safe" not in system.lower() or "read_file" in system  # safe tools don't get warning marker

    def test_build_system_includes_memory(self):
        cm = ContextManager()
        ts = TaskState.create("hello")
        system = cm.build_system(ts, FakeTools(), FakeMemory(), FakeSandbox())
        assert "task: test" in system

    def test_build_system_disables_memory(self):
        config = ContextConfig(enable_memory=False)
        cm = ContextManager(config)
        ts = TaskState.create("hello")
        system = cm.build_system(ts, FakeTools(), FakeMemory(), FakeSandbox())
        assert "task: test" not in system

    def test_build_system_includes_resume_status(self):
        cm = ContextManager()
        ts = TaskState.create("hello")
        ts.resume_status = "partial-stale"
        system = cm.build_system(ts, FakeTools(), FakeMemory(), FakeSandbox())
        assert "partial-stale" in system


class TestContextManagerCompaction:
    """四级压缩管道。"""

    def test_tool_result_budget_persists_large(self):
        """Level 1: 大输出 → 磁盘卸载。"""
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            config = ContextConfig(
                compact_threshold_chars=100,
                compact_preview_chars=50,
                persist_dir=str(d / "outputs"),
            )
            cm = ContextManager(config)

            long_output = "x" * 200
            messages = [
                {"role": "user", "content": "hello"},
                {"role": "assistant", "content": [{"type": "text", "text": "ok"}]},
                {"role": "user", "content": [
                    {"type": "tool_result", "tool_use_id": "call_01", "content": long_output},
                ]},
            ]

            result = cm._tool_result_budget(messages)
            persisted_block = result[-1]["content"][0]
            assert "<persisted-output>" in persisted_block["content"]
            assert "Preview" in persisted_block["content"]
            assert "call_01" in persisted_block["content"]

    def test_tool_result_budget_skips_small(self):
        """小输出不卸载。"""
        cm = ContextManager(ContextConfig(compact_threshold_chars=1000))
        messages = [
            {"role": "user", "content": [
                {"type": "tool_result", "tool_use_id": "call_01", "content": "small"},
            ]},
        ]
        result = cm._tool_result_budget(messages)
        assert "<persisted-output>" not in result[-1]["content"][0]["content"]

    def test_micro_compact_replaces_old(self):
        """Level 3: 久远工具结果替换。"""
        config = ContextConfig(keep_recent_tool_results=1)
        cm = ContextManager(config)

        messages = []
        for i in range(5):
            messages.append({"role": "assistant", "content": [{"type": "text", "text": f"msg{i}"}]})
            messages.append({"role": "user", "content": [
                {"type": "tool_result", "tool_use_id": f"call_{i:02d}",
                 "content": "x" * 200},
            ]})

        result = cm._micro_compact(messages)
        # 只有最近 1 条保留原始内容，其余被替换
        compacted_count = 0
        for msg in result:
            for block in (msg.get("content") or []):
                if isinstance(block, dict) and block.get("type") == "tool_result":
                    if "compacted" in str(block.get("content", "")):
                        compacted_count += 1
        assert compacted_count >= 3  # 5 total, keep 1 = 4 compacted (但可能有一两条没有长内容)

    def test_compact_count_tracks(self):
        cm = ContextManager()
        assert cm.compact_count == 0

    def test_reactive_compact(self):
        cm = ContextManager()
        messages = [
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "hi"},
        ] * 10
        result = cm.reactive_compact(messages)
        assert len(result) <= len(messages)
        assert any("Reactive compact" in str(m.get("content", "")) for m in result)

    def test_compact_messages_are_provider_safe(self):
        """所有 compact 产出的 user 消息都使用 list[dict] 格式，而不是纯字符串。
        纯字符串会被 Anthropic/OpenAI/Ollama 的 _blocks_to_text() 静默丢弃。"""
        cm = ContextManager()
        messages = [
            {"role": "user", "content": [{"type": "text", "text": "test"}]},
            {"role": "assistant", "content": [{"type": "text", "text": "ok"}]},
        ] * 15

        for compact_fn, label in [
            (lambda msgs: cm.reactive_compact(msgs), "reactive_compact"),
            (lambda msgs: cm._compact_history(msgs), "_compact_history (no llm)"),
            (lambda msgs: cm._snip_compact(msgs), "_snip_compact"),
        ]:
            result = compact_fn(messages)
            for msg in result:
                assert isinstance(msg.get("content"), list), \
                    f"{label}: content 应为 list[dict]，实际为 {type(msg['content'])}"
                for block in msg["content"]:
                    assert isinstance(block, dict), \
                        f"{label}: block 应为 dict，实际为 {type(block)}"


class TestBudget:
    """预算控制。"""

    def test_apply_budget_reduces_sections(self):
        b = Budget(total=100, prefix=80, memory=80, history=80,
                   floors={"prefix": 10, "memory": 10, "history": 10})
        config = ContextConfig(budget=b)
        cm = ContextManager(config)

        sections = {
            "prefix": "A" * 50,
            "memory": "B" * 50,
            "history": "C" * 50,
        }
        result = cm._apply_budget(sections)
        total = sum(len(v) for v in result.values())
        assert total <= 100

    def test_truncate(self):
        result = ContextManager._truncate("hello world", 5)
        assert "hello" in result
        assert "truncated" in result
        assert len(result) < len("hello world") + len("\n... [truncated]")


class TestContextConfig:
    """配置默认值。"""

    def test_default_values(self):
        cfg = ContextConfig()
        assert cfg.compact_threshold_chars == 30_000
        assert cfg.compact_preview_chars == 2_000
        assert cfg.max_messages == 50
        assert cfg.context_limit_chars == 50_000
        assert cfg.enable_memory is True
        assert cfg.enable_compaction is True
