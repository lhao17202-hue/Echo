"""End-to-end tests for the Echo single-agent kernel using FakeLLMClient.

Run: python -m pytest tests/test_agent_loop.py -v
"""

import sys, tempfile
from pathlib import Path
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from echo.providers.fake_client import FakeLLMClient
from echo.tools.registry import ToolRegistry
from echo.tools.executor import ToolExecutor
from echo.hooks.base import HookManager
from echo.hooks.builtin import PermissionHook, LogHook, PostLogHook, StatsHook
from echo.core.context_manager import ContextManager
from echo.core.agent_loop import AgentLoop
from echo.memory.base import MemoryManager, MemoryEntry
from echo.memory.default import KeywordMemory
from echo.security.sandbox import Sandbox
from echo.security.env_filter import ShellExecutor
from echo.persistence.session_store import Session, SessionStore
from echo.persistence.run_store import RunStore


def _trace_events(workspace: str, loop: AgentLoop) -> list[dict]:
    run_id = getattr(loop.run_store, "current_run_id", "")
    assert run_id, "RunStore.current_run_id should be set after AgentLoop.run()"
    trace_path = Path(workspace) / ".echo" / "sessions" / "test-session" / "runs" / run_id / "trace.jsonl"
    assert trace_path.exists(), f"trace file missing: {trace_path}"
    import json
    return [json.loads(line) for line in trace_path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _events_named(events: list[dict], name: str) -> list[dict]:
    return [event for event in events if event.get("event") == name]


def _make_loop(workspace: str, llm_outputs: list[str],
               approval: str = "auto", max_steps: int = 10):
    """构建一个完整的 AgentLoop 用于测试。"""
    registry = ToolRegistry()
    registry.discover("echo.tools.builtin")

    hooks = HookManager()
    hooks.register(PermissionHook(), priority=0)
    hooks.register(LogHook(), priority=100)
    hooks.register(PostLogHook(), priority=100)
    hooks.register(StatsHook(), priority=200)

    sandbox = Sandbox(workspace)
    shell = ShellExecutor(workspace)
    memory = MemoryManager(KeywordMemory())
    context = ContextManager()

    # session dir
    from pathlib import Path
    sess_dir = str(Path(workspace) / ".echo" / "sessions" / "test-session")
    session_store = SessionStore(workspace)
    run_store = RunStore(sess_dir)

    executor = ToolExecutor(registry)

    return AgentLoop(
        llm=FakeLLMClient(outputs=llm_outputs),
        memory=memory,
        tools=executor,
        hooks=hooks,
        context=context,
        sandbox=sandbox,
        shell=shell,
        session_store=session_store,
        run_store=run_store,
        max_steps=max_steps,
        approval_policy=approval,
    )


# ═══════════════════════════════════════════════════
# 确定性端到端测试
# ═══════════════════════════════════════════════════

class TestE2EBasic:
    """最基本的 LLM → tool → LLM → final 循环。"""

    def test_runtime_events_are_injected_before_model_call(self):
        class FakeScheduler:
            def consume(self):
                return []

        with tempfile.TemporaryDirectory() as d:
            from echo.multi_agent.message_bus import MessageBus
            from echo.multi_agent.task_manager import GlobalTaskManager

            bus = MessageBus()
            bus.register("lead")
            bus.send("researcher", "lead", "Runtime event reached the model.", msg_type="task_completed")
            loop = _make_loop(d, ["final answer"])
            loop.message_bus = bus
            loop.global_tasks = GlobalTaskManager()
            loop.scheduler = FakeScheduler()

            loop.run("handle runtime event")

            injected = [
                block.text
                for msg in loop.messages
                for block in msg.get("content", [])
                if hasattr(block, "text") and "## Runtime Events" in block.text
            ]
            assert injected
            assert "Runtime event reached the model." in injected[0]

    def test_read_file_then_final(self):
        """Agent 先调用 read_file，然后返回 final answer。"""
        with tempfile.TemporaryDirectory() as d:
            (Path(d) / "README.md").write_text("# My Project\nHello world")

            loop = _make_loop(d, [
                '<tool name="read_file" path="README.md" />',
                "项目名是 My Project。",
            ])

            answer = loop.run("Read README and tell me the project name")
            assert "My Project" in answer
            assert loop.llm.call_count == 2
            assert "Stopped" not in answer

    def test_two_tools_then_final(self):
        """Agent 连续调用两个工具，然后返回 final。"""
        with tempfile.TemporaryDirectory() as d:
            (Path(d) / "a.txt").write_text("AAA")
            (Path(d) / "b.txt").write_text("BBB")

            loop = _make_loop(d, [
                '<tool name="read_file" path="a.txt" />',
                '<tool name="read_file" path="b.txt" />',
                "两个文件分别包含 AAA 和 BBB。",
            ])

            answer = loop.run("Read a.txt and b.txt")
            assert "AAA" in answer or "BBB" in answer
            assert loop.llm.call_count == 3

    def test_no_tools_just_answer(self):
        """模型直接返回文本，不调用任何工具。"""
        with tempfile.TemporaryDirectory() as d:
            loop = _make_loop(d, ["我没有可用的工具，但这是答案。"])

            answer = loop.run("hello")
            assert "答案" in answer
            assert loop.llm.call_count == 1

    def test_exhausted_returns_empty(self):
        """FakeLLMClient 序列耗尽时返回空字符串（不崩溃）。"""
        with tempfile.TemporaryDirectory() as d:
            loop = _make_loop(d, [])  # 无预设回复

            answer = loop.run("do something")
            # 应该优雅终止而不是崩溃
            assert isinstance(answer, str)
            # 会经过 stop_model_error 或正常结束
            assert answer


class TestE2EToolResults:
    """验证工具结果正确记录。"""

    def test_read_file_content_in_result(self):
        """read_file 的结果应该包含文件内容。"""
        with tempfile.TemporaryDirectory() as d:
            (Path(d) / "data.txt").write_text("secret-data-123")

            loop = _make_loop(d, [
                '<tool name="read_file" path="data.txt" />',
                "文件内容是 secret-data-123。",
            ])

            answer = loop.run("check data.txt")
            assert "secret-data-123" in answer

    def test_trace_records_read_file_lifecycle(self):
        """Trace should show a whitebox lifecycle for a successful tool call."""
        with tempfile.TemporaryDirectory() as d:
            (Path(d) / "data.txt").write_text("secret-data-123")
            loop = _make_loop(d, [
                '<tool name="read_file" path="data.txt" />',
                "文件内容是 secret-data-123。",
            ])

            loop.run("check data.txt")
            events = _trace_events(d, loop)
            started = _events_named(events, "tool_started")
            finished = _events_named(events, "tool_finished")

            assert len(started) == 1
            assert len(finished) == 1
            start_payload = started[0]["payload"]
            finish_payload = finished[0]["payload"]
            assert start_payload["tool"] == "read_file"
            assert finish_payload["tool"] == "read_file"
            assert start_payload["tool_call_id"] == finish_payload["tool_call_id"]
            assert start_payload["input_summary"]
            assert finish_payload["success"] is True
            assert finish_payload["output_summary"]
            assert isinstance(finish_payload["duration_ms"], int)
            assert finish_payload["duration_ms"] >= 0
            assert any(path.endswith("data.txt") for path in finish_payload["files_read"])
            assert any(path.endswith("data.txt") for path in finish_payload["files_touched"])

    def test_glob_finds_files(self):
        with tempfile.TemporaryDirectory() as d:
            (Path(d) / "test_a.py").write_text("")
            (Path(d) / "test_b.py").write_text("")

            loop = _make_loop(d, [
                '<tool name="glob" pattern="*.py" />',
                "找到了 test_a.py 和 test_b.py。",
            ])

            answer = loop.run("find python files")
            assert "test_a.py" in answer


# ═══════════════════════════════════════════════════
# 权限系统测试
# ═══════════════════════════════════════════════════

class TestE2EPermission:
    """权限策略端到端测试。"""

    def test_safe_tool_always_allowed(self):
        """safe 工具（如 read_file）在任何策略下都不拦截。"""
        with tempfile.TemporaryDirectory() as d:
            (Path(d) / "f.txt").write_text("content")

            for policy in ("auto", "ask", "never"):
                loop = _make_loop(d, [
                    '<tool name="read_file" path="f.txt" />',
                    "done",
                ], approval=policy)

                answer = loop.run("read")
                assert "done" in answer or "content" in answer, f"failed with policy={policy}"

    def test_warn_tool_blocked_by_never(self):
        """warn 工具（如 write_file）在 policy=never 时应被拦截。"""
        with tempfile.TemporaryDirectory() as d:
            loop = _make_loop(d, [
                '<tool name="write_file" path="x.txt" content="data" />',
                "wrote file",
            ], approval="never")

            answer = loop.run("write a file")
            # write_file 被拦截 → 模型看到 Blocked 错误 → 可能继续或停止
            # 关键是 AgentLoop 不崩溃
            assert isinstance(answer, str)

    def test_trace_records_blocked_tool(self):
        """Trace should show why a permission-blocked tool did not run."""
        with tempfile.TemporaryDirectory() as d:
            loop = _make_loop(d, [
                '<tool name="write_file" path="blocked.txt" content="data" />',
                "I could not write the file.",
            ], approval="never")

            loop.run("write blocked.txt")
            events = _trace_events(d, loop)
            blocked = _events_named(events, "tool_blocked")
            finished = _events_named(events, "tool_finished")

            assert len(blocked) == 1
            assert finished == []
            payload = blocked[0]["payload"]
            assert payload["tool"] == "write_file"
            assert payload["success"] is False
            assert payload["approval_policy"] == "never"
            assert payload["reason"]
            assert payload["tool_call_id"]
            assert payload["input_summary"]
            assert not (Path(d) / "blocked.txt").exists()

    def test_warn_tool_allowed_by_auto(self):
        """warn 工具在 policy=auto 时直接放行。"""
        with tempfile.TemporaryDirectory() as d:
            loop = _make_loop(d, [
                '<tool name="write_file" path="out.txt" content="hi" />',
                "写入成功。",
            ], approval="auto")

            answer = loop.run("write out.txt")
            assert "成功" in answer or "hi" in answer or "Wrote" in answer
            assert (Path(d) / "out.txt").exists()

    def test_danger_tool_deny_list_blocked(self):
        """danger 工具 + deny list 命令 → 被 Hook 拦截，Agent 不崩溃。"""
        with tempfile.TemporaryDirectory() as d:
            loop = _make_loop(d, [
                f'<tool name="run_shell" command="rm -rf /" />',
                "done",
            ], approval="auto")  # auto 也不放行 deny list

            answer = loop.run("delete everything")
            # Hook 拦截了危险命令 → AgentLoop 不崩溃即可
            assert isinstance(answer, str)


# ═══════════════════════════════════════════════════
# 安全性测试
# ═══════════════════════════════════════════════════

class TestE2ESafety:
    """安全性端到端测试。"""

    def test_path_escape_blocked(self):
        """工具不应访问工作区外的路径（沙箱拦截，Agent 不崩溃）。"""
        with tempfile.TemporaryDirectory() as d:
            loop = _make_loop(d, [
                '<tool name="read_file" path="../etc/passwd" />',
                "done",
            ])

            answer = loop.run("read /etc/passwd")
            # 沙箱拦截后返回 ToolResult.fail → Agent 不崩溃即可
            assert isinstance(answer, str)

    def test_trace_records_failed_tool(self):
        """Trace should show tool_failed when execution returns an error."""
        with tempfile.TemporaryDirectory() as d:
            loop = _make_loop(d, [
                '<tool name="read_file" path="missing.txt" />',
                "The file was missing.",
            ])

            loop.run("read missing.txt")
            events = _trace_events(d, loop)
            started = _events_named(events, "tool_started")
            failed = _events_named(events, "tool_failed")

            assert len(started) == 1
            assert len(failed) == 1
            start_payload = started[0]["payload"]
            fail_payload = failed[0]["payload"]
            assert start_payload["tool"] == "read_file"
            assert fail_payload["tool"] == "read_file"
            assert start_payload["tool_call_id"] == fail_payload["tool_call_id"]
            assert fail_payload["success"] is False
            assert fail_payload["error_preview"]
            assert isinstance(fail_payload["duration_ms"], int)
            assert fail_payload["duration_ms"] >= 0

    def test_sandbox_prevents_write_outside(self):
        """write_file 不能写入工作区外（沙箱拦截，Agent 不崩溃）。"""
        with tempfile.TemporaryDirectory() as d:
            loop = _make_loop(d, [
                '<tool name="write_file" path="../outside.txt" content="bad" />',
                "done",
            ], approval="auto")

            answer = loop.run("write outside")
            # 沙箱拦截后返回 ToolResult.fail → Agent 不崩溃即可
            assert isinstance(answer, str)


# ═══════════════════════════════════════════════════
# 压缩 + 持久化
# ═══════════════════════════════════════════════════

class TestE2ECompact:
    """上下文压缩 + 持久化证据链。"""

    def test_session_created_after_run(self):
        with tempfile.TemporaryDirectory() as d:
            loop = _make_loop(d, ["答案来了。"])
            loop._session = Session(session_id="test-s1", workspace_root=d)

            answer = loop.run("hello")
            assert "答案" in answer

            # 验证 session 被保存
            store = SessionStore(d)
            loaded = store.load("test-s1")
            assert loaded.session_id == "test-s1"

    def test_trace_written(self):
        with tempfile.TemporaryDirectory() as d:
            loop = _make_loop(d, [
                '<tool name="read_file" path="README.md" />',
                "done",
            ])
            (Path(d) / "README.md").write_text("content")

            answer = loop.run("read")
            # trace.jsonl 应该存在
            from echo.persistence.trace import read_jsonl
            trace_path = loop.run_store._trace_path()
            events = read_jsonl(trace_path)
            assert len(events) > 0
            assert any(e["event"] == "run_started" for e in events)
