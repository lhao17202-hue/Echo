"""Tests for Persistent Teammates V1 — task manager, teammate agent, manager, tools, state, loop wiring, and e2e."""

import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import tempfile
from pathlib import Path

from echo.multi_agent.task_manager import GlobalTaskManager


class TestGlobalTaskManagerTeammateV1:
    def test_assign_keeps_task_pending_and_sets_owner(self):
        tasks = GlobalTaskManager()
        task_id = tasks.create("Inspect README", "Find the project name")

        assert tasks.assign(task_id, "researcher") is True

        task = tasks.get(task_id)
        assert task is not None
        assert task.status == "pending"
        assert task.owner_agent == "researcher"

    def test_assign_rejects_missing_or_non_pending_task(self):
        tasks = GlobalTaskManager()
        task_id = tasks.create("Inspect README")
        assert tasks.assign("missing", "researcher") is False

        assert tasks.claim(task_id, "researcher") is True
        assert tasks.assign(task_id, "other") is False
        assert tasks.get(task_id).owner_agent == "researcher"

    def test_list_all_returns_all_tasks(self):
        tasks = GlobalTaskManager()
        first = tasks.create("First")
        second = tasks.create("Second")
        tasks.complete(second, "done")

        all_tasks = tasks.list_all()
        assert [t.task_id for t in all_tasks] == [first, second]
        assert [t.status for t in all_tasks] == ["pending", "completed"]

    def test_list_available_filters_by_owner_when_agent_name_given(self):
        tasks = GlobalTaskManager()
        unowned = tasks.create("Unowned")
        owned_by_a = tasks.create("Owned by A")
        owned_by_b = tasks.create("Owned by B")
        tasks.assign(owned_by_a, "agent-a")
        tasks.assign(owned_by_b, "agent-b")

        visible_to_a = tasks.list_available("agent-a")
        assert [t.task_id for t in visible_to_a] == [unowned, owned_by_a]


from echo.tools.base import ToolContext


class TestToolContextMultiAgentHandles:
    def test_tool_context_exposes_optional_multi_agent_handles(self):
        ctx = ToolContext(agent_name="lead")

        assert ctx.agent_name == "lead"
        assert ctx.message_bus is None
        assert ctx.teammate_manager is None
        assert ctx.global_tasks is None


from echo.providers.fake_client import FakeLLMClient
from echo.tools.registry import ToolRegistry
from echo.tools.executor import ToolExecutor
from echo.security.sandbox import Sandbox
from echo.security.env_filter import ShellExecutor
from echo.memory.base import MemoryManager
from echo.memory.default import KeywordMemory
from echo.multi_agent.message_bus import MessageBus
from echo.multi_agent.teammate import TeammateAgent


def _readonly_executor_and_ctx(workspace: str):
    registry = ToolRegistry()
    registry.discover("echo.tools.builtin")
    allowed = {"read_file", "glob", "grep", "list_files", "search_memory"}
    for name in list(registry.get_names()):
        if name not in allowed:
            registry.unregister(name)
    sandbox = Sandbox(workspace)
    shell = ShellExecutor(workspace)
    memory = MemoryManager(KeywordMemory())
    ctx = ToolContext(
        workspace_root=workspace,
        sandbox=sandbox,
        shell=shell,
        memory=memory,
        llm=FakeLLMClient([]),
        tool_registry=registry,
        agent_name="researcher",
    )
    return ToolExecutor(registry), ctx


class TestTeammateAgent:
    def test_snapshot_starts_idle(self):
        with tempfile.TemporaryDirectory() as d:
            tools, ctx = _readonly_executor_and_ctx(d)
            bus = MessageBus()
            bus.register("lead")
            bus.register("researcher")
            tasks = GlobalTaskManager()

            agent = TeammateAgent(
                name="researcher",
                role="research assistant",
                prompt="Focus on concise findings.",
                llm=FakeLLMClient([]),
                tools=tools,
                ctx=ctx,
                bus=bus,
                tasks=tasks,
                poll_interval=0.01,
            )

            snap = agent.snapshot()
            assert snap["name"] == "researcher"
            assert snap["role"] == "research assistant"
            assert snap["status"] == "idle"
            assert snap["current_task_id"] == ""

    def test_tick_claims_task_completes_it_and_sends_lead_message(self):
        with tempfile.TemporaryDirectory() as d:
            Path(d, "README.md").write_text("# Echo\nPersistent teammates", encoding="utf-8")
            tools, ctx = _readonly_executor_and_ctx(d)
            bus = MessageBus()
            bus.register("lead")
            bus.register("researcher")
            tasks = GlobalTaskManager()
            task_id = tasks.create("Read README", "Report the project title")
            tasks.assign(task_id, "researcher")

            agent = TeammateAgent(
                name="researcher",
                role="research assistant",
                prompt="Focus on concise findings.",
                llm=FakeLLMClient([
                    '<tool name="read_file" path="README.md" />',
                    "The project title is Echo.",
                ]),
                tools=tools,
                ctx=ctx,
                bus=bus,
                tasks=tasks,
                poll_interval=0.01,
            )

            agent._tick()

            task = tasks.get(task_id)
            assert task.status == "completed"
            assert "Echo" in task.result
            messages = bus.receive("lead")
            assert len(messages) == 1
            assert messages[0].from_agent == "researcher"
            assert messages[0].msg_type == "task_completed"
            assert "Echo" in messages[0].content
