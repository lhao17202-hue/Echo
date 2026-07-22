"""Tests for Persistent Teammates V1 — task manager, teammate agent, manager, tools, state, loop wiring, and e2e."""

import sys
import tempfile
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

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


from echo.multi_agent.teammate_manager import TeammateManager


def _manager_fixture(workspace: str, llm_outputs=None):
    registry = ToolRegistry()
    registry.discover("echo.tools.builtin")
    sandbox = Sandbox(workspace)
    shell = ShellExecutor(workspace)
    memory = MemoryManager(KeywordMemory())
    bus = MessageBus()
    bus.register("lead")
    tasks = GlobalTaskManager()
    manager = TeammateManager(
        llm=FakeLLMClient(llm_outputs or []),
        tool_registry=registry,
        sandbox=sandbox,
        shell=shell,
        memory=memory,
        bus=bus,
        tasks=tasks,
    )
    return manager, registry, bus, tasks


class TestTeammateManager:
    def test_spawn_creates_teammate_and_duplicate_is_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            manager, _registry, _bus, _tasks = _manager_fixture(d)

            first = manager.spawn("researcher", "research assistant", "Be concise")
            duplicate = manager.spawn("researcher", "other", "")

            assert first["success"] is True
            assert first["teammate"]["name"] == "researcher"
            assert duplicate["success"] is False
            assert "already exists" in duplicate["error"]
            assert len(manager.list()) == 1
            manager.stop("researcher")

    def test_stop_unknown_returns_false_and_stop_existing_returns_true(self):
        with tempfile.TemporaryDirectory() as d:
            manager, _registry, _bus, _tasks = _manager_fixture(d)

            assert manager.stop("missing") is False
            manager.spawn("researcher", "research assistant", "")
            assert manager.stop("researcher") is True
            assert manager.snapshot()["researcher"]["status"] == "stopped"

    def test_assign_task_creates_pending_task_for_teammate(self):
        with tempfile.TemporaryDirectory() as d:
            manager, _registry, _bus, tasks = _manager_fixture(d)
            manager.spawn("researcher", "research assistant", "")

            task_id = manager.assign_task("researcher", "Read README", "Find title")

            task = tasks.get(task_id)
            assert task is not None
            assert task.status == "pending"
            assert task.owner_agent == "researcher"
            manager.stop("researcher")

    def test_selected_tools_are_read_only_and_exclude_delegate_and_writes(self):
        with tempfile.TemporaryDirectory() as d:
            manager, _registry, _bus, _tasks = _manager_fixture(d)

            names = [tool.name for tool in manager._select_teammate_tools()]

            assert "read_file" in names
            assert "grep" in names
            assert "delegate" not in names
            assert "write_file" not in names
            assert "patch_file" not in names
            assert "run_shell" not in names
            assert "spawn_teammate" not in names


from echo.tools.builtin import (
    SpawnTeammateTool,
    AssignTaskTool,
    ListTeammatesTool,
    StopTeammateTool,
    ListGlobalTasksTool,
)


class TestTeammateBuiltinTools:
    def test_teammate_tools_fail_when_manager_unavailable(self):
        ctx = ToolContext()

        result = SpawnTeammateTool().execute(ctx, {"name": "researcher", "role": "assistant", "prompt": ""})

        assert result.error
        assert "Teammate manager unavailable" in result.error

    def test_spawn_assign_list_stop_tools_use_manager(self):
        with tempfile.TemporaryDirectory() as d:
            manager, _registry, _bus, tasks = _manager_fixture(d)
            ctx = ToolContext(teammate_manager=manager, global_tasks=tasks)

            spawn = SpawnTeammateTool().execute(ctx, {
                "name": "researcher",
                "role": "research assistant",
                "prompt": "Be concise",
            })
            assign = AssignTaskTool().execute(ctx, {
                "teammate": "researcher",
                "subject": "Read README",
                "description": "Find the title",
            })
            listed = ListTeammatesTool().execute(ctx, {})
            tasks_listed = ListGlobalTasksTool().execute(ctx, {})
            stopped = StopTeammateTool().execute(ctx, {"name": "researcher"})

            assert spawn.success
            assert "researcher" in spawn.output
            assert assign.success
            assert "Assigned task" in assign.output
            assert listed.success
            assert "researcher" in listed.output
            assert tasks_listed.success
            assert "Read README" in tasks_listed.output
            assert stopped.success
            assert "stopped" in stopped.output.lower()

    def test_registry_discovers_teammate_tools(self):
        registry = ToolRegistry()
        registry.discover("echo.tools.builtin")

        names = registry.get_names()
        assert "spawn_teammate" in names
        assert "assign_task" in names
        assert "list_teammates" in names
        assert "stop_teammate" in names
        assert "list_global_tasks" in names


from echo.core.task_state import TaskState


class TestTaskStateTeammateSnapshots:
    def test_task_state_serializes_teammate_snapshot_fields(self):
        state = TaskState.create("coordinate teammates")
        state.active_teammates = {"researcher": {"status": "idle"}}
        state.global_task_ids = ["abc123"]
        state.unprocessed_messages = [{"from_agent": "researcher", "content": "done"}]

        restored = TaskState.from_dict(state.to_dict())

        assert restored.active_teammates == {"researcher": {"status": "idle"}}
        assert restored.global_task_ids == ["abc123"]
        assert restored.unprocessed_messages == [{"from_agent": "researcher", "content": "done"}]


from echo.hooks.base import HookManager
from echo.core.context_manager import ContextManager
from echo.persistence.session_store import SessionStore
from echo.persistence.run_store import RunStore
from echo.core.agent_loop import AgentLoop


def _bare_loop_for_inbox(workspace: str, bus: MessageBus, tasks: GlobalTaskManager, teammate_manager=None):
    registry = ToolRegistry()
    registry.discover("echo.tools.builtin")
    sandbox = Sandbox(workspace)
    shell = ShellExecutor(workspace)
    memory = MemoryManager(KeywordMemory())
    return AgentLoop(
        llm=FakeLLMClient(["done"]),
        memory=memory,
        tools=ToolExecutor(registry),
        hooks=HookManager(),
        context=ContextManager(),
        sandbox=sandbox,
        shell=shell,
        session_store=SessionStore(workspace),
        run_store=RunStore(str(Path(workspace) / ".echo" / "sessions" / "test-session")),
        message_bus=bus,
        teammate_manager=teammate_manager,
        global_tasks=tasks,
    )


class TestAgentLoopTeammateInbox:
    def test_inject_inbox_messages_appends_user_message_and_logs_trace(self):
        with tempfile.TemporaryDirectory() as d:
            bus = MessageBus()
            bus.register("lead")
            tasks = GlobalTaskManager()
            loop = _bare_loop_for_inbox(d, bus, tasks)
            state = TaskState.create("lead task")
            loop.run_store.start_run(state)

            bus.send("researcher", "lead", "The README title is Echo.", msg_type="task_completed")
            loop._inject_inbox_messages(state)

            assert len(loop.messages) == 1
            block = loop.messages[0]["content"][0]
            assert "## Teammate Messages" in block.text
            assert "researcher" in block.text
            assert "The README title is Echo." in block.text

    def test_sync_multi_agent_state_records_snapshots(self):
        with tempfile.TemporaryDirectory() as d:
            manager, _registry, bus, tasks = _manager_fixture(d)
            manager.spawn("researcher", "research assistant", "")
            task_id = manager.assign_task("researcher", "Read README", "")
            loop = _bare_loop_for_inbox(d, bus, tasks, teammate_manager=manager)
            state = TaskState.create("lead task")

            loop._sync_multi_agent_state(state)

            assert "researcher" in state.active_teammates
            assert task_id in state.global_task_ids
            manager.stop("researcher")


from echo.config import EchoConfig
from echo.core.echo import Echo


class TestEchoFacadeTeammateWiring:
    @pytest.mark.skip(reason="Requires anthropic/openai/ollama SDK to be installed")
    def test_echo_initializes_teammate_runtime(self):
        with tempfile.TemporaryDirectory() as d:
            config = EchoConfig(provider="anthropic", model="claude-sonnet-4-6", api_key="fake-key")
            echo = Echo(workspace_root=d, config=config)

            assert echo.message_bus is not None
            assert echo.global_tasks is not None
            assert echo.teammates is not None
            echo.message_bus.send("tester", "lead", "hello")
            assert echo.message_bus.receive("lead")[0].content == "hello"


class TestPersistentTeammatesE2E:
    def test_lead_spawns_assigns_and_receives_teammate_result(self):
        """E2E: lead spawns/assigns, teammate thread completes, lead receives via inbox.

        The lead's final FakeLLM output is generic (not mentioning "Echo"), so the
        "Echo" in the answer MUST come from inbox injection — proving the pipeline works.
        """
        with tempfile.TemporaryDirectory() as d:
            Path(d, "README.md").write_text("# Echo\nPersistent teammates", encoding="utf-8")
            registry = ToolRegistry()
            registry.discover("echo.tools.builtin")
            sandbox = Sandbox(d)
            shell = ShellExecutor(d)
            memory = MemoryManager(KeywordMemory())
            bus = MessageBus()
            bus.register("lead")
            tasks = GlobalTaskManager()
            # Separate LLM clients to avoid thread race on shared sequence
            lead_llm = FakeLLMClient([
                '<tool name="spawn_teammate" name="researcher" role="research assistant" prompt="Be concise" />',
                '<tool name="assign_task" teammate="researcher" subject="Read README" description="Find the title" />',
                "Task assigned, waiting for teammate results.",  # generic — does NOT contain "Echo"
            ])
            teammate_llm = FakeLLMClient([
                '<tool name="read_file" path="README.md" />',
                "The project title is Echo.",
            ])
            manager = TeammateManager(teammate_llm, registry, sandbox, shell, memory, bus, tasks)
            run_store = RunStore(str(Path(d) / ".echo" / "sessions" / "test-session"))
            loop = AgentLoop(
                llm=lead_llm,
                memory=memory,
                tools=ToolExecutor(registry),
                hooks=HookManager(),
                context=ContextManager(),
                sandbox=sandbox,
                shell=shell,
                session_store=SessionStore(d),
                run_store=run_store,
                max_steps=6,
                approval_policy="auto",
                message_bus=bus,
                teammate_manager=manager,
                global_tasks=tasks,
            )

            loop.run("Create a teammate and have them inspect README")

            # Wait for the teammate daemon thread to complete the task
            import time
            time.sleep(0.3)

            # Manually inject inbox to simulate what the next loop iteration would do
            state = TaskState.create("lead task")
            loop.run_store.start_run(state)
            loop._inject_inbox_messages(state)

            # Verify inbox injection actually inserted the teammate result
            inbox_texts = [m["content"][0].text for m in loop.messages
                           if isinstance(m.get("content"), list)
                           and len(m["content"]) > 0
                           and hasattr(m["content"][0], "text")
                           and "## Teammate Messages" in m["content"][0].text]
            assert len(inbox_texts) > 0, "Inbox messages were not injected into lead messages"
            assert "Echo" in inbox_texts[0]
            assert "researcher" in inbox_texts[0]
            assert any(task.status == "completed" for task in tasks.list_all())
            for snap in manager.snapshot().values():
                if snap["status"] != "stopped":
                    manager.stop(snap["name"])

    def test_manager_task_completion_then_lead_inbox_injection(self):
        with tempfile.TemporaryDirectory() as d:
            Path(d, "README.md").write_text("# Echo\nPersistent teammates", encoding="utf-8")
            manager, registry, bus, tasks = _manager_fixture(d, [
                '<tool name="read_file" path="README.md" />',
                "The title is Echo.",
            ])
            manager.spawn("researcher", "research assistant", "")
            task_id = manager.assign_task("researcher", "Read README", "Find the title")
            manager._teammates["researcher"]._tick()

            loop = _bare_loop_for_inbox(d, bus, tasks, teammate_manager=manager)
            state = TaskState.create("lead task")
            loop.run_store.start_run(state)
            loop._inject_inbox_messages(state)

            assert tasks.get(task_id).status == "completed"
            assert "Echo" in loop.messages[-1]["content"][0].text
            manager.stop("researcher")
