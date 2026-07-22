"""Tests for Persistent Teammates V1 — task manager, teammate agent, manager, tools, state, loop wiring, and e2e."""

import json
import sys
import tempfile
import threading
import time
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

    def test_wait_returns_completed_task_result(self):
        tasks = GlobalTaskManager()
        task_id = tasks.create("Async work")

        def complete_later():
            time.sleep(0.02)
            tasks.complete(task_id, "done")

        worker = threading.Thread(target=complete_later)
        worker.start()
        task = tasks.wait(task_id, timeout=1)
        worker.join()

        assert task is not None
        assert task.status == "completed"
        assert task.result == "done"

    def test_wait_returns_pending_task_after_timeout(self):
        tasks = GlobalTaskManager()
        task_id = tasks.create("Slow work")

        task = tasks.wait(task_id, timeout=0.01, interval=0.005)

        assert task is not None
        assert task.status == "pending"

    def test_wait_returns_none_for_unknown_task(self):
        tasks = GlobalTaskManager()

        assert tasks.wait("missing", timeout=0.01) is None


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

    def test_tick_fails_task_on_empty_llm_response(self):
        with tempfile.TemporaryDirectory() as d:
            tools, ctx = _readonly_executor_and_ctx(d)
            bus = MessageBus()
            bus.register("lead")
            bus.register("researcher")
            tasks = GlobalTaskManager()
            task_id = tasks.create("Do something", "Just do it")
            tasks.assign(task_id, "researcher")

            agent = TeammateAgent(
                name="researcher",
                role="research assistant",
                prompt="",
                llm=FakeLLMClient([""]),  # empty response — no text, no tools
                tools=tools,
                ctx=ctx,
                bus=bus,
                tasks=tasks,
                poll_interval=0.01,
            )

            agent._tick()

            task = tasks.get(task_id)
            assert task.status == "failed"
            assert "no text response" in task.result.lower()
            messages = bus.receive("lead")
            assert len(messages) == 1
            assert messages[0].msg_type == "task_failed"

    def test_tick_fails_task_when_step_limit_reached(self):
        with tempfile.TemporaryDirectory() as d:
            tools, ctx = _readonly_executor_and_ctx(d)
            bus = MessageBus()
            bus.register("lead")
            bus.register("researcher")
            tasks = GlobalTaskManager()
            task_id = tasks.create("Loop forever", "")
            tasks.assign(task_id, "researcher")

            # Each Fake output is a tool call (keeps looping), never returns text
            agent = TeammateAgent(
                name="researcher",
                role="research assistant",
                prompt="",
                llm=FakeLLMClient(['<tool name="read_file" path="README.md" />'] * 12),
                tools=tools,
                ctx=ctx,
                bus=bus,
                tasks=tasks,
                poll_interval=0.01,
            )

            agent._tick()

            task = tasks.get(task_id)
            assert task.status == "failed"
            assert "step limit" in task.result.lower()


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
            assert "wait_global_task" not in names


from echo.tools.builtin import (
    SpawnTeammateTool,
    AssignTaskTool,
    ListTeammatesTool,
    StopTeammateTool,
    ListGlobalTasksTool,
    WaitGlobalTaskTool,
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
        assert "wait_global_task" in names

    def test_wait_global_task_tool_returns_completed_result(self):
        tasks = GlobalTaskManager()
        task_id = tasks.create("Research")
        tasks.complete(task_id, "The answer is 42.")
        ctx = ToolContext(global_tasks=tasks)

        result = WaitGlobalTaskTool().execute(ctx, {
            "task_id": task_id,
            "timeout_seconds": 0.01,
        })

        assert result.success
        assert "completed" in result.output
        assert "The answer is 42." in result.output

    def test_wait_global_task_tool_returns_failed_task_as_failure(self):
        tasks = GlobalTaskManager()
        task_id = tasks.create("Research")
        tasks.fail(task_id, "failed loudly")
        ctx = ToolContext(global_tasks=tasks)

        result = WaitGlobalTaskTool().execute(ctx, {
            "task_id": task_id,
            "timeout_seconds": 0.01,
        })

        assert not result.success
        assert "Global task failed" in result.error
        assert "failed loudly" in result.output

    def test_wait_global_task_tool_returns_partial_on_timeout(self):
        tasks = GlobalTaskManager()
        task_id = tasks.create("Slow task")
        ctx = ToolContext(global_tasks=tasks)

        result = WaitGlobalTaskTool().execute(ctx, {
            "task_id": task_id,
            "timeout_seconds": 0.01,
        })

        assert not result.success
        assert result.is_partial
        assert "not complete" in result.error
        assert "pending" in result.output

    def test_manager_events_appear_in_run_trace(self):
        """spawn and assign_task must log into the current run's trace when trace_logger+run_id are provided."""
        with tempfile.TemporaryDirectory() as d:
            run_store = RunStore(str(Path(d) / ".echo" / "sessions" / "test-session"))
            state = TaskState.create("trace test", run_id="run-trace-test-001")
            run_store.start_run(state)

            manager, _registry, _bus, _tasks = _manager_fixture(d)
            # Pass run_store as the per-spawn trace_logger (simulates what builtin tools do via ToolContext)
            manager.spawn("researcher", "research assistant", "Be concise",
                          run_id=state.run_id, trace_logger=run_store)
            manager.assign_task("researcher", "Read README", "Find the title",
                                run_id=state.run_id, trace_logger=run_store)

            # Read the trace.jsonl written by RunStore (path: .../runs/{run_id}/trace.jsonl)
            trace_path = Path(d) / ".echo" / "sessions" / "test-session" / "runs" / state.run_id / "trace.jsonl"
            assert trace_path.exists(), f"trace.jsonl not found at {trace_path}"
            lines = [l.strip() for l in trace_path.read_text(encoding="utf-8").splitlines() if l.strip()]
            events = []
            for l in lines:
                import json
                events.append(json.loads(l))

            # helper to find events
            event_types = [e["event"] for e in events]

            assert "teammate_spawned" in event_types, f"missing teammate_spawned in {event_types}"
            assert "global_task_created" in event_types, f"missing global_task_created in {event_types}"
            assert "global_task_assigned" in event_types, f"missing global_task_assigned in {event_types}"

            # Verify run_id propagation: each manager event must carry the correct run_id
            for e in events:
                if e["event"] in ("teammate_spawned", "global_task_created", "global_task_assigned"):
                    assert e.get("run_id") == "run-trace-test-001", \
                        f"wrong run_id in {e['event']}: {e.get('run_id')}"

            manager.stop("researcher")

    def test_teammate_task_events_follow_assigned_run_trace_across_runs(self):
        with tempfile.TemporaryDirectory() as d:
            run_a = RunStore(str(Path(d) / ".echo" / "sessions" / "session-a"))
            state_a = TaskState.create("spawn teammate", run_id="run-a")
            run_a.start_run(state_a)

            manager, _registry, _bus, tasks = _manager_fixture(d, ["Run B result"])
            manager.spawn("researcher", "research assistant", "",
                          run_id=state_a.run_id, trace_logger=run_a)
            manager._teammates["researcher"].poll_interval = 999

            run_b = RunStore(str(Path(d) / ".echo" / "sessions" / "session-b"))
            state_b = TaskState.create("assign work", run_id="run-b")
            run_b.start_run(state_b)

            task_id = manager.assign_task("researcher", "Do run B work", "",
                                          run_id=state_b.run_id, trace_logger=run_b)
            manager._teammates["researcher"]._tick()

            trace_a = Path(d) / ".echo" / "sessions" / "session-a" / "runs" / "run-a" / "trace.jsonl"
            trace_b = Path(d) / ".echo" / "sessions" / "session-b" / "runs" / "run-b" / "trace.jsonl"

            events_a = [json.loads(line) for line in trace_a.read_text(encoding="utf-8").splitlines() if line.strip()]
            events_b = [json.loads(line) for line in trace_b.read_text(encoding="utf-8").splitlines() if line.strip()]

            task_events_a = [e for e in events_a if e.get("task_id") == task_id]
            task_events_b = [e for e in events_b if e.get("task_id") == task_id]
            task_event_types_b = [e["event"] for e in task_events_b]

            assert "teammate_task_claimed" in task_event_types_b
            assert "teammate_task_completed" in task_event_types_b
            assert all(e.get("run_id") == "run-b" for e in task_events_b)
            assert not any(e["event"] in ("teammate_task_claimed", "teammate_task_completed")
                           for e in task_events_a)

            assert tasks.get(task_id).status == "completed"
            manager.stop("researcher")

    def test_teammate_exception_failure_uses_assigned_run_trace_before_clear(self):
        class _RaisingLLM(FakeLLMClient):
            def chat(self, messages=None, tools=None, system="", max_tokens=8000, temperature=0.0):
                raise RuntimeError("boom")

        with tempfile.TemporaryDirectory() as d:
            run_a = RunStore(str(Path(d) / ".echo" / "sessions" / "session-a"))
            state_a = TaskState.create("spawn teammate", run_id="run-a")
            run_a.start_run(state_a)

            registry = ToolRegistry()
            registry.discover("echo.tools.builtin")
            sandbox = Sandbox(d)
            shell = ShellExecutor(d)
            memory = MemoryManager(KeywordMemory())
            bus = MessageBus()
            bus.register("lead")
            tasks = GlobalTaskManager()
            manager = TeammateManager(_RaisingLLM([]), registry, sandbox, shell, memory, bus, tasks)
            manager.spawn("researcher", "research assistant", "",
                          run_id=state_a.run_id, trace_logger=run_a)
            manager._teammates["researcher"].poll_interval = 999

            run_b = RunStore(str(Path(d) / ".echo" / "sessions" / "session-b"))
            state_b = TaskState.create("assign work", run_id="run-b")
            run_b.start_run(state_b)

            task_id = manager.assign_task("researcher", "Fail in run B", "",
                                          run_id=state_b.run_id, trace_logger=run_b)
            manager._teammates["researcher"]._tick()

            trace_b = Path(d) / ".echo" / "sessions" / "session-b" / "runs" / "run-b" / "trace.jsonl"
            events_b = [json.loads(line) for line in trace_b.read_text(encoding="utf-8").splitlines() if line.strip()]
            failed = [e for e in events_b
                      if e.get("task_id") == task_id and e.get("event") == "teammate_task_failed"]

            assert failed
            assert all(e.get("run_id") == "run-b" for e in failed)
            assert tasks.get(task_id).status == "failed"
            manager.stop("researcher")


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

    def test_lead_can_wait_for_teammate_task_result(self):
        import re

        class _WaitAwareLeadLLM(FakeLLMClient):
            def chat(self, messages=None, tools=None, system="", max_tokens=8000, temperature=0.0):
                if self.call_count == 2:
                    task_id = ""
                    for msg in reversed(messages or []):
                        for block in msg.get("content", []) or []:
                            text = ""
                            if isinstance(block, dict):
                                text = str(block.get("content", ""))
                            elif hasattr(block, "text"):
                                text = block.text
                            match = re.search(r"Assigned task ([0-9a-f]+)", text)
                            if match:
                                task_id = match.group(1)
                                break
                        if task_id:
                            break
                    self.feed(f'<tool name="wait_global_task" task_id="{task_id}" timeout_seconds="3" />')
                elif self.call_count == 3:
                    self.feed("Lead received teammate result.")
                return super().chat(messages, tools, system, max_tokens, temperature)

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
            lead_llm = _WaitAwareLeadLLM([
                '<tool name="spawn_teammate" name="researcher" role="research assistant" prompt="Be concise" />',
                '<tool name="assign_task" teammate="researcher" subject="Read README" description="Find the title" />',
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
                max_steps=8,
                approval_policy="auto",
                message_bus=bus,
                teammate_manager=manager,
                global_tasks=tasks,
            )

            answer = loop.run("Create a teammate and wait for README findings")

            wait_results = [
                block.get("content", "")
                for msg in loop.messages
                for block in (msg.get("content") or [])
                if isinstance(block, dict) and block.get("tool_name") == "wait_global_task"
            ]
            assert answer == "Lead received teammate result."
            assert any("The project title is Echo." in text for text in wait_results)
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


class TestSharedLLMLock:
    """Prove the shared llm_lock serialises lead + teammate LLM calls."""

    def test_agent_loop_applies_shared_lock_to_external_context_manager(self):
        import threading as _th

        with tempfile.TemporaryDirectory() as d:
            lock = _th.Lock()
            context = ContextManager()
            loop = _bare_loop_for_inbox(d, MessageBus(), GlobalTaskManager())
            loop_with_lock = AgentLoop(
                llm=loop.llm,
                memory=loop.memory,
                tools=loop.tools,
                hooks=loop.hooks,
                context=context,
                sandbox=loop.sandbox,
                shell=loop.shell,
                session_store=loop.session_store,
                run_store=loop.run_store,
                llm_lock=lock,
            )

            assert loop_with_lock._llm_lock is lock
            assert context._llm_lock is lock

    def test_lead_and_teammate_never_call_llm_concurrently(self):
        """When lead and teammate share one lock, peak concurrency must be 1."""
        import threading as _th
        lock = _th.Lock()
        peak_concurrent = [0]
        current_concurrent = [0]

        class _CountingFakeLLM(FakeLLMClient):
            def chat(self, messages=None, tools=None, system="", max_tokens=8000, temperature=0.0):
                current_concurrent[0] += 1
                peak_concurrent[0] = max(peak_concurrent[0], current_concurrent[0])
                # Simulate a small amount of work
                _th.Event().wait(0.05)
                result = super().chat(messages, tools, system, max_tokens, temperature)
                current_concurrent[0] -= 1
                return result

        with tempfile.TemporaryDirectory() as d:
            Path(d, "README.md").write_text("# Test", encoding="utf-8")
            registry = ToolRegistry()
            registry.discover("echo.tools.builtin")
            sandbox = Sandbox(d)
            shell = ShellExecutor(d)
            memory = MemoryManager(KeywordMemory())
            bus = MessageBus()
            bus.register("lead")
            tasks = GlobalTaskManager()

            lead_llm = _CountingFakeLLM([
                '<tool name="spawn_teammate" name="researcher" role="research assistant" prompt="" />',
                '<tool name="assign_task" teammate="researcher" subject="Read README" description="" />',
                "Lead done.",
            ])
            # Teammate gets several tool calls so it loops while lead is also looping
            teammate_llm = _CountingFakeLLM([
                '<tool name="read_file" path="README.md" />',
                "Found some content.",
            ])

            manager = TeammateManager(teammate_llm, registry, sandbox, shell, memory, bus, tasks,
                                      llm_lock=lock)
            run_store = RunStore(str(Path(d) / ".echo" / "sessions" / "test-session"))
            loop = AgentLoop(
                llm=lead_llm, memory=memory,
                tools=ToolExecutor(registry),
                hooks=HookManager(), context=ContextManager(),
                sandbox=sandbox, shell=shell,
                session_store=SessionStore(d), run_store=run_store,
                max_steps=6, approval_policy="auto",
                message_bus=bus, teammate_manager=manager, global_tasks=tasks,
                llm_lock=lock,
            )

            loop.run("Test concurrency")

            # Wait for teammate thread to finish
            import time
            time.sleep(0.3)

            manager.stop("researcher")
            assert peak_concurrent[0] <= 1, \
                f"peak concurrent LLM calls = {peak_concurrent[0]}, expected <= 1"
