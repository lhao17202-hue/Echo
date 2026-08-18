from echo.runtime.background import BackgroundManager
import tempfile
import time
from pathlib import Path


def test_background_manager_completes_shell_task_and_emits_event():
    manager = BackgroundManager()

    bg_id = manager.start_shell_task(
        command='python -c "print(12345)"',
        cwd='.',
        timeout_seconds=5,
    )
    assert bg_id

    events = []
    for _ in range(50):
        events = manager.poll_completed()
        if events:
            break
        time.sleep(0.02)

    assert events, "background task should eventually emit a completion event"
    event = events[0]
    assert event.source == "background"
    assert event.event_type == "completed"
    assert event.metadata["bg_id"] == bg_id
    assert "12345" in event.content


def test_background_manager_lists_active_task():
    manager = BackgroundManager()

    bg_id = manager.start_shell_task(
        command='python -c "import time; time.sleep(0.1)"',
        cwd='.',
        timeout_seconds=5,
    )

    listed = manager.list()
    assert any(task.bg_id == bg_id for task in listed)


def test_run_shell_background_tool_starts_background_task():
    from echo.tools.base import ToolContext
    from echo.tools.builtin import RunShellBackgroundTool

    manager = BackgroundManager()
    ctx = ToolContext(background_manager=manager)

    result = RunShellBackgroundTool().execute(ctx, {
        "command": 'python -c "print(12345)"',
        "timeout": 5,
        "cwd": ".",
    })

    assert result.success
    assert result.output.startswith("Started background shell task")
    assert "bg_" in result.output


def test_echo_facade_provides_background_manager_to_agent_loop(monkeypatch):
    from echo.config import EchoConfig
    from echo.core import echo as echo_module
    from echo.core.echo import Echo
    from echo.providers.fake_client import FakeLLMClient

    captured = {}
    real_agent_loop = echo_module.AgentLoop

    class CapturingAgentLoop(real_agent_loop):
        def __init__(self, *args, **kwargs):
            captured["background_manager"] = kwargs.get("background_manager")
            super().__init__(*args, **kwargs)

    monkeypatch.setattr(echo_module, "AnthropicClient", lambda **_kwargs: FakeLLMClient(["done"]))
    monkeypatch.setattr(echo_module, "AgentLoop", CapturingAgentLoop)

    with tempfile.TemporaryDirectory() as d:
        config = EchoConfig(provider="anthropic", model="fake-model", api_key="fake-key")
        echo = Echo(workspace_root=d, config=config)

        try:
            echo.ask("hello")
        finally:
            echo.close()

        assert captured["background_manager"] is echo.background_manager


    from types import SimpleNamespace

    from echo.core.agent_loop import AgentLoop
    from echo.core.context_manager import ContextManager
    from echo.hooks.base import HookManager
    from echo.memory.base import MemoryManager
    from echo.memory.default import KeywordMemory
    from echo.persistence.run_store import RunStore
    from echo.persistence.session_store import SessionStore
    from echo.providers.fake_client import FakeLLMClient
    from echo.runtime.events import RuntimeEvent
    from echo.security.env_filter import ShellExecutor
    from echo.security.sandbox import Sandbox
    from echo.tools.executor import ToolExecutor
    from echo.tools.registry import ToolRegistry

    class ImmediateBackgroundManager:
        def __init__(self):
            self.bg_id = "bg_test"
            self._events = []

        def start_shell_task(self, command: str, cwd: str = ".", timeout_seconds: float = 20.0) -> str:
            self._events.append(RuntimeEvent(
                source="background",
                event_type="completed",
                content="12345",
                metadata={"bg_id": self.bg_id, "kind": "shell", "command": command},
            ))
            return self.bg_id

        def poll_completed(self):
            events = list(self._events)
            self._events.clear()
            return events

        def list(self):
            return [SimpleNamespace(bg_id=self.bg_id)]

    with tempfile.TemporaryDirectory() as d:
        registry = ToolRegistry().discover("echo.tools.builtin")
        manager = ImmediateBackgroundManager()
        loop = AgentLoop(
            llm=FakeLLMClient([
                '<tool name="run_shell_background" command="python -c print(12345)" timeout="5" />',
                "background finished",
            ]),
            memory=MemoryManager(KeywordMemory()),
            tools=ToolExecutor(registry),
            hooks=HookManager(),
            context=ContextManager(),
            sandbox=Sandbox(d),
            shell=ShellExecutor(d),
            session_store=SessionStore(d),
            run_store=RunStore(str(Path(d) / ".echo" / "sessions" / "test-session")),
            background_manager=manager,
            max_steps=3,
        )

        answer = loop.run("start a background command")

        injected = [
            block.text
            for msg in loop.messages
            for block in msg.get("content", [])
            if hasattr(block, "text") and "## Runtime Events" in block.text
        ]
        assert "background finished" in answer
        assert injected
        assert "[background/completed]" in injected[-1]
        assert "12345" in injected[-1]
        assert manager.bg_id not in loop._last_state.active_background_tasks
