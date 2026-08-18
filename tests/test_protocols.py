import tempfile
from pathlib import Path

from echo.runtime.events import RuntimeEvent


def test_protocol_manager_creates_resolves_and_emits_events():
    from echo.runtime.protocols import ProtocolManager

    manager = ProtocolManager()
    request = manager.create(
        protocol_type="approval",
        prompt="Approve the deploy",
        payload={"target": "prod"},
    )

    assert request.request_id in manager.pending_ids()
    assert request.status == "pending"

    resolved = manager.resolve(request.request_id, result="approved")
    assert resolved.status == "resolved"

    events = manager.poll_events()
    assert len(events) == 1
    event = events[0]
    assert isinstance(event, RuntimeEvent)
    assert event.source == "protocol"
    assert event.event_type == "resolved"
    assert request.request_id in event.metadata["request_id"]
    assert event.content == "approved"
    assert manager.poll_events() == []
    assert manager.pending_ids() == []


def test_agent_loop_injects_protocol_resolution_events_before_model_call():
    from echo.core.agent_loop import AgentLoop
    from echo.core.context_manager import ContextManager
    from echo.hooks.base import HookManager
    from echo.memory.base import MemoryManager
    from echo.memory.default import KeywordMemory
    from echo.persistence.run_store import RunStore
    from echo.persistence.session_store import SessionStore
    from echo.providers.fake_client import FakeLLMClient
    from echo.runtime.protocols import ProtocolManager
    from echo.security.env_filter import ShellExecutor
    from echo.security.sandbox import Sandbox
    from echo.tools.executor import ToolExecutor
    from echo.tools.registry import ToolRegistry

    with tempfile.TemporaryDirectory() as d:
        registry = ToolRegistry().discover("echo.tools.builtin")
        manager = ProtocolManager()
        request = manager.create(
            protocol_type="approval",
            prompt="Approve the deploy",
            payload={"target": "prod"},
        )
        manager.resolve(request.request_id, result="approved")

        loop = AgentLoop(
            llm=FakeLLMClient(["done"]),
            memory=MemoryManager(KeywordMemory()),
            tools=ToolExecutor(registry),
            hooks=HookManager(),
            context=ContextManager(),
            sandbox=Sandbox(d),
            shell=ShellExecutor(d),
            session_store=SessionStore(d),
            run_store=RunStore(str(Path(d) / ".echo" / "sessions" / "test-session")),
            protocol_manager=manager,
            max_steps=1,
        )

        answer = loop.run("handle protocol")

        injected = [
            block.text
            for msg in loop.messages
            for block in msg.get("content", [])
            if hasattr(block, "text") and "## Runtime Events" in block.text
        ]

        assert answer == "done"
        assert injected
        assert "[protocol/resolved]" in injected[0]
        assert request.request_id in injected[0]
        assert "approved" in injected[0]
        assert loop._last_state.pending_protocols == []
