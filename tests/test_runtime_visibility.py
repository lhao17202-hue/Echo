from pathlib import Path

from echo.core.context_manager import ContextManager
from echo.core.task_state import TaskState
from echo.security.sandbox import Sandbox
from echo.tools.registry import ToolRegistry


class _FakeBackgroundManager:
    def list(self):
        return [type("Task", (), {"bg_id": "bg_1", "status": "running", "command": "python job.py"})()]


class _FakeProtocolManager:
    def pending(self):
        return [type("Request", (), {"request_id": "proto_1", "protocol_type": "approval", "prompt": "Approve deploy"})()]


class _FakeMcpManager:
    def snapshot(self):
        return {"servers": [{"name": "demo", "status": "running", "registered_tools": ["mcp_demo_get_status"], "last_error": ""}]}


def test_context_manager_renders_runtime_state_sections(tmp_path):
    registry = ToolRegistry().discover("echo.tools.builtin")
    state = TaskState.create("inspect runtime")
    state.active_background_tasks.append("bg_1")
    state.pending_protocols.append("proto_1")

    system = ContextManager().build_system(
        state,
        registry,
        memory=type("Memory", (), {
            "render_working": lambda self: "",
            "retrieve": lambda self, *_args, **_kwargs: [],
            "relevant_for_prompt": lambda self, *_args, **_kwargs: "",
        })(),
        sandbox=Sandbox(str(tmp_path)),
        background_manager=_FakeBackgroundManager(),
        protocol_manager=_FakeProtocolManager(),
        mcp_manager=_FakeMcpManager(),
    )

    assert "## Runtime State" in system
    assert "bg_1" in system
    assert "python job.py" in system
    assert "proto_1" in system
    assert "Approve deploy" in system
    assert "## MCP Servers" in system
    assert "demo" in system
    assert "mcp_demo_get_status" in system
