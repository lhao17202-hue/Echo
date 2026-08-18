"""Tests for MCP stdio tool integration."""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest

from echo.mcp.adapter import McpToolAdapter, normalize_mcp_result
from echo.mcp.client import McpClientSession, McpToolDefinition
from echo.mcp.config import McpConfig, McpConfigError, McpServerConfig, load_mcp_config, normalize_mcp_name
from echo.mcp.manager import McpManager
from echo.tools.base import ToolContext
from echo.tools.registry import ToolRegistry


def test_missing_mcp_config_loads_empty_config(tmp_path):
    config = load_mcp_config(tmp_path / ".echo" / "mcp.json")

    assert config.servers == []


def test_claude_desktop_mcp_servers_config_loads_correctly(tmp_path):
    config_path = tmp_path / ".echo" / "mcp.json"
    config_path.parent.mkdir()
    config_path.write_text(
        json.dumps({
            "mcpServers": {
                "demo": {
                    "command": sys.executable,
                    "args": ["server.py", "--flag"],
                    "env": {"DEMO_TOKEN": "test-token"},
                }
            }
        }),
        encoding="utf-8",
    )

    config = load_mcp_config(config_path)

    assert len(config.servers) == 1
    server = config.servers[0]
    assert server.name == "demo"
    assert server.command == sys.executable
    assert server.args == ["server.py", "--flag"]
    assert server.env == {"DEMO_TOKEN": "test-token"}


def test_mcp_config_defaults_args_and_env(tmp_path):
    config_path = tmp_path / ".echo" / "mcp.json"
    config_path.parent.mkdir()
    config_path.write_text(
        json.dumps({"mcpServers": {"demo": {"command": "python"}}}),
        encoding="utf-8",
    )

    config = load_mcp_config(config_path)

    assert config.servers[0].args == []
    assert config.servers[0].env == {}


def test_mcp_config_rejects_invalid_json(tmp_path):
    config_path = tmp_path / ".echo" / "mcp.json"
    config_path.parent.mkdir()
    config_path.write_text("{not valid json", encoding="utf-8")

    with pytest.raises(McpConfigError, match="Invalid MCP config JSON"):
        load_mcp_config(config_path)


def test_mcp_config_requires_mcp_servers_object(tmp_path):
    config_path = tmp_path / ".echo" / "mcp.json"
    config_path.parent.mkdir()
    config_path.write_text(json.dumps({"mcpServers": []}), encoding="utf-8")

    with pytest.raises(McpConfigError, match="mcpServers must be an object"):
        load_mcp_config(config_path)


def test_mcp_config_validates_command_args_and_env_types(tmp_path):
    config_path = tmp_path / ".echo" / "mcp.json"
    config_path.parent.mkdir()

    config_path.write_text(json.dumps({"mcpServers": {"demo": {}}}), encoding="utf-8")
    with pytest.raises(McpConfigError, match="demo.command must be a non-empty string"):
        load_mcp_config(config_path)

    config_path.write_text(
        json.dumps({"mcpServers": {"demo": {"command": "python", "args": "bad"}}}),
        encoding="utf-8",
    )
    with pytest.raises(McpConfigError, match="demo.args must be a list of strings"):
        load_mcp_config(config_path)

    config_path.write_text(
        json.dumps({"mcpServers": {"demo": {"command": "python", "env": []}}}),
        encoding="utf-8",
    )
    with pytest.raises(McpConfigError, match="demo.env must be an object with string keys and values"):
        load_mcp_config(config_path)


def test_normalize_mcp_name_rules_and_empty_rejection():
    assert normalize_mcp_name("filesystem") == "filesystem"
    assert normalize_mcp_name("GitHub") == "github"
    assert normalize_mcp_name("create-pull-request") == "create_pull_request"
    assert normalize_mcp_name("  Weird Name!! ") == "weird_name"

    with pytest.raises(McpConfigError, match="normalizes to an empty name"):
        normalize_mcp_name("!!!")


@dataclass(frozen=True)
class _RemoteTool:
    name: str
    description: str
    input_schema: dict[str, Any]


class _FakeSession:
    def __init__(self, result):
        self.result = result
        self.calls = []

    def call_tool(self, tool_name, arguments):
        self.calls.append((tool_name, arguments))
        return self.result


def test_mcp_manager_snapshot_reports_configured_and_live_servers():
    manager = McpManager.from_file(Path("missing-mcp-config.json"))
    manager.config = type(manager.config)(servers=[_demo_server_config()])

    snapshot = manager.snapshot()

    assert snapshot["servers"][0]["name"] == "demo"
    assert snapshot["servers"][0]["status"] == "configured"
    assert snapshot["servers"][0]["registered_tools"] == []
    assert snapshot["servers"][0]["last_error"] == ""


def test_mcp_manager_snapshot_reports_registered_tools():
    registry = ToolRegistry()
    manager = McpManager(McpConfig(servers=[_demo_server_config()]))
    try:
        manager.register_tools(registry)
        snapshot = manager.snapshot()
    finally:
        manager.close()

    demo = snapshot["servers"][0]
    assert demo["name"] == "demo"
    assert demo["status"] == "running"
    assert "mcp_demo_get_status" in demo["registered_tools"]
    assert demo["last_error"] == ""


    remote = _RemoteTool(
        name="get_status",
        description="Return current status.",
        input_schema={
            "type": "object",
            "properties": {"verbose": {"type": "boolean"}},
            "required": ["verbose"],
        },
    )
    adapter = McpToolAdapter(
        echo_name="mcp_demo_get_status",
        server_name="demo",
        tool=remote,
        session=_FakeSession(result="ok"),
    )

    schema = adapter.to_schema()

    assert schema["name"] == "mcp_demo_get_status"
    assert schema["description"].startswith("[MCP: demo] Original tool: get_status")
    assert "Return current status." in schema["description"]
    assert schema["input_schema"] == remote.input_schema
    assert adapter.risk_level == "danger"
    assert adapter.is_read_only is False


def test_mcp_tool_adapter_execute_forwards_arguments_and_returns_text():
    session = _FakeSession(result={"content": [{"type": "text", "text": "status ok"}]})
    remote = _RemoteTool(name="get_status", description="", input_schema={"type": "object"})
    adapter = McpToolAdapter("mcp_demo_get_status", "demo", remote, session)

    result = adapter.run(ToolContext(), {"verbose": True})

    assert result.success
    assert result.output == "status ok"
    assert session.calls == [("get_status", {"verbose": True})]


def test_mcp_tool_adapter_execute_returns_fail_on_mcp_error_result():
    session = _FakeSession(result={"isError": True, "content": [{"type": "text", "text": "boom"}]})
    remote = _RemoteTool(name="explode", description="", input_schema={"type": "object"})
    adapter = McpToolAdapter("mcp_demo_explode", "demo", remote, session)

    result = adapter.run(ToolContext(), {})

    assert not result.success
    assert result.error == "MCP tool error: boom"


def test_mcp_tool_adapter_execute_returns_fail_when_session_raises():
    class RaisingSession:
        def call_tool(self, tool_name, arguments):
            raise RuntimeError("server unavailable")

    remote = _RemoteTool(name="get_status", description="", input_schema={"type": "object"})
    adapter = McpToolAdapter("mcp_demo_get_status", "demo", remote, RaisingSession())

    result = adapter.run(ToolContext(), {})

    assert not result.success
    assert result.error == "MCP tool error: server unavailable"


def test_normalize_mcp_result_handles_structured_and_unknown_blocks():
    assert normalize_mcp_result({"content": [{"type": "text", "text": "hello"}]}) == "hello"

    structured = normalize_mcp_result({"structuredContent": {"b": 2, "a": 1}})
    assert structured == '{\n  "a": 1,\n  "b": 2\n}'

    multiple = normalize_mcp_result({
        "content": [
            {"type": "text", "text": "first"},
            {"type": "image", "data": "..."},
        ]
    })
    assert multiple == "first\n\n[Unsupported MCP content block: image]"


def _demo_server_config(env=None):
    server_path = Path(__file__).parent / "fixtures" / "demo_mcp_server.py"
    return McpServerConfig(
        name="demo",
        command=sys.executable,
        args=[str(server_path)],
        env=env or {},
    )


def test_fake_stdio_server_initializes_and_lists_tools():
    session = McpClientSession(_demo_server_config())
    try:
        session.initialize()
        tools = session.list_tools()
    finally:
        session.close()

    assert any(tool.name == "get_status" for tool in tools)
    status = next(tool for tool in tools if tool.name == "get_status")
    assert isinstance(status, McpToolDefinition)
    assert status.description == "Return demo status."
    assert status.input_schema["type"] == "object"


def test_mcp_client_session_calls_tool_and_returns_result():
    session = McpClientSession(_demo_server_config())
    try:
        session.initialize()
        result = session.call_tool("get_status", {"verbose": True})
    finally:
        session.close()

    assert normalize_mcp_result(result) == "status ok verbose"


def test_mcp_client_session_passes_explicit_env_and_filters_host_env(monkeypatch):
    monkeypatch.setenv("HOST_SECRET", "should-not-leak")
    session = McpClientSession(_demo_server_config(env={"DEMO_TOKEN": "test-token"}))
    try:
        session.initialize()
        result = session.call_tool("read_env", {})
    finally:
        session.close()

    env_seen = json.loads(normalize_mcp_result(result))
    assert env_seen["DEMO_TOKEN"] == "test-token"
    assert env_seen["HOST_SECRET"] == ""
    assert env_seen["PATH_PRESENT"] is True


def _write_mcp_config(root: Path, servers: dict):
    config_path = root / ".echo" / "mcp.json"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(json.dumps({"mcpServers": servers}), encoding="utf-8")
    return config_path


def test_mcp_manager_registers_tools_with_prefixed_names(tmp_path):
    config_path = _write_mcp_config(tmp_path, {
        "demo": {
            "command": sys.executable,
            "args": [str(Path(__file__).parent / "fixtures" / "demo_mcp_server.py")],
        }
    })
    manager = McpManager.from_file(config_path)
    registry = ToolRegistry()
    try:
        manager.register_tools(registry)
        names = {schema["name"] for schema in registry.list_schemas()}
    finally:
        manager.close()

    assert "mcp_demo_get_status" in names
    assert "mcp_demo_echo_structured" in names


def test_mcp_manager_registered_tool_executes_through_registry(tmp_path):
    config_path = _write_mcp_config(tmp_path, {
        "demo": {
            "command": sys.executable,
            "args": [str(Path(__file__).parent / "fixtures" / "demo_mcp_server.py")],
        }
    })
    manager = McpManager.from_file(config_path)
    registry = ToolRegistry()
    try:
        manager.register_tools(registry)
        tool = registry.get("mcp_demo_get_status")
        result = tool.run(ToolContext(), {"verbose": False})
    finally:
        manager.close()

    assert result.success
    assert result.output == "status ok"


def test_mcp_manager_continues_when_one_server_fails_to_start(tmp_path):
    config_path = _write_mcp_config(tmp_path, {
        "bad": {"command": "definitely-missing-command-for-echo-tests"},
        "demo": {
            "command": sys.executable,
            "args": [str(Path(__file__).parent / "fixtures" / "demo_mcp_server.py")],
        },
    })
    manager = McpManager.from_file(config_path)
    registry = ToolRegistry()
    try:
        manager.register_tools(registry)
        assert registry.get("mcp_demo_get_status") is not None
        assert registry.get("mcp_bad_anything") is None
    finally:
        manager.close()


def test_mcp_manager_skips_server_when_tool_names_collide(tmp_path, monkeypatch):
    from echo.mcp import manager as manager_module

    class CollidingSession:
        def __init__(self, config):
            self.config = config

        def initialize(self):
            return None

        def list_tools(self):
            return [
                McpToolDefinition("same-name", "first", {"type": "object"}),
                McpToolDefinition("same_name", "second", {"type": "object"}),
            ]

        def close(self):
            return None

    monkeypatch.setattr(manager_module, "McpClientSession", CollidingSession)
    config_path = _write_mcp_config(tmp_path, {"demo": {"command": sys.executable}})
    manager = McpManager.from_file(config_path)
    registry = ToolRegistry()

    manager.register_tools(registry)

    assert registry.get("mcp_demo_same_name") is None


def test_mcp_manager_missing_config_registers_no_tools(tmp_path):
    manager = McpManager.from_file(tmp_path / ".echo" / "mcp.json")
    registry = ToolRegistry()

    manager.register_tools(registry)

    assert registry.list_schemas() == []


def test_mcp_tools_are_danger_and_not_read_only_by_default(tmp_path):
    config_path = _write_mcp_config(tmp_path, {
        "demo": {
            "command": sys.executable,
            "args": [str(Path(__file__).parent / "fixtures" / "demo_mcp_server.py")],
        }
    })
    manager = McpManager.from_file(config_path)
    registry = ToolRegistry()
    try:
        manager.register_tools(registry)
        tool = registry.get("mcp_demo_get_status")
    finally:
        manager.close()

    assert tool.risk_level == "danger"
    assert tool.is_read_only is False


def test_permission_hook_rejects_mcp_tool_when_policy_never(tmp_path):
    from echo.hooks.builtin import PermissionHook

    config_path = _write_mcp_config(tmp_path, {
        "demo": {
            "command": sys.executable,
            "args": [str(Path(__file__).parent / "fixtures" / "demo_mcp_server.py")],
        }
    })
    manager = McpManager.from_file(config_path)
    registry = ToolRegistry()
    try:
        manager.register_tools(registry)
        tool = registry.get("mcp_demo_get_status")
        message = PermissionHook().handle(
            tool=tool,
            tool_input={"verbose": True},
            approval_policy="never",
        )
    finally:
        manager.close()

    assert "需要显式授权" in message


def test_echo_registers_mcp_tools_from_workspace_config(tmp_path, monkeypatch):
    from echo.config import EchoConfig
    from echo.core.echo import Echo
    from echo.providers.fake_client import FakeLLMClient

    monkeypatch.setattr(
        "echo.core.echo.OllamaClient",
        lambda *args, **kwargs: FakeLLMClient(outputs=["done"]),
    )
    _write_mcp_config(tmp_path, {
        "demo": {
            "command": sys.executable,
            "args": [str(Path(__file__).parent / "fixtures" / "demo_mcp_server.py")],
        }
    })

    echo = Echo(workspace_root=str(tmp_path), config=EchoConfig(provider="ollama"))
    try:
        assert echo.tool_registry.get("mcp_demo_get_status") is not None
        assert echo.tool_registry.get("read_file") is not None
    finally:
        echo.close()


def test_echo_starts_without_mcp_config(tmp_path, monkeypatch):
    from echo.config import EchoConfig
    from echo.core.echo import Echo
    from echo.providers.fake_client import FakeLLMClient

    monkeypatch.setattr(
        "echo.core.echo.OllamaClient",
        lambda *args, **kwargs: FakeLLMClient(outputs=["done"]),
    )

    echo = Echo(workspace_root=str(tmp_path), config=EchoConfig(provider="ollama"))
    try:
        assert echo.tool_registry.get("read_file") is not None
        assert not any(schema["name"].startswith("mcp_") for schema in echo.tool_registry.list_schemas())
    finally:
        echo.close()


def test_agent_loop_can_use_mcp_tool_without_provider_specific_branch(tmp_path):
    from echo.core.agent_loop import AgentLoop
    from echo.core.context_manager import ContextManager
    from echo.hooks.base import HookManager
    from echo.hooks.builtin import LogHook, PermissionHook, PostLogHook, StatsHook
    from echo.memory.base import MemoryManager
    from echo.memory.default import KeywordMemory
    from echo.persistence.run_store import RunStore
    from echo.persistence.session_store import SessionStore
    from echo.providers.fake_client import FakeLLMClient
    from echo.security.env_filter import ShellExecutor
    from echo.security.sandbox import Sandbox
    from echo.tools.executor import ToolExecutor

    config_path = _write_mcp_config(tmp_path, {
        "demo": {
            "command": sys.executable,
            "args": [str(Path(__file__).parent / "fixtures" / "demo_mcp_server.py")],
        }
    })
    manager = McpManager.from_file(config_path)
    registry = ToolRegistry()
    registry.discover("echo.tools.builtin")
    manager.register_tools(registry)

    hooks = HookManager()
    hooks.register(PermissionHook(), priority=0)
    hooks.register(LogHook(), priority=100)
    hooks.register(PostLogHook(), priority=100)
    hooks.register(StatsHook(), priority=200)

    try:
        loop = AgentLoop(
            llm=FakeLLMClient(outputs=[
                '<tool name="mcp_demo_get_status" verbose="true" />',
                "Demo status is ok.",
            ]),
            memory=MemoryManager(KeywordMemory()),
            tools=ToolExecutor(registry),
            hooks=hooks,
            context=ContextManager(),
            sandbox=Sandbox(str(tmp_path)),
            shell=ShellExecutor(str(tmp_path)),
            session_store=SessionStore(str(tmp_path)),
            run_store=RunStore(str(tmp_path / ".echo" / "sessions" / "test-session")),
            max_steps=5,
            approval_policy="auto",
        )

        answer = loop.run("check demo status")
    finally:
        manager.close()

    assert "Demo status is ok" in answer
    assert "mcp_demo_get_status" in loop.llm.last_system
