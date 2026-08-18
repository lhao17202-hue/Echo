import tempfile

from echo.config import EchoConfig
from echo.core.echo import Echo
from echo.providers.fake_client import FakeLLMClient


class _FakeTeammates:
    def __init__(self):
        self.stopped = False

    def stop_all(self):
        self.stopped = True


class _FakeMcpManager:
    def __init__(self):
        self.closed = False

    def close(self):
        self.closed = True

    def snapshot(self):
        return {"servers": []}


def test_echo_passes_runtime_managers_to_agent_loop(monkeypatch):
    from echo.core import echo as echo_module

    captured = {}
    real_agent_loop = echo_module.AgentLoop

    class CapturingAgentLoop(real_agent_loop):
        def __init__(self, *args, **kwargs):
            captured["background_manager"] = kwargs.get("background_manager")
            captured["protocol_manager"] = kwargs.get("protocol_manager")
            captured["mcp_manager"] = kwargs.get("mcp_manager")
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
        assert captured["protocol_manager"] is echo.protocol_manager
        assert captured["mcp_manager"] is echo.mcp_manager


def test_echo_close_stops_teammates_and_mcp(monkeypatch):
    from echo.core import echo as echo_module

    monkeypatch.setattr(echo_module, "AnthropicClient", lambda **_kwargs: FakeLLMClient(["done"]))

    with tempfile.TemporaryDirectory() as d:
        config = EchoConfig(provider="anthropic", model="fake-model", api_key="fake-key")
        echo = Echo(workspace_root=d, config=config)
        teammates = _FakeTeammates()
        mcp = _FakeMcpManager()
        echo.teammates = teammates
        echo.mcp_manager = mcp

        echo.close()

        assert teammates.stopped is True
        assert mcp.closed is True
