import tempfile
from pathlib import Path

from echo.config import EchoConfig
from echo.core.echo import Echo
from echo.providers.fake_client import FakeLLMClient
from echo.scheduler.cron_scheduler import CronJob


class FakeScheduler:
    def __init__(self, jobs=None):
        self.jobs = list(jobs or [])
        self.started = False
        self.stopped = False

    def start(self):
        self.started = True

    def stop(self):
        self.stopped = True

    def consume(self):
        jobs = list(self.jobs)
        self.jobs.clear()
        return jobs


def test_agent_loop_consumes_due_cron_jobs_as_runtime_events():
    from echo.tools.registry import ToolRegistry
    from echo.tools.executor import ToolExecutor
    from echo.hooks.base import HookManager
    from echo.core.context_manager import ContextManager
    from echo.core.agent_loop import AgentLoop
    from echo.memory.base import MemoryManager
    from echo.memory.default import KeywordMemory
    from echo.security.sandbox import Sandbox
    from echo.security.env_filter import ShellExecutor
    from echo.persistence.session_store import SessionStore
    from echo.persistence.run_store import RunStore

    with tempfile.TemporaryDirectory() as d:
        registry = ToolRegistry().discover("echo.tools.builtin")
        scheduler = FakeScheduler([
            CronJob(job_id="cron_1", cron_expr="* * * * *", prompt="check status"),
        ])
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
            scheduler=scheduler,
        )

        loop.run("handle cron")

        injected = [
            block.text
            for msg in loop.messages
            for block in msg.get("content", [])
            if hasattr(block, "text") and "## Runtime Events" in block.text
        ]
        assert injected
        assert "[cron/fired]" in injected[0]
        assert "job_id=cron_1" in injected[0]
        assert "check status" in injected[0]


def test_echo_facade_starts_and_stops_scheduler_when_cron_enabled(monkeypatch):
    from echo.core import echo as echo_module

    created = []

    def factory():
        scheduler = FakeScheduler()
        created.append(scheduler)
        return scheduler

    monkeypatch.setattr(echo_module, "CronScheduler", factory)
    monkeypatch.setattr(echo_module, "AnthropicClient", lambda **_kwargs: FakeLLMClient(["done"]))

    with tempfile.TemporaryDirectory() as d:
        config = EchoConfig(provider="anthropic", model="fake-model", api_key="fake-key", enable_cron=True)
        echo = Echo(workspace_root=d, config=config)

        assert created
        assert created[0].started is True
        echo.close()
        assert created[0].stopped is True
