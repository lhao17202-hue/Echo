import json
from pathlib import Path

from fastapi.testclient import TestClient

from echo.server.app import create_app
from echo.server.dependencies import EchoService, get_echo_service
from echo.server.schemas import ChatResponse, TraceEventDTO


class FakeEchoService(EchoService):
    def chat(self, message: str, session_id: str | None = None) -> ChatResponse:
        return ChatResponse(
            session_id=session_id or "session_fake",
            run_id="run_fake",
            answer=f"echo: {message}",
            status="completed",
            trace=[TraceEventDTO(event="run_started", run_id="run_fake")],
            tools=[],
            files_touched=[],
        )


def test_chat_endpoint_uses_injected_echo_service():
    app = create_app()
    app.dependency_overrides[get_echo_service] = lambda: FakeEchoService()
    client = TestClient(app)

    response = client.post("/api/chat", json={"message": "hello", "session_id": None})

    assert response.status_code == 200
    assert response.json()["session_id"] == "session_fake"
    assert response.json()["run_id"] == "run_fake"
    assert response.json()["answer"] == "echo: hello"
    assert response.json()["status"] == "completed"
    assert response.json()["trace"][0]["event"] == "run_started"


from echo.server.dependencies import DefaultEchoService


class FakeSessionStore:
    def __init__(self, latest_id: str = "session_real"):
        self.latest_id = latest_id

    def latest(self) -> str:
        return self.latest_id


class FakeEchoRuntime:
    def __init__(self):
        self.calls = []
        self.session_store = FakeSessionStore()
        self.run_store = type("FakeRunStore", (), {"current_run_id": "run_stale"})()
        self.config = type("FakeConfig", (), {"approval_policy": "ask"})()
        self.workspace_root: Path | None = None
        self.last_run_id = "run_real"

    def ask(self, message: str) -> str:
        self.calls.append(("ask", message))
        self.run_store.current_run_id = self.last_run_id
        return "final answer"

    def resume(self, session_id: str, message: str) -> str:
        self.calls.append(("resume", session_id, message))
        return "resumed answer"


def test_default_echo_service_calls_ask_without_session_id():
    runtime = FakeEchoRuntime()
    service = DefaultEchoService(runtime=runtime)

    response = service.chat("hello", session_id=None)

    assert runtime.calls == [("ask", "hello")]
    assert response.answer == "final answer"
    assert response.status == "completed"
    assert response.session_id == "session_real"
    assert response.run_id


def test_default_echo_service_uses_returned_session_id_for_resume():
    runtime = FakeEchoRuntime()
    service = DefaultEchoService(runtime=runtime)

    first = service.chat("hello", session_id=None)
    second = service.chat("continue", session_id=first.session_id)

    assert runtime.calls == [("ask", "hello"), ("resume", "session_real", "continue")]
    assert second.answer == "resumed answer"
    assert second.session_id == "session_real"


def test_default_echo_service_calls_resume_with_session_id():
    runtime = FakeEchoRuntime()
    service = DefaultEchoService(runtime=runtime)

    response = service.chat("continue", session_id="session_123")

    assert runtime.calls == [("resume", "session_123", "continue")]
    assert response.answer == "resumed answer"
    assert response.session_id == "session_123"


def test_default_echo_service_includes_trace_tools_and_files(tmp_path: Path):
    trace_dir = tmp_path / ".echo" / "sessions" / "session_real" / "runs" / "run_real"
    trace_dir.mkdir(parents=True)
    (trace_dir / "trace.jsonl").write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "event": "tool_started",
                        "run_id": "run_real",
                        "payload": {"tool": "write_file", "tool_call_id": "toolu_1", "input_summary": "hello.txt"},
                    }
                ),
                json.dumps(
                    {
                        "event": "tool_finished",
                        "run_id": "run_real",
                        "payload": {
                            "tool": "write_file",
                            "tool_call_id": "toolu_1",
                            "input_summary": "hello.txt",
                            "success": True,
                            "output_summary": "created",
                            "files_touched": ["hello.txt"],
                            "files_written": ["hello.txt"],
                        },
                    }
                ),
            ]
        ),
        encoding="utf-8",
    )
    runtime = FakeEchoRuntime()
    runtime.workspace_root = tmp_path
    service = DefaultEchoService(runtime=runtime)

    response = service.chat("hello", session_id=None)

    assert response.run_id == "run_real"
    assert [event.event for event in response.trace] == ["tool_started", "tool_finished"]
    assert response.trace[0].payload["tool"] == "write_file"
    assert response.tools[0].name == "write_file"
    assert response.tools[0].input_summary == "hello.txt"
    assert response.tools[0].success is True
    assert response.tools[0].output_summary == "created"
    assert response.files_touched == ["hello.txt"]


class MissingSessionRuntime(FakeEchoRuntime):
    def resume(self, session_id: str, message: str) -> str:
        self.calls.append(("resume", session_id, message))
        raise FileNotFoundError(session_id)


def test_default_echo_service_returns_failed_response_for_missing_session():
    runtime = MissingSessionRuntime()
    service = DefaultEchoService(runtime=runtime)

    response = service.chat("continue", session_id="missing_session")

    assert runtime.calls == [("resume", "missing_session", "continue")]
    assert response.session_id == "missing_session"
    assert response.status == "failed"
    assert "会话不存在" in response.answer


class FailingEchoRuntime:
    def __init__(self):
        self.session_store = FakeSessionStore("session_failed")

    def ask(self, message: str) -> str:
        return "Stopped: model_error"


def test_default_echo_service_marks_stopped_model_errors_as_failed():
    service = DefaultEchoService(runtime=FailingEchoRuntime())

    response = service.chat("hello")

    assert response.answer == "Stopped: model_error"
    assert response.status == "failed"


def test_default_echo_service_updates_runtime_approval_policy():
    runtime = FakeEchoRuntime()
    service = DefaultEchoService(runtime=runtime)

    response = service.update_approval_policy("danger")

    assert response.approval_policy == "danger"
    assert runtime.config.approval_policy == "danger"
