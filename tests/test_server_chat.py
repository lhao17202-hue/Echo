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


class FakeEchoRuntime:
    def __init__(self):
        self.calls = []

    def ask(self, message: str) -> str:
        self.calls.append(("ask", message))
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
    assert response.session_id
    assert response.run_id


def test_default_echo_service_calls_resume_with_session_id():
    runtime = FakeEchoRuntime()
    service = DefaultEchoService(runtime=runtime)

    response = service.chat("continue", session_id="session_123")

    assert runtime.calls == [("resume", "session_123", "continue")]
    assert response.answer == "resumed answer"
    assert response.session_id == "session_123"


class FailingEchoRuntime:
    def ask(self, message: str) -> str:
        return "Stopped: model_error"


def test_default_echo_service_marks_stopped_model_errors_as_failed():
    service = DefaultEchoService(runtime=FailingEchoRuntime())

    response = service.chat("hello")

    assert response.answer == "Stopped: model_error"
    assert response.status == "failed"
