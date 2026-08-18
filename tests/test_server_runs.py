from fastapi.testclient import TestClient

from echo.server.app import create_app
from echo.server.dependencies import EchoService, get_echo_service
from echo.server.schemas import ChatResponse, TraceEventDTO


class FakeRunService(EchoService):
    def chat(self, message: str, session_id: str | None = None) -> ChatResponse:
        raise AssertionError("not used")

    def get_run_trace(self, run_id: str) -> list[TraceEventDTO]:
        return [
            TraceEventDTO(event="run_started", run_id=run_id, payload={"status": "running"}),
            TraceEventDTO(event="tool_executed", run_id=run_id, payload={"tools": [{"name": "read_file"}]}),
        ]


def test_get_run_trace_returns_timeline_events():
    app = create_app()
    app.dependency_overrides[get_echo_service] = lambda: FakeRunService()
    client = TestClient(app)

    response = client.get("/api/runs/run_1/trace")

    assert response.status_code == 200
    assert response.json()[0]["event"] == "run_started"
    assert response.json()[1]["event"] == "tool_executed"
    assert response.json()[1]["payload"] == {"tools": [{"name": "read_file"}]}
