from fastapi.testclient import TestClient

from echo.server.app import create_app
from echo.server.dependencies import EchoService, get_echo_service
from echo.server.schemas import ChatResponse, MessageDTO, SessionDetail, SessionSummary


class FakeSessionService(EchoService):
    def chat(self, message: str, session_id: str | None = None) -> ChatResponse:
        raise AssertionError("not used")

    def list_sessions(self) -> list[SessionSummary]:
        return [SessionSummary(session_id="session_1", title="First session", updated_at="2026-08-18T10:00:00", run_count=2)]

    def get_session(self, session_id: str) -> SessionDetail:
        return SessionDetail(
            session_id=session_id,
            title="First session",
            messages=[MessageDTO(role="user", content="hello"), MessageDTO(role="assistant", content="hi")],
        )


def test_list_sessions_returns_sidebar_summaries():
    app = create_app()
    app.dependency_overrides[get_echo_service] = lambda: FakeSessionService()
    client = TestClient(app)

    response = client.get("/api/sessions")

    assert response.status_code == 200
    assert response.json() == [
        {"session_id": "session_1", "title": "First session", "updated_at": "2026-08-18T10:00:00", "run_count": 2}
    ]


def test_get_session_returns_message_history():
    app = create_app()
    app.dependency_overrides[get_echo_service] = lambda: FakeSessionService()
    client = TestClient(app)

    response = client.get("/api/sessions/session_1")

    assert response.status_code == 200
    assert response.json()["session_id"] == "session_1"
    assert response.json()["messages"] == [
        {"role": "user", "content": "hello"},
        {"role": "assistant", "content": "hi"},
    ]
