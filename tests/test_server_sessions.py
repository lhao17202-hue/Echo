from pathlib import Path

from fastapi.testclient import TestClient

from echo.server.app import create_app
from echo.server.dependencies import DefaultEchoService, EchoService, get_echo_service
from echo.server.schemas import ChatResponse, MessageDTO, SessionDetail, SessionSummary, SessionUpdateRequest


class FakeSessionService(EchoService):
    def chat(self, message: str, session_id: str | None = None) -> ChatResponse:
        raise AssertionError("not used")

    def list_sessions(self, query: str | None = None) -> list[SessionSummary]:
        sessions = [
            SessionSummary(session_id="session_1", title="First session", updated_at="2026-08-18T10:00:00", run_count=2),
            SessionSummary(session_id="session_2", title="Second note", updated_at="2026-08-18T11:00:00", run_count=1),
        ]
        if query:
            return [session for session in sessions if query.lower() in session.title.lower()]
        return sessions

    def get_session(self, session_id: str) -> SessionDetail:
        return SessionDetail(
            session_id=session_id,
            title="First session",
            messages=[MessageDTO(role="user", content="hello"), MessageDTO(role="assistant", content="hi")],
        )

    def rename_session(self, session_id: str, title: str) -> SessionSummary:
        return SessionSummary(session_id=session_id, title=title, updated_at="2026-08-18T10:00:00", run_count=2)

    def delete_session(self, session_id: str) -> None:
        return None
class FakeRuntimeSessionStore:
    def load(self, session_id: str):
        if session_id == "missing_session":
            raise FileNotFoundError(session_id)
        return type(
            "FakeSession",
            (),
            {
                "session_id": session_id,
                "history": [
                    {"role": "user", "content": [{"type": "text", "text": "hello"}]},
                    {
                        "role": "assistant",
                        "content": [
                            {"type": "tool_use", "name": "read_file", "input": {"path": "README.md"}},
                            {"type": "text", "text": "hi"},
                        ],
                    },
                    {"role": "user", "content": [{"type": "tool_result", "content": "hidden tool output"}]},
                ],
            },
        )()


class FakeSessionRuntime:
    def __init__(self, workspace_root: Path | None = None):
        self.session_store = FakeRuntimeSessionStore()
        self.workspace_root = workspace_root

    def list_sessions(self, limit: int = 20) -> list[dict]:
        return [
            {
                "session_id": "session_1",
                "created_at": "2026-08-18T09:00:00",
                "modified_at": "2026-08-18T10:00:00",
            }
        ]


def test_list_sessions_returns_sidebar_summaries():
    app = create_app()
    app.dependency_overrides[get_echo_service] = lambda: FakeSessionService()
    client = TestClient(app)

    response = client.get("/api/sessions")

    assert response.status_code == 200
    assert response.json() == [
        {"session_id": "session_1", "title": "First session", "updated_at": "2026-08-18T10:00:00", "run_count": 2},
        {"session_id": "session_2", "title": "Second note", "updated_at": "2026-08-18T11:00:00", "run_count": 1},
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


def test_list_sessions_filters_by_query():
    app = create_app()
    app.dependency_overrides[get_echo_service] = lambda: FakeSessionService()
    client = TestClient(app)

    response = client.get("/api/sessions?query=second")

    assert response.status_code == 200
    assert [session["session_id"] for session in response.json()] == ["session_2"]


def test_rename_session_returns_updated_summary():
    app = create_app()
    app.dependency_overrides[get_echo_service] = lambda: FakeSessionService()
    client = TestClient(app)

    response = client.patch("/api/sessions/session_1", json={"title": "Renamed"})

    assert response.status_code == 200
    assert response.json()["title"] == "Renamed"


def test_delete_session_returns_no_content():
    app = create_app()
    app.dependency_overrides[get_echo_service] = lambda: FakeSessionService()
    client = TestClient(app)

    response = client.delete("/api/sessions/session_1")

    assert response.status_code == 204


def test_default_echo_service_lists_runtime_sessions():
    service = DefaultEchoService(runtime=FakeSessionRuntime())

    sessions = service.list_sessions()

    assert sessions == [
        SessionSummary(session_id="session_1", title="hello", updated_at="2026-08-18T10:00:00", run_count=0)
    ]


def test_default_echo_service_returns_empty_detail_for_missing_session():
    service = DefaultEchoService(runtime=FakeSessionRuntime())

    session = service.get_session("missing_session")

    assert session == SessionDetail(session_id="missing_session", title="会话不存在或已失效", messages=[])


def test_default_echo_service_gets_text_history_without_tool_blocks():
    service = DefaultEchoService(runtime=FakeSessionRuntime())

    session = service.get_session("session_1")

    assert session.session_id == "session_1"
    assert session.title == "hello"
    assert session.messages == [
        MessageDTO(role="user", content="hello"),
        MessageDTO(role="assistant", content="hi"),
    ]


def test_default_echo_service_filters_session_summaries_by_query():
    service = DefaultEchoService(runtime=FakeSessionRuntime())

    assert service.list_sessions(query="hell") == [
        SessionSummary(session_id="session_1", title="hello", updated_at="2026-08-18T10:00:00", run_count=0)
    ]
    assert service.list_sessions(query="missing") == []


def test_default_echo_service_renames_session_with_web_metadata(tmp_path: Path):
    session_dir = tmp_path / ".echo" / "sessions" / "session_1"
    session_dir.mkdir(parents=True)
    service = DefaultEchoService(runtime=FakeSessionRuntime(workspace_root=tmp_path))

    renamed = service.rename_session("session_1", "Renamed title")

    assert renamed.title == "Renamed title"
    assert service.list_sessions()[0].title == "Renamed title"


def test_default_echo_service_deletes_session_directory(tmp_path: Path):
    session_dir = tmp_path / ".echo" / "sessions" / "session_1"
    session_dir.mkdir(parents=True)
    service = DefaultEchoService(runtime=FakeSessionRuntime(workspace_root=tmp_path))

    service.delete_session("session_1")

    assert not session_dir.exists()
