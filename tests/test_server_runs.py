import json
from pathlib import Path

from fastapi.testclient import TestClient

from echo.server.app import create_app
from echo.server.dependencies import DefaultEchoService, EchoService, get_echo_service
from echo.server.schemas import ChatResponse, RunFileDiff, RunFileSummary, TraceEventDTO


class FakeRunService(EchoService):
    def chat(self, message: str, session_id: str | None = None) -> ChatResponse:
        raise AssertionError("not used")

    def get_run_trace(self, run_id: str) -> list[TraceEventDTO]:
        return [
            TraceEventDTO(event="run_started", run_id=run_id, payload={"status": "running"}),
            TraceEventDTO(event="tool_executed", run_id=run_id, payload={"tools": [{"name": "read_file"}]}),
        ]

    def get_run_files(self, run_id: str) -> list[RunFileSummary]:
        return [RunFileSummary(path="hello.txt", status="modified")]

    def get_run_file_diff(self, run_id: str, file_path: str) -> RunFileDiff:
        return RunFileDiff(path=file_path, status="modified", diff="diff --git a/hello.txt b/hello.txt")



def test_get_run_trace_returns_timeline_events():
    app = create_app()
    app.dependency_overrides[get_echo_service] = lambda: FakeRunService()
    client = TestClient(app)

    response = client.get("/api/runs/run_1/trace")

    assert response.status_code == 200
    assert response.json()[0]["event"] == "run_started"
    assert response.json()[1]["event"] == "tool_executed"
    assert response.json()[1]["payload"] == {"tools": [{"name": "read_file"}]}


def test_get_run_files_returns_file_summaries():
    app = create_app()
    app.dependency_overrides[get_echo_service] = lambda: FakeRunService()
    client = TestClient(app)

    response = client.get("/api/runs/run_1/files")

    assert response.status_code == 200
    assert response.json() == [{"path": "hello.txt", "status": "modified"}]


def test_get_run_file_diff_returns_diff_text():
    app = create_app()
    app.dependency_overrides[get_echo_service] = lambda: FakeRunService()
    client = TestClient(app)

    response = client.get("/api/runs/run_1/files/diff", params={"path": "hello.txt"})

    assert response.status_code == 200
    assert response.json()["path"] == "hello.txt"
    assert response.json()["diff"].startswith("diff --git")


def test_default_echo_service_reads_persisted_run_trace(tmp_path: Path):
    trace_dir = tmp_path / ".echo" / "sessions" / "session_1" / "runs" / "run_1"
    trace_dir.mkdir(parents=True)
    (trace_dir / "trace.jsonl").write_text(
        "\n".join(
            [
                json.dumps({"event": "run_started", "run_id": "run_1", "created_at": "2026-08-18T10:00:00", "request": "hello"}),
                json.dumps(
                    {
                        "event": "tool_executed",
                        "run_id": "run_1",
                        "tools": [{"name": "write_file", "input_summary": "hello.txt"}],
                        "file_changes": ["hello.txt"],
                    }
                ),
            ]
        ),
        encoding="utf-8",
    )
    runtime = type("FakeRuntime", (), {"workspace_root": tmp_path})()
    service = DefaultEchoService(runtime=runtime)

    trace = service.get_run_trace("run_1")

    assert [event.event for event in trace] == ["run_started", "tool_executed"]
    assert trace[0].payload == {"request": "hello"}
    assert trace[1].payload == {
        "tools": [{"name": "write_file", "input_summary": "hello.txt"}],
        "file_changes": ["hello.txt"],
    }


def test_default_echo_service_gets_run_files_from_trace(tmp_path: Path):
    trace_dir = tmp_path / ".echo" / "sessions" / "session_1" / "runs" / "run_1"
    trace_dir.mkdir(parents=True)
    (trace_dir / "trace.jsonl").write_text(
        json.dumps({"event": "tool_executed", "run_id": "run_1", "file_changes": ["hello.txt", "hello.txt"]}),
        encoding="utf-8",
    )
    runtime = type("FakeRuntime", (), {"workspace_root": tmp_path})()
    service = DefaultEchoService(runtime=runtime)

    assert service.get_run_files("run_1") == [RunFileSummary(path="hello.txt", status="modified")]


def test_default_echo_service_returns_file_preview_when_git_diff_is_empty(tmp_path: Path):
    (tmp_path / "hello.txt").write_text("hello", encoding="utf-8")
    runtime = type("FakeRuntime", (), {"workspace_root": tmp_path})()
    service = DefaultEchoService(runtime=runtime)

    diff = service.get_run_file_diff("run_1", "hello.txt")

    assert diff == RunFileDiff(path="hello.txt", status="current", diff="hello")


def test_default_echo_service_marks_missing_file_diff(tmp_path: Path):
    runtime = type("FakeRuntime", (), {"workspace_root": tmp_path})()
    service = DefaultEchoService(runtime=runtime)

    diff = service.get_run_file_diff("run_1", "missing.txt")

    assert diff == RunFileDiff(path="missing.txt", status="missing", diff="")
