from fastapi.testclient import TestClient

from echo.server.app import create_app
from echo.server.dependencies import EchoService, get_echo_service
from echo.server.schemas import ConfigSummary, GitStatus, RuntimeStatus, WorkspaceInfo


class FakeWorkspaceService(EchoService):
    def chat(self, message: str, session_id: str | None = None):
        raise AssertionError("not used")

    def get_workspace_info(self) -> WorkspaceInfo:
        return WorkspaceInfo(name="Echo", root="C:/Users/lihm/Desktop/Echo")

    def get_git_status(self) -> GitStatus:
        return GitStatus(branch="master", dirty=True, changed_files=["frontend/src/App.tsx"])

    def get_config_summary(self) -> ConfigSummary:
        return ConfigSummary(
            provider="deepseek",
            model="deepseek-v4-pro",
            base_url="https://api.deepseek.com/anthropic",
            approval_policy="auto",
            api_key_configured=True,
        )

    def get_runtime_status(self) -> RuntimeStatus:
        return RuntimeStatus(background_tasks=1, cron_tasks=0, mcp_servers=2, tools=12, approval_policy="auto")


def test_workspace_endpoint_returns_current_workspace():
    app = create_app()
    app.dependency_overrides[get_echo_service] = lambda: FakeWorkspaceService()
    client = TestClient(app)

    response = client.get("/api/workspace")

    assert response.status_code == 200
    assert response.json() == {"name": "Echo", "root": "C:/Users/lihm/Desktop/Echo"}


def test_git_status_endpoint_returns_branch_and_changes():
    app = create_app()
    app.dependency_overrides[get_echo_service] = lambda: FakeWorkspaceService()
    client = TestClient(app)

    response = client.get("/api/git/status")

    assert response.status_code == 200
    assert response.json() == {"branch": "master", "dirty": True, "changed_files": ["frontend/src/App.tsx"]}


def test_config_endpoint_redacts_secret_values():
    app = create_app()
    app.dependency_overrides[get_echo_service] = lambda: FakeWorkspaceService()
    client = TestClient(app)

    response = client.get("/api/config")

    assert response.status_code == 200
    assert response.json()["api_key_configured"] is True
    assert "api_key" not in response.json()


def test_runtime_status_endpoint_returns_counts():
    app = create_app()
    app.dependency_overrides[get_echo_service] = lambda: FakeWorkspaceService()
    client = TestClient(app)

    response = client.get("/api/runtime/status")

    assert response.status_code == 200
    assert response.json()["tools"] == 12
