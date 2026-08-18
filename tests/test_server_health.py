from fastapi.testclient import TestClient

from echo.server.app import create_app


def test_health_endpoint_reports_echo_server_ready():
    client = TestClient(create_app())

    response = client.get("/api/health")

    assert response.status_code == 200
    assert response.json() == {
        "ok": True,
        "name": "Echo",
        "version": "0.1.0",
    }
