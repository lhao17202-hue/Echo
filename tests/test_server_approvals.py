from fastapi.testclient import TestClient

from echo.server.app import create_app
from echo.server.dependencies import EchoService, get_echo_service
from echo.server.schemas import ApprovalDecisionResponse, ApprovalRequestDTO


class FakeApprovalService(EchoService):
    def __init__(self):
        self.decisions = []

    def pending_approvals(self) -> list[ApprovalRequestDTO]:
        return [
            ApprovalRequestDTO(
                request_id="approval_1",
                tool_name="run_shell",
                risk_level="danger",
                tool_input={"command": "python --version"},
                command="python --version",
            )
        ]

    def decide_approval(self, request_id: str, approved: bool) -> ApprovalDecisionResponse:
        self.decisions.append((request_id, approved))
        return ApprovalDecisionResponse(
            request_id=request_id,
            status="approved" if approved else "denied",
        )


def test_pending_approvals_endpoint_returns_waiting_tool_requests():
    service = FakeApprovalService()
    app = create_app()
    app.dependency_overrides[get_echo_service] = lambda: service
    client = TestClient(app)

    response = client.get("/api/approvals/pending")

    assert response.status_code == 200
    body = response.json()
    assert body == [
        {
            "request_id": "approval_1",
            "tool_name": "run_shell",
            "risk_level": "danger",
            "tool_input": {"command": "python --version"},
            "command": "python --version",
            "status": "pending",
        }
    ]


def test_approval_decision_endpoint_forwards_yes_no_to_service():
    service = FakeApprovalService()
    app = create_app()
    app.dependency_overrides[get_echo_service] = lambda: service
    client = TestClient(app)

    response = client.post("/api/approvals/approval_1/decision", json={"approved": False})

    assert response.status_code == 200
    assert response.json() == {"request_id": "approval_1", "status": "denied"}
    assert service.decisions == [("approval_1", False)]
