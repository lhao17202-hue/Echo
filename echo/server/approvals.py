"""Web approval request coordination."""

from __future__ import annotations

from dataclasses import dataclass, field
from threading import Condition
from uuid import uuid4
from typing import Any


@dataclass
class ApprovalRequest:
    request_id: str
    tool_name: str
    risk_level: str
    tool_input: dict[str, Any] = field(default_factory=dict)
    command: str = ""
    status: str = "pending"
    approved: bool | None = None


class WebApprovalManager:
    """Stores pending tool approvals and blocks until the web UI decides."""

    def __init__(self):
        self._condition = Condition()
        self._requests: dict[str, ApprovalRequest] = {}

    def request_approval(self, request: dict[str, Any]) -> bool:
        approval = ApprovalRequest(
            request_id=f"approval_{uuid4().hex[:8]}",
            tool_name=str(request.get("tool_name", "")),
            risk_level=str(request.get("risk_level", "")),
            tool_input=dict(request.get("tool_input") or {}),
            command=str(request.get("command", "")),
        )
        with self._condition:
            self._requests[approval.request_id] = approval
            self._condition.notify_all()
            while approval.status == "pending":
                self._condition.wait()
            return approval.approved is True

    def pending(self) -> list[ApprovalRequest]:
        with self._condition:
            return [request for request in self._requests.values() if request.status == "pending"]

    def decide(self, request_id: str, approved: bool) -> ApprovalRequest:
        with self._condition:
            request = self._requests[request_id]
            request.approved = bool(approved)
            request.status = "approved" if approved else "denied"
            self._condition.notify_all()
            return request
