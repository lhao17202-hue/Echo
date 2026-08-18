"""Protocol request runtime for Echo."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any
from uuid import uuid4

from echo.runtime.events import RuntimeEvent


@dataclass
class ProtocolRequest:
    request_id: str
    protocol_type: str
    status: str
    prompt: str
    payload: dict[str, Any] = field(default_factory=dict)
    result: str = ""
    created_at: float = field(default_factory=time.time)
    resolved_at: float | None = None


class ProtocolManager:
    def __init__(self):
        self._requests: dict[str, ProtocolRequest] = {}
        self._events: list[RuntimeEvent] = []

    def create(self, protocol_type: str, prompt: str, payload: dict[str, Any] | None = None) -> ProtocolRequest:
        request = ProtocolRequest(
            request_id=f"proto_{uuid4().hex[:8]}",
            protocol_type=protocol_type,
            status="pending",
            prompt=prompt,
            payload=dict(payload or {}),
        )
        self._requests[request.request_id] = request
        return request

    def resolve(self, request_id: str, result: str) -> ProtocolRequest:
        request = self._requests[request_id]
        request.status = "resolved"
        request.result = str(result)
        request.resolved_at = time.time()
        self._events.append(RuntimeEvent(
            source="protocol",
            event_type="resolved",
            content=request.result,
            metadata={
                "request_id": request.request_id,
                "protocol_type": request.protocol_type,
            },
        ))
        return request

    def poll_events(self) -> list[RuntimeEvent]:
        events = list(self._events)
        self._events.clear()
        return events

    def pending_ids(self) -> list[str]:
        return [
            request.request_id
            for request in self._requests.values()
            if request.status == "pending"
        ]

    def pending(self) -> list[ProtocolRequest]:
        return [
            request
            for request in self._requests.values()
            if request.status == "pending"
        ]
