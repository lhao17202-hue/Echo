"""Dependency providers for Echo Web API."""

from __future__ import annotations

from uuid import uuid4

from echo.server.schemas import ChatResponse, SessionDetail, SessionSummary, TraceEventDTO


class EchoService:
    """Thin service boundary around the Echo runtime."""

    def chat(self, message: str, session_id: str | None = None) -> ChatResponse:
        raise NotImplementedError

    def list_sessions(self) -> list[SessionSummary]:
        raise NotImplementedError

    def get_session(self, session_id: str) -> SessionDetail:
        raise NotImplementedError

    def get_run_trace(self, run_id: str) -> list[TraceEventDTO]:
        raise NotImplementedError


class DefaultEchoService(EchoService):
    def __init__(self, runtime=None):
        self.runtime = runtime

    def chat(self, message: str, session_id: str | None = None) -> ChatResponse:
        if self.runtime is None:
            from echo.core.echo import Echo

            self.runtime = Echo()

        if session_id:
            answer = self.runtime.resume(session_id, message)
            resolved_session_id = session_id
        else:
            answer = self.runtime.ask(message)
            resolved_session_id = getattr(getattr(self.runtime, "session", None), "session_id", "") or f"session_{uuid4().hex[:8]}"

        run_id = getattr(getattr(self.runtime, "run_store", None), "current_run_id", "") or f"run_{uuid4().hex[:8]}"
        answer_text = str(answer)
        status = "failed" if answer_text.startswith("Stopped:") else "completed"
        return ChatResponse(
            session_id=resolved_session_id,
            run_id=run_id,
            answer=answer_text,
            status=status,
            trace=[],
            tools=[],
            files_touched=[],
        )

    def list_sessions(self) -> list[SessionSummary]:
        return []

    def get_session(self, session_id: str) -> SessionDetail:
        return SessionDetail(session_id=session_id, title=session_id, messages=[])

    def get_run_trace(self, run_id: str) -> list[TraceEventDTO]:
        return []


_service: DefaultEchoService | None = None


def get_echo_service() -> EchoService:
    global _service
    if _service is None:
        _service = DefaultEchoService()
    return _service
