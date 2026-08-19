"""Session endpoints for Echo Web."""

from fastapi import APIRouter, Depends, Response, status

from echo.server.dependencies import EchoService, get_echo_service
from echo.server.schemas import SessionDetail, SessionSummary, SessionUpdateRequest

router = APIRouter(prefix="/api", tags=["sessions"])


@router.get("/sessions", response_model=list[SessionSummary])
def list_sessions(query: str | None = None, service: EchoService = Depends(get_echo_service)) -> list[SessionSummary]:
    return service.list_sessions(query=query)


@router.get("/sessions/{session_id}", response_model=SessionDetail)
def get_session(session_id: str, service: EchoService = Depends(get_echo_service)) -> SessionDetail:
    return service.get_session(session_id)


@router.patch("/sessions/{session_id}", response_model=SessionSummary)
def rename_session(
    session_id: str,
    request: SessionUpdateRequest,
    service: EchoService = Depends(get_echo_service),
) -> SessionSummary:
    return service.rename_session(session_id, request.title)


@router.delete("/sessions/{session_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_session(session_id: str, service: EchoService = Depends(get_echo_service)) -> Response:
    service.delete_session(session_id)
    return Response(status_code=status.HTTP_204_NO_CONTENT)
