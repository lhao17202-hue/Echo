"""Session endpoints for Echo Web."""

from fastapi import APIRouter, Depends

from echo.server.dependencies import EchoService, get_echo_service
from echo.server.schemas import SessionDetail, SessionSummary

router = APIRouter(prefix="/api", tags=["sessions"])


@router.get("/sessions", response_model=list[SessionSummary])
def list_sessions(service: EchoService = Depends(get_echo_service)) -> list[SessionSummary]:
    return service.list_sessions()


@router.get("/sessions/{session_id}", response_model=SessionDetail)
def get_session(session_id: str, service: EchoService = Depends(get_echo_service)) -> SessionDetail:
    return service.get_session(session_id)
