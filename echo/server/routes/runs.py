"""Run endpoints for Echo Web."""

from fastapi import APIRouter, Depends

from echo.server.dependencies import EchoService, get_echo_service
from echo.server.schemas import TraceEventDTO

router = APIRouter(prefix="/api", tags=["runs"])


@router.get("/runs/{run_id}/trace", response_model=list[TraceEventDTO])
def get_run_trace(run_id: str, service: EchoService = Depends(get_echo_service)) -> list[TraceEventDTO]:
    return service.get_run_trace(run_id)
