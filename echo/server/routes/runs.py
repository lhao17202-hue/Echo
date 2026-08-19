"""Run endpoints for Echo Web."""

from fastapi import APIRouter, Depends

from echo.server.dependencies import EchoService, get_echo_service
from echo.server.schemas import RunFileDiff, RunFileSummary, TraceEventDTO

router = APIRouter(prefix="/api", tags=["runs"])


@router.get("/runs/{run_id}/trace", response_model=list[TraceEventDTO])
def get_run_trace(run_id: str, service: EchoService = Depends(get_echo_service)) -> list[TraceEventDTO]:
    return service.get_run_trace(run_id)


@router.get("/runs/{run_id}/files", response_model=list[RunFileSummary])
def get_run_files(run_id: str, service: EchoService = Depends(get_echo_service)) -> list[RunFileSummary]:
    return service.get_run_files(run_id)


@router.get("/runs/{run_id}/files/diff", response_model=RunFileDiff)
def get_run_file_diff(
    run_id: str,
    path: str,
    service: EchoService = Depends(get_echo_service),
) -> RunFileDiff:
    return service.get_run_file_diff(run_id, path)
