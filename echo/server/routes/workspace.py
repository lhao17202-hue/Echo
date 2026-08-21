"""Workspace endpoints for Echo Web V3."""

from fastapi import APIRouter, Depends

from echo.server.dependencies import EchoService, get_echo_service
from echo.server.schemas import ApprovalPolicyUpdateRequest, ConfigSummary, GitStatus, RuntimeStatus, WorkspaceInfo

router = APIRouter(prefix="/api", tags=["workspace"])


@router.get("/workspace", response_model=WorkspaceInfo)
def get_workspace_info(service: EchoService = Depends(get_echo_service)) -> WorkspaceInfo:
    return service.get_workspace_info()


@router.get("/git/status", response_model=GitStatus)
def get_git_status(service: EchoService = Depends(get_echo_service)) -> GitStatus:
    return service.get_git_status()


@router.get("/config", response_model=ConfigSummary)
def get_config_summary(service: EchoService = Depends(get_echo_service)) -> ConfigSummary:
    return service.get_config_summary()


@router.patch("/config/approval", response_model=ConfigSummary)
def update_approval_policy(
    request: ApprovalPolicyUpdateRequest,
    service: EchoService = Depends(get_echo_service),
) -> ConfigSummary:
    return service.update_approval_policy(request.approval_policy)


@router.get("/runtime/status", response_model=RuntimeStatus)
def get_runtime_status(service: EchoService = Depends(get_echo_service)) -> RuntimeStatus:
    return service.get_runtime_status()
