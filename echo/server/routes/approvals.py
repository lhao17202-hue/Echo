"""Approval endpoints for Echo Web."""

from fastapi import APIRouter, Depends, HTTPException

from echo.server.dependencies import EchoService, get_echo_service
from echo.server.schemas import ApprovalDecisionRequest, ApprovalDecisionResponse, ApprovalRequestDTO

router = APIRouter(prefix="/api/approvals", tags=["approvals"])


@router.get("/pending", response_model=list[ApprovalRequestDTO])
def pending_approvals(service: EchoService = Depends(get_echo_service)) -> list[ApprovalRequestDTO]:
    return service.pending_approvals()


@router.post("/{request_id}/decision", response_model=ApprovalDecisionResponse)
def decide_approval(
    request_id: str,
    decision: ApprovalDecisionRequest,
    service: EchoService = Depends(get_echo_service),
) -> ApprovalDecisionResponse:
    try:
        return service.decide_approval(request_id, decision.approved)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="approval request not found") from exc
