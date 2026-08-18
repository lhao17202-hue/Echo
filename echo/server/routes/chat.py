"""Chat endpoint for Echo Web."""

from fastapi import APIRouter, Depends

from echo.server.dependencies import EchoService, get_echo_service
from echo.server.schemas import ChatRequest, ChatResponse

router = APIRouter(prefix="/api", tags=["chat"])


@router.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest, service: EchoService = Depends(get_echo_service)) -> ChatResponse:
    return service.chat(message=request.message, session_id=request.session_id)
