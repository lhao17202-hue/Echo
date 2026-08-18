"""Health endpoint for Echo Web."""

from fastapi import APIRouter

from echo.server.schemas import HealthResponse

router = APIRouter(prefix="/api", tags=["health"])


@router.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(ok=True, name="Echo", version="0.1.0")
