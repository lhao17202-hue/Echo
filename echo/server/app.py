"""FastAPI app factory for Echo Web."""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from echo.server.routes.approvals import router as approvals_router
from echo.server.routes.chat import router as chat_router
from echo.server.routes.health import router as health_router
from echo.server.routes.runs import router as runs_router
from echo.server.routes.sessions import router as sessions_router
from echo.server.routes.workspace import router as workspace_router


def create_app() -> FastAPI:
    app = FastAPI(title="Echo Web API", version="0.1.0")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.include_router(health_router)
    app.include_router(approvals_router)
    app.include_router(chat_router)
    app.include_router(sessions_router)
    app.include_router(runs_router)
    app.include_router(workspace_router)
    return app
