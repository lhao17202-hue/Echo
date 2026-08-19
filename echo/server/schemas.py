"""Pydantic schemas for Echo Web API."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    ok: bool
    name: str
    version: str


class ChatRequest(BaseModel):
    message: str = Field(min_length=1)
    session_id: str | None = None


class TraceEventDTO(BaseModel):
    event: str
    run_id: str | None = None
    event_id: str | None = None
    created_at: str | None = None
    timestamp: float | None = None
    payload: dict[str, Any] = Field(default_factory=dict)


class ToolCallSummary(BaseModel):
    name: str
    input_summary: str = ""
    success: bool | None = None
    output_summary: str = ""


class ChatResponse(BaseModel):
    session_id: str
    run_id: str
    answer: str
    status: str
    trace: list[TraceEventDTO] = Field(default_factory=list)
    tools: list[ToolCallSummary] = Field(default_factory=list)
    files_touched: list[str] = Field(default_factory=list)


class SessionSummary(BaseModel):
    session_id: str
    title: str
    updated_at: str | None = None
    run_count: int = 0


class MessageDTO(BaseModel):
    role: str
    content: str


class SessionDetail(BaseModel):
    session_id: str
    title: str
    messages: list[MessageDTO] = Field(default_factory=list)


class SessionUpdateRequest(BaseModel):
    title: str = Field(min_length=1, max_length=120)


class WorkspaceInfo(BaseModel):
    name: str
    root: str


class GitStatus(BaseModel):
    branch: str
    dirty: bool
    changed_files: list[str] = Field(default_factory=list)


class ConfigSummary(BaseModel):
    provider: str
    model: str
    base_url: str
    approval_policy: str
    api_key_configured: bool


class RuntimeStatus(BaseModel):
    background_tasks: int = 0
    cron_tasks: int = 0
    mcp_servers: int = 0
    tools: int = 0
    approval_policy: str = "auto"


class RunFileSummary(BaseModel):
    path: str
    status: str = "modified"


class RunFileDiff(BaseModel):
    path: str
    status: str
    diff: str
