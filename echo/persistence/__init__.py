"""Persistence — session, run, checkpoint, trace."""

from echo.persistence.session_store import Session, SessionStore
from echo.persistence.run_store import RunStore
from echo.persistence.checkpoint import (
    CheckpointManager,
    CHECKPOINT_FULL_VALID,
    CHECKPOINT_PARTIAL_STALE,
    CHECKPOINT_WORKSPACE_MISMATCH,
    CHECKPOINT_NONE,
)
from echo.persistence.trace import (
    TraceEvent, TraceEmitter, TraceReader, read_jsonl, EventType,
)

__all__ = [
    "Session", "SessionStore",
    "RunStore",
    "CheckpointManager",
    "CHECKPOINT_FULL_VALID", "CHECKPOINT_PARTIAL_STALE",
    "CHECKPOINT_WORKSPACE_MISMATCH", "CHECKPOINT_NONE",
    "TraceEvent", "TraceEmitter", "TraceReader", "read_jsonl", "EventType",
]
