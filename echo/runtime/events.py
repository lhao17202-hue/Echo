"""Runtime event primitives for integrating external runtime sources into AgentLoop."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class RuntimeEvent:
    """A model-visible event collected from a runtime subsystem."""

    source: str
    event_type: str
    content: str
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_cron_job(cls, job: Any) -> "RuntimeEvent":
        """Convert a due CronJob into a model-visible runtime event."""
        return cls(
            source="cron",
            event_type="fired",
            content=str(getattr(job, "prompt", "")),
            metadata={
                "job_id": getattr(job, "job_id", ""),
                "cron_expr": getattr(job, "cron_expr", ""),
                "recurring": getattr(job, "recurring", True),
            },
        )


def render_runtime_events(events: list[RuntimeEvent]) -> str:
    """Render runtime events into a compact user-message block for the model."""
    if not events:
        return ""

    lines = ["## Runtime Events"]
    for event in events:
        metadata = " ".join(
            f"{key}={value}"
            for key, value in sorted(event.metadata.items())
            if value not in (None, "")
        )
        prefix = f"- [{event.source}/{event.event_type}]"
        if metadata:
            prefix += f" {metadata}:"
        else:
            prefix += ":"
        lines.append(f"{prefix} {event.content}")
    return "\n".join(lines)
