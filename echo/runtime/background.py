"""Background task runtime for Echo."""

from __future__ import annotations

import subprocess
import threading
import time
from dataclasses import dataclass, field
from typing import Any

from echo.runtime.events import RuntimeEvent


@dataclass
class BackgroundTask:
    bg_id: str
    kind: str
    command: str
    cwd: str
    timeout_seconds: float
    status: str = "running"
    result: str = ""
    error: str = ""
    started_at: float = field(default_factory=time.time)
    completed_at: float | None = None


class BackgroundManager:
    def __init__(self):
        self._tasks: dict[str, BackgroundTask] = {}
        self._completed: list[RuntimeEvent] = []
        self._lock = threading.Lock()
        self._counter = 0

    def start_shell_task(self, command: str, cwd: str = ".", timeout_seconds: float = 20.0) -> str:
        with self._lock:
            self._counter += 1
            bg_id = f"bg_{self._counter:04d}"
            task = BackgroundTask(bg_id=bg_id, kind="shell", command=command, cwd=cwd, timeout_seconds=timeout_seconds)
            self._tasks[bg_id] = task

        thread = threading.Thread(target=self._run_shell_task, args=(task,), daemon=True)
        thread.start()
        return bg_id

    def poll_completed(self) -> list[RuntimeEvent]:
        with self._lock:
            events = list(self._completed)
            self._completed.clear()
        return events

    def list(self) -> list[BackgroundTask]:
        with self._lock:
            return list(self._tasks.values())

    def get(self, bg_id: str) -> BackgroundTask | None:
        with self._lock:
            return self._tasks.get(bg_id)

    def _run_shell_task(self, task: BackgroundTask) -> None:
        try:
            result = subprocess.run(
                task.command,
                cwd=task.cwd,
                shell=True,
                capture_output=True,
                text=True,
                timeout=task.timeout_seconds,
            )
            output = (result.stdout or "").strip()
            error = (result.stderr or "").strip()
            with self._lock:
                task.status = "completed" if result.returncode == 0 else "failed"
                task.result = output
                task.error = error
                task.completed_at = time.time()
                self._completed.append(RuntimeEvent(
                    source="background",
                    event_type="completed" if result.returncode == 0 else "failed",
                    content=output or error or f"exit code {result.returncode}",
                    metadata={"bg_id": task.bg_id, "kind": task.kind, "command": task.command},
                ))
        except Exception as exc:
            with self._lock:
                task.status = "failed"
                task.error = str(exc)
                task.completed_at = time.time()
                self._completed.append(RuntimeEvent(
                    source="background",
                    event_type="failed",
                    content=str(exc),
                    metadata={"bg_id": task.bg_id, "kind": task.kind, "command": task.command},
                ))
