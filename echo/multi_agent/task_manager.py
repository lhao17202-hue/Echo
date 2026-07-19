"""Global task manager — cross-agent task pool with locking."""

import json
import logging
import os
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger("echo.tasks")


@dataclass
class GlobalTask:
    task_id: str = field(default_factory=lambda: uuid.uuid4().hex[:8])
    subject: str = ""
    description: str = ""
    status: str = "pending"
    owner_agent: str | None = None
    blocked_by: list[str] = field(default_factory=list)
    worktree: str | None = None
    created_at: str = ""
    completed_at: str | None = None
    result: str = ""


class GlobalTaskManager:
    """跨 Agent 共享任务池。

    线程安全：所有写操作加 threading.Lock。
    存储：单文件 JSON（.echo/global/tasks.json）。
    """

    def __init__(self, storage_path: str = ""):
        self._path = Path(storage_path) if storage_path else None
        self._tasks: dict[str, GlobalTask] = {}
        self._lock = threading.Lock()
        if self._path and self._path.exists():
            self._load()

    def create(self, subject: str, description: str = "",
               blocked_by: list[str] | None = None,
               worktree: str | None = None) -> str:
        task = GlobalTask(
            subject=subject,
            description=description,
            blocked_by=blocked_by or [],
            worktree=worktree,
            created_at=time.strftime("%Y-%m-%dT%H:%M:%S"),
        )
        with self._lock:
            self._tasks[task.task_id] = task
            self._save()
        return task.task_id

    def claim(self, task_id: str, agent_name: str) -> bool:
        """认领任务。加锁保证原子性。"""
        with self._lock:
            task = self._tasks.get(task_id)
            if task and task.status == "pending":
                # 检查依赖
                for dep_id in task.blocked_by:
                    dep = self._tasks.get(dep_id)
                    if dep and dep.status != "completed":
                        return False
                task.status = "in_progress"
                task.owner_agent = agent_name
                self._save()
                return True
            return False

    def complete(self, task_id: str, result: str = "") -> None:
        with self._lock:
            task = self._tasks.get(task_id)
            if task:
                task.status = "completed"
                task.completed_at = time.strftime("%Y-%m-%dT%H:%M:%S")
                task.result = result
                self._save()

    def fail(self, task_id: str, error: str = "") -> None:
        with self._lock:
            task = self._tasks.get(task_id)
            if task:
                task.status = "failed"
                task.result = error
                self._save()

    def list_available(self) -> list[GlobalTask]:
        """列出可认领任务（pending + 依赖已满足）。"""
        available = []
        with self._lock:
            for task in self._tasks.values():
                if task.status != "pending":
                    continue
                if all(
                    self._tasks.get(d) and self._tasks[d].status == "completed"
                    for d in task.blocked_by
                ):
                    available.append(task)
        return available

    def get(self, task_id: str) -> GlobalTask | None:
        return self._tasks.get(task_id)

    def _save(self) -> None:
        if not self._path:
            return
        data = {}
        for tid, task in self._tasks.items():
            data[tid] = {
                "task_id": task.task_id,
                "subject": task.subject,
                "description": task.description,
                "status": task.status,
                "owner_agent": task.owner_agent,
                "blocked_by": task.blocked_by,
                "worktree": task.worktree,
                "created_at": task.created_at,
                "completed_at": task.completed_at,
                "result": task.result,
            }
        tmp = self._path.with_suffix(".tmp")
        tmp.parent.mkdir(parents=True, exist_ok=True)
        tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False))
        os.replace(tmp, self._path)

    def _load(self) -> None:
        try:
            data = json.loads(self._path.read_text())
            for tid, d in data.items():
                self._tasks[tid] = GlobalTask(**d)
        except Exception:
            pass

import os  # for os.replace
