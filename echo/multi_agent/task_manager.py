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
    run_id: str = ""  # the run that created/assigned this task (for trace routing)

    def to_dict(self) -> dict:
        return {
            "task_id": self.task_id,
            "subject": self.subject,
            "description": self.description,
            "status": self.status,
            "owner_agent": self.owner_agent,
            "blocked_by": list(self.blocked_by),
            "worktree": self.worktree,
            "created_at": self.created_at,
            "completed_at": self.completed_at,
            "result": self.result,
            "run_id": self.run_id,
        }


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
               worktree: str | None = None,
               run_id: str = "") -> str:
        task = GlobalTask(
            subject=subject,
            description=description,
            blocked_by=blocked_by or [],
            worktree=worktree,
            run_id=run_id,
            created_at=time.strftime("%Y-%m-%dT%H:%M:%S"),
        )
        with self._lock:
            self._tasks[task.task_id] = task
            self._save()
        return task.task_id

    def claim(self, task_id: str, agent_name: str) -> bool:
        """认领任务。加锁保证原子性。

        只有任务 owner 为 None 或与当前 agent 一致时才能认领，
        防止已分配给其他 agent 的任务被抢走。
        """
        with self._lock:
            task = self._tasks.get(task_id)
            if task and task.status == "pending":
                # 检查 owner：只能认领未分配或分配给自己的任务
                if task.owner_agent not in (None, agent_name):
                    return False
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

    def assign(self, task_id: str, agent_name: str) -> bool:
        """Assign a pending task to an agent without claiming it."""
        with self._lock:
            task = self._tasks.get(task_id)
            if not task or task.status != "pending":
                return False
            task.owner_agent = agent_name
            self._save()
            return True

    def list_all(self) -> list[GlobalTask]:
        """Return all tasks in insertion order."""
        with self._lock:
            return list(self._tasks.values())

    def list_available(self, agent_name: str | None = None) -> list[GlobalTask]:
        """List claimable pending tasks, optionally filtered by owner.

        If agent_name is provided, return unowned tasks and tasks assigned to that agent.
        """
        available = []
        with self._lock:
            for task in self._tasks.values():
                if task.status != "pending":
                    continue
                if agent_name and task.owner_agent not in (None, agent_name):
                    continue
                if all(
                    self._tasks.get(d) and self._tasks[d].status == "completed"
                    for d in task.blocked_by
                ):
                    available.append(task)
        return available

    def get(self, task_id: str) -> GlobalTask | None:
        with self._lock:
            return self._tasks.get(task_id)

    def _save(self) -> None:
        if not self._path:
            return
        data = {tid: task.to_dict() for tid, task in self._tasks.items()}
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


