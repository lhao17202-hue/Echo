"""Global task manager — cross-agent task pool with locking."""

import json
import logging
import os
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

logger = logging.getLogger("echo.tasks")

TaskStatus = Literal["pending", "in_progress", "completed", "failed"]
TaskValidationLevel = Literal["warning", "error"]
VALID_TASK_STATUSES = {"pending", "in_progress", "completed", "failed"}


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


@dataclass(frozen=True)
class TaskValidationIssue:
    """Validation issue discovered in the global task graph."""

    task_id: str
    level: TaskValidationLevel
    message: str


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

    def _blocked_dependency_ids(self, task: GlobalTask) -> list[str]:
        """Return dependency IDs that are missing or not completed."""
        blocked = []
        for dep_id in task.blocked_by:
            dep = self._tasks.get(dep_id)
            if dep is None or dep.status != "completed":
                blocked.append(dep_id)
        return blocked

    def _deps_satisfied(self, task: GlobalTask) -> bool:
        """Return True when every dependency exists and is completed."""
        return not self._blocked_dependency_ids(task)

    def blocked_dependencies(self, task_id: str) -> list[str]:
        """Return dependency IDs that are missing or not completed."""
        with self._lock:
            task = self._tasks.get(task_id)
            if not task:
                return []
            return self._blocked_dependency_ids(task)

    def can_start(self, task_id: str) -> bool:
        """Return True when a pending task exists and all dependencies completed."""
        with self._lock:
            task = self._tasks.get(task_id)
            return bool(task and task.status == "pending" and self._deps_satisfied(task))

    def list_unblocked_after(self, task_id: str) -> list[GlobalTask]:
        """Return pending downstream tasks that are now claimable."""
        with self._lock:
            return [
                task for task in self._tasks.values()
                if task.status == "pending"
                and task_id in task.blocked_by
                and self._deps_satisfied(task)
            ]

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
                # 检查依赖：缺失或未完成都视为 blocked
                if not self._deps_satisfied(task):
                    return False
                task.status = "in_progress"
                task.owner_agent = agent_name
                self._save()
                return True
            return False

    def claim_task(self, task_id: str, agent_name: str) -> tuple[bool, str]:
        """Claim a task and return a user-facing reason on failure."""
        task = self.get(task_id)
        if task is None:
            return False, f"Unknown global task: {task_id}"
        blocked = self.blocked_dependencies(task_id)
        if blocked:
            return False, f"Blocked by: {blocked}"
        if not self.claim(task_id, agent_name):
            current = self.get(task_id)
            status = current.status if current else "missing"
            return False, f"Task {task_id} cannot be claimed; status={status}"
        return True, f"Claimed task {task_id} for {agent_name}"

    def complete_task(self, task_id: str, result: str = "") -> tuple[bool, str]:
        """Complete a task and report newly unblocked downstream tasks."""
        task = self.get(task_id)
        if task is None:
            return False, f"Unknown global task: {task_id}"
        if not self.complete(task_id, result):
            current = self.get(task_id)
            status = current.status if current else "missing"
            return False, f"Task {task_id} is {status}, cannot complete"
        output = f"Completed task {task_id}"
        unblocked = self.list_unblocked_after(task_id)
        if unblocked:
            output += "\nUnblocked:\n" + "\n".join(
                f"- {self.format_task(task, include_result=False)}"
                for task in unblocked
            )
        return True, output

    def fail_task(self, task_id: str, error: str = "") -> tuple[bool, str]:
        """Fail a task and return a user-facing reason on failure."""
        task = self.get(task_id)
        if task is None:
            return False, f"Unknown global task: {task_id}"
        if not self.fail(task_id, error):
            current = self.get(task_id)
            status = current.status if current else "missing"
            return False, f"Task {task_id} is {status}, cannot fail"
        return True, f"Failed task {task_id}"

    def complete(self, task_id: str, result: str = "") -> bool:
        with self._lock:
            task = self._tasks.get(task_id)
            if not task or task.status != "in_progress":
                return False
            task.status = "completed"
            task.completed_at = time.strftime("%Y-%m-%dT%H:%M:%S")
            task.result = result
            self._save()
            return True

    def fail(self, task_id: str, error: str = "") -> bool:
        with self._lock:
            task = self._tasks.get(task_id)
            if not task or task.status == "completed":
                return False
            task.status = "failed"
            task.result = error
            self._save()
            return True

    def wait(self, task_id: str, timeout: float = 10.0,
             interval: float = 0.1) -> GlobalTask | None:
        """Wait until a task reaches a terminal status or timeout expires.

        This method does not mutate task state. It returns the current task
        snapshot when the task completes/fails, or the latest known task after
        timeout so callers can report pending/in_progress status.
        """
        timeout = max(0.0, float(timeout))
        interval = max(0.01, float(interval))
        deadline = time.time() + timeout

        while True:
            task = self.get(task_id)
            if task is None:
                return None
            if task.status in ("completed", "failed"):
                return task
            if time.time() >= deadline:
                return task
            time.sleep(min(interval, max(0.0, deadline - time.time())))

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
                if self._deps_satisfied(task):
                    available.append(task)
        return available

    def get(self, task_id: str) -> GlobalTask | None:
        with self._lock:
            return self._tasks.get(task_id)

    def validate(self) -> list[TaskValidationIssue]:
        """Return dependency, status, and shape issues without mutating tasks."""
        with self._lock:
            tasks = dict(self._tasks)

        issues: list[TaskValidationIssue] = []
        for task_id, task in tasks.items():
            if task.status not in VALID_TASK_STATUSES:
                issues.append(TaskValidationIssue(
                    task_id=task_id,
                    level="error",
                    message=f"Unknown status: {task.status}",
                ))
            if not str(task.subject or "").strip():
                issues.append(TaskValidationIssue(
                    task_id=task_id,
                    level="error",
                    message="Missing subject",
                ))
            for dep_id in task.blocked_by:
                if dep_id not in tasks:
                    issues.append(TaskValidationIssue(
                        task_id=task_id,
                        level="error",
                        message=f"Unknown dependency: {dep_id}",
                    ))

        visiting: set[str] = set()
        visited: set[str] = set()

        def visit(task_id: str, trail: list[str]) -> None:
            if task_id in visiting:
                cycle = [*trail, task_id]
                issues.append(TaskValidationIssue(
                    task_id=task_id,
                    level="error",
                    message="Dependency cycle: " + " -> ".join(cycle),
                ))
                return
            if task_id in visited or task_id not in tasks:
                return
            visiting.add(task_id)
            for dep_id in tasks[task_id].blocked_by:
                visit(dep_id, [*trail, task_id])
            visiting.remove(task_id)
            visited.add(task_id)

        for task_id in tasks:
            visit(task_id, [])
        return issues

    @staticmethod
    def format_task(task: GlobalTask, include_result: bool = True) -> str:
        """Format one task as a compact prompt/CLI line."""
        deps = ", ".join(task.blocked_by)
        result_preview = (task.result or "")[:120]
        line = (
            f"{task.task_id} [{task.status}] owner={task.owner_agent or '-'} "
            f"subject={task.subject} blocked_by=[{deps}]"
        )
        if include_result and result_preview:
            line += f" result={result_preview}"
        return line

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

