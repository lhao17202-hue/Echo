"""Manage persistent teammate agents for Echo V1."""

from __future__ import annotations

import threading

from echo.tools.base import ToolContext
from echo.tools.executor import ToolExecutor
from echo.tools.registry import ToolRegistry
from echo.multi_agent.teammate import TeammateAgent


class TeammateManager:
    """Owns same-process persistent teammate daemon threads."""

    BLOCKED_TOOLS = {
        "delegate",
        "compact",
        "spawn_teammate",
        "assign_task",
        "list_teammates",
        "stop_teammate",
        "list_global_tasks",
        "write_file",
        "patch_file",
        "run_shell",
        "todo_write",
        "save_memory",
    }

    def __init__(self, llm, tool_registry, sandbox, shell, memory, bus, tasks, trace_logger=None):
        self.llm = llm
        self.tool_registry = tool_registry
        self.sandbox = sandbox
        self.shell = shell
        self.memory = memory
        self.bus = bus
        self.tasks = tasks
        self.trace_logger = trace_logger
        self._teammates: dict[str, TeammateAgent] = {}
        self._lock = threading.Lock()

    def spawn(self, name: str, role: str, prompt: str = "") -> dict:
        name = str(name or "").strip()
        role = str(role or "assistant").strip() or "assistant"
        prompt = str(prompt or "")
        if not name:
            return {"success": False, "error": "teammate name is required"}

        with self._lock:
            if name in self._teammates:
                return {"success": False, "error": f"teammate '{name}' already exists"}

            registry = ToolRegistry()
            for tool in self._select_teammate_tools():
                registry.register(tool)
            executor = ToolExecutor(registry)
            ctx = ToolContext(
                workspace_root=str(self.sandbox.root),
                sandbox=self.sandbox,
                shell=self.shell,
                memory=self.memory,
                llm=self.llm,
                tool_registry=registry,
                message_bus=self.bus,
                teammate_manager=self,
                global_tasks=self.tasks,
                agent_name=name,
                trace_logger=self.trace_logger,
                depth=0,
                max_depth=0,
            )
            self.bus.register(name)
            teammate = TeammateAgent(
                name=name,
                role=role,
                prompt=prompt,
                llm=self.llm,
                tools=executor,
                ctx=ctx,
                bus=self.bus,
                tasks=self.tasks,
                trace_logger=self.trace_logger,
            )
            self._teammates[name] = teammate
            teammate.start()
            self._log("teammate_spawned", teammate=name, role=role)
            return {"success": True, "teammate": teammate.snapshot()}

    def stop(self, name: str) -> bool:
        with self._lock:
            teammate = self._teammates.get(name)
            if not teammate:
                return False
            teammate.stop()
            return True

    def list(self) -> list[dict]:
        with self._lock:
            return [t.snapshot() for t in self._teammates.values()]

    def assign_task(self, teammate: str, subject: str, description: str = "") -> str:
        with self._lock:
            if teammate not in self._teammates:
                raise ValueError(f"unknown teammate: {teammate}")
        task_id = self.tasks.create(subject, description)
        if not self.tasks.assign(task_id, teammate):
            raise RuntimeError(f"failed to assign task {task_id} to {teammate}")
        self._log("global_task_created", task_id=task_id, teammate=teammate, subject=subject[:200])
        self._log("global_task_assigned", task_id=task_id, teammate=teammate)
        return task_id

    def snapshot(self) -> dict:
        with self._lock:
            return {name: agent.snapshot() for name, agent in self._teammates.items()}

    def _select_teammate_tools(self) -> list:
        return [
            tool for tool in self.tool_registry.get_all()
            if tool.is_read_only and tool.name not in self.BLOCKED_TOOLS
        ]

    def _log(self, event_type: str, **payload) -> None:
        if self.trace_logger:
            self.trace_logger.log(event_type, **payload)
