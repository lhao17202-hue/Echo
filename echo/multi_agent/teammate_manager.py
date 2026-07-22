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
        "wait_global_task",
        "write_file",
        "patch_file",
        "run_shell",
        "todo_write",
        "save_memory",
    }

    def __init__(self, llm, tool_registry, sandbox, shell, memory, bus, tasks,
                 trace_logger=None, llm_lock=None):
        self.llm = llm
        self.tool_registry = tool_registry
        self.sandbox = sandbox
        self.shell = shell
        self.memory = memory
        self.bus = bus
        self.tasks = tasks
        self.trace_logger = trace_logger
        self._teammates: dict[str, TeammateAgent] = {}
        self._task_traces: dict[str, object] = {}  # task_id → trace_logger (per-task routing)
        self._lock = threading.Lock()
        # Use external lock when provided (Echo facade shares one lock with AgentLoop);
        # otherwise create a private lock for standalone/test use.
        self._llm_lock = llm_lock or threading.Lock()

    def spawn(self, name: str, role: str, prompt: str = "",
              run_id: str = "", trace_logger=None) -> dict:
        """Spawn a new teammate daemon thread.

        Accepts optional run_id and trace_logger from the lead's tool context so
        teammate lifecycle events are written into the current run's trace.
        """
        name = str(name or "").strip()
        role = str(role or "assistant").strip() or "assistant"
        prompt = str(prompt or "")
        if not name:
            return {"success": False, "error": "teammate name is required"}

        # Per-spawn trace_logger takes precedence over manager-level default
        effective_trace = trace_logger or self.trace_logger

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
                run_id=run_id,
                trace_logger=effective_trace,
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
                trace_logger=effective_trace,
                llm_lock=self._llm_lock,
            )
            self._teammates[name] = teammate
            teammate.start()
            self._log("teammate_spawned", trace_logger=effective_trace, teammate=name, role=role, run_id=run_id)
            return {"success": True, "teammate": teammate.snapshot()}

    def stop(self, name: str) -> bool:
        """Signal a teammate to stop after its current tick boundary.

        V1 limitations (intentional):
        - Does not join() the thread (daemon threads exit with the process).
        - Does not remove the teammate from _teammates (name stays reserved;
          stopped teammates remain visible in list()/snapshot()).
        - Stopped teammates cannot be re-spawned with the same name. A V2
          restart/recover primitive would need explicit destroy + clear.
        """
        with self._lock:
            teammate = self._teammates.get(name)
            if not teammate:
                return False
            teammate.stop()
            return True

    def list(self) -> list[dict]:
        with self._lock:
            return [t.snapshot() for t in self._teammates.values()]

    def assign_task(self, teammate: str, subject: str, description: str = "",
                    run_id: str = "", trace_logger=None) -> str:
        with self._lock:
            if teammate not in self._teammates:
                raise ValueError(f"unknown teammate: {teammate}")
        effective_trace = trace_logger or self.trace_logger
        task_id = self.tasks.create(subject, description, run_id=run_id)
        # Store per-task trace so teammate events (claimed/completed/failed)
        # write to the correct run even across multiple ask() calls.
        with self._lock:
            self._task_traces[task_id] = effective_trace
        if not self.tasks.assign(task_id, teammate):
            raise RuntimeError(f"failed to assign task {task_id} to {teammate}")
        self._log("global_task_created", trace_logger=effective_trace,
                  task_id=task_id, teammate=teammate, subject=subject[:200], run_id=run_id)
        self._log("global_task_assigned", trace_logger=effective_trace,
                  task_id=task_id, teammate=teammate, run_id=run_id)
        return task_id

    def snapshot(self) -> dict:
        with self._lock:
            return {name: agent.snapshot() for name, agent in self._teammates.items()}

    def get_task_trace(self, task_id: str) -> object | None:
        """Return the trace_logger stored for a given task_id, or None."""
        with self._lock:
            return self._task_traces.get(task_id)

    def clear_task_trace(self, task_id: str) -> None:
        """Remove the trace_logger entry after a task is complete/failed."""
        with self._lock:
            self._task_traces.pop(task_id, None)

    def _select_teammate_tools(self) -> list:
        return [
            tool for tool in self.tool_registry.get_all()
            if tool.is_read_only and tool.name not in self.BLOCKED_TOOLS
        ]

    def _log(self, event_type: str, trace_logger=None, **payload) -> None:
        """Emit a trace event. Uses the per-call trace_logger if provided,
        falling back to the manager-level default."""
        logger = trace_logger or self.trace_logger
        if logger:
            logger.log(event_type, **payload)

    def _log_at(self, trace_logger, event_type: str, **payload) -> None:
        """Convenience: log with an explicit trace_logger (no fallback)."""
        if trace_logger:
            trace_logger.log(event_type, **payload)
