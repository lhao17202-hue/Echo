"""Long-lived teammate agent for Echo's persistent teammate V1."""

from __future__ import annotations

import logging
import threading
import time

from echo.providers.base import TextBlock, ToolUseBlock
from echo.tools.base import ToolContext

logger = logging.getLogger("echo.teammate")


class TeammateState:
    """Snapshot of a teammate agent's current state."""

    def __init__(self, name: str, role: str):
        self.name = name
        self.role = role
        self.status: str = "idle"  # idle | running | stopped | failed
        self.current_task_id: str = ""
        self.session_id: str = ""
        self.last_error: str = ""
        self.started_at: str = ""
        self.stopped_at: str = ""

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "role": self.role,
            "status": self.status,
            "current_task_id": self.current_task_id,
            "session_id": self.session_id,
            "last_error": self.last_error,
            "started_at": self.started_at,
            "stopped_at": self.stopped_at,
        }


class TeammateAgent:
    """A same-process daemon-thread teammate that claims global tasks."""

    def __init__(
        self,
        name: str,
        role: str,
        prompt: str,
        llm,
        tools,
        ctx: ToolContext,
        bus,
        tasks,
        trace_logger=None,
        poll_interval: float = 0.2,
        llm_lock: threading.Lock | None = None,
    ):
        self.name = name
        self.role = role
        self.prompt = prompt or ""
        self.llm = llm
        self.tools = tools
        self.ctx = ctx
        self.bus = bus
        self.tasks = tasks
        self.trace_logger = trace_logger
        self.poll_interval = poll_interval
        self._llm_lock = llm_lock or threading.Lock()
        self.state = TeammateState(name=name, role=role)
        self.state.started_at = time.strftime("%Y-%m-%dT%H:%M:%S")
        self._stop_requested = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._thread = threading.Thread(
            target=self.run_loop,
            name=f"echo-teammate-{self.name}",
            daemon=True,
        )
        self._thread.start()
        self._log("teammate_started")

    def stop(self) -> None:
        self._stop_requested.set()
        self.state.status = "stopped"
        self.state.stopped_at = time.strftime("%Y-%m-%dT%H:%M:%S")
        self._log("teammate_stopped")

    def snapshot(self) -> dict:
        return self.state.to_dict()

    def run_loop(self) -> None:
        while not self._stop_requested.is_set():
            try:
                self._tick()
            except Exception as exc:
                self.state.status = "failed"
                self.state.last_error = str(exc)
                self._log("teammate_task_failed", error=str(exc)[:300])
                logger.exception("Teammate %s tick failed", self.name)
            time.sleep(self.poll_interval)
        self.state.status = "stopped"
        if not self.state.stopped_at:
            self.state.stopped_at = time.strftime("%Y-%m-%dT%H:%M:%S")

    def _tick(self) -> None:
        self._handle_inbox()
        if self._stop_requested.is_set():
            return
        task = self._claim_available_task()
        if task is not None:
            self._run_task(task)

    def _handle_inbox(self) -> None:
        for msg in self.bus.receive(self.name):
            if msg.msg_type == "stop":
                self.stop()
            elif msg.msg_type == "message":
                self._log(
                    "message_received",
                    from_agent=msg.from_agent,
                    to_agent=msg.to_agent,
                    msg_type=msg.msg_type,
                )

    def _claim_available_task(self):
        for task in self.tasks.list_available(self.name):
            if self.tasks.claim(task.task_id, self.name):
                claimed = self.tasks.get(task.task_id)
                self.state.status = "running"
                self.state.current_task_id = task.task_id
                # Resolve per-task trace_logger so teammate events write to the
                # correct run even when the teammate persists across ask() calls.
                task_trace = None
                if hasattr(self.ctx, 'teammate_manager') and self.ctx.teammate_manager:
                    task_trace = self.ctx.teammate_manager.get_task_trace(task.task_id)
                self._current_task_trace = task_trace or self.trace_logger
                self._log("teammate_task_claimed", task_id=task.task_id)
                return claimed
        return None

    def _run_task(self, task) -> None:
        try:
            result, success = self._run_llm_task(task)
            if success:
                self.tasks.complete(task.task_id, result)
                self.bus.send(
                    self.name,
                    "lead",
                    result,
                    msg_type="task_completed",
                    metadata={"task_id": task.task_id},
                )
                self._log("teammate_task_completed", task_id=task.task_id, result_preview=result[:300])
            else:
                # empty response or step limit — treat as failure, not completion
                self.tasks.fail(task.task_id, result)
                self.bus.send(
                    self.name,
                    "lead",
                    result,
                    msg_type="task_failed",
                    metadata={"task_id": task.task_id},
                )
                self.state.last_error = result
                self._log("teammate_task_failed", task_id=task.task_id, error=result[:300])
            self.state.status = "idle"
            self.state.current_task_id = ""
            self._clear_task_trace(task.task_id)
        except Exception as exc:
            error = str(exc)
            self.tasks.fail(task.task_id, error)
            self.bus.send(
                self.name,
                "lead",
                error,
                msg_type="task_failed",
                metadata={"task_id": task.task_id},
            )
            self.state.status = "failed"
            self.state.last_error = error
            self.state.current_task_id = ""
            self._clear_task_trace(task.task_id)
            self._log("teammate_task_failed", task_id=task.task_id, error=error[:300])

    def _run_llm_task(self, task, max_steps: int = 8) -> tuple[str, bool]:
        """Run the LLM loop for a task. Returns (text, success).

        success=False means: empty response, step limit, or tool failures —
        these should be marked as task_failed by the caller, not completed.
        """
        request = self._render_task(task)
        messages = [{"role": "user", "content": [TextBlock(text=request)]}]
        system = self._build_system()
        final_text = ""

        for _ in range(max_steps):
            with self._llm_lock:
                response = self.llm.chat(
                    messages,
                    self.tools.registry.list_schemas(),
                    system,
                    max_tokens=8000,
                )
            messages.append({"role": "assistant", "content": response.content})
            tool_blocks = [b for b in response.content if isinstance(b, ToolUseBlock)]
            if not tool_blocks:
                texts = [b.text for b in response.content if isinstance(b, TextBlock)]
                final_text = " ".join(t for t in texts if t).strip()
                if final_text:
                    return final_text, True
                return "Teammate produced no text response.", False

            tool_results = []
            for block in tool_blocks:
                result = self.tools.execute(block.name, block.input, self._child_ctx())
                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "tool_name": block.name,
                    "tool_input": block.input,
                    "content": result.output if not result.error else f"Error: {result.error}\n{result.output}",
                })
            messages.append({"role": "user", "content": tool_results})

        return (final_text or "Stopped after teammate step limit.", False)

    def _render_task(self, task) -> str:
        parts = [f"Task: {task.subject}"]
        if task.description:
            parts.append(f"Description: {task.description}")
        parts.append("Return concise findings for the lead agent.")
        return "\n\n".join(parts)

    def _build_system(self) -> str:
        base = (
            f"You are teammate '{self.name}', role: {self.role}. "
            "You are a read-only research teammate. Use available read-only tools when useful. "
            "Do not attempt to modify files or run shell commands."
        )
        if self.prompt:
            return base + "\n\n" + self.prompt
        return base

    def _child_ctx(self):
        self.ctx.agent_name = self.name
        return self.ctx

    def _clear_task_trace(self, task_id: str) -> None:
        """Release per-task trace reference and clean up manager mapping."""
        self._current_task_trace = None
        if hasattr(self.ctx, 'teammate_manager') and self.ctx.teammate_manager:
            self.ctx.teammate_manager.clear_task_trace(task_id)

    def _log(self, event_type: str, **payload) -> None:
        # Per-task trace takes precedence (routes to the assign_task's run),
        # then instance-level trace_logger (set at spawn time).
        trace = getattr(self, '_current_task_trace', None) or self.trace_logger
        if trace:
            trace.log(event_type, run_id=getattr(self.ctx, "run_id", ""), teammate=self.name, **payload)
