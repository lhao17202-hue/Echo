"""单次运行的生产物存储。

设计思路：
  session.json 负责保存"可恢复的会话状态"；
  RunStore 负责保存"单次运行的审计工件"——
    task_state.json  （运行状态机快照，覆盖写，原子替换）
    trace.jsonl      （事件日志，追加写，用于调试和审计）
    report.json      （最终报告，一次性写入，用于复盘）

存储位置：.echo/sessions/{session_id}/runs/{run_id}/
"""

import json
import os
import time
from pathlib import Path
from echo.core.task_state import TaskState
from echo.security.redaction import redact_artifact


class RunStore:
    """单次运行的工件仓库。

    trace.jsonl 直接在此类中管理（不依赖外部 TraceLogger），
    避免循环导入。TraceEmitter 通过 append_trace() 方法写入。
    """

    def __init__(self, session_dir: str):
        self._base = Path(session_dir) / "runs"
        self._run_dir: Path | None = None

    def _run_path(self, run_id: str) -> Path:
        return self._base / run_id

    def _state_path(self) -> Path:
        self._ensure_run_started()
        return self._run_dir / "task_state.json"

    def _report_path(self) -> Path:
        self._ensure_run_started()
        return self._run_dir / "report.json"

    def _trace_path(self) -> Path:
        self._ensure_run_started()
        return self._run_dir / "trace.jsonl"

    def _ensure_run_started(self) -> None:
        """确保 start_run() 已被调用。"""
        if self._run_dir is None:
            raise RuntimeError("start_run() 必须先于任何写操作调用")

    # ── 生命周期 ──────────────────────────────────

    def start_run(self, task_state: TaskState) -> Path:
        """创建 run 目录并写入初始状态。"""
        self._run_dir = self._run_path(task_state.run_id)
        self._run_dir.mkdir(parents=True, exist_ok=True)
        self.update_state(task_state)
        return self._run_dir

    # ── 状态快照 ──────────────────────────────────

    def update_state(self, task_state: TaskState) -> Path:
        """覆盖写 task_state.json（原子写入）。"""
        path = self._state_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        self._atomic_write(path, json.dumps(task_state.to_dict(), indent=2, ensure_ascii=False))
        return path

    def load_task_state(self, run_id: str) -> TaskState:
        """从磁盘加载 task_state。"""
        path = self._run_path(run_id) / "task_state.json"
        if not path.exists():
            raise FileNotFoundError(f"task_state 不存在: {path}")
        return TaskState.from_dict(json.loads(path.read_text(encoding="utf-8")))

    # ── Trace（JSONL 追加）────────────────────────

    def append_trace(self, event) -> Path:
        """追加 TraceEvent 到 trace.jsonl（写入前强制脱敏）。

        供 TraceEmitter 调用。接收有 to_dict() 方法的 TraceEvent 对象。
        无论调用方是否传入 redact_fn，这是最后一道全局脱敏关卡。

        Args:
            event: TraceEvent 实例。

        Returns:
            trace.jsonl 的路径。
        """
        path = self._trace_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        raw = event.to_dict()
        safe = redact_artifact(raw)
        line = json.dumps(safe, ensure_ascii=False)
        with open(path, "a", encoding="utf-8") as f:
            f.write(line + "\n")
        return path

    def log(self, event_type: str, run_id: str = "", **payload) -> Path:
        """快捷 trace 写入（无需构造 TraceEvent 对象）。

        Args:
            event_type: 事件类型字符串。
            run_id: 运行 ID。
            **payload: 事件载荷（自动过 redact_artifact 脱敏）。
        """
        path = self._trace_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        import uuid as _uuid
        safe_payload = redact_artifact(payload)
        line = json.dumps({
            "event": event_type,
            "run_id": run_id,
            "event_id": _uuid.uuid4().hex[:8],
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "timestamp": time.time(),
            **safe_payload,
        }, ensure_ascii=False)
        with open(path, "a", encoding="utf-8") as f:
            f.write(line + "\n")
        return path

    # ── 报告 ──────────────────────────────────────

    def write_report(self, task_state: TaskState,
                     total_tokens: dict | None = None,
                     durable_promotions: list | None = None) -> Path:
        """写最终 report.json（原子写入）。"""
        path = self._report_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        report = {
            "run_id": task_state.run_id, "task_id": task_state.task_id,
            "agent_type": task_state.agent_type, "agent_name": task_state.agent_name,
            "status": task_state.status.value, "stop_reason": task_state.stop_reason,
            "user_request": task_state.user_request[:200],
            "tool_steps": task_state.tool_steps, "attempts": task_state.attempts,
            "compact_count": task_state.compact_count,
            "duration_s": task_state.duration_seconds,
            "total_tokens": total_tokens or {},
            "durable_promotions": durable_promotions or [],
            "final_answer": (task_state.final_answer[:500] if task_state.final_answer else None),
            "errors": task_state.errors,
            "started_at": task_state.started_at, "finished_at": task_state.finished_at,
            "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        }
        self._atomic_write(path, json.dumps(report, indent=2, ensure_ascii=False))
        return path

    def load_report(self, run_id: str) -> dict:
        """加载 report.json。"""
        path = self._run_path(run_id) / "report.json"
        if not path.exists():
            raise FileNotFoundError(f"report 不存在: {path}")
        return json.loads(path.read_text(encoding="utf-8"))

    # ── 原子写入 ──────────────────────────────────

    @staticmethod
    def _atomic_write(path: Path, content: str) -> None:
        tmp = path.with_suffix(".tmp")
        tmp.parent.mkdir(parents=True, exist_ok=True)
        tmp.write_text(content, encoding="utf-8")
        os.replace(tmp, path)
