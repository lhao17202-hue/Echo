"""追踪日志系统 —— TraceEvent 数据模型 + TraceEmitter + TraceReader。

设计思路：
  trace 是运行中的逐事件时间线，适合回答"这一轮 agent 到底做了什么"。
  采用 JSONL 追加写入，逐条落盘比"最后一次性写整份 trace"更稳。

  trace 和 report 的区别：
    trace 关注过程 —— 每个事件的时间线，用于调试和审计
    report 关注结果 —— 最终的状态快照，用于复盘和指标统计

安全保证：
  所有事件在写入前经过 Redactor 脱敏 —— 密钥值在落盘前已被替换。
  prompt_metadata 中的 secret_env_summary 只包含名字，不含值。

事件类型：
  运行生命周期: run_started, run_finished, run_error
  模型交互:     prompt_built, model_requested, model_parsed
  工具执行:     tool_executed（含 workspace_changes）
  检查点:       checkpoint_created, runtime_identity_mismatch
  上下文压缩:   compaction_triggered
  多 Agent:     message_sent, message_received, teammate_spawned, teammate_stopped
  Cron:         cron_fired
  后台任务:     background_started, background_completed
  记忆:         memory_promoted
"""

import json
import time
import uuid
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

logger = logging.getLogger("echo.trace")


# ═══════════════════════════════════════════════════════
# 事件类型常量
# ═══════════════════════════════════════════════════════

class EventType:
    """Trace 事件类型 —— 覆盖 Agent 运行全生命周期的所有关键节点。"""

    # ── 运行生命周期 ──────────────────────────────
    RUN_STARTED = "run_started"                # 新的 ask() 开始
    RUN_FINISHED = "run_finished"              # ask() 正常结束（含 final_answer 摘要）
    RUN_ERROR = "run_error"                    # ask() 异常终止

    # ── 上下文与提示词 ────────────────────────────
    PROMPT_BUILT = "prompt_built"              # 上下文组装完成（含 prompt_metadata）
    COMPACTION_TRIGGERED = "compaction_triggered"  # 触发上下文压缩（含 compact_count）

    # ── 模型交互 ──────────────────────────────────
    MODEL_REQUESTED = "model_requested"        # 发起 LLM 调用（含 attempts / tool_steps）
    MODEL_PARSED = "model_parsed"              # LLM 响应解析完成（含 kind / usage / duration）

    # ── 工具执行 ──────────────────────────────────
    TOOL_EXECUTED = "tool_executed"            # 工具执行完成（含 name / args / success / workspace_changes / duration）

    # ── 检查点与恢复 ──────────────────────────────
    CHECKPOINT_CREATED = "checkpoint_created"         # 新建检查点（含 checkpoint_id / trigger）
    RUNTIME_IDENTITY_MISMATCH = "runtime_identity_mismatch"  # 运行时环境不一致

    # ── 多 Agent────────────────────────────
    MESSAGE_SENT = "message_sent"              # MessageBus 发送消息
    MESSAGE_RECEIVED = "message_received"      # MessageBus 收到消息
    TEAMMATE_SPAWNED = "teammate_spawned"      # 队友启动
    TEAMMATE_STOPPED = "teammate_stopped"      # 队友停止

    # ── Cron────────────────────────────────
    CRON_FIRED = "cron_fired"                  # 定时任务触发

    # ── 后台任务────────────────────────────
    BACKGROUND_STARTED = "background_started"   # 后台任务提交
    BACKGROUND_COMPLETED = "background_completed"  # 后台任务完成

    # ── 记忆───────────────────────────────
    MEMORY_PROMOTED = "memory_promoted"        # 短期记忆提升为持久记忆


# ═══════════════════════════════════════════════════════
# TraceEvent 数据模型
# ═══════════════════════════════════════════════════════

@dataclass
class TraceEvent:
    """追踪事件 —— 所有事件统一为此结构。

    每个事件一行 JSON，追加写入 trace.jsonl。
    写入前通过 redact_artifact() 脱敏 —— 密钥值在落盘前已被替换。

    字段设计：
      - event_type: 放在 payload 里的 "event" 字段，方便按类型过滤
      - run_id: 关联到具体的一次 ask() 运行
      - agent_name / agent_type: 多 Agent 场景下区分事件来源
      - payload: 事件详情，因 event_type 不同而异
      - timestamp / created_at: 双重时间戳（浮点数精确排序 + 人类可读）
      - event_id: 全局唯一，用于去重和引用
    """

    # ── 事件身份 ──────────────────────────────────
    event_type: str = ""          # EventType 常量
    event_id: str = ""            # 全局唯一事件 ID（8 位 hex）
    run_id: str = ""              # 关联的运行 ID

    # ── 来源 Agent ────────────────────────────────
    agent_name: str | None = None # Agent 名称（如 "code-reviewer"）
    agent_type: str | None = None # "lead" | "teammate" | "subagent"

    # ── 事件载荷（写入前必须脱敏！）───────────────
    payload: dict = field(default_factory=dict)

    # ── 时间 ──────────────────────────────────────
    timestamp: float = field(default_factory=time.time)  # 精确时间（用于排序）
    created_at: str = ""          # ISO 格式（人类可读），自动生成

    def __post_init__(self):
        """自动生成 event_id、created_at、timestamp。"""
        if not self.event_id:
            self.event_id = uuid.uuid4().hex[:8]
        if not self.created_at:
            self.created_at = time.strftime("%Y-%m-%dT%H:%M:%S")
        if not self.timestamp:
            self.timestamp = time.time()

    def to_dict(self) -> dict:
        """序列化为 JSON 兼容字典。

        与 Echo trace 格式对齐：
          event 放在顶层，payload 展开。
        """
        return {
            "event": self.event_type,
            "event_id": self.event_id,
            "run_id": self.run_id,
            "agent_name": self.agent_name,
            "agent_type": self.agent_type,
            "created_at": self.created_at,
            "timestamp": self.timestamp,
            **self.payload,  # 展开 payload，与 Echo trace 格式对齐
        }


# ═══════════════════════════════════════════════════════
# TraceEmitter — 事件发射器
# ═══════════════════════════════════════════════════════

class TraceEmitter:
    """追踪事件发射器 —— Agent 中所有 emit_trace 调用的统一入口。

    职责：
      1. 构造 TraceEvent 并填充公共字段（run_id, agent_name, agent_type）
      2. 调用 redact_artifact() 脱敏
      3. 追加写入 trace.jsonl
      4. 可选：同步写入 stderr 用于实时调试

    TraceEmitter 设计：
      - event 类型放在 payload["event"] 中
      - 所有 payload 写入前经过脱敏
      - 返回 payload 供调用者进一步使用

    使用模式：
      emitter = TraceEmitter(run_store, run_id, redact_fn, agent_name="lead")
      emitter.emit(EventType.RUN_STARTED, task_id="t1", user_request="...")
      emitter.emit(EventType.TOOL_EXECUTED, tool="read_file", success=True, duration_ms=42)
    """

    def __init__(self,
                 run_store,               # RunStore 实例
                 run_id: str,
                 redact_fn: Callable | None = None,  # redact_artifact 函数
                 agent_name: str | None = None,
                 agent_type: str | None = "lead",
                 debug: bool = False,      # True → 同时打印到 stderr
                 ):
        """初始化事件发射器。

        Args:
            run_store: RunStore 实例（用于 append_trace）。
            run_id: 当前运行的 run_id。
            redact_fn: 脱敏函数（通常传 redact_artifact）。
            agent_name: 当前 Agent 名称。
            agent_type: 当前 Agent 类型。
            debug: 是否同步输出到 stderr（开发调试用）。
        """
        self._run_store = run_store
        self._run_id = run_id
        self._redact = redact_fn or (lambda x: x)
        self._agent_name = agent_name
        self._agent_type = agent_type
        self._debug = debug
        self._event_count = 0  # 已发送的事件计数

    # ── 核心 emit 方法 ────────────────────────────

    def emit(self, event_type: str, **payload) -> dict:
        """发射一个追踪事件。

        1. 构造 TraceEvent
        2. 脱敏 payload
        3. 写入 trace.jsonl
        4. 可选 debug 输出

        Args:
            event_type: EventType 常量。
            **payload: 事件载荷（关键字参数自动转为 dict）。

        Returns:
            脱敏后的 payload dict（调用者可以继续使用）。
        """
        self._event_count += 1

        # 填充公共字段
        payload.setdefault("run_id", self._run_id)
        payload.setdefault("agent_name", self._agent_name)
        payload.setdefault("agent_type", self._agent_type)

        # 脱敏
        redacted = self._redact(payload)

        # 构造 TraceEvent
        event = TraceEvent(
            event_type=event_type,
            run_id=self._run_id,
            agent_name=self._agent_name,
            agent_type=self._agent_type,
            payload=redacted,
        )

        # 写入 trace.jsonl
        try:
            self._run_store.append_trace(event)
        except Exception as e:
            logger.warning(f"Trace 写入失败: {e}")

        # Debug 输出
        if self._debug:
            import sys
            preview = json.dumps(redacted, ensure_ascii=False)[:200]
            print(f"[trace #{self._event_count}] {event_type} {preview}", file=sys.stderr)

        return redacted

    # ── 便捷方法（按事件类型分类）─────────────────

    def run_started(self, task_id: str, user_request: str, **extra) -> dict:
        """发射 run_started 事件。"""
        return self.emit(EventType.RUN_STARTED,
                         task_id=task_id,
                         user_request=user_request[:300],
                         **extra)

    def run_finished(self, status: str, stop_reason: str = "",
                     tool_steps: int = 0, attempts: int = 0,
                     final_answer_preview: str = "",
                     total_tokens: dict | None = None, **extra) -> dict:
        """发射 run_finished 事件。"""
        return self.emit(EventType.RUN_FINISHED,
                         status=status,
                         stop_reason=stop_reason,
                         tool_steps=tool_steps,
                         attempts=attempts,
                         final_answer_preview=final_answer_preview[:200],
                         total_tokens=total_tokens or {},
                         **extra)

    def run_error(self, error: str, **extra) -> dict:
        """发射 run_error 事件。"""
        return self.emit(EventType.RUN_ERROR, error=str(error), **extra)

    def prompt_built(self, prompt_metadata: dict,
                     duration_ms: int = 0, **extra) -> dict:
        """发射 prompt_built 事件。

        prompt_metadata 应包含：secret_env_summary, model, prompt_cache_key,
        resume_status, budget_reductions 等信息。
        方便解释"这一轮 prompt 为什么长这样"。
        """
        return self.emit(EventType.PROMPT_BUILT,
                         prompt_metadata=prompt_metadata,
                         duration_ms=duration_ms,
                         **extra)

    def model_requested(self, attempts: int = 0, tool_steps: int = 0,
                        max_tokens: int = 0, prompt_cache_key: str = "",
                        **extra) -> dict:
        """发射 model_requested 事件。

        记录发起 LLM 调用时的状态：当前第几轮、调了多少工具、cache key 等。
        """
        return self.emit(EventType.MODEL_REQUESTED,
                         attempts=attempts,
                         tool_steps=tool_steps,
                         max_tokens=max_tokens,
                         prompt_cache_key=prompt_cache_key,
                         **extra)

    def model_parsed(self, kind: str,  # "tool" | "final" | "retry"
                     usage: dict | None = None,
                     duration_ms: int = 0,
                     stop_reason: str = "", **extra) -> dict:
        """发射 model_parsed 事件。

        kind: 模型返回的类型（tool → 工具调用, final → 最终回复, retry → 需重试）。
        usage: {"input_tokens", "output_tokens", "cache_read", "cache_write"}。
        """
        return self.emit(EventType.MODEL_PARSED,
                         kind=kind,
                         usage=usage or {},
                         duration_ms=duration_ms,
                         stop_reason=stop_reason,
                         **extra)

    def tool_executed(self, tool_name: str,
                      success: bool = True,
                      error: str = "",
                      workspace_changes: list[str] | None = None,
                      duration_ms: int = 0,
                      preview: str = "", **extra) -> dict:
        """发射 tool_executed 事件。

        Args:
            tool_name: 工具名称。
            success: 是否成功执行。
            error: 错误信息（失败时）。
            workspace_changes: 工作区变更列表（如 ["modified:src/main.py", "created:new.txt"]）。
            duration_ms: 执行耗时（毫秒）。
            preview: 输出前 200 字符预览。
        """
        return self.emit(EventType.TOOL_EXECUTED,
                         tool=tool_name,
                         success=success,
                         error=error,
                         workspace_changes=workspace_changes or [],
                         duration_ms=duration_ms,
                         preview=str(preview)[:200],
                         **extra)

    def checkpoint_created(self, checkpoint_id: str, trigger: str = "",
                           key_files_count: int = 0, **extra) -> dict:
        """发射 checkpoint_created 事件。"""
        return self.emit(EventType.CHECKPOINT_CREATED,
                         checkpoint_id=checkpoint_id,
                         trigger=trigger,
                         key_files_count=key_files_count,
                         **extra)

    def runtime_identity_mismatch(self, fields: list[str], **extra) -> dict:
        """发射 runtime_identity_mismatch 事件。"""
        return self.emit(EventType.RUNTIME_IDENTITY_MISMATCH,
                         fields=fields,
                         **extra)

    def compaction_triggered(self, trigger: str = "auto",
                             compact_count: int = 0,
                             message_count_before: int = 0,
                             message_count_after: int = 0, **extra) -> dict:
        """发射 compaction_triggered 事件。"""
        return self.emit(EventType.COMPACTION_TRIGGERED,
                         trigger=trigger,
                         compact_count=compact_count,
                         message_count_before=message_count_before,
                         message_count_after=message_count_after,
                         **extra)

    # ── 多 Agent ──────────────────────────────────

    def message_sent(self, from_agent: str = "", to_agent: str = "",
                     msg_type: str = "message", preview: str = "", **extra) -> dict:
        """发射 message_sent 事件。"""
        return self.emit(EventType.MESSAGE_SENT,
                         from_agent=from_agent or self._agent_name,
                         to_agent=to_agent,
                         msg_type=msg_type,
                         preview=str(preview)[:200],
                         **extra)

    def message_received(self, from_agent: str = "", to_agent: str = "",
                         msg_type: str = "message", preview: str = "", **extra) -> dict:
        """发射 message_received 事件。"""
        return self.emit(EventType.MESSAGE_RECEIVED,
                         from_agent=from_agent,
                         to_agent=to_agent or self._agent_name,
                         msg_type=msg_type,
                         preview=str(preview)[:200],
                         **extra)

    def teammate_spawned(self, name: str = "", role: str = "",
                         session_id: str = "", **extra) -> dict:
        """发射 teammate_spawned 事件。"""
        return self.emit(EventType.TEAMMATE_SPAWNED,
                         name=name, role=role, session_id=session_id, **extra)

    def teammate_stopped(self, name: str = "", reason: str = "", **extra) -> dict:
        """发射 teammate_stopped 事件。"""
        return self.emit(EventType.TEAMMATE_STOPPED,
                         name=name, reason=reason, **extra)

    # ── Cron / 后台 / 记忆 ────────────────────────

    def cron_fired(self, job_id: str = "", prompt_preview: str = "", **extra) -> dict:
        """发射 cron_fired 事件。"""
        return self.emit(EventType.CRON_FIRED,
                         job_id=job_id,
                         prompt_preview=str(prompt_preview)[:200],
                         **extra)

    def background_started(self, bg_id: str = "", tool_name: str = "", **extra) -> dict:
        """发射 background_started 事件。"""
        return self.emit(EventType.BACKGROUND_STARTED,
                         bg_id=bg_id, tool=tool_name, **extra)

    def background_completed(self, bg_id: str = "", tool_name: str = "",
                             success: bool = True, **extra) -> dict:
        """发射 background_completed 事件。"""
        return self.emit(EventType.BACKGROUND_COMPLETED,
                         bg_id=bg_id, tool=tool_name, success=success, **extra)

    def memory_promoted(self, entry_preview: str = "",
                        topic: str = "", **extra) -> dict:
        """发射 memory_promoted 事件。"""
        return self.emit(EventType.MEMORY_PROMOTED,
                         entry_preview=str(entry_preview)[:200],
                         topic=topic, **extra)

    # ── 属性 ──────────────────────────────────────

    @property
    def event_count(self) -> int:
        """已发射的事件数。"""
        return self._event_count


# ═══════════════════════════════════════════════════════
# RunStore.append_trace 适配
# ═══════════════════════════════════════════════════════

# TraceEmitter 调用 self._run_store.append_trace(event)。
# 需要在 RunStore 中添加此方法（如果还没有的话）。
# 如果 RunStore 已经有 log_trace(event)，则别名兼容。


# ═══════════════════════════════════════════════════════
# TraceReader — 读取和分析 trace 文件
# ═══════════════════════════════════════════════════════

class TraceReader:
    """Trace 文件读取和分析工具。

    用于事后复盘、指标统计和调试。

    使用模式：
      reader = TraceReader(trace_path)
      events = reader.load()
      model_events = reader.filter_by_type(EventType.MODEL_REQUESTED)
      stats = reader.stats()
    """

    def __init__(self, path: Path | str):
        """初始化读取器。

        Args:
            path: trace.jsonl 文件路径或 run 目录路径（自动查找 trace.jsonl）。
        """
        path = Path(path)
        if path.is_dir():
            path = path / "trace.jsonl"
        self._path = path
        self._events: list[dict] | None = None  # 懒加载

    def load(self, force: bool = False) -> list[dict]:
        """加载所有 trace 事件。

        Args:
            force: 强制重新加载（即使已缓存）。

        Returns:
            事件 dict 列表。
        """
        if self._events is not None and not force:
            return self._events
        self._events = read_jsonl(self._path)
        return self._events

    def filter_by_type(self, event_type: str) -> list[dict]:
        """按事件类型过滤。

        Args:
            event_type: EventType 常量。

        Returns:
            匹配的事件列表。
        """
        return [e for e in self.load() if e.get("event") == event_type]

    def filter_by_agent(self, agent_name: str) -> list[dict]:
        """按 Agent 名称过滤。"""
        return [e for e in self.load() if e.get("agent_name") == agent_name]

    def stats(self) -> dict:
        """生成 trace 统计摘要。

        Returns:
            {
                "total_events": ...,
                "event_counts": {"run_started": 1, "tool_executed": 5, ...},
                "duration_seconds": ...,
                "tool_executions": [...],
                "model_calls": ...,
                "compactions": ...,
                "errors": [...],
                "agents": {"lead": 20, "reviewer": 5, ...},
            }
        """
        events = self.load()
        if not events:
            return {"total_events": 0}

        event_counts: dict[str, int] = {}
        tool_executions: list[dict] = []
        model_calls = 0
        compactions = 0
        errors: list[str] = []
        agents: dict[str, int] = {}

        first_ts = None
        last_ts = None

        for e in events:
            event_type = e.get("event", "unknown")
            event_counts[event_type] = event_counts.get(event_type, 0) + 1

            if event_type == EventType.TOOL_EXECUTED:
                tool_executions.append({
                    "tool": e.get("tool", ""),
                    "success": e.get("success", True),
                    "duration_ms": e.get("duration_ms", 0),
                })
            elif event_type == EventType.MODEL_REQUESTED:
                model_calls += 1
            elif event_type == EventType.COMPACTION_TRIGGERED:
                compactions += 1
            elif event_type == EventType.RUN_ERROR:
                errors.append(e.get("error", ""))

            agent = e.get("agent_name") or e.get("agent_type") or "unknown"
            agents[agent] = agents.get(agent, 0) + 1

            ts = e.get("timestamp", 0)
            if ts:
                if first_ts is None or ts < first_ts:
                    first_ts = ts
                if last_ts is None or ts > last_ts:
                    last_ts = ts

        duration = (last_ts - first_ts) if first_ts and last_ts else 0

        return {
            "total_events": len(events),
            "event_counts": event_counts,
            "duration_seconds": round(duration, 2),
            "tool_executions": tool_executions,
            "tool_count": len(tool_executions),
            "model_calls": model_calls,
            "compactions": compactions,
            "errors": errors,
            "agents": agents,
        }

    def replay_timeline(self) -> str:
        """生成人类可读的事件时间线。

        Returns:
            多行字符串，按时间顺序展示事件。
        """
        events = self.load()
        if not events:
            return "(empty trace)"

        lines = []
        for e in events:
            ts = e.get("created_at", "")[:19]
            event = e.get("event", "?")
            detail = ""

            if event == EventType.TOOL_EXECUTED:
                tool = e.get("tool", "?")
                ok = "OK" if e.get("success", True) else "FAIL"
                ms = e.get("duration_ms", 0)
                detail = f" {tool} [{ok}] {ms}ms"
            elif event == EventType.MODEL_REQUESTED:
                detail = f" attempt={e.get('attempts', 0)} step={e.get('tool_steps', 0)}"
            elif event == EventType.MODEL_PARSED:
                detail = f" kind={e.get('kind', '?')} {e.get('duration_ms', 0)}ms"
            elif event == EventType.CHECKPOINT_CREATED:
                detail = f" {e.get('checkpoint_id', '?')} ({e.get('trigger', '?')})"
            elif event == EventType.RUN_FINISHED:
                detail = f" status={e.get('status', '?')} steps={e.get('tool_steps', 0)}"

            lines.append(f"{ts} [{event}]{detail}")

        return "\n".join(lines)


# ═══════════════════════════════════════════════════════
# JSONL 容错读取
# ═══════════════════════════════════════════════════════

def read_jsonl(path: Path | str) -> list[dict]:
    """读取 JSONL 文件，逐行解析，跳过损坏行。

    写入中途进程崩溃可能产生半行残缺 JSON —— 如果用 json.load()
    读取整个文件会直接抛 JSONDecodeError。逐行解析可以把损坏控制在一行。

    Args:
        path: JSONL 文件路径。

    Returns:
        成功解析的 dict 列表。损坏行被跳过并在日志中记录警告。
    """
    path = Path(path)
    if not path.exists():
        return []

    results = []
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            try:
                results.append(json.loads(line))
            except json.JSONDecodeError:
                logger.warning(f"跳过损坏行 {i} in {path}")
                continue

    return results
