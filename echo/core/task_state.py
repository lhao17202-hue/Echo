"""一次 ask() 运行过程中的状态机快照。

回答三个问题：
  这次用户请求当前进行到哪了？（status / tool_steps / attempts）
  调了多少次工具、最后为什么停下？（stop_reason / last_tool）
  多 Agent 协作了什么？（agent_type / bound_global_task_id / pending_protocols）

这个对象在运行中不断被 RunStore 写入 task_state.json，
供运行中观察、运行后复盘、以及 Checkpoint 断点恢复使用。

Echo run-state model with local task and collaboration metadata.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import ClassVar
from datetime import datetime
from uuid import uuid4
import time


# ═══════════════════════════════════════════════════════
# 状态常量
# ═══════════════════════════════════════════════════════

class Status(str, Enum):
    """运行状态。

    running:   主循环正在执行
    completed: 正常结束，模型返回了 final_answer
    stopped:   被终止（步数上限 / 审批被拒 / 用户中断）
    failed:    异常失败（模型错误 / 工具超时 / 持久化错误）
    """
    RUNNING = "running"
    COMPLETED = "completed"
    STOPPED = "stopped"
    FAILED = "failed"


class StopReason(str, Enum):
    """终止原因。status 说"停了"，stop_reason 说"为什么停了"。

    设计原则：
      stop_reason 和 status 分开存，是为了区分"怎么停的"和"停下时是什么状态"。
      例如 step_limit 是正常停止(stop_reason)，但状态是 stopped 而不是 completed。
    """
    FINAL_ANSWER = "final_answer_returned"     # 正常结束，模型给出了 final_answer
    STEP_LIMIT = "step_limit_reached"          # 工具调用次数超过上限
    ATTEMPT_LIMIT = "attempt_limit_reached"    # 模型调用轮次超过上限
    RETRY_LIMIT = "retry_limit_reached"        # 模型调用重试次数超过上限
    MODEL_ERROR = "model_error"                # 模型返回错误（限流 / 过载 / 空响应）
    TOOL_TIMEOUT = "tool_timeout"              # 工具执行超时
    APPROVAL_DENIED = "approval_denied"        # 用户拒绝了高危操作
    USER_INTERRUPT = "user_interrupt"          # Ctrl+C 中断
    MAX_TOKENS = "max_tokens_exceeded"         # 模型 max_tokens 达到上限
    EMPTY_RESPONSE = "empty_response"          # 模型返回了空内容
    DELEGATE_FAILED = "delegate_failed"          # 子 Agent 委托失败
    PERSISTENCE_ERROR = "persistence_error"    # 持久化写入失败
    RESUME_LOAD_ERROR = "resume_load_error"    # Checkpoint 恢复加载失败
    CONTEXT_OVERFLOW = "context_overflow"      # 上下文超出模型上限无法恢复


# ═══════════════════════════════════════════════════════
# TaskState
# ═══════════════════════════════════════════════════════

@dataclass
class TaskState:
    """单次 ask() 运行时的完整状态快照。

    字段分四组：
      身份标识  — 谁发起的、属于哪个 Agent、运行在哪
      进度追踪  — 调了多少次工具、模型调了几轮
      多 Agent  — 队友、全局任务、协议、后台任务
      终止信息  — 怎么停的、最终回复是什么

    Echo 扩展字段：
      agent_type / agent_name  — 区分 lead / teammate / subagent
      worktree_name            — 队友绑定到哪个工作区
      bound_global_task_id     — 关联到 GlobalTaskManager 的任务 ID
      active_background_tasks  — 运行中的后台任务列表
      pending_protocols        — 待处理的协议请求 ID
      compact_count            — 上下文压缩次数
      todos                    — 模型自我规划的 todo 列表
    """

    # ── 身份标识 ──────────────────────────────────
    run_id: str = ""              # 本次运行的唯一 ID（run_20260715-143052-a1b2c3）
    task_id: str = ""             # 用户视角的任务 ID（可跨 run 追踪）

    agent_type: str = "lead"      # 代理类型："lead" | "teammate" | "subagent"
    agent_name: str | None = None # 代理名称（队友的名称，如 "code-reviewer"）
    worktree_name: str | None = None  # 绑定的 git worktree 名称

    # ── 状态 ──────────────────────────────────────
    status: Status = Status.RUNNING
    stop_reason: str = ""         # StopReason 的值

    # ── 输入 ──────────────────────────────────────
    user_request: str = ""        # 用户原始请求（单次 ask 的 prompt）

    # ── 进度追踪 ──────────────────────────────────
    tool_steps: int = 0           # 已执行的工具调用步数
    attempts: int = 0             # 模型被调用的总轮次数（不等于 tool_steps，因为一轮可能执行多个工具）
    last_tool: str = ""           # 上一个被调用的工具名

    # ── 多 Agent & 高级特性 ────────────────────────
    bound_global_task_id: str | None = None   # 关联的 GlobalTask ID
    active_background_tasks: list[str] = field(default_factory=list)  # 运行中的后台任务 bg_id 列表
    pending_protocols: list[str] = field(default_factory=list)        # 待处理的协议请求 ID
    todos: list[dict] = field(default_factory=list)                   # 模型自我规划的 todo 列表
    active_teammates: dict = field(default_factory=dict)              # {name: teammate snapshot}
    global_task_ids: list[str] = field(default_factory=list)          # GlobalTask IDs relevant to this run
    unprocessed_messages: list = field(default_factory=list)          # Message snapshots not yet injected

    # ── 上下文 & 检查点 ──────────────────────────
    compact_count: int = 0        # 上下文压缩触发次数
    checkpoint_id: str = ""       # 当前检查点 ID
    resume_status: str = ""       # 恢复状态："full-valid" | "partial-stale" | "workspace-mismatch"
    depth: int = 0                # 代理嵌套深度（0 = 主 Agent，1 = 子 Agent）

    # ── 结果 ──────────────────────────────────────
    final_answer: str = ""        # 模型的最终回复
    errors: list[str] = field(default_factory=list)  # 异常信息列表

    # ── 时间 ──────────────────────────────────────
    started_at: float = field(default_factory=time.time)
    finished_at: float | None = None

    # ═══════════════════════════════════════════════
    # 工厂方法
    # ═══════════════════════════════════════════════

    @classmethod
    def create(cls, user_request: str, task_id: str = "", run_id: str = "",
               agent_type: str = "lead", agent_name: str | None = None) -> "TaskState":
        """创建一个新的 TaskState 实例。

        Args:
            user_request: 用户原始请求文本。
            task_id: 用户视角的任务 ID，为空则自动生成。
            run_id: 本次运行 ID，为空则自动生成（格式：run_YYYYMMDD-HHMMSS-随机6位hex）。
            agent_type: 代理类型。
            agent_name: 代理名称。

        Returns:
            新的 TaskState 实例，状态为 RUNNING。
        """
        if not run_id:
            run_id = "run_" + datetime.now().strftime("%Y%m%d-%H%M%S") + "-" + uuid4().hex[:6]
        if not task_id:
            task_id = "task_" + uuid4().hex[:8]
        return cls(
            run_id=run_id,
            task_id=task_id,
            user_request=user_request,
            agent_type=agent_type,
            agent_name=agent_name,
        )

    @classmethod
    def _safe_status(cls, value: str) -> Status:
        """安全地解析状态值，无效值回退到 RUNNING。"""
        try:
            return Status(value)
        except ValueError:
            return Status.RUNNING

    @classmethod
    def from_dict(cls, data: dict) -> "TaskState":
        """从字典反序列化 TaskState。

        用于从 task_state.json 恢复状态。缺失字段使用默认值。

        Args:
            data: 序列化后的字典。

        Returns:
            反序列化的 TaskState 实例。
        """
        return cls(
            run_id=str(data.get("run_id", "")),
            task_id=str(data.get("task_id", "")),
            agent_type=str(data.get("agent_type", "lead")),
            agent_name=data.get("agent_name"),
            worktree_name=data.get("worktree_name"),
            status=cls._safe_status(data.get("status", Status.RUNNING.value)),
            stop_reason=str(data.get("stop_reason", "")),
            user_request=str(data.get("user_request", "")),
            tool_steps=int(data.get("tool_steps", 0)),
            attempts=int(data.get("attempts", 0)),
            last_tool=str(data.get("last_tool", "")),
            bound_global_task_id=data.get("bound_global_task_id"),
            active_background_tasks=list(data.get("active_background_tasks", [])),
            pending_protocols=list(data.get("pending_protocols", [])),
            todos=list(data.get("todos", [])),
            active_teammates=dict(data.get("active_teammates", {})),
            global_task_ids=list(data.get("global_task_ids", [])),
            unprocessed_messages=list(data.get("unprocessed_messages", [])),
            compact_count=int(data.get("compact_count", 0)),
            checkpoint_id=str(data.get("checkpoint_id", "")),
            resume_status=str(data.get("resume_status", "")),
            depth=int(data.get("depth", 0)),
            final_answer=str(data.get("final_answer", "")),
            errors=list(data.get("errors", [])),
            started_at=float(data.get("started_at", time.time())),
            finished_at=float(data["finished_at"]) if data.get("finished_at") is not None else None,
        )

    # ═══════════════════════════════════════════════
    # 进度记录方法（链式调用，返回 self）
    # ═══════════════════════════════════════════════

    def record_attempt(self) -> "TaskState":
        """记录一次模型调用。

        attempt 统计的是"模型被调用了几轮"，不等于 tool_steps。
        一轮模型调用可能返回多个 tool_use block。
        """
        self.attempts += 1
        return self

    def record_tool(self, name: str) -> "TaskState":
        """记录一次成功的工具执行。

        只有 result.success=True 时才调用（由 AgentLoop 判断）。
        被 Hook 拦截、沙箱逃逸、工具内部失败的调用均不计入。
        """
        self.tool_steps += 1
        self.last_tool = str(name or "")
        return self

    def add_background_task(self, bg_id: str) -> "TaskState":
        """记录一个正在运行的后台任务。"""
        self.active_background_tasks.append(bg_id)
        return self

    def remove_background_task(self, bg_id: str) -> "TaskState":
        """标记一个后台任务已完成。"""
        if bg_id in self.active_background_tasks:
            self.active_background_tasks.remove(bg_id)
        return self

    # ═══════════════════════════════════════════════
    # 终止方法
    # ═══════════════════════════════════════════════

    def finish_success(self, final_answer: str) -> "TaskState":
        """标记任务正常完成。

        Args:
            final_answer: 模型的最终回复文本。

        Raises:
            ValueError: 如果当前状态不允许转移到 COMPLETED。
        """
        ok, msg = self.validate_transition(Status.COMPLETED)
        if not ok:
            raise ValueError(msg)
        self.status = Status.COMPLETED
        self.stop_reason = StopReason.FINAL_ANSWER.value
        self.final_answer = str(final_answer)
        self.finished_at = time.time()
        return self

    def stop(self, reason: str, status: Status = Status.STOPPED,
             final_answer: str = "") -> "TaskState":
        """通用终止方法。

        Args:
            reason: StopReason 的值。
            status: 终止后的状态（默认 STOPPED）。
            final_answer: 如果模型在终止前已给出部分回复，可传入。

        Raises:
            ValueError: 如果当前状态不允许转移到目标状态。
        """
        ok, msg = self.validate_transition(status)
        if not ok:
            raise ValueError(msg)
        self.status = status
        self.stop_reason = reason
        self.finished_at = time.time()
        if final_answer:
            self.final_answer = final_answer
        return self

    def stop_step_limit(self) -> "TaskState":
        """步数上限终止。"""
        return self.stop(StopReason.STEP_LIMIT.value)

    def stop_attempt_limit(self) -> "TaskState":
        """模型调用轮次上限终止。"""
        return self.stop(StopReason.ATTEMPT_LIMIT.value)

    def stop_retry_limit(self) -> "TaskState":
        """重试上限终止。"""
        return self.stop(StopReason.RETRY_LIMIT.value)

    def stop_model_error(self, error: str = "") -> "TaskState":
        """模型错误终止（属于 FAILED 状态）。"""
        # 先验证转移，再修改状态（避免验证失败时 errors 已被污染）
        result = self.stop(StopReason.MODEL_ERROR.value, status=Status.FAILED)
        if error:
            self.errors.append(error)
        return result

    def stop_approval_denied(self) -> "TaskState":
        """用户拒绝高危操作终止。"""
        return self.stop(StopReason.APPROVAL_DENIED.value)

    def stop_user_interrupt(self) -> "TaskState":
        """用户 Ctrl+C 中断。"""
        return self.stop(StopReason.USER_INTERRUPT.value)

    # ═══════════════════════════════════════════════
    # 查询属性
    # ═══════════════════════════════════════════════

    @property
    def is_running(self) -> bool:
        """是否仍在运行中。"""
        return self.status == Status.RUNNING

    @property
    def is_terminal(self) -> bool:
        """是否已终止（无论成功/停止/失败）。"""
        return self.status != Status.RUNNING

    @property
    def is_success(self) -> bool:
        """是否正常完成。"""
        return self.status == Status.COMPLETED

    @property
    def is_failed(self) -> bool:
        """是否异常失败。"""
        return self.status == Status.FAILED

    @property
    def is_stopped(self) -> bool:
        """是否被终止（非失败）。"""
        return self.status == Status.STOPPED

    @property
    def duration_seconds(self) -> float | None:
        """运行时长（秒），运行中则为 None。"""
        if self.finished_at is None:
            return None
        return self.finished_at - self.started_at

    @property
    def can_resume(self) -> bool:
        """此状态是否可以通过 --resume 恢复。"""
        return self.status in (Status.STOPPED, Status.FAILED)

    @property
    def has_errors(self) -> bool:
        """是否有错误记录。"""
        return len(self.errors) > 0

    # ═══════════════════════════════════════════════
    # 状态转移验证
    # ═══════════════════════════════════════════════

    # 允许的状态转移映射：{当前状态: {允许转移到的目标状态}}
    # 设计原则：
    #   - RUNNING 可以转移到任何状态
    #   - COMPLETED/STOPPED/FAILED 是终态，不可再转移
    TRANSITIONS: ClassVar[dict] = {
        Status.RUNNING:   {Status.COMPLETED, Status.STOPPED, Status.FAILED},
        Status.COMPLETED: set(),   # 终态，不可转移
        Status.STOPPED:   set(),   # 终态，不可转移
        Status.FAILED:    set(),   # 终态，不可转移
    }

    def validate_transition(self, target: Status) -> tuple[bool, str]:
        """验证从当前状态到目标状态的转移是否合法。

        Args:
            target: 目标状态。

        Returns:
            (是否合法, 原因描述)。
        """
        if target not in self.TRANSITIONS:
            return False, f"未知的目标状态: {target.value}"
        allowed = self.TRANSITIONS.get(self.status, set())
        if target not in allowed:
            return False, (
                f"非法的状态转移: {self.status.value} → {target.value}。"
                f"当前状态是终态，不允许再转移。"
            )
        return True, ""

    # ═══════════════════════════════════════════════
    # 序列化
    # ═══════════════════════════════════════════════

    def to_dict(self) -> dict:
        """将 TaskState 序列化为字典。

        Returns:
            可被 JSON 序列化的字典，包含所有状态字段。
        """
        return {
            "run_id": self.run_id,
            "task_id": self.task_id,
            "agent_type": self.agent_type,
            "agent_name": self.agent_name,
            "worktree_name": self.worktree_name,
            "status": self.status.value,
            "stop_reason": self.stop_reason,
            "user_request": self.user_request,
            "tool_steps": self.tool_steps,
            "attempts": self.attempts,
            "last_tool": self.last_tool,
            "bound_global_task_id": self.bound_global_task_id,
            "active_background_tasks": list(self.active_background_tasks),
            "pending_protocols": list(self.pending_protocols),
            "todos": list(self.todos),
            "active_teammates": self.active_teammates,
            "global_task_ids": list(self.global_task_ids),
            "unprocessed_messages": list(self.unprocessed_messages),
            "compact_count": self.compact_count,
            "checkpoint_id": self.checkpoint_id,
            "resume_status": self.resume_status,
            "depth": self.depth,
            "final_answer": self.final_answer,
            "errors": list(self.errors),
            "started_at": self.started_at,
            "finished_at": self.finished_at,
        }


# ═══════════════════════════════════════════════════════
# Checkpoint — 检查点数据模型
# ═══════════════════════════════════════════════════════

@dataclass
class Checkpoint:
    """检查点：用于断点恢复。

    恢复边界（重要）：
      主 Agent Checkpoint → 恢复主对话 + 全局任务映射 + 未处理消息 + 待处理协议 ID
      队友 Checkpoint     → 每个队友独立维护，由 TeammateManager 逐个恢复

    不支持"单 Checkpoint 恢复全链路状态"——每个 Agent 的 Checkpoint 是独立的。
    """

    # ── 身份 ──────────────────────────────────────
    checkpoint_id: str = field(default_factory=lambda: "ckpt_" + uuid4().hex[:8])
    parent_id: str | None = None      # 上一个检查点的 ID（形成链表）
    schema_version: str = "v1"        # 检查点 schema 版本（升级时用于兼容判断）

    # ── 任务进度 ──────────────────────────────────
    current_goal: str = ""            # 用户原始请求（从 TaskState.user_request 复制）
    completed: list[str] = field(default_factory=list)   # 已完成的事项
    excluded: list[str] = field(default_factory=list)    # 已排除的方案
    current_blocker: str | None = None  # 当前阻塞原因（如果任务未完成）
    next_step: str = ""               # 推断的下一步操作

    # ── 文件状态 ──────────────────────────────────
    key_files: dict[str, str] = field(default_factory=dict)  # {路径: SHA-256 哈希}

    # ── 运行环境指纹 ──────────────────────────────
    runtime_identity: dict = field(default_factory=dict)
    # 包含：cwd, model, model_client_class, approval_policy,
    #       read_only, max_steps, max_new_tokens, feature_flags,
    #       shell_env_allowlist, workspace_fingerprint, tool_signature

    # ── 多 Agent 恢复字段──────────────
    snapshot_teammates: dict = field(default_factory=dict)
    # 活跃队友快照：{name: {session_id, worktree, status}}

    unprocessed_messages: list = field(default_factory=list)
    # 未处理的消息（来自 MessageBus，尚未注入主 Agent 对话历史）

    pending_protocols: list[str] = field(default_factory=list)
    # 待处理的协议请求 ID 列表（如 plan_approval、shutdown）

    created_at: str = ""              # 创建时间 ISO 格式


# ═══════════════════════════════════════════════════════
# ResumeStatus — 断点恢复状态枚举
# ═══════════════════════════════════════════════════════

class ResumeStatus(str, Enum):
    """检查点恢复状态。

    用于标记从断点恢复时，上次的运行环境和当前环境是否一致。
    不同状态影响 Agent 的恢复行为：
      full-valid          → 直接续跑，无需提示
      partial-stale       → 注入提示告知模型某些文件可能已被外部修改
      workspace-mismatch  → 警告环境不一致，但仍可尝试续跑
      schema-mismatch     → 检查点格式不兼容，无法恢复
      no-checkpoint       → 没有检查点，从头开始
    """
    FULL_VALID = "full-valid"
    PARTIAL_STALE = "partial-stale"
    WORKSPACE_MISMATCH = "workspace-mismatch"
    SCHEMA_MISMATCH = "schema-mismatch"
    NO_CHECKPOINT = "no-checkpoint"

    @classmethod
    def can_continue(cls, status: str) -> bool:
        """此恢复状态是否允许继续执行。

        full-valid 和 partial-stale 都可以续跑；
        workspace-mismatch 严格来说不推荐但可尝试；
        schema-mismatch 绝对不能。
        """
        return status in (cls.FULL_VALID.value, cls.PARTIAL_STALE.value,
                          cls.WORKSPACE_MISMATCH.value)

    @classmethod
    def needs_warning(cls, status: str) -> bool:
        """此恢复状态是否需要给模型注入警告提示。"""
        return status in (cls.PARTIAL_STALE.value, cls.WORKSPACE_MISMATCH.value)


# ═══════════════════════════════════════════════════════
# 状态机总结辅助工具
# ═══════════════════════════════════════════════════════

def state_summary(state: TaskState) -> str:
    """生成单行状态摘要（用于日志和调试）。

    Args:
        state: TaskState 实例。

    Returns:
        格式化的单行摘要字符串。
    """
    parts = [
        f"[{state.run_id}]",
        f"status={state.status.value}",
    ]
    if state.stop_reason:
        parts.append(f"reason={state.stop_reason}")
    parts.append(f"steps={state.tool_steps}")
    parts.append(f"attempts={state.attempts}")
    if state.agent_type != "lead":
        parts.append(f"agent={state.agent_type}")
        if state.agent_name:
            parts[-1] += f"/{state.agent_name}"
    if state.bound_global_task_id:
        parts.append(f"gtask={state.bound_global_task_id}")
    if state.errors:
        parts.append(f"errors={len(state.errors)}")
    return " ".join(parts)


def is_terminal_status(status_value: str) -> bool:
    """检查状态值是否为终态。

    Args:
        status_value: Status 的字符串值。

    Returns:
        True 表示 COMPLETED / STOPPED / FAILED。
    """
    return status_value in (
        Status.COMPLETED.value,
        Status.STOPPED.value,
        Status.FAILED.value,
    )
