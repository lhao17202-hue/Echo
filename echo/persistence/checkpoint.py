"""检查点与断点恢复系统。

设计思路：
  每个工具执行后自动创建一个检查点，记录：
    文件新鲜度    — 关键文件的 SHA-256 哈希（检测是否被外部修改）
    运行时身份    — 当前环境指纹（模型、工具集、配置等）
    任务进度      — 当前目标、已排除方案、阻塞原因、下一步

  恢复时：
    文件新鲜度校验 → stale 则标记 partial-stale
    运行时身份比对 → 不匹配则标记 workspace-mismatch
    全部一致       → 标记 full-valid，直接续跑

Echo 扩展：
  snapshot_teammates    — 活跃队友快照（恢复时重建队友线程）
  unprocessed_messages  — MessageBus 中尚未消费的消息
  pending_protocols     — 待处理的协议 ID
"""

import hashlib
import time
import uuid
from pathlib import Path
from typing import Any
from echo.core.task_state import TaskState, Checkpoint


# ═══════════════════════════════════════════════════════
# 常量
# ═══════════════════════════════════════════════════════

CHECKPOINT_SCHEMA_VERSION = "v1"

# 检查点恢复状态
CHECKPOINT_NONE = "no-checkpoint"
CHECKPOINT_FULL_VALID = "full-valid"
CHECKPOINT_PARTIAL_STALE = "partial-stale"
CHECKPOINT_WORKSPACE_MISMATCH = "workspace-mismatch"
CHECKPOINT_SCHEMA_MISMATCH = "schema-mismatch"

# 恢复时比对的运行时身份字段
RUNTIME_IDENTITY_KEYS = (
    "cwd",                    # 工作区根目录
    "model",                  # 模型名称
    "model_client",           # 客户端类名
    "approval_policy",        # 审批策略
    "read_only",              # 只读模式
    "max_steps",              # 最大步数
    "max_new_tokens",         # 最大 token 数
    "feature_flags",          # 功能开关
    "shell_env_allowlist",    # Shell 环境白名单
    "workspace_fingerprint",  # 工作区指纹
    "tool_signature",         # 工具签名
)


# ═══════════════════════════════════════════════════════
# CheckpointManager
# ═══════════════════════════════════════════════════════

class CheckpointManager:
    """检查点管理器。

    核心职责：
      1. 创建检查点 —— 在每个工具执行后自动调用
      2. 评估恢复状态 —— 文件新鲜度 + 运行时身份比对
      3. 渲染检查点文本 —— 注入 system prompt 供 LLM 理解上下文

    恢复边界（重要）：
      主 Agent Checkpoint → 恢复主对话 + 全局任务映射 + 未处理消息 + 待处理协议
      队友 Checkpoint     → 每个队友独立维护，由 TeammateManager 逐个恢复

    使用模式：
      cm = CheckpointManager("/path/to/workspace")
      ckpt = cm.create(task_state, key_files=["src/main.py"])
      status = cm.evaluate(ckpt)         # → "full-valid"
      prompt_text = cm.render(ckpt)      # → 注入 system prompt 的文本
    """

    def __init__(self, workspace_root: str):
        """初始化检查点管理器。

        Args:
            workspace_root: 工作区根目录（用于捕获运行时身份中的 cwd）。
        """
        self._workspace = Path(workspace_root)

    # ── 创建检查点 ────────────────────────────────

    def create(self,
               state: TaskState,
               key_files: list[str] | None = None,
               recent_files: list[str] | None = None,
               snapshot_teammates: dict | None = None,
               unprocessed_messages: list | None = None,
               trigger: str = "tool_executed") -> Checkpoint:
        """创建一个新的检查点。

        在每次工具执行后调用。记录当前文件状态、运行环境和任务进度。

        Args:
            state: 当前 TaskState。
            key_files: 要跟踪的关键文件路径列表（通常是 recent_files）。
            recent_files: 最近访问的文件列表（也纳入新鲜度快照）。
            snapshot_teammates: 活跃队友的状态快照。
            unprocessed_messages: MessageBus 中尚未消费的消息。
            trigger: 触发原因（tool_executed / context_reduction / run_finished）。

        Returns:
            新创建的 Checkpoint 实例。
        """
        all_files = list(key_files or [])
        if recent_files:
            for f in recent_files:
                if f not in all_files:
                    all_files.append(f)

        # 计算每个文件的新鲜度
        key_files_hashes: dict[str, str] = {}
        for fpath in all_files:
            p = Path(fpath) if isinstance(fpath, str) else fpath
            if p.exists() and p.is_file():
                try:
                    key_files_hashes[str(p)] = hashlib.sha256(p.read_bytes()).hexdigest()
                except (OSError, PermissionError):
                    # 文件被锁定或权限不足，跳过
                    continue

        # 推断下一步
        next_step = self._infer_next_step(state)

        # 当前阻塞原因
        blocker = None
        if state.status.value != "running":
            blocker = state.stop_reason
        elif state.stop_reason and state.stop_reason not in ("", "final_answer_returned"):
            blocker = state.stop_reason

        ckpt = Checkpoint(
            parent_id=state.checkpoint_id or None,
            schema_version=CHECKPOINT_SCHEMA_VERSION,
            current_goal=state.user_request,
            completed=[state.final_answer] if state.final_answer else [],
            excluded=[],
            current_blocker=blocker,
            next_step=next_step,
            key_files=key_files_hashes,
            runtime_identity=self._get_identity(),
            snapshot_teammates=snapshot_teammates or {},
            unprocessed_messages=unprocessed_messages or [],
            pending_protocols=list(state.pending_protocols),
            created_at=time.strftime("%Y-%m-%dT%H:%M:%S"),
        )

        state.checkpoint_id = ckpt.checkpoint_id
        return ckpt

    # ── 评估恢复状态 ──────────────────────────────

    def evaluate(self, checkpoint: Checkpoint,
                 invalidated_files: list[str] | None = None) -> dict:
        """评估检查点的恢复状态。

        三步检查：
          1. Schema 兼容性 —— 检查点格式版本是否匹配
          2. 文件新鲜度    —— 关键文件自检查点后是否被修改
          3. 运行时身份    —— 当前环境是否与创建时一致

        Args:
            checkpoint: 要评估的 Checkpoint。
            invalidated_files: 已知的失效文件列表（来自记忆系统的新鲜度检查）。

        Returns:
            {
                "status": "full-valid" | "partial-stale" | "workspace-mismatch" | "schema-mismatch",
                "stale_paths": [...],
                "mismatch_fields": [...],
            }
        """
        stale_paths = list(invalidated_files or [])
        mismatch_fields: list[str] = []

        # 1. Schema 版本检查
        if checkpoint.schema_version != CHECKPOINT_SCHEMA_VERSION:
            return {
                "status": CHECKPOINT_SCHEMA_MISMATCH,
                "stale_paths": stale_paths,
                "mismatch_fields": [],
            }

        # 2. 文件新鲜度检查
        for fpath, old_hash in checkpoint.key_files.items():
            p = Path(fpath)
            if not p.exists():
                if fpath not in stale_paths:
                    stale_paths.append(fpath)
                continue

            try:
                new_hash = hashlib.sha256(p.read_bytes()).hexdigest()
                if new_hash != old_hash and fpath not in stale_paths:
                    stale_paths.append(fpath)
            except (OSError, PermissionError):
                # 文件现在不可读但之前可以 → 视为变化
                if fpath not in stale_paths:
                    stale_paths.append(fpath)

        # 3. 运行时身份检查
        # 注意：必须用 _get_identity()（完整身份），不能用 _capture_identity()（只有 3 个基础字段）。
        # create() 存的是 _get_identity() 的完整结果，evaluate() 必须用同样的方法比对。
        current_identity = self._get_identity()
        saved_identity = checkpoint.runtime_identity or {}
        for key in RUNTIME_IDENTITY_KEYS:
            if key not in saved_identity:
                continue
            if saved_identity.get(key) != current_identity.get(key):
                mismatch_fields.append(key)

        mismatch_fields.sort()

        # 判定最终状态
        if stale_paths:
            status = CHECKPOINT_PARTIAL_STALE
        elif mismatch_fields:
            status = CHECKPOINT_WORKSPACE_MISMATCH
        else:
            status = CHECKPOINT_FULL_VALID

        return {
            "status": status,
            "stale_paths": stale_paths,
            "mismatch_fields": mismatch_fields,
        }

    # ── 持久化（SessionStore 集成）─────────────────

    def save_to_session(self, checkpoint: Checkpoint, session) -> None:
        """将检查点写入 Session 的 checkpoints 字典中持久化。

        必须在创建检查点后调用，否则重启后检查点丢失。

        Args:
            checkpoint: 要持久化的 Checkpoint。
            session: Session 实例（含 checkpoints dict）。
        """
        session.checkpoints.setdefault("items", {})
        session.checkpoints["items"][checkpoint.checkpoint_id] = {
            "checkpoint_id": checkpoint.checkpoint_id,
            "parent_id": checkpoint.parent_id,
            "schema_version": checkpoint.schema_version,
            "current_goal": checkpoint.current_goal,
            "completed": checkpoint.completed,
            "excluded": checkpoint.excluded,
            "current_blocker": checkpoint.current_blocker,
            "next_step": checkpoint.next_step,
            "key_files": checkpoint.key_files,
            "runtime_identity": checkpoint.runtime_identity,
            "snapshot_teammates": checkpoint.snapshot_teammates,
            "unprocessed_messages": checkpoint.unprocessed_messages,
            "pending_protocols": checkpoint.pending_protocols,
            "created_at": checkpoint.created_at,
        }
        session.checkpoints["current_id"] = checkpoint.checkpoint_id

    def load_from_session(self, session, checkpoint_id: str = "") -> Checkpoint | None:
        """从 Session 中恢复检查点。

        Args:
            session: Session 实例。
            checkpoint_id: 要加载的检查点 ID。为空则加载 current_id 指向的检查点。

        Returns:
            恢复的 Checkpoint，如果不存在则返回 None。
        """
        items = session.checkpoints.get("items", {})
        cid = checkpoint_id or session.checkpoints.get("current_id", "")
        if not cid or cid not in items:
            return None

        data = items[cid]
        return Checkpoint(
            checkpoint_id=data.get("checkpoint_id", cid),
            parent_id=data.get("parent_id"),
            schema_version=data.get("schema_version", "v1"),
            current_goal=data.get("current_goal", ""),
            completed=data.get("completed", []),
            excluded=data.get("excluded", []),
            current_blocker=data.get("current_blocker"),
            next_step=data.get("next_step", ""),
            key_files=data.get("key_files", {}),
            runtime_identity=data.get("runtime_identity", {}),
            snapshot_teammates=data.get("snapshot_teammates", {}),
            unprocessed_messages=data.get("unprocessed_messages", []),
            pending_protocols=data.get("pending_protocols", []),
            created_at=data.get("created_at", ""),
        )

    def can_resume(self, checkpoint: Checkpoint,
                   invalidated_files: list[str] | None = None) -> bool:
        """快速检查：是否可以从此检查点恢复。

        Args:
            checkpoint: 检查点。
            invalidated_files: 已知的失效文件。

        Returns:
            full-valid 或 partial-stale 都可以续跑（partial-stale 只是注入提示）。
            只有 workspace-mismatch 和 schema-mismatch 不能恢复。
        """
        result = self.evaluate(checkpoint, invalidated_files)
        return result["status"] in (CHECKPOINT_FULL_VALID, CHECKPOINT_PARTIAL_STALE)

    # ── 渲染检查点文本 ────────────────────────────

    def render(self, checkpoint: Checkpoint,
               invalidated_files: list[str] | None = None) -> str:
        """将检查点渲染为 LLM 可读的文本。

        这段文本会被注入 system prompt，告诉模型：
          - 之前做到哪了
          - 为什么停下的
          - 哪些文件可能被外部修改了

        Args:
            checkpoint: 检查点。
            invalidated_files: 已知的失效文件列表。

        Returns:
            Markdown 格式的检查点文本（注入到 system prompt 中）。
        """
        result = self.evaluate(checkpoint, invalidated_files)
        lines = [
            "## Task Checkpoint",
            f"- Resume status: {result['status']}",
            f"- Current goal: {checkpoint.current_goal or '-'}",
            f"- Current blocker: {checkpoint.current_blocker or '-'}",
            f"- Next step: {checkpoint.next_step or '-'}",
        ]

        if checkpoint.key_files:
            files = list(checkpoint.key_files.keys())[:10]
            lines.append(f"- Key files: {', '.join(files)}")

        if checkpoint.completed:
            lines.append("- Completed: " + " | ".join(
                str(item)[:100] for item in checkpoint.completed
            ))

        if checkpoint.excluded:
            lines.append("- Excluded: " + " | ".join(
                str(item)[:100] for item in checkpoint.excluded
            ))

        if result["stale_paths"]:
            lines.append(f"- Stale paths (may have been modified externally): "
                         f"{', '.join(result['stale_paths'][:10])}")

        if result["mismatch_fields"]:
            lines.append(f"- Runtime changes: {', '.join(result['mismatch_fields'])}")

        if checkpoint.unprocessed_messages:
            lines.append(f"- Unprocessed messages from teammates: "
                         f"{len(checkpoint.unprocessed_messages)}")

        return "\n".join(lines)

    # ═══════════════════════════════════════════════
    # 内部方法
    # ═══════════════════════════════════════════════

    def _infer_next_step(self, state: TaskState) -> str:
        """根据 TaskState 推断下一步应该做什么。

        Args:
            state: 当前 TaskState。

        Returns:
            推断的下一步描述字符串。
        """
        if state.status.value == "completed":
            return "任务已完成，无需继续。"
        if state.stop_reason and "step_limit" in str(state.stop_reason):
            return "从上个检查点恢复，继续完成任务。"
        if state.last_tool:
            return f"执行完 {state.last_tool} 后，决定下一步操作。"
        return "从上个检查点恢复，继续执行任务。"

    def _capture_identity(self) -> dict:
        """捕获当前运行环境指纹。

        用于恢复时比对——如果环境变了（不同模型、不同配置），
        恢复可能不安全。

        Returns:
            当前环境的身份信息字典。
        """
        return {
            "cwd": str(self._workspace),
            "timestamp": time.time(),
            "checkpoint_schema": CHECKPOINT_SCHEMA_VERSION,
        }

    def update_identity(self, agent_config: dict) -> dict:
        """用 Echo Agent 的实际配置更新运行时身份。

        这个方法在 Agent 启动时调用一次，把完整的运行环境信息
        注入到身份指纹中，用于后续所有检查点的创建和对比。

        Args:
            agent_config: Echo Agent 配置字典，包含：
                model, model_client, approval_policy, read_only,
                max_steps, max_new_tokens, feature_flags,
                shell_env_allowlist, workspace_fingerprint, tool_signature

        Returns:
            更新后的完整运行时身份字典。
        """
        identity = self._capture_identity()
        identity.update(agent_config)
        # 保存完整身份，供 create() 时使用
        self._agent_identity = identity
        return identity

    def _get_identity(self) -> dict:
        """获取当前的完整运行时身份。

        优先使用 update_identity() 设置的 Agent 配置，
        否则回退到基础身份（只有 cwd + timestamp）。
        """
        return getattr(self, "_agent_identity", None) or self._capture_identity()
