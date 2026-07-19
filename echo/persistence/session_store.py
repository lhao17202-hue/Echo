"""会话级持久化存储。

设计思路：
  SessionStore 负责保存"可恢复的会话状态"——对话历史、记忆、队友、定时任务。
  它回答的问题是：下次启动时，Agent 还记得什么？

  RunStore 负责保存"单次运行的审计工件"——task_state、trace、report。
  两者分开后，恢复现场和复盘证据不会混在一起。

目录结构：
  .echo/
    sessions/{session_id}/
      session.json          ← 会话主状态（原子写入）
      runs/{run_id}/        ← 单次运行工件（由 RunStore 管理）
      checkpoints/          ← 断点快照（由 CheckpointManager 管理）
    global/
      tasks.json            ← 全局任务池（由 GlobalTaskManager 管理）
      mailboxes/            ← Agent 消息邮箱（由 MessageBus 管理）
      scheduled_tasks.json  ← 持久化定时任务（由 CronScheduler 管理）
"""

import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path


# ═══════════════════════════════════════════════════════
# Session 数据模型
# ═══════════════════════════════════════════════════════

@dataclass
class Session:
    """会话数据 —— Agent 的持久化记忆体。

    一次"打开终端 → 执行多轮对话 → 关闭终端"就是一个 Session。
    保存位置：.echo/sessions/{session_id}/session.json

    Echo 扩展字段：
      teammates          — 队友名称 → 子会话 ID 映射
      cron_jobs           — 持久化的定时任务列表
      checkpoints         — 断点 ID 列表（用于恢复）
      pending_protocols   — 待处理的协议状态（计划审批 / 关机等）
      feature_flags       — 功能开关（可跨会话持久化用户偏好）
    """

    # ── 身份 ──────────────────────────────────────
    session_id: str = ""          # 全局唯一会话 ID，格式：20260715-143052-a1b2c3
    created_at: str = field(default_factory=lambda: time.strftime("%Y-%m-%dT%H:%M:%S"))

    # ── 环境 ──────────────────────────────────────
    workspace_root: str = ""      # 工作区根目录（启动时注入，用于恢复时校验）

    # ── 模型配置 ──────────────────────────────────
    model_config: dict = field(default_factory=lambda: {
        "provider": "deepseek",
        "model": "deepseek-v4-pro",
        "max_tokens": 8000,
        "temperature": 0.0,
    })

    # ── 安全配置 ──────────────────────────────────
    security_config: dict = field(default_factory=lambda: {
        "approval_policy": "ask",       # "ask" | "auto" | "never"
        "read_only": False,             # 只读模式（子 Agent 用）
        "shell_env_allowlist": [        # Shell 环境变量白名单
            "HOME", "LANG", "PATH", "USER", "VIRTUAL_ENV",
        ],
        "secret_env_names": [],         # 用户显式标记的密钥环境变量名
    })

    # ── 对话历史 ──────────────────────────────────
    history: list = field(default_factory=list)
    # 格式：[{"role": "user"|"assistant", "content": [...]}, ...]
    # 这是完整的消息列表，可用于 --resume 恢复对话上下文

    # ── 短期记忆 ──────────────────────────────────
    short_term_memory: dict = field(default_factory=dict)
    # 格式：{"task_summary": "...", "recent_files": [...], "file_summaries": {...}}
    # 会话结束时从 DefaultMemory 序列化，恢复时重新加载

    # ── 多 Agent 状态 ──────────────────────────────
    teammates: dict = field(default_factory=dict)
    # 格式：{队友名称: 子会话 ID}，用于恢复时找到队友的 Session

    # ── Cron 定时任务 ─────────────────────────────
    cron_jobs: list = field(default_factory=list)
    # 格式：[{"job_id": "...", "cron_expr": "...", "prompt": "...", ...}, ...]
    # durable=True 的定时任务在 Session 关闭后仍然保留

    # ── 检查点 ────────────────────────────────────
    checkpoints: dict = field(default_factory=lambda: {
        "current_id": "",            # 当前活跃的检查点 ID
        "items": {},                 # {checkpoint_id: Checkpoint数据}
    })

    # ── 协议状态 ──────────────────────────────────
    pending_protocols: list[dict] = field(default_factory=list)
    # 格式：[{"request_id": "...", "type": "plan_approval", "status": "waiting", ...}, ...]
    # 断点恢复时用于重建 ProtocolEngine 的待处理协议

    # ── 功能开关 ──────────────────────────────────
    feature_flags: dict = field(default_factory=lambda: {
        "memory": True,              # 是否启用记忆系统
        "compaction": True,          # 是否启用上下文压缩
        "hooks": True,               # 是否启用 Hook 管道
        "cron": False,               # 是否启用定时任务
        "background_tasks": True,    # 是否启用后台任务
    })

    # ── 运行时身份（恢复校验用）───────────────────
    runtime_identity: dict = field(default_factory=dict)
    # 运行时身份字段：
    # cwd, model, model_client, approval_policy, read_only,
    # max_steps, max_new_tokens, feature_flags, shell_env_allowlist,
    # workspace_fingerprint, tool_signature
    # 恢复时比对，如果有变化标记为 workspace-mismatch

    # ── 恢复状态（运行时填充，不持久化）───────────
    resume_state: dict = field(default_factory=dict)
    # 格式：{"status": "full-valid"|"partial-stale"|..., "stale_paths": [...], ...}

    # ═══════════════════════════════════════════════
    # 序列化
    # ═══════════════════════════════════════════════

    def to_dict(self) -> dict:
        """将 Session 序列化为 JSON 兼容的字典。

        运行时状态（resume_state）不序列化——那是临时数据。
        """
        return {
            "session_id": self.session_id,
            "created_at": self.created_at,
            "workspace_root": self.workspace_root,
            "model_config": self.model_config,
            "security_config": self.security_config,
            "history": self.history,
            "short_term_memory": self.short_term_memory,
            "teammates": self.teammates,
            "cron_jobs": self.cron_jobs,
            "checkpoints": self.checkpoints,
            "pending_protocols": self.pending_protocols,
            "feature_flags": self.feature_flags,
            "runtime_identity": self.runtime_identity,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Session":
        """从字典反序列化 Session。

        Args:
            data: JSON 反序列化后的字典。

        Returns:
            Session 实例。
        """
        return cls(
            session_id=str(data.get("session_id", "")),
            created_at=str(data.get("created_at", "")),
            workspace_root=str(data.get("workspace_root", "")),
            model_config=dict(data.get("model_config", {})),
            security_config=dict(data.get("security_config", {})),
            history=list(data.get("history", [])),
            short_term_memory=dict(data.get("short_term_memory", {})),
            teammates=dict(data.get("teammates", {})),
            cron_jobs=list(data.get("cron_jobs", [])),
            checkpoints=dict(data.get("checkpoints", {"current_id": "", "items": {}})),
            pending_protocols=list(data.get("pending_protocols", [])),
            feature_flags=dict(data.get("feature_flags", {})),
            runtime_identity=dict(data.get("runtime_identity", {})),
        )


# ═══════════════════════════════════════════════════════
# SessionStore
# ═══════════════════════════════════════════════════════

class SessionStore:
    """会话持久化仓库。

    存储位置：.echo/sessions/{session_id}/session.json

    核心特性：
      原子写入  — 先写 .tmp 临时文件，再 os.replace 原子替换，杜绝半截 JSON
      断电安全  — 替换操作是操作系统级别的原子操作
      可列表    — list_sessions() 按修改时间倒序列出最近会话

    使用模式：
      store = SessionStore("/path/to/workspace")
      store.save(session)                    # 持久化
      session = store.load(session_id)       # 恢复
      latest_id = store.latest()             # 最新会话
    """

    def __init__(self, workspace_root: str):
        """初始化会话仓库。

        Args:
            workspace_root: 工作区根目录。sessions 将保存在 .echo/sessions/ 下。
        """
        self._base = Path(workspace_root) / ".echo" / "sessions"

    # ── 路径工具 ──────────────────────────────────

    def _session_dir(self, session_id: str) -> Path:
        """获取会话目录路径。"""
        return self._base / session_id

    def _session_path(self, session_id: str) -> Path:
        """获取 session.json 完整路径。"""
        return self._session_dir(session_id) / "session.json"

    # ── CRUD ──────────────────────────────────────

    def save(self, session: Session) -> Path:
        """保存会话到磁盘（原子写入）。

        Args:
            session: Session 实例。

        Returns:
            写入的文件路径。
        """
        path = self._session_path(session.session_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        data = session.to_dict()
        self._atomic_write(path, json.dumps(data, indent=2, ensure_ascii=False))
        return path

    def load(self, session_id: str) -> Session:
        """加载会话。

        Args:
            session_id: 会话 ID。

        Returns:
            Session 实例。

        Raises:
            FileNotFoundError: session 不存在。
        """
        path = self._session_path(session_id)
        if not path.exists():
            raise FileNotFoundError(f"会话不存在: {session_id}")
        data = json.loads(path.read_text(encoding="utf-8"))
        return Session.from_dict(data)

    def delete(self, session_id: str) -> bool:
        """删除会话目录及其所有内容。

        Args:
            session_id: 会话 ID。

        Returns:
            成功删除返回 True，会话不存在返回 False。
        """
        import shutil
        session_dir = self._session_dir(session_id)
        if not session_dir.exists():
            return False
        shutil.rmtree(session_dir)
        return True

    def latest(self) -> str | None:
        """获取最近修改的会话 ID。

        Returns:
            最新 session ID，如果没有会话则返回 None。
        """
        if not self._base.exists():
            return None
        files = sorted(self._base.glob("*/session.json"),
                       key=lambda p: p.stat().st_mtime)
        if not files:
            return None
        return files[-1].parent.name

    def list_sessions(self, limit: int = 10) -> list[dict]:
        """列出最近的会话（按修改时间倒序）。

        Args:
            limit: 最多返回的会话数。

        Returns:
            [{"session_id": ..., "created_at": ..., "modified_at": ...}, ...]
        """
        if not self._base.exists():
            return []
        dirs = sorted(
            [d for d in self._base.iterdir() if d.is_dir()],
            key=lambda d: d.stat().st_mtime,
            reverse=True,
        )
        result = []
        for d in dirs[:limit]:
            session_path = d / "session.json"
            if session_path.exists():
                try:
                    data = json.loads(session_path.read_text(encoding="utf-8"))
                    result.append({
                        "session_id": d.name,
                        "created_at": data.get("created_at", ""),
                        "workspace_root": data.get("workspace_root", ""),
                        "modified_at": time.strftime(
                            "%Y-%m-%dT%H:%M:%S",
                            time.localtime(session_path.stat().st_mtime),
                        ),
                    })
                except (json.JSONDecodeError, OSError):
                    # 跳过损坏的会话文件
                    continue
        return result

    # ── 原子写入 ──────────────────────────────────

    @staticmethod
    def _atomic_write(path: Path, content: str) -> None:
        """原子写入文件。

        先写入 .tmp 临时文件（同一目录下，保证同文件系统），
        再 os.replace（POSIX rename，原子操作）。
        这样即使中途断电或异常退出，也不会留下半截 JSON。

        使用 tempfile + os.replace 模式。
        """
        tmp = path.with_suffix(".tmp")
        tmp.parent.mkdir(parents=True, exist_ok=True)
        tmp.write_text(content, encoding="utf-8")
        os.replace(tmp, path)
