"""记忆系统 — MemoryEntry 数据模型 + MemoryBackend 存储抽象 + MemoryManager 管理层。

两层设计：
  MemoryBackend（存储层）— 纯 CRUD，不包含智能判断。
    可替换为 ChromaDB / FAISS / Qdrant 等任何后端。
    默认实现：KeywordMemory（关键词匹配 + SHA-256 新鲜度）。

  MemoryManager（管理层）— 决定「哪些记忆该记、哪些该忘、何时提升为持久记忆」。
    当前使用基于规则的简单策略，后续可替换为 AI 驱动的智能判断。
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
import time
import uuid


# ═══════════════════════════════════════════════════════
# MemoryEntry — 记忆条目
# ═══════════════════════════════════════════════════════

@dataclass
class MemoryEntry:
    """记忆条目。

    Attributes:
        text: 记忆内容。
        tags: 标签列表（用于关键词匹配）。
        source: 来源（read_file / write_file / user / system）。
        kind: "episodic"（短期）或 "durable"（持久）。
        created_at: 创建时间戳。
        freshness_hash: 文件记忆的 SHA-256（用于检测过期）。
        file_path: 关联的文件路径。
        entry_id: 唯一 ID。
        metadata: 扩展元数据。
    """

    text: str = ""
    tags: list[str] = field(default_factory=list)
    source: str = ""
    kind: str = "episodic"
    created_at: float = field(default_factory=time.time)
    freshness_hash: str | None = None
    file_path: str | None = None
    entry_id: str = field(default_factory=lambda: uuid.uuid4().hex[:8])
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "text": self.text, "tags": self.tags, "source": self.source,
            "kind": self.kind, "created_at": self.created_at,
            "freshness_hash": self.freshness_hash, "file_path": self.file_path,
            "entry_id": self.entry_id, "metadata": self.metadata,
        }


# ═══════════════════════════════════════════════════════
# MemoryBackend — 存储层抽象（可替换为向量数据库）
# ═══════════════════════════════════════════════════════

class MemoryBackend(ABC):
    """记忆存储后端抽象。

    纯 CRUD，不包含智能判断。上层 MemoryManager 负责决策逻辑。

    内置实现：
      KeywordMemory       — 关键词匹配 + SHA-256 新鲜度（当前默认）
      未来可替换:
      ChromaMemory        — ChromaDB 向量检索
      FAISSMemory         — FAISS 向量检索
      QdrantMemory        — Qdrant 向量检索
    """

    @abstractmethod
    def store(self, entry: MemoryEntry) -> str:
        """存入一条记忆，返回 entry_id。"""
        ...

    @abstractmethod
    def search(self, query: str, top_k: int = 5) -> list[MemoryEntry]:
        """语义搜索记忆（具体语义取决于后端实现）。

        纯关键词后端做关键词匹配；向量后端做 embedding 相似度。
        """
        ...

    @abstractmethod
    def get(self, entry_id: str) -> MemoryEntry | None:
        """按 ID 获取记忆。"""
        ...

    @abstractmethod
    def list_all(self, limit: int = 50) -> list[MemoryEntry]:
        """列出所有记忆（按时间倒序）。"""
        ...

    @abstractmethod
    def delete(self, entry_id: str) -> bool:
        """删除一条记忆。"""
        ...

    @abstractmethod
    def clear(self) -> None:
        """清空所有短期记忆。"""
        ...

    @abstractmethod
    def count(self) -> int:
        """记忆总数。"""
        ...


# ═══════════════════════════════════════════════════════
# MemoryManager — 管理层（规则驱动，可替换为 AI 驱动）
# ═══════════════════════════════════════════════════════

class MemoryManager:
    """记忆管理器 —— AgentLoop 与记忆存储之间的中间层。

    职责：
      1. 委派 CRUD 到 MemoryBackend
      2. 从工具输出中自动提取记忆（observe_tool_result）
      3. 管理持久记忆的提升与衰减（promote / decay）
      4. 渲染工作记忆为 LLM prompt 文本（render_working）
      5. 文件摘要的新鲜度管理

    当前使用基于规则的简单策略：
      - read_file → 自动记录文件摘要
      - write_file/patch_file → 失效旧摘要
      - run_shell 的错误输出 → 记录为笔记
    """

    def __init__(self, backend: MemoryBackend,
                 durable_backend: MemoryBackend | None = None):
        """初始化管理器。

        Args:
            backend: 工作记忆后端（如 KeywordMemory）。
            durable_backend: 持久记忆后端（如 JsonDurableMemoryBackend），可选。
        """
        self._backend = backend
        self._durable = durable_backend
        self._task_summary: str = ""
        self._recent_files: list[str] = []
        self._file_summaries: dict[str, MemoryEntry] = {}

       # ── 委派到后端 ──────────────────────────────

    def add(self, content: str, metadata: dict | None = None) -> str:
        """快捷添加一条纯文本记忆。"""
        md = metadata or {}
        entry = MemoryEntry(
            text=content,
            tags=md.get("tags", []),
            source=md.get("source", "system"),
            metadata=md,
        )
        return self._backend.store(entry)

    def search(self, query: str, top_k: int = 5,
               include_durable: bool = True) -> list[MemoryEntry]:
        """搜索记忆（working + durable 合并排序，保底 durable 不会被挤掉）。"""
        query_lower = query.lower()
        query_words = set(query_lower.split())

        def _score(entry: MemoryEntry) -> float:
            text_lower = entry.text.lower()
            s = sum(2.0 for w in query_words if w in text_lower)
            s += sum(3.0 for t in entry.tags if any(w in t.lower() for w in query_words))
            s += 1.0 if any(w in entry.source.lower() for w in query_words) else 0.0
            s += min(entry.metadata.get("usage_count", 0), 10) * 0.05 if entry.metadata else 0.0
            s += (entry.metadata.get("score", 1.0) * 0.1) if entry.metadata else 0.0
            return s

        # 各自取 top_k * 2 再统一排序，保证 durable 有机会进入
        working = self._backend.search(query, top_k * 2)
        durable = []
        if include_durable and self._durable:
            durable = self._durable.search(query, top_k * 2)

        # 合并去重，统一评分
        seen: set[str] = set()
        all_entries: list[MemoryEntry] = []
        for e in working + durable:
            if e.entry_id not in seen:
                seen.add(e.entry_id)
                all_entries.append(e)

        all_entries.sort(key=_score, reverse=True)
        return all_entries[:top_k]

    def retrieve(self, query: str, top_k: int = 5) -> list[dict]:
        """检索 working memory（只查近期，不混入 durable）。"""
        entries = self._backend.search(query, top_k)
        return [e.to_dict() for e in entries]

    def save_durable(self, text: str, metadata: dict | None = None) -> str:
        """保存一条持久记忆（写入 durable backend）。

        安全规则：拒绝 secret-shaped 内容。
        """
        from echo.security.redaction import is_secret_shaped
        if is_secret_shaped(text):
            raise ValueError("拒绝保存：内容疑似密钥或敏感信息")

        md = metadata or {}
        entry = MemoryEntry(
            text=text,
            tags=md.get("tags", []),
            source=md.get("source", "manual"),
            kind="durable",
            metadata=md,
        )
        if self._durable:
            return self._durable.store(entry)
        return ""

    def relevant_for_prompt(self, query: str, limit: int = 5) -> str:
        """检索相关长期记忆并渲染为 prompt 片段。"""
        if not self._durable:
            return ""
        entries = self._durable.search(query, limit)
        if not entries:
            return ""
        lines = ["## Relevant Long-term Memory"]
        for e in entries:
            src = f"[{e.source}]" if e.source else ""
            lines.append(f"- {src} {e.text[:300]}")
        return "\n".join(lines)

    def list_all(self, limit: int = 50) -> list[MemoryEntry]:
        """列出所有记忆。"""
        return self._backend.list_all(limit)

    def count(self) -> int:
        """记忆总数。"""
        return self._backend.count()

    def delete(self, entry_id: str) -> bool:
        """删除记忆。"""
        return self._backend.delete(entry_id)

    # ── 工具观察（自动提取记忆）───────────────────

    def observe_tool_result(self, tool_name: str, params: dict,
                            result, ctx=None) -> None:
        """从工具执行结果中自动提取记忆。

        ctx: ToolContext 实例（可选，用于 workspace-resolved 路径）。
        """
        # 用 ctx 解析文件路径（相对路径 → 绝对路径）
        def _resolve(p: str) -> str:
            if ctx and hasattr(ctx, "resolve_path"):
                try:
                    return str(ctx.resolve_path(p))
                except Exception:
                    return p
            return p

        if tool_name == "read_file":
            path = _resolve(params.get("path", ""))
            output = getattr(result, "output", str(result))
            self._add_file_summary(path, output[:500])
        elif tool_name in ("write_file", "patch_file"):
            path = _resolve(params.get("path", ""))
            self._invalidate_file(path)
        elif tool_name == "run_shell":
            error = getattr(result, "error", None)
            if error:
                self.add(
                    f"Shell 错误: {error[:300]}",
                    {"tags": ["shell", "error"], "source": "run_shell"},
                )

    def observe_user_message(self, text: str) -> None:
        """从用户消息中提取记忆。

        当前简单记录用户请求摘要。后续可接入 LLM 做智能提取。
        """
        self._task_summary = text[:300]

    # ── 持久记忆管理 ──────────────────────────────

    def promote(self, entry_id: str) -> bool:
        """将一条短期记忆提升为持久记忆（修改 kind + 写入 durable backend）。"""
        entry = self._backend.get(entry_id)
        if entry is None:
            return False
        from echo.security.redaction import is_secret_shaped
        if is_secret_shaped(entry.text):
            return False
        entry.kind = "durable"
        if self._durable:
            self._durable.store(entry)
        return True

    def decay_old_entries(self, max_age_seconds: float = 3600.0) -> int:
        """清理过期的短期记忆。

        Args:
            max_age_seconds: 最大保留时间（秒）。

        Returns:
            清理的条目数。
        """
        now = time.time()
        removed = 0
        for entry in self._backend.list_all(limit=1000):
            if entry.kind == "episodic" and (now - entry.created_at) > max_age_seconds:
                self._backend.delete(entry.entry_id)
                removed += 1
        return removed

    # ── 渲染为 prompt ─────────────────────────────

    def retrieve_context(self, query: str, top_k: int = 5) -> str:
        """检索相关记忆并渲染为 LLM 可读文本。"""
        entries = self._backend.search(query, top_k)
        if not entries:
            return ""

        lines = ["## 相关记忆"]
        for e in entries:
            lines.append(
                f"- [{e.source}] {e.text[:300]}"
                + (f" ({', '.join(e.tags)})" if e.tags else "")
            )
        return "\n".join(lines)

    def render_working(self) -> str:
        """渲染工作记忆为 LLM 可读文本。"""
        parts = []
        if self._task_summary:
            parts.append(f"当前任务: {self._task_summary}")
        if self._recent_files:
            parts.append(f"近期文件: {', '.join(self._recent_files[-5:])}")
        if self._file_summaries:
            parts.append("文件摘要:")
            for path, entry in list(self._file_summaries.items())[-5:]:
                parts.append(f"  {path}: {entry.text[:180]}")
        return "\n".join(parts)

    def render_for_prompt(self) -> str:
        """为 LLM prompt 渲染完整记忆上下文。

        包含：工作记忆 + 相关记忆检索。
        """
        working = self.render_working()
        context = self.retrieve_context(self._task_summary or "")
        parts = [p for p in [working, context] if p]
        return "\n\n".join(parts)

    # ── 文件摘要管理 ──────────────────────────────

    def _add_file_summary(self, path: str, content: str) -> None:
        import hashlib
        from pathlib import Path
        p = Path(path)
        fhash = ""
        if p.exists():
            try:
                fhash = hashlib.sha256(p.read_bytes()).hexdigest()
            except (OSError, PermissionError):
                pass
        entry = MemoryEntry(
            text=content[:500],
            tags=["file", p.name],
            source="read_file",
            freshness_hash=fhash,
            file_path=str(p),
        )
        self._file_summaries[str(p)] = entry
        self._recent_files.append(str(p))
        if len(self._recent_files) > 8:
            self._recent_files = self._recent_files[-8:]
        self._backend.store(entry)

    def _invalidate_file(self, path: str) -> None:
        self._file_summaries.pop(path, None)

    # ── 序列化（session 持久化用）──────────────────

    def to_dict(self) -> dict:
        """序列化短期记忆为 dict。用于 Session.save()。"""
        summaries = {}
        for p, entry in self._file_summaries.items():
            summaries[p] = entry.to_dict()
        return {
            "task_summary": self._task_summary,
            "recent_files": list(self._recent_files[-8:]),
            "file_summaries": summaries,
        }

    def load_dict(self, data: dict) -> None:
        """从 dict 恢复短期记忆。用于 Session.load()。"""
        self._task_summary = data.get("task_summary", "") or ""
        self._recent_files = data.get("recent_files", []) or []
        self._file_summaries = {}
        for p, d in (data.get("file_summaries", {}) or {}).items():
            entry = MemoryEntry(
                text=d.get("text", ""),
                tags=d.get("tags", []),
                source=d.get("source", ""),
                kind=d.get("kind", "episodic"),
                freshness_hash=d.get("freshness_hash"),
                file_path=d.get("file_path"),
                entry_id=d.get("entry_id", ""),
            )
            self._file_summaries[p] = entry
            # 同时写回检索后端，使 search/retrieve 能找到恢复的记忆
            self._backend.store(entry)
