"""默认记忆后端 — KeywordMemory（关键词匹配 + SHA-256 新鲜度）。

实现 MemoryBackend 接口。提供纯关键词匹配的搜索，不含向量检索。
后续可替换为 ChromaMemory / FAISSMemory 等向量后端。
"""

import time
from echo.memory.base import MemoryBackend, MemoryEntry


class KeywordMemory(MemoryBackend):
    """关键词匹配记忆后端。

    检索算法：精确标签匹配（权重 10）> 关键词重叠（权重 1）> 时间新近（权重 0.1）。

    特性：
      - 工作记忆：task_summary + recent_files + file_summaries + episodic_notes
      - 持久记忆：存储到 .echo/memory/ 目录（MEMORY.md 索引 + topics/*.md）
      - 文件新鲜度：read_file 后记录 SHA-256，write/patch 后失效
    """

    def __init__(self):
        self._entries: dict[str, MemoryEntry] = {}
        self._max_entries = 200

    # ── MemoryBackend 接口 ─────────────────────────

    def store(self, entry: MemoryEntry) -> str:
        """存入一条记忆。超出上限时移除最旧的。"""
        self._entries[entry.entry_id] = entry
        if len(self._entries) > self._max_entries:
            oldest = min(self._entries.values(), key=lambda e: e.created_at)
            del self._entries[oldest.entry_id]
        return entry.entry_id

    def search(self, query: str, top_k: int = 5) -> list[MemoryEntry]:
        """关键词搜索。"""
        query_lower = query.lower()
        scored: list[tuple[float, MemoryEntry]] = []

        for entry in self._entries.values():
            score = 0.0
            # 精确标签匹配
            for tag in entry.tags:
                if tag.lower() in query_lower:
                    score += 10
            # 关键词重叠
            query_words = set(query_lower.split())
            entry_words = set(entry.text.lower().split())
            score += len(query_words & entry_words)
            # 时间新近
            score += (entry.created_at / max(time.time(), 1)) * 0.1
            scored.append((score, entry))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [e for _, e in scored[:top_k]]

    def get(self, entry_id: str) -> MemoryEntry | None:
        return self._entries.get(entry_id)

    def list_all(self, limit: int = 50) -> list[MemoryEntry]:
        entries = sorted(self._entries.values(),
                         key=lambda e: e.created_at, reverse=True)
        return entries[:limit]

    def delete(self, entry_id: str) -> bool:
        if entry_id in self._entries:
            del self._entries[entry_id]
            return True
        return False

    def clear(self) -> None:
        self._entries.clear()

    def count(self) -> int:
        return len(self._entries)
