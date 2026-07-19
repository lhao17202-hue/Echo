"""JSON-file-based durable memory backend.

Stores persistent memories in .echo/memory/durable.json.
Simple keyword scoring for search — no embeddings / vector index.
"""

import json
import os
import time
import uuid
from pathlib import Path
from echo.memory.base import MemoryBackend, MemoryEntry


class JsonDurableMemoryBackend(MemoryBackend):
    """JSON 文件持久记忆后端。

    - 存储位置: .echo/memory/durable.json
    - 检索: 关键词评分（query 词命中数 + tag/source 加分 + 使用计数/时间加分）
    - 原子写入: tmp → replace
    """

    def __init__(self, path: str):
        self._path = Path(path)
        self._items: dict[str, dict] = {}
        self._load()

    # ── MemoryBackend 接口 ─────────────────────────

    def store(self, entry: MemoryEntry) -> str:
        item = entry.to_dict()
        item["id"] = item.get("entry_id", "mem_" + uuid.uuid4().hex[:8])
        item.setdefault("score", 1.0)
        item.setdefault("usage_count", 0)
        item["updated_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
        if "created_at" not in item or not item["created_at"]:
            item["created_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
        self._items[item["id"]] = item
        self._save()
        return item["id"]

    def search(self, query: str, top_k: int = 5) -> list[MemoryEntry]:
        """关键词评分搜索。"""
        query_lower = query.lower()
        query_words = set(query_lower.split())
        scored = []

        for item in self._items.values():
            text_lower = item.get("text", "").lower()
            tags = [t.lower() for t in item.get("tags", [])]
            source = item.get("source", "").lower()

            score = 0.0
            # 文本命中词
            for w in query_words:
                if w in text_lower:
                    score += 2.0
            # tag 命中
            for t in tags:
                if any(w in t for w in query_words):
                    score += 3.0
            # source 命中
            if any(w in source for w in query_words):
                score += 1.0
            # 使用次数轻轻加分
            score += min(item.get("usage_count", 0), 10) * 0.05
            # 基础分数
            score += item.get("score", 1.0) * 0.1

            if score > 0:
                scored.append((score, item))

        scored.sort(key=lambda x: x[0], reverse=True)
        results = []
        for _, item in scored[:top_k]:
            # 记录一次"被检索"
            item["usage_count"] = item.get("usage_count", 0) + 1
            results.append(MemoryEntry(
                text=item.get("text", ""),
                tags=item.get("tags", []),
                source=item.get("source", ""),
                kind="durable",
                entry_id=item.get("id", ""),
                metadata=item,
            ))
        if results:
            self._save()  # 更新 usage_count
        return results

    def get(self, entry_id: str) -> MemoryEntry | None:
        item = self._items.get(entry_id)
        if item is None:
            return None
        return MemoryEntry(
            text=item.get("text", ""), tags=item.get("tags", []),
            source=item.get("source", ""), kind="durable",
            entry_id=item.get("id", ""), metadata=item,
        )

    def list_all(self, limit: int = 50) -> list[MemoryEntry]:
        items = sorted(self._items.values(),
                       key=lambda x: x.get("created_at", ""), reverse=True)
        return [
            MemoryEntry(
                text=i.get("text", ""), tags=i.get("tags", []),
                source=i.get("source", ""), kind="durable",
                entry_id=i.get("id", ""), metadata=i,
            )
            for i in items[:limit]
        ]

    def delete(self, entry_id: str) -> bool:
        if entry_id in self._items:
            del self._items[entry_id]
            self._save()
            return True
        return False

    def clear(self) -> None:
        self._items.clear()
        self._save()

    def count(self) -> int:
        return len(self._items)

    # ── 持久化 ─────────────────────────────────────

    def _load(self) -> None:
        if not self._path.exists():
            self._items = {}
            return
        try:
            data = json.loads(self._path.read_text(encoding="utf-8"))
            self._items = {}
            for item in data.get("items", []):
                if "id" in item:
                    self._items[item["id"]] = item
        except (json.JSONDecodeError, KeyError):
            self._items = {}

    def _save(self) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        data = {"version": 1, "items": list(self._items.values())}
        tmp = self._path.with_suffix(".tmp")
        tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
        os.replace(tmp, self._path)
