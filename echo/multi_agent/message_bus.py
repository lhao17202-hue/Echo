"""Message bus — inter-agent communication (queue + JSONL fallback)."""

import json
import queue
import time
import logging
from pathlib import Path
from dataclasses import dataclass, field
import uuid

logger = logging.getLogger("echo.bus")


@dataclass
class MessageItem:
    msg_id: str = field(default_factory=lambda: uuid.uuid4().hex[:8])
    from_agent: str = ""
    to_agent: str = ""
    msg_type: str = "message"
    content: str = ""
    metadata: dict = field(default_factory=dict)
    ts: float = field(default_factory=time.time)


class MessageBus:
    """多 Agent 消息总线。

    进程内：queue.Queue（线程安全快速通道）。
    跨进程兜底：JSONL 文件邮箱（崩溃恢复用）。
    """

    def __init__(self, storage_dir: str | None = None):
        self._queues: dict[str, queue.Queue] = {}
        self._storage_dir = Path(storage_dir) if storage_dir else None

    def register(self, agent_name: str) -> None:
        if agent_name not in self._queues:
            self._queues[agent_name] = queue.Queue()

    def send(self, from_agent: str, to_agent: str, content: str,
             msg_type: str = "message", metadata: dict | None = None) -> None:
        msg = MessageItem(
            from_agent=from_agent,
            to_agent=to_agent,
            msg_type=msg_type,
            content=content,
            metadata=metadata or {},
        )
        if to_agent in self._queues:
            self._queues[to_agent].put(msg)
        if self._storage_dir:
            self._append_jsonl(to_agent, msg)

    def receive(self, agent_name: str) -> list[MessageItem]:
        if agent_name not in self._queues:
            return []
        msgs = []
        while not self._queues[agent_name].empty():
            try:
                msgs.append(self._queues[agent_name].get_nowait())
            except queue.Empty:
                break
        return msgs

    def _append_jsonl(self, agent_name: str, msg: MessageItem) -> None:
        try:
            path = self._storage_dir / f"{agent_name}.jsonl"
            path.parent.mkdir(parents=True, exist_ok=True)
            with open(path, "a", encoding="utf-8") as f:
                f.write(json.dumps({
                    "msg_id": msg.msg_id,
                    "from_agent": msg.from_agent,
                    "to_agent": msg.to_agent,
                    "msg_type": msg.msg_type,
                    "content": msg.content,
                    "metadata": msg.metadata,
                    "ts": msg.ts,
                }, ensure_ascii=False) + "\n")
        except Exception as e:
            logger.warning(f"Failed to persist message: {e}")
