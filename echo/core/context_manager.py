"""Context assembly and multi-level compaction engine.

Context assembly and multi-level compaction engine for Echo.

Five-level compaction pipeline:
  L1: tool_result_budget — persist large (>30KB) tool outputs to disk, keep preview
  L2: dedupe_file_reads   — same file read multiple times → keep most recent only
  L3: micro_compact       — old tool results → replace with stubs
  L4: snip_compact        — too many messages → trim middle rounds
  L5: compact_history     — context oversize → LLM summary + keep recent tail

Three entry points:
  compact(messages, llm)         — passive: every LLM turn, lightweight, only summarize if over
  force_compact(messages, llm)   — active: after compact tool, always summarize + transcript
  reactive_compact(messages, llm)— recovery: after prompt-too-long, most aggressive + retry
"""

import json
import hashlib
import time
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from echo.core.task_state import TaskState

logger = logging.getLogger("echo.context")


@dataclass
class Budget:
    prefix: int = 3600
    memory: int = 1600
    relevant_memory: int = 1200
    history: int = 5200
    total: int = 12000

    floors: dict[str, int] = field(default_factory=lambda: {
        "prefix": 1200, "memory": 400,
        "relevant_memory": 300, "durable_memory": 300,
        "history": 1500,
    })
    priority: list[str] = field(default_factory=lambda: [
        "relevant_memory", "durable_memory", "history", "memory", "prefix",
    ])


@dataclass
class ContextConfig:
    budget: Budget = field(default_factory=Budget)

    compact_threshold_chars: int = 30_000
    compact_preview_chars: int = 2_000
    max_messages: int = 50
    keep_recent_tool_results: int = 3
    keep_recent_messages: int = 8        # tail messages kept after compact_history
    summary_max_tokens: int = 2000

    context_limit_chars: int = 50_000

    enable_memory: bool = True
    enable_relevant_memory: bool = True
    enable_compaction: bool = True
    enable_transcript: bool = True

    persist_dir: str = ""
    transcript_dir: str = ""


class ContextManager:

    def __init__(self, config: ContextConfig | None = None, llm_lock=None):
        self.config = config or ContextConfig()
        self._compact_count = 0
        self._llm_lock = llm_lock  # shared lock for lead+teammate LLM serialisation

    # ═══════════════════════════════════════════════
    # System Prompt
    # ═══════════════════════════════════════════════

    def build_system(self, state, tools, memory, sandbox,
                     checkpoint_manager=None) -> str:
        sections = {}
        sections["prefix"] = self._build_prefix(tools, sandbox, state, checkpoint_manager)
        if self.config.enable_memory:
            sections["memory"] = memory.render_working()
        if self.config.enable_relevant_memory:
            relevant = memory.retrieve(state.user_request, top_k=3)
            if relevant:
                sections["relevant_memory"] = self._render_relevant(relevant)
            # 注入持久记忆
            durable = memory.relevant_for_prompt(state.user_request, limit=5)
            if durable:
                sections["durable_memory"] = durable
        if self.config.enable_compaction:
            sections = self._apply_budget(sections)
        return "\n\n".join(sections.values())

    def _build_prefix(self, tools, sandbox, state, checkpoint_manager=None) -> str:
        lines = [
            "You are Echo, a coding agent. Use tools to accomplish tasks. Act directly.",
            "",
            "## Workspace",
            f"Root: {sandbox.root}",
            f"Branch: {sandbox.git_branch}",
            "",
            "## Available Tools",
        ]
        for tool in tools.get_all():
            risk_mark = ""
            if tool.risk_level == "warn":
                risk_mark = " [APPROVAL REQUIRED]"
            elif tool.risk_level == "danger":
                risk_mark = " [DANGER]"
            lines.append(f"- **{tool.name}**: {tool.description}{risk_mark}")
        if state.resume_status:
            lines.append(f"\nResume status: {state.resume_status}")
        # 注入当前未完成的 todo
        active_todos = [t for t in (state.todos or [])
                        if t.get("status") in ("pending", "in_progress")]
        if active_todos:
            lines.append("\n## Current Todos")
            for t in active_todos:
                status = t.get("status", "pending")
                mark = "⏳" if status == "in_progress" else "⬜"
                lines.append(f"- {mark} {t.get('content', '')}")
        return "\n".join(lines)

    def _render_relevant(self, entries: list[dict]) -> str:
        lines = ["## Relevant Memory"]
        for entry in entries[:5]:
            text = str(entry.get("text", ""))[:300]
            source = entry.get("source", "")
            line = f"- {text}"
            if source:
                line += f"  (from: {source})"
            lines.append(line)
        return "\n".join(lines)

    # ═══════════════════════════════════════════════
    # Three entry points
    # ═══════════════════════════════════════════════

    def compact(self, messages: list[dict], llm_client=None) -> list[dict]:
        """Passive: every LLM turn. Only LLM-summarizes if context_limit exceeded."""
        if not self.config.enable_compaction:
            return messages
        messages = self._tool_result_budget(messages)
        messages = self._dedupe_file_reads(messages)
        messages = self._micro_compact(messages)
        messages = self._snip_compact(messages)
        if self._estimate_size(messages) > self.config.context_limit_chars:
            messages = self._compact_history(messages, llm_client)
            self._compact_count += 1
        return messages

    def force_compact(self, messages: list[dict], llm_client=None,
                      reason: str = "tool_requested") -> list[dict]:
        """Active: model called compact tool. Always writes transcript + LLM summary."""
        if not self.config.enable_compaction:
            return messages
        messages = self._tool_result_budget(messages)
        messages = self._dedupe_file_reads(messages)
        messages = self._micro_compact(messages)
        if self.config.enable_transcript:
            self._write_transcript(messages)
        messages = self._compact_history(messages, llm_client)
        self._compact_count += 1
        return messages

    def reactive_compact(self, messages: list[dict], llm_client=None) -> list[dict]:
        """Recovery: prompt-too-long error. Most aggressive — transcript + summary, keep ~5 tail."""
        if self.config.enable_transcript:
            self._write_transcript(messages)
        tail_start = max(0, len(messages) - 5)
        if (tail_start > 0 and tail_start < len(messages)
                and self._is_tool_result(messages[tail_start])
                and self._message_has_tool_use(messages[tail_start - 1])):
            tail_start -= 1
        if llm_client and tail_start > 0:
            try:
                summary = self._summarize_history(messages[:tail_start], llm_client)
            except Exception:
                summary = "Earlier conversation trimmed after prompt-too-long error."
        elif llm_client:
            try:
                summary = self._summarize_history(messages, llm_client)
            except Exception:
                summary = "Earlier conversation trimmed."
        else:
            summary = "Earlier conversation trimmed."
        self._compact_count += 1
        if tail_start == 0:
            return [self._user_msg(f"[Reactive compact]\n\n{summary}")]
        return [
            self._user_msg(f"[Reactive compact]\n\n{summary}"),
            *messages[tail_start:],
        ]

    # ═══════════════════════════════════════════════
    # L1: tool_result_budget
    # ═══════════════════════════════════════════════

    def _tool_result_budget(self, messages: list[dict]) -> list[dict]:
        if not messages:
            return messages
        last = messages[-1]
        content = last.get("content")
        if last.get("role") != "user" or not isinstance(content, list):
            return messages
        blocks = [(i, b) for i, b in enumerate(content)
                  if isinstance(b, dict) and b.get("type") == "tool_result"]
        total_size = sum(len(str(b.get("content", ""))) for _, b in blocks)
        if total_size <= self.config.compact_threshold_chars:
            return messages
        for _, block in sorted(blocks,
                               key=lambda pair: len(str(pair[1].get("content", ""))),
                               reverse=True):
            if total_size <= self.config.compact_threshold_chars:
                break
            text = str(block.get("content", ""))
            block["content"] = self._persist_large_output(
                block.get("tool_use_id", "unknown"), text, force=True)
            total_size = sum(len(str(b.get("content", ""))) for _, b in blocks)
        return messages

    def _persist_large_output(self, tool_use_id: str, output: str,
                              force: bool = False) -> str:
        """Persist tool output to disk. Returns preview stub.

        Args:
            tool_use_id: tool call id, used as filename.
            output: full output text.
            force: if True, persist even if below threshold (used when
                   total budget exceeded by multiple medium-sized outputs).
        """
        if not force and len(output) <= self.config.compact_threshold_chars:
            return output
        persist_dir = Path(self.config.persist_dir) if self.config.persist_dir else None
        if persist_dir:
            persist_dir.mkdir(parents=True, exist_ok=True)
            path = persist_dir / f"{tool_use_id}.txt"
            if not path.exists():
                path.write_text(output, encoding="utf-8")
            # 文件级显示路径，短占位符只显示文件名（避免 Windows 长路径问题）
            path_str = str(path)
            file_name = path.name
        else:
            file_name = f"{tool_use_id}.txt"
            path_str = f".echo/tool_outputs/{file_name}"

        preview = output[:self.config.compact_preview_chars]
        stub = (
            f"<persisted-output>\n"
            f"Full output: {path_str}\n"
            f"Preview:\n{preview}\n"
            f"</persisted-output>"
        )
        # 不变量：替换后的 stub 必须短于原内容，否则用最小占位符
        if len(stub) >= len(output):
            stub = f"[Large output persisted: {file_name}]"
        return stub

    # ═══════════════════════════════════════════════
    # L2: dedupe_file_reads
    # ═══════════════════════════════════════════════

    def _dedupe_file_reads(self, messages: list[dict]) -> list[dict]:
        """Replace older read_file results for the same path with short summaries.

        Logic: for each read_file tool_result, find the LAST occurrence of that path
        and replace earlier occurrences with "[Earlier read of {path} — see recent result.]"
        """
        reads: list[tuple[int, int, str]] = []
        for mi, msg in enumerate(messages):
            content = msg.get("content") if isinstance(msg.get("content"), list) else []
            for bi, block in enumerate(content):
                if isinstance(block, dict) and block.get("type") == "tool_result":
                    path = (
                        (block.get("tool_input") or {}).get("path", "")
                        if isinstance(block.get("tool_input"), dict)
                        else ""
                    )
                    if block.get("tool_name") == "read_file" and path:
                        reads.append((mi, bi, path))

        if len(reads) <= 1:
            return messages

        # 每个 path 保留最后一次（按 reads 列表位置），其余替换
        last_pos: dict[str, int] = {}
        for i, (_mi, _bi, path) in enumerate(reads):
            last_pos[path] = i  # position in reads list (in order)

        for i, (mi, bi, path) in enumerate(reads):
            if i != last_pos[path]:
                msg = messages[mi]
                block = msg["content"][bi]
                block["content"] = (
                    f"[Earlier read of {path} — "
                    f"see most recent result in a later message.]"
                )

        return messages

    # ═══════════════════════════════════════════════
    # L3: micro_compact
    # ═══════════════════════════════════════════════

    def _micro_compact(self, messages: list[dict]) -> list[dict]:
        collected = self._collect_tool_results(messages)
        if len(collected) <= self.config.keep_recent_tool_results:
            return messages
        for _mi, _bi, block in collected[:-self.config.keep_recent_tool_results]:
            if len(str(block.get("content", ""))) > 120:
                block["content"] = "[Earlier tool result compacted. Re-run the tool if needed.]"
        return messages

    def _collect_tool_results(self, messages: list[dict]) -> list[tuple[int, int, dict]]:
        results = []
        for mi, msg in enumerate(messages):
            content = msg.get("content", [])
            if not isinstance(content, list):
                continue
            for bi, block in enumerate(content):
                if isinstance(block, dict) and block.get("type") == "tool_result":
                    results.append((mi, bi, block))
        return results

    # ═══════════════════════════════════════════════
    # L4: snip_compact
    # ═══════════════════════════════════════════════

    def _snip_compact(self, messages: list[dict]) -> list[dict]:
        if len(messages) <= self.config.max_messages:
            return messages
        max_msgs = self.config.max_messages
        head_end = 3
        tail_start = len(messages) - (max_msgs - 3)
        if head_end > 0 and self._message_has_tool_use(messages[head_end - 1]):
            while head_end < len(messages) and self._is_tool_result(messages[head_end]):
                head_end += 1
        if (tail_start > 0 and tail_start < len(messages)
                and self._is_tool_result(messages[tail_start])
                and self._message_has_tool_use(messages[tail_start - 1])):
            tail_start -= 1
        if head_end >= tail_start:
            return messages
        snipped = tail_start - head_end
        return (
            messages[:head_end]
            + [self._user_msg(f"[snipped {snipped} messages]")]
            + messages[tail_start:]
        )

    # ═══════════════════════════════════════════════
    # L5: compact_history
    # ═══════════════════════════════════════════════

    def _compact_history(self, messages: list[dict], llm_client=None) -> list[dict]:
        if llm_client:
            tail_start = max(0, len(messages) - self.config.keep_recent_messages)
            if (tail_start > 0 and tail_start < len(messages)
                    and self._is_tool_result(messages[tail_start])
                    and self._message_has_tool_use(messages[tail_start - 1])):
                tail_start -= 1
            if tail_start > 0:
                summary = self._summarize_history(messages[:tail_start], llm_client)
                return [
                    self._user_msg(f"[Compacted conversation]\n\n{summary}"),
                    *messages[tail_start:],
                ]
            else:
                summary = self._summarize_history(messages, llm_client)
                return [self._user_msg(f"[Compacted conversation]\n\n{summary}")]

        keep = 10 if len(messages) > 10 else max(3, len(messages) // 2)
        tail_start = max(0, len(messages) - keep)
        if (tail_start > 0 and tail_start < len(messages)
                and self._is_tool_result(messages[tail_start])
                and self._message_has_tool_use(messages[tail_start - 1])):
            tail_start -= 1
        if tail_start == 0:
            return messages
        return [self._user_msg("[Earlier conversation trimmed]"),
                *messages[tail_start:]]

    def _summarize_history(self, messages: list[dict], llm_client) -> str:
        conversation = json.dumps(messages, default=str)[:80_000]
        prompt = (
            "Summarize this coding-agent conversation so work can continue.\n"
            "Preserve exactly:\n"
            "- current user goal and explicit user constraints\n"
            "- files read, written, modified, or patched (with paths)\n"
            "- commands run and important outputs\n"
            "- decisions already made and their rationale\n"
            "- current blockers, errors, or unresolved issues\n"
            "- remaining work to complete\n"
            "- tool_use/tool_result pairs needed to continue (with IDs)\n"
            "- exact paths, symbols, names, and values referenced\n\n"
            + conversation
        )
        try:
            if self._llm_lock:
                with self._llm_lock:
                    response = llm_client.chat(
                        messages=[{"role": "user", "content": [{"type": "text", "text": prompt}]}],
                        max_tokens=self.config.summary_max_tokens,
                    )
            else:
                response = llm_client.chat(
                    messages=[{"role": "user", "content": [{"type": "text", "text": prompt}]}],
                    max_tokens=self.config.summary_max_tokens,
                )
            texts = []
            for block in response.content:
                if hasattr(block, "text"):
                    texts.append(block.text)
            return " ".join(texts) or "(empty summary)"
        except Exception as e:
            logger.warning(f"History summarization failed: {e}")
            return f"(summarization failed)"

    def _write_transcript(self, messages: list[dict]) -> Path:
        transcript_dir = Path(self.config.transcript_dir) if self.config.transcript_dir else None
        if not transcript_dir:
            transcript_dir = Path(".echo/transcripts")
        transcript_dir.mkdir(parents=True, exist_ok=True)
        path = transcript_dir / f"transcript_{int(time.time())}.jsonl"
        with path.open("w", encoding="utf-8") as f:
            for msg in messages:
                f.write(json.dumps(msg, default=str, ensure_ascii=False) + "\n")
        return path

    # ═══════════════════════════════════════════════
    # Budget
    # ═══════════════════════════════════════════════

    def _apply_budget(self, sections: dict[str, str]) -> dict[str, str]:
        total = sum(len(v) for v in sections.values())
        if total <= self.config.budget.total:
            return sections
        for key in self.config.budget.priority:
            if total <= self.config.budget.total:
                break
            if key not in sections:
                continue
            floor = self.config.budget.floors.get(key, 0)
            if len(sections[key]) > floor:
                sections[key] = self._truncate(sections[key], floor)
                total = sum(len(v) for v in sections.values())
        return sections

    @staticmethod
    def _truncate(text: str, max_chars: int) -> str:
        if len(text) <= max_chars:
            return text
        return text[:max_chars] + "\n... [truncated]"

    # ═══════════════════════════════════════════════
    # Helpers
    # ═══════════════════════════════════════════════

    @staticmethod
    def _user_msg(text: str) -> dict:
        """创建 provider-safe 的 user 消息（TextBlock 列表，而非纯字符串）。

        所有 provider 的 _blocks_to_text / _convert_messages 都期望
        content 是 list[dict]，纯字符串会被静默丢弃。
        """
        return {"role": "user", "content": [{"type": "text", "text": text}]}

    @staticmethod
    def _estimate_size(messages: list[dict]) -> int:
        try:
            return len(json.dumps(messages, default=str))
        except Exception:
            return sum(len(str(m)) for m in messages)

    @staticmethod
    def _message_has_tool_use(msg: dict) -> bool:
        if msg.get("role") != "assistant":
            return False
        content = msg.get("content", [])
        if not isinstance(content, list):
            return False
        return any(
            (isinstance(b, dict) and b.get("type") == "tool_use")
            or (hasattr(b, "name") and hasattr(b, "input"))
            for b in content
        )

    @staticmethod
    def _is_tool_result(msg: dict) -> bool:
        content = msg.get("content", [])
        if not isinstance(content, list):
            return False
        return any(
            isinstance(b, dict) and b.get("type") == "tool_result"
            for b in content
        )

    @property
    def compact_count(self) -> int:
        return self._compact_count
