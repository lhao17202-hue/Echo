"""Unit tests for echo.persistence.trace.

Run: python -m pytest tests/test_trace.py -v
"""

import json
import sys
import tempfile
import pytest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from echo.persistence.trace import (
    TraceEvent, TraceEmitter, TraceReader, EventType, read_jsonl,
)
from echo.persistence.run_store import RunStore
from echo.core.task_state import TaskState


# ── helpers ────────────────────────────────────────

def _make_emitter():
    """创建测试用的 TraceEmitter + RunStore。"""
    d = Path(tempfile.mkdtemp())
    rs = RunStore(str(d))
    ts = TaskState.create("test", run_id="r1")
    rs.start_run(ts)
    return TraceEmitter(rs, "r1"), rs, d


# ═══════════════════════════════════════════════════
# TraceEvent
# ═══════════════════════════════════════════════════

class TestTraceEvent:
    """TraceEvent 数据模型。"""

    def test_auto_generates_fields(self):
        e = TraceEvent(event_type=EventType.RUN_STARTED, run_id="r1")
        assert e.event_id
        assert len(e.event_id) == 8
        assert e.created_at
        assert e.timestamp > 0

    def test_to_dict(self):
        e = TraceEvent(event_type=EventType.TOOL_EXECUTED, run_id="r1",
                       payload={"tool": "read_file", "success": True})
        d = e.to_dict()
        assert d["event"] == "tool_executed"
        assert d["run_id"] == "r1"
        assert d["tool"] == "read_file"
        assert d["success"] is True

    def test_event_id_unique(self):
        e1 = TraceEvent()
        e2 = TraceEvent()
        assert e1.event_id != e2.event_id


# ═══════════════════════════════════════════════════
# TraceEmitter — 基本功能
# ═══════════════════════════════════════════════════

class TestTraceEmitterBasic:
    """TraceEmitter 基础功能。"""

    def test_emit_counts(self):
        emitter, rs, d = _make_emitter()
        emitter.emit("test_event", key="val")
        emitter.emit("test_event2", a=1)
        assert emitter.event_count == 2

    def test_run_lifecycle(self):
        emitter, rs, d = _make_emitter()
        emitter.run_started(task_id="t1", user_request="hello")
        emitter.run_finished(status="completed", tool_steps=3, attempts=2)

        events = read_jsonl(rs._trace_path())
        assert len(events) == 2
        assert events[0]["event"] == "run_started"
        assert events[0]["task_id"] == "t1"
        assert events[1]["event"] == "run_finished"
        assert events[1]["status"] == "completed"

    def test_model_lifecycle(self):
        emitter, rs, d = _make_emitter()
        emitter.model_requested(attempts=1, tool_steps=0, max_tokens=8000)
        emitter.model_parsed(kind="tool", usage={"input": 100, "output": 50},
                             duration_ms=1200, stop_reason="tool_use")

        events = read_jsonl(rs._trace_path())
        assert len(events) == 2
        assert events[0]["event"] == "model_requested"
        assert events[0]["attempts"] == 1
        assert events[1]["event"] == "model_parsed"
        assert events[1]["kind"] == "tool"
        assert events[1]["usage"]["input"] == 100

    def test_tool_executed(self):
        emitter, rs, d = _make_emitter()
        emitter.tool_executed("read_file", success=True, duration_ms=42,
                              preview="line 1\nline 2")
        emitter.tool_executed("write_file", success=False, error="permission denied",
                              duration_ms=100)

        events = read_jsonl(rs._trace_path())
        assert events[0]["tool"] == "read_file"
        assert events[0]["success"] is True
        assert events[1]["tool"] == "write_file"
        assert events[1]["success"] is False
        assert "permission" in events[1]["error"]

    def test_checkpoint(self):
        emitter, rs, d = _make_emitter()
        emitter.checkpoint_created(checkpoint_id="ckpt_abc", trigger="tool_executed",
                                   key_files_count=3)
        events = read_jsonl(rs._trace_path())
        assert events[0]["event"] == "checkpoint_created"
        assert events[0]["checkpoint_id"] == "ckpt_abc"


# ═══════════════════════════════════════════════════
# TraceEmitter events
# ═══════════════════════════════════════════════════

class TestTraceEmitterMultiAgent:
    """多 Agent 追踪事件。"""

    def test_message_events(self):
        emitter, rs, d = _make_emitter()
        emitter.message_sent(to_agent="reviewer", msg_type="plan_request",
                             preview="Please review this plan")
        emitter.message_received(from_agent="reviewer", msg_type="plan_response",
                                 preview="Plan approved")

        events = read_jsonl(rs._trace_path())
        assert events[0]["event"] == "message_sent"
        assert events[0]["to_agent"] == "reviewer"
        assert events[1]["event"] == "message_received"
        assert events[1]["from_agent"] == "reviewer"

    def test_teammate_events(self):
        emitter, rs, d = _make_emitter()
        emitter.teammate_spawned(name="reviewer", role="code reviewer",
                                 session_id="sub-session-1")
        emitter.teammate_stopped(name="reviewer", reason="task completed")

        events = read_jsonl(rs._trace_path())
        assert events[0]["event"] == "teammate_spawned"
        assert events[0]["name"] == "reviewer"
        assert events[1]["event"] == "teammate_stopped"

    def test_cron_background_memory(self):
        emitter, rs, d = _make_emitter()
        emitter.cron_fired(job_id="cron_123", prompt_preview="check status")
        emitter.background_started(bg_id="bg_1", tool_name="run_shell")
        emitter.background_completed(bg_id="bg_1", tool_name="run_shell", success=True)
        emitter.memory_promoted(entry_preview="Python version is 3.12", topic="project-conventions")

        events = read_jsonl(rs._trace_path())
        assert len(events) == 4
        assert events[0]["event"] == "cron_fired"
        assert events[1]["event"] == "background_started"
        assert events[2]["event"] == "background_completed"
        assert events[3]["event"] == "memory_promoted"


# ═══════════════════════════════════════════════════
# TraceEmitter — 脱敏 + debug
# ═══════════════════════════════════════════════════

class TestTraceEmitterRedaction:
    """Trace 脱敏。"""

    def test_redaction_applied(self):
        """payload 在写入前被脱敏。"""
        def fake_redact(payload):
            result = dict(payload)
            if "secret" in result:
                result["secret"] = "<redacted>"
            return result

        d = Path(tempfile.mkdtemp())
        rs = RunStore(str(d))
        ts = TaskState.create("test", run_id="r1")
        rs.start_run(ts)

        emitter = TraceEmitter(rs, "r1", redact_fn=fake_redact)
        emitter.emit("test", secret="sk-abc123", normal="hello")

        events = read_jsonl(rs._trace_path())
        assert events[0]["secret"] == "<redacted>"
        assert events[0]["normal"] == "hello"

    def test_append_trace_global_redaction(self):
        """append_trace() 作为最后一道关卡，即使 TraceEmitter 没传 redact_fn 也要脱敏。

        为验证这一点，直接调用 RunStore.append_trace() 而非通过 TraceEmitter。
        """
        d = Path(tempfile.mkdtemp())
        rs = RunStore(str(d))
        ts = TaskState.create("test", run_id="r-redact")
        rs.start_run(ts)

        # 直接构造带 secret 的 TraceEvent，绕过 TraceEmitter 的 redact_fn
        from echo.persistence.trace import TraceEvent
        event = TraceEvent(
            event_type="test",
            run_id="r-redact",
            payload={"api_key": "sk-live-deadbeef", "safe_data": "hello"},
        )
        rs.append_trace(event)

        events = read_jsonl(rs._trace_path())
        assert len(events) == 1
        # 全局脱敏：原始 secret 不应出现
        assert "sk-live-deadbeef" not in json.dumps(events[0])
        assert events[0]["api_key"] == "<redacted>"
        assert events[0]["safe_data"] == "hello"


# ═══════════════════════════════════════════════════
# TraceReader
# ═══════════════════════════════════════════════════

class TestTraceReader:
    """TraceReader 分析工具。"""

    def test_filter_by_type(self):
        emitter, rs, d = _make_emitter()
        emitter.tool_executed("a", success=True)
        emitter.tool_executed("b", success=False)
        emitter.model_requested(attempts=1)

        reader = TraceReader(rs._trace_path())
        tool_events = reader.filter_by_type(EventType.TOOL_EXECUTED)
        assert len(tool_events) == 2

        model_events = reader.filter_by_type(EventType.MODEL_REQUESTED)
        assert len(model_events) == 1

    def test_stats(self):
        emitter, rs, d = _make_emitter()
        emitter.run_started(task_id="t1", user_request="x")
        emitter.model_requested(attempts=1)
        emitter.model_parsed(kind="tool")
        emitter.tool_executed("read", success=True)
        emitter.tool_executed("write", success=False, error="fail")
        emitter.run_finished(status="completed", tool_steps=2, attempts=1)

        reader = TraceReader(rs._trace_path())
        stats = reader.stats()
        assert stats["total_events"] == 6
        assert stats["tool_count"] == 2
        assert stats["model_calls"] == 1

    def test_replay_timeline(self):
        emitter, rs, d = _make_emitter()
        emitter.run_started(task_id="t1", user_request="fix bug")
        emitter.tool_executed("read_file", success=True, duration_ms=50, preview="code")
        emitter.run_finished(status="completed", tool_steps=1, attempts=1)

        reader = TraceReader(rs._trace_path())
        timeline = reader.replay_timeline()
        assert "run_started" in timeline
        assert "read_file" in timeline
        assert "run_finished" in timeline

    def test_empty_trace(self):
        d = Path(tempfile.mkdtemp())
        reader = TraceReader(d)
        assert reader.stats() == {"total_events": 0}
        assert reader.replay_timeline() == "(empty trace)"


# ═══════════════════════════════════════════════════
# read_jsonl 容错
# ═══════════════════════════════════════════════════

class TestReadJsonl:
    """JSONL 容错读取。"""

    def test_normal_read(self):
        d = Path(tempfile.mkdtemp())
        path = d / "test.jsonl"
        path.write_text(
            '{"event":"a","id":1}\n{"event":"b","id":2}\n',
            encoding="utf-8"
        )
        events = read_jsonl(path)
        assert len(events) == 2

    def test_corrupted_skipped(self):
        d = Path(tempfile.mkdtemp())
        path = d / "test.jsonl"
        path.write_text(
            '{"event":"a","id":1}\ncorrupted line\n{"event":"b","id":2}\n',
            encoding="utf-8"
        )
        events = read_jsonl(path)
        assert len(events) == 2  # corrupted skipped

    def test_empty_file(self):
        d = Path(tempfile.mkdtemp())
        path = d / "nonexistent.jsonl"
        assert read_jsonl(path) == []

    def test_blank_lines_skipped(self):
        d = Path(tempfile.mkdtemp())
        path = d / "test.jsonl"
        path.write_text(
            '\n{"event":"a"}\n\n{"event":"b"}\n\n',
            encoding="utf-8"
        )
        events = read_jsonl(path)
        assert len(events) == 2
