"""Unit tests for echo.persistence + echo.core.task_state.

Run: python -m pytest tests/test_persistence.py -v
"""

import os
import sys
import json
import time
import tempfile
import pytest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from echo.core.task_state import (
    TaskState, Checkpoint, Status, StopReason,
)
from echo.persistence.session_store import Session, SessionStore
from echo.persistence.run_store import RunStore
from echo.persistence.checkpoint import (
    CheckpointManager,
    CHECKPOINT_FULL_VALID,
    CHECKPOINT_PARTIAL_STALE,
    CHECKPOINT_WORKSPACE_MISMATCH,
    CHECKPOINT_SCHEMA_MISMATCH,
    CHECKPOINT_NONE,
    RUNTIME_IDENTITY_KEYS,
)
from echo.persistence.trace import (
    TraceEvent, TraceEmitter, TraceReader, EventType, read_jsonl,
)


# ═══════════════════════════════════════════════════
# TaskState
# ═══════════════════════════════════════════════════

class TestTaskStateCreation:
    """TaskState 工厂方法和初始化。"""

    def test_create_generates_ids(self):
        ts = TaskState.create("hello")
        assert ts.run_id.startswith("run_")
        assert ts.task_id.startswith("task_")
        assert ts.status == Status.RUNNING
        assert ts.user_request == "hello"

    def test_create_custom_ids(self):
        ts = TaskState.create("x", task_id="my-task", run_id="my-run")
        assert ts.task_id == "my-task"
        assert ts.run_id == "my-run"

    def test_create_sets_agent_info(self):
        ts = TaskState.create("x", agent_type="subagent", agent_name="helper")
        assert ts.agent_type == "subagent"
        assert ts.agent_name == "helper"
        assert ts.depth == 0


class TestTaskStateProgress:
    """TaskState 进度记录方法。"""

    def test_record_attempt(self):
        ts = TaskState.create("x")
        ts.record_attempt()
        ts.record_attempt()
        assert ts.attempts == 2

    def test_record_tool(self):
        ts = TaskState.create("x")
        ts.record_tool("read_file")
        ts.record_tool("glob")
        assert ts.tool_steps == 2
        assert ts.last_tool == "glob"

    def test_attempt_vs_tool_steps(self):
        """attempt 统计模型调用轮数，tool_steps 统计工具执行次数——两者不同。"""
        ts = TaskState.create("x")
        ts.record_attempt()   # 第1轮模型调用
        ts.record_tool("a")
        ts.record_tool("b")   # 一轮调了两个工具
        ts.record_attempt()   # 第2轮模型调用
        ts.record_tool("c")
        assert ts.attempts == 2
        assert ts.tool_steps == 3


class TestTaskStateTermination:
    """TaskState 终止方法。"""

    def test_finish_success(self):
        ts = TaskState.create("x")
        ts.finish_success("all done")
        assert ts.is_success
        assert ts.final_answer == "all done"
        assert ts.stop_reason == StopReason.FINAL_ANSWER.value
        assert ts.finished_at is not None

    def test_stop_step_limit(self):
        ts = TaskState.create("x")
        ts.stop_step_limit()
        assert ts.status == Status.STOPPED
        assert ts.stop_reason == StopReason.STEP_LIMIT.value

    def test_stop_model_error(self):
        ts = TaskState.create("x")
        ts.stop_model_error("connection refused")
        assert ts.status == Status.FAILED
        assert "connection refused" in ts.errors

    def test_is_running_terminal_success(self):
        ts = TaskState.create("x")
        assert ts.is_running
        assert not ts.is_terminal
        ts.finish_success("ok")
        assert not ts.is_running
        assert ts.is_terminal

    def test_duration(self):
        ts = TaskState.create("x")
        assert ts.duration_seconds is None
        ts.finish_success("ok")
        assert ts.duration_seconds is not None
        assert ts.duration_seconds >= 0


class TestTaskStateSerialization:
    """TaskState 序列化/反序列化。"""

    def test_to_dict_basic(self):
        ts = TaskState.create("hello", task_id="t1", run_id="r1")
        d = ts.to_dict()
        assert d["run_id"] == "r1"
        assert d["task_id"] == "t1"
        assert d["status"] == "running"

    def test_from_dict_roundtrip(self):
        ts = TaskState.create("x", agent_type="subagent", agent_name="bot")
        ts.record_attempt()
        ts.record_tool("glob")
        ts.pending_protocols.append("p1")
        ts.active_background_tasks.append("bg1")
        ts.bound_global_task_id = "gt1"
        ts.stop_step_limit()

        restored = TaskState.from_dict(ts.to_dict())
        assert restored.run_id == ts.run_id
        assert restored.agent_type == "subagent"
        assert restored.tool_steps == 1
        assert restored.stop_reason == StopReason.STEP_LIMIT.value
        assert "p1" in restored.pending_protocols
        assert "bg1" in restored.active_background_tasks
        assert restored.bound_global_task_id == "gt1"

    def test_from_dict_missing_fields(self):
        """缺失字段使用默认值。"""
        restored = TaskState.from_dict({"run_id": "r1"})
        assert restored.run_id == "r1"
        assert restored.tool_steps == 0
        assert restored.agent_type == "lead"

    def test_background_task_lifecycle(self):
        ts = TaskState.create("x")
        ts.add_background_task("bg1")
        assert "bg1" in ts.active_background_tasks
        ts.remove_background_task("bg1")
        assert "bg1" not in ts.active_background_tasks


# ═══════════════════════════════════════════════════
# Session / SessionStore
# ═══════════════════════════════════════════════════

class TestSession:
    """Session 数据模型。"""

    def test_defaults(self):
        s = Session(session_id="s1")
        assert s.model_config["provider"] == "deepseek"
        assert s.security_config["approval_policy"] == "ask"
        assert s.feature_flags["memory"] is True

    def test_to_dict_from_dict(self):
        s = Session(session_id="s1", workspace_root="/tmp")
        s.history = [{"role": "user", "content": "hello"}]
        s.teammates = {"reviewer": "sub-session-1"}
        s.pending_protocols = [{"request_id": "r1", "type": "plan_approval"}]

        restored = Session.from_dict(s.to_dict())
        assert restored.session_id == "s1"
        assert restored.history == s.history
        assert restored.teammates == s.teammates
        assert restored.pending_protocols == s.pending_protocols


class TestSessionStore:
    """SessionStore CRUD 操作。"""

    def test_save_and_load(self):
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            store = SessionStore(d)
            s = Session(session_id="test-session", workspace_root=str(d))
            store.save(s)
            loaded = store.load("test-session")
            assert loaded.session_id == "test-session"
            assert loaded.workspace_root == str(d)

    def test_latest(self):
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            store = SessionStore(d)
            s1 = Session(session_id="s1")
            s2 = Session(session_id="s2")
            store.save(s1)
            time.sleep(0.1)
            store.save(s2)
            assert store.latest() == "s2"

    def test_list_sessions(self):
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            store = SessionStore(d)
            store.save(Session(session_id="a"))
            store.save(Session(session_id="b"))
            sessions = store.list_sessions()
            ids = [s["session_id"] for s in sessions]
            assert "a" in ids
            assert "b" in ids

    def test_delete(self):
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            store = SessionStore(d)
            store.save(Session(session_id="to-delete"))
            assert store.delete("to-delete") is True
            assert store.delete("nonexistent") is False

    def test_load_nonexistent_raises(self):
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            store = SessionStore(d)
            with pytest.raises(FileNotFoundError):
                store.load("nope")


# ═══════════════════════════════════════════════════
# RunStore
# ═══════════════════════════════════════════════════

class TestRunStore:
    """RunStore 运行工件管理。"""

    def test_start_run_creates_artifacts(self):
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            store = RunStore(d)
            ts = TaskState.create("test")
            run_dir = store.start_run(ts)
            assert run_dir.exists()
            assert (run_dir / "task_state.json").exists()

    def test_update_and_load_state(self):
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            store = RunStore(d)
            ts = TaskState.create("test", run_id="r1")
            store.start_run(ts)
            ts.record_tool("read_file")
            ts.finish_success("done")
            store.update_state(ts)

            loaded = store.load_task_state("r1")
            assert loaded.final_answer == "done"
            assert loaded.tool_steps == 1

    def test_write_and_load_report(self):
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            store = RunStore(d)
            ts = TaskState.create("test", run_id="r1")
            store.start_run(ts)
            ts.finish_success("answer")
            store.write_report(ts, {"total": 42})

            report = store.load_report("r1")
            assert report["status"] == "completed"
            assert report["total_tokens"]["total"] == 42
            assert "answer" in report["final_answer"]

    def test_atomic_write_no_corruption(self):
        """原子写入：不会产生半截 JSON。"""
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            store = RunStore(d)
            ts = TaskState.create("test", run_id="r1")
            store.start_run(ts)
            store.update_state(ts)

            # 读取的文件应该是完整 JSON
            path = store._state_path()
            data = json.loads(path.read_text())
            assert data["run_id"] == "r1"


# ═══════════════════════════════════════════════════
# Trace / TraceLogger
# ═══════════════════════════════════════════════════

class TestTraceEvent:
    """TraceEvent 数据模型。"""

    def test_auto_generated_fields(self):
        te = TraceEvent(event_type=EventType.RUN_STARTED, run_id="r1")
        assert te.event_id
        assert te.created_at
        assert te.timestamp > 0


class TestTracePersistence:
    """Trace 写入（通过 RunStore）和 JSONL 容错。"""

    def test_log_events(self):
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            rs = RunStore(str(d))
            ts = TaskState.create("test", run_id="r1")
            rs.start_run(ts)
            rs.log(EventType.RUN_STARTED, run_id="r1", msg="start")
            rs.log(EventType.TOOL_EXECUTED, run_id="r1", tool="read_file")

            events = read_jsonl(rs._trace_path())
            assert len(events) == 2
            assert events[0]["event"] == EventType.RUN_STARTED
            assert events[1]["tool"] == "read_file"

    def test_corrupted_line_skipped(self):
        """崩溃产生的半行 JSON 应被跳过。"""
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            rs = RunStore(str(d))
            ts = TaskState.create("test", run_id="r1")
            rs.start_run(ts)
            rs.log(EventType.RUN_STARTED, run_id="r1")

            path = rs._trace_path()
            with open(path, "a") as f:
                f.write('{"broken": "json"\n')

            rs.log(EventType.RUN_FINISHED, run_id="r1")
            events = read_jsonl(path)
            assert len(events) == 2

    def test_read_jsonl_empty(self):
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            assert read_jsonl(d / "nonexistent.jsonl") == []


# ═══════════════════════════════════════════════════
# Checkpoint / CheckpointManager
# ═══════════════════════════════════════════════════

class TestCheckpointManager:
    """CheckpointManager 创建、评估、渲染。"""

    def test_create_and_evaluate_valid(self):
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            cm = CheckpointManager(d)
            ts = TaskState.create("test task")
            ts.record_tool("read_file")

            ckpt = cm.create(ts)
            result = cm.evaluate(ckpt)
            assert result["status"] == CHECKPOINT_FULL_VALID
            assert cm.can_resume(ckpt)

    def test_file_change_detected(self):
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            test_file = os.path.join(d, "test.py")
            Path(test_file).write_text("v1")

            cm = CheckpointManager(d)
            ts = TaskState.create("test")
            ckpt = cm.create(ts, recent_files=[test_file])

            # 文件被外部修改
            Path(test_file).write_text("v2")

            result = cm.evaluate(ckpt)
            assert result["status"] == CHECKPOINT_PARTIAL_STALE
            assert test_file in result["stale_paths"]

    def test_file_deleted_detected(self):
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            test_file = os.path.join(d, "test.py")
            Path(test_file).write_text("v1")

            cm = CheckpointManager(d)
            ts = TaskState.create("test")
            ckpt = cm.create(ts, recent_files=[test_file])

            # 文件被删除
            os.remove(test_file)

            result = cm.evaluate(ckpt)
            assert result["status"] == CHECKPOINT_PARTIAL_STALE

    def test_schema_mismatch(self):
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            cm = CheckpointManager(d)
            ts = TaskState.create("test")
            ckpt = cm.create(ts)
            ckpt.schema_version = "v0-old"

            result = cm.evaluate(ckpt)
            assert result["status"] == CHECKPOINT_SCHEMA_MISMATCH
            assert not cm.can_resume(ckpt)

    def test_render_includes_key_info(self):
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            cm = CheckpointManager(d)
            ts = TaskState.create("fix the login bug")
            ts.record_tool("read_file")

            text = cm.render(cm.create(ts))
            assert "fix the login bug" in text
            assert "full-valid" in text

    def test_checkpoint_ids_form_chain(self):
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            cm = CheckpointManager(d)
            ts = TaskState.create("task")

            ckpt1 = cm.create(ts)
            ckpt2 = cm.create(ts)

            assert ckpt2.checkpoint_id != ckpt1.checkpoint_id
            assert ckpt2.parent_id == ckpt1.checkpoint_id

    def test_runtime_identity_keys_defined(self):
        """确保 RUNTIME_IDENTITY_KEYS 覆盖了关键环境字段。"""
        assert "cwd" in RUNTIME_IDENTITY_KEYS
        assert "model" in RUNTIME_IDENTITY_KEYS
        assert "max_steps" in RUNTIME_IDENTITY_KEYS
        assert "feature_flags" in RUNTIME_IDENTITY_KEYS


class TestCheckpointDataModel:
    """Checkpoint 数据模型测试。"""

    def test_fields_have_defaults(self):
        ckpt = Checkpoint()
        assert ckpt.checkpoint_id.startswith("ckpt_")
        assert ckpt.schema_version == "v1"
        assert ckpt.key_files == {}
        assert ckpt.pending_protocols == []
        assert ckpt.snapshot_teammates == {}
        assert ckpt.unprocessed_messages == []
