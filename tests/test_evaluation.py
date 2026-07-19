"""Tests for evaluation framework — Evaluator, VerifierRegistry, MetricsCollector.

Run: python -m pytest tests/test_evaluation.py -v
"""

import json
import sys
import tempfile
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from echo.evaluation.benchmark import BenchmarkTask, BenchmarkResult, BenchmarkSuite
from echo.evaluation.evaluator import Evaluator, VerifierRegistry
from echo.evaluation.metrics import MetricsCollector, RunRecord, RunComparison
from echo.persistence.trace import read_jsonl


# ═══════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════

def _make_task(**kwargs) -> BenchmarkTask:
    """Quick task builder with sensible defaults."""
    defaults = {
        "id": "test-task",
        "prompt": "Do something.",
        "step_budget": 10,
    }
    defaults.update(kwargs)
    return BenchmarkTask(**defaults)


def _make_result(**kwargs) -> BenchmarkResult:
    """Quick result builder."""
    defaults = {"task_id": "test-task", "passed": False, "score": 0.0}
    defaults.update(kwargs)
    return BenchmarkResult(**defaults)


# ═══════════════════════════════════════════════════
# VerifierRegistry — 直接调用（不经过 Evaluator）
# ═══════════════════════════════════════════════════

class TestVerifierFileExists:
    """file_exists 验证器。"""

    def test_file_exists_passes(self):
        v = VerifierRegistry()
        with tempfile.TemporaryDirectory() as d:
            workspace = Path(d)
            (workspace / "hello.txt").write_text("hi")
            task = _make_task(expected_artifact="hello.txt", verifier="file_exists")
            passed, score, metrics = v.verify("file_exists", task, _make_result(), workspace)
            assert passed
            assert score == 1.0

    def test_file_not_exists_fails(self):
        v = VerifierRegistry()
        with tempfile.TemporaryDirectory() as d:
            task = _make_task(expected_artifact="missing.txt", verifier="file_exists")
            passed, score, metrics = v.verify("file_exists", task, _make_result(), Path(d))
            assert not passed
            assert score == 0.0


class TestVerifierFileContains:
    """file_contains 验证器。"""

    def test_contains_passes(self):
        v = VerifierRegistry()
        with tempfile.TemporaryDirectory() as d:
            workspace = Path(d)
            (workspace / "out.txt").write_text("line1\nHello World\nline3")
            task = _make_task(
                expected_artifact="out.txt",
                verifier="file_contains",
                verifier_params={"content": "Hello World"},
            )
            passed, score, _ = v.verify("file_contains", task, _make_result(), workspace)
            assert passed

    def test_not_contains_fails(self):
        v = VerifierRegistry()
        with tempfile.TemporaryDirectory() as d:
            workspace = Path(d)
            (workspace / "out.txt").write_text("just some text")
            task = _make_task(
                expected_artifact="out.txt",
                verifier="file_contains",
                verifier_params={"content": "missing string"},
            )
            passed, score, _ = v.verify("file_contains", task, _make_result(), workspace)
            assert not passed

    def test_missing_file_fails(self):
        v = VerifierRegistry()
        with tempfile.TemporaryDirectory() as d:
            task = _make_task(
                expected_artifact="nope.txt",
                verifier="file_contains",
                verifier_params={"content": "x"},
            )
            passed, score, metrics = v.verify("file_contains", task, _make_result(), Path(d))
            assert not passed
            assert "error" in metrics


class TestVerifierFileEquals:
    """file_equals 验证器。"""

    def test_exact_match_passes(self):
        v = VerifierRegistry()
        with tempfile.TemporaryDirectory() as d:
            workspace = Path(d)
            (workspace / "out.txt").write_text("exact content")
            task = _make_task(
                expected_artifact="out.txt",
                verifier="file_equals",
                verifier_params={"content": "exact content"},
            )
            passed, _, _ = v.verify("file_equals", task, _make_result(), workspace)
            assert passed

    def test_partial_match_fails(self):
        """file_equals 要求精确匹配，子串不算。"""
        v = VerifierRegistry()
        with tempfile.TemporaryDirectory() as d:
            workspace = Path(d)
            (workspace / "out.txt").write_text("extra stuff exact content more")
            task = _make_task(
                expected_artifact="out.txt",
                verifier="file_equals",
                verifier_params={"content": "exact content"},
            )
            passed, _, _ = v.verify("file_equals", task, _make_result(), workspace)
            assert not passed


class TestVerifierOutputContains:
    """output_contains 验证器。"""

    def test_output_contains_passes(self):
        v = VerifierRegistry()
        task = _make_task(verifier_params={"content": "success"})
        result = _make_result(final_answer="Task completed with success.")
        passed, _, _ = v.verify("output_contains", task, result, Path("."))
        assert passed

    def test_output_not_contains_fails(self):
        v = VerifierRegistry()
        task = _make_task(verifier_params={"content": "success"})
        result = _make_result(final_answer="Task failed.")
        passed, _, _ = v.verify("output_contains", task, result, Path("."))
        assert not passed

    def test_empty_final_answer_fails(self):
        v = VerifierRegistry()
        task = _make_task(verifier_params={"content": "anything"})
        result = _make_result(final_answer="")
        passed, _, _ = v.verify("output_contains", task, result, Path("."))
        assert not passed


class TestVerifierOutputMatches:
    """output_matches 验证器。"""

    def test_matches_passes(self):
        v = VerifierRegistry()
        task = _make_task(verifier_params={"pattern": r"found \d+ files"})
        result = _make_result(final_answer="Scanned and found 42 files in total.")
        passed, _, _ = v.verify("output_matches", task, result, Path("."))
        assert passed

    def test_not_matches_fails(self):
        v = VerifierRegistry()
        task = _make_task(verifier_params={"pattern": r"Error:.*"})
        result = _make_result(final_answer="All good.")
        passed, _, _ = v.verify("output_matches", task, result, Path("."))
        assert not passed


class TestVerifierNoErrors:
    """no_errors 验证器。"""

    def test_no_errors_passes(self):
        v = VerifierRegistry()
        result = _make_result(errors=[])
        passed, _, _ = v.verify("no_errors", _make_task(), result, Path("."))
        assert passed

    def test_has_errors_fails(self):
        v = VerifierRegistry()
        result = _make_result(errors=["Something went wrong"])
        passed, _, _ = v.verify("no_errors", _make_task(), result, Path("."))
        assert not passed


class TestVerifierTraceHasEvent:
    """trace_has_event 验证器。"""

    def test_event_found_passes(self):
        v = VerifierRegistry()
        with tempfile.TemporaryDirectory() as d:
            trace_path = Path(d) / "trace.jsonl"
            trace_path.write_text(
                '{"event":"run_started"}\n{"event":"tool_executed"}\n{"event":"run_finished"}\n',
                encoding="utf-8",
            )
            task = _make_task(verifier_params={"event_type": "tool_executed"})
            result = _make_result(trace_path=str(trace_path))
            passed, _, metrics = v.verify("trace_has_event", task, result, Path(d))
            assert passed
            assert metrics["total_events"] == 3

    def test_event_not_found_fails(self):
        v = VerifierRegistry()
        with tempfile.TemporaryDirectory() as d:
            trace_path = Path(d) / "trace.jsonl"
            trace_path.write_text(
                '{"event":"run_started"}\n',
                encoding="utf-8",
            )
            task = _make_task(verifier_params={"event_type": "model_requested"})
            result = _make_result(trace_path=str(trace_path))
            passed, _, _ = v.verify("trace_has_event", task, result, Path(d))
            assert not passed

    def test_missing_trace_file_fails(self):
        v = VerifierRegistry()
        task = _make_task(verifier_params={"event_type": "any"})
        result = _make_result(trace_path="/no/such/trace.jsonl")
        passed, _, _ = v.verify("trace_has_event", task, result, Path("."))
        assert not passed


class TestUnregisteredVerifier:
    """未注册的 verifier。"""

    def test_raises_not_implemented(self):
        v = VerifierRegistry()
        with pytest.raises(NotImplementedError, match="unknown_verifier"):
            v.verify("unknown_verifier", _make_task(), _make_result(), Path("."))

    def test_list_names(self):
        v = VerifierRegistry()
        names = v.list_names()
        assert "file_exists" in names
        assert "file_contains" in names
        assert "output_matches" in names
        assert "trace_has_event" in names
        assert len(names) == 7


# ═══════════════════════════════════════════════════
# Evaluator.run_one() — 端到端
# ═══════════════════════════════════════════════════

class TestEvaluatorSingle:
    """Evaluator.run_one() 单任务测试。"""

    def test_write_file_and_verify_content(self):
        """Agent 写文件 → file_contains verifier 检查内容。"""
        evaluator = Evaluator()
        task = BenchmarkTask(
            id="write-hello",
            prompt="Write hello.txt with 'Hello World'",
            model_outputs=[
                '<tool name="write_file" path="hello.txt" content="Hello World"/>',
                "Done.",
            ],
            expected_artifact="hello.txt",
            verifier="file_contains",
            verifier_params={"content": "Hello World"},
        )
        result = evaluator.run_one(task)
        assert result.task_id == "write-hello"
        assert result.passed
        assert result.score == 1.0
        assert result.tool_steps == 1
        assert len(result.errors) == 0

    def test_file_exists_verifier(self):
        """Agent 写文件 → file_exists 验证器。"""
        evaluator = Evaluator()
        task = BenchmarkTask(
            id="create-config",
            prompt="Create config.json",
            model_outputs=[
                '<tool name="write_file" path="config.json" content="{}"/>',
                "Created config.",
            ],
            expected_artifact="config.json",
            verifier="file_exists",
        )
        result = evaluator.run_one(task)
        assert result.passed
        assert result.tool_steps == 1

    def test_verifier_fails_on_wrong_content(self):
        """Agent 写错内容 → file_contains 失败。"""
        evaluator = Evaluator()
        task = BenchmarkTask(
            id="wrong-content",
            prompt="Write a file.",
            model_outputs=[
                '<tool name="write_file" path="out.txt" content="wrong content"/>',
                "Done.",
            ],
            expected_artifact="out.txt",
            verifier="file_contains",
            verifier_params={"content": "expected content"},
        )
        result = evaluator.run_one(task)
        assert not result.passed
        assert result.score == 0.0

    def test_output_contains_verifier(self):
        """Agent 最终回复包含预期文本。"""
        evaluator = Evaluator()
        task = BenchmarkTask(
            id="say-hello",
            prompt="Greet the user.",
            model_outputs=["Hello, this project contains 3 Python files."],
            verifier="output_contains",
            verifier_params={"content": "Python files"},
        )
        result = evaluator.run_one(task)
        assert result.passed
        assert result.tool_steps == 0

    def test_output_matches_verifier(self):
        """Agent 最终回复匹配正则。"""
        evaluator = Evaluator()
        task = BenchmarkTask(
            id="scan-report",
            prompt="Scan and report.",
            model_outputs=["Scan complete: found 7 issues."],
            verifier="output_matches",
            verifier_params={"pattern": r"found \d+ issues"},
        )
        result = evaluator.run_one(task)
        assert result.passed

    def test_no_errors_verifier_passes(self):
        """无错误时 no_errors 通过。"""
        evaluator = Evaluator()
        task = BenchmarkTask(
            id="just-reply",
            prompt="Just say hello.",
            model_outputs=["Hello World."],
            verifier="no_errors",
        )
        result = evaluator.run_one(task)
        assert result.passed
        assert len(result.errors) == 0

    def test_trace_has_event_verifier(self):
        """trace.jsonl 包含 run_started 事件。"""
        evaluator = Evaluator()
        task = BenchmarkTask(
            id="trace-check",
            prompt="Do one thing.",
            model_outputs=["Done."],
            verifier="trace_has_event",
            verifier_params={"event_type": "run_started"},
        )
        result = evaluator.run_one(task)
        assert result.passed

    def test_no_verifier_defaults_to_no_errors_passed(self):
        """无 verifier 时，只要没异常就通过。"""
        evaluator = Evaluator()
        task = BenchmarkTask(
            id="no-verify",
            prompt="Just answer.",
            model_outputs=["OK"],
        )
        result = evaluator.run_one(task)
        assert result.passed

    def test_no_verifier_fails_on_tool_error(self):
        """Agent 调用不存在的工具 → _scan_tool_errors 捕获 → no_errors 失败。"""
        evaluator = Evaluator()
        task = BenchmarkTask(
            id="bad-tool",
            prompt="Try an invalid tool.",
            model_outputs=[
                '<tool name="nonexistent_tool" x="y"/>',
                "Tried.",
            ],
            verifier="no_errors",
        )
        result = evaluator.run_one(task)
        assert not result.passed
        assert len(result.errors) >= 1
        assert any("Error:" in e or "not found" in e.lower() for e in result.errors)

    def test_allowed_tools_restricts_registry(self):
        """allowed_tools 只允许 read_file → write_file 被拦截 → error 被捕获。"""
        evaluator = Evaluator()
        task = BenchmarkTask(
            id="restricted-tools",
            prompt="Only read is allowed.",
            allowed_tools=["read_file"],
            model_outputs=[
                '<tool name="read_file" path="notes.txt"/>',
                '<tool name="write_file" path="out.txt" content="should fail"/>',
                "Done.",
            ],
            verifier="no_errors",
        )
        result = evaluator.run_one(task)
        assert not result.passed
        assert any("not found" in e.lower() or "write_file" in e for e in result.errors)

    def test_missing_fixture_repo_fails_immediately(self):
        """fixture_repo 路径不存在 → 立即返回 failed，不执行 AgentLoop。"""
        evaluator = Evaluator()
        task = BenchmarkTask(
            id="missing-fixture",
            prompt="This should never run.",
            fixture_repo="/no/such/path/xyz",
            verifier="no_errors",
        )
        result = evaluator.run_one(task)
        assert not result.passed
        assert result.score == 0.0
        assert any("Fixture repo not found" in e for e in result.errors)
        assert result.tool_steps == 0
        assert result.trace_path == ""

    def test_workspace_root_respected(self):
        """workspace_root 生效 → benchmark workspace 放在用户指定目录下。"""
        with tempfile.TemporaryDirectory() as custom_root:
            evaluator = Evaluator(workspace_root=str(custom_root))
            task = BenchmarkTask(
                id="custom-root",
                prompt="Just answer.",
                model_outputs=["OK"],
                verifier="no_errors",
            )
            result = evaluator.run_one(task)
            assert result.passed
            # trace 路径在 custom_root 下，不在系统 temp 下
            assert result.trace_path
            assert str(custom_root) in result.trace_path

    def test_verifier_exception_does_not_break_suite(self):
        """自定义 verifier 崩溃 → 单 task 标记 failed，不炸整个 suite。"""
        evaluator = Evaluator()

        # 注册一个会崩溃的 verifier
        def _crash(task, result, workspace):
            raise RuntimeError("boom")

        evaluator.verifiers.register("crashy", _crash)

        task = BenchmarkTask(
            id="crash-test",
            prompt="Anything.",
            model_outputs=["OK"],
            verifier="crashy",
        )
        result = evaluator.run_one(task)
        assert not result.passed
        assert any("Verifier" in e and "crashy" in e for e in result.errors)

    def test_exhausted_outputs_returns_empty_final_answer(self):
        """FakeLLMClient 序列耗尽 → 返回空回复。"""
        evaluator = Evaluator()
        task = BenchmarkTask(
            id="exhausted",
            prompt="Anything.",
            model_outputs=["OK I'm done."],
            verifier="output_contains",
            verifier_params={"content": "done"},
        )
        result = evaluator.run_one(task)
        # 第一次调用返回 "OK I'm done."，没有 tool_use → finish_success
        assert result.passed

    def test_multiple_tool_calls_in_sequence(self):
        """Agent 多轮工具调用（全部成功才算 step）。"""
        evaluator = Evaluator()
        task = BenchmarkTask(
            id="multi-tool",
            prompt="Write two files.",
            model_outputs=[
                '<tool name="write_file" path="first.txt" content="AAA"/>',
                '<tool name="write_file" path="second.txt" content="BBB"/>',
                "Both written.",
            ],
            expected_artifact="second.txt",
            verifier="file_contains",
            verifier_params={"content": "BBB"},
        )
        result = evaluator.run_one(task)
        assert result.passed
        assert result.tool_steps == 2


# ═══════════════════════════════════════════════════
# Evaluator.run() — 多任务套件
# ═══════════════════════════════════════════════════

class TestEvaluatorSuite:
    """Evaluator.run() 多任务测试。"""

    def test_run_suite_all_pass(self):
        evaluator = Evaluator()
        suite = BenchmarkSuite(
            name="test-suite",
            tasks=[
                BenchmarkTask(
                    id="task-a",
                    prompt="Write a.txt",
                    model_outputs=[
                        '<tool name="write_file" path="a.txt" content="AAA"/>',
                        "Ok.",
                    ],
                    expected_artifact="a.txt",
                    verifier="file_contains",
                    verifier_params={"content": "AAA"},
                ),
                BenchmarkTask(
                    id="task-b",
                    prompt="Write b.txt",
                    model_outputs=[
                        '<tool name="write_file" path="b.txt" content="BBB"/>',
                        "Ok.",
                    ],
                    expected_artifact="b.txt",
                    verifier="file_contains",
                    verifier_params={"content": "BBB"},
                ),
            ],
        )
        results = evaluator.run(suite)
        assert len(results) == 2
        assert results[0].task_id == "task-a"
        assert results[1].task_id == "task-b"
        assert results[0].passed
        assert results[1].passed

    def test_run_suite_mixed_results(self):
        evaluator = Evaluator()
        suite = BenchmarkSuite(
            name="mixed-suite",
            tasks=[
                BenchmarkTask(
                    id="good",
                    prompt="Create good.txt",
                    model_outputs=[
                        '<tool name="write_file" path="good.txt" content="correct"/>',
                        "Done.",
                    ],
                    expected_artifact="good.txt",
                    verifier="file_contains",
                    verifier_params={"content": "correct"},
                ),
                BenchmarkTask(
                    id="bad",
                    prompt="Create bad.txt",
                    model_outputs=[
                        '<tool name="write_file" path="bad.txt" content="wrong"/>',
                        "Done.",
                    ],
                    expected_artifact="bad.txt",
                    verifier="file_contains",
                    verifier_params={"content": "expected_but_missing"},
                ),
            ],
        )
        results = evaluator.run(suite)
        assert results[0].passed
        assert not results[1].passed

    def test_suite_serialization_roundtrip(self):
        """Suite → dict → Suite 往返。"""
        original = BenchmarkSuite(
            name="roundtrip",
            schema_version=1,
            tasks=[
                BenchmarkTask(
                    id="t1",
                    prompt="Do X.",
                    model_outputs=["OK"],
                    verifier="no_errors",
                    verifier_params={"foo": "bar"},
                ),
            ],
        )
        restored = BenchmarkSuite.from_dict(original.to_dict())
        assert restored.name == "roundtrip"
        assert len(restored.tasks) == 1
        assert restored.tasks[0].verifier_params == {"foo": "bar"}

    def test_save_and_load_results(self):
        evaluator = Evaluator()
        results = [
            BenchmarkResult(task_id="t1", passed=True, score=1.0, tool_steps=2),
            BenchmarkResult(task_id="t2", passed=False, score=0.3, errors=["fail"]),
        ]
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "results.json"
            evaluator.save_results(results, str(path))
            loaded = evaluator.load_results(str(path))
            assert len(loaded) == 2
            assert loaded[0].task_id == "t1"
            assert loaded[0].passed
            assert loaded[1].task_id == "t2"
            assert not loaded[1].passed
            assert loaded[1].errors == ["fail"]


# ═══════════════════════════════════════════════════
# MetricsCollector.compare()
# ═══════════════════════════════════════════════════

class TestMetricsCollector:
    """MetricsCollector 指标收集和对比。"""

    def test_add_and_summary(self):
        mc = MetricsCollector()
        results = [
            BenchmarkResult(task_id="a", passed=True, score=1.0, tool_steps=3),
            BenchmarkResult(task_id="b", passed=False, score=0.0, tool_steps=5),
        ]
        mc.add("v1", results, model="fake")
        summary = mc.summary("v1")
        assert summary["total_tasks"] == 2
        assert summary["passed_count"] == 1
        assert summary["pass_rate"] == 0.5
        assert summary["total_duration"] == 0.0

    def test_compare_pass_rate_delta(self):
        mc = MetricsCollector()
        baseline = [
            BenchmarkResult(task_id="a", passed=True, score=1.0),
            BenchmarkResult(task_id="b", passed=False, score=0.0),
            BenchmarkResult(task_id="c", passed=True, score=1.0),
        ]
        experiment = [
            BenchmarkResult(task_id="a", passed=True, score=1.0),   # same
            BenchmarkResult(task_id="b", passed=True, score=1.0),   # improved
            BenchmarkResult(task_id="c", passed=False, score=0.0),  # regressed
        ]
        mc.add("baseline", baseline)
        mc.add("experiment", experiment)
        comp = mc.compare("baseline", "experiment")

        # baseline: 2/3 ≈ 0.667, experiment: 2/3 ≈ 0.667 → delta 0
        # Wait: experiment has a=True, b=True, c=False → 2/3 ≈ 0.667
        # baseline has a=True, b=False, c=True → 2/3 ≈ 0.667
        # So delta is 0.0, passed_diff = ["b"], failed_diff = ["c"]
        assert comp.pass_rate_delta == 0.0
        assert comp.passed_diff == ["b"]
        assert comp.failed_diff == ["c"]
        assert len(comp.per_task) == 3

    def test_compare_improvement(self):
        mc = MetricsCollector()
        baseline = [
            BenchmarkResult(task_id="x", passed=False, score=0.0, tool_steps=10),
            BenchmarkResult(task_id="y", passed=False, score=0.0, tool_steps=8),
        ]
        experiment = [
            BenchmarkResult(task_id="x", passed=True, score=1.0, tool_steps=5),
            BenchmarkResult(task_id="y", passed=True, score=1.0, tool_steps=4),
        ]
        mc.add("baseline", baseline)
        mc.add("experiment", experiment)
        comp = mc.compare("baseline", "experiment")

        assert comp.pass_rate_delta == 1.0
        assert comp.avg_steps_delta == -4.5  # baseline avg=9, experiment avg=4.5
        assert set(comp.passed_diff) == {"x", "y"}
        assert comp.failed_diff == []

    def test_compare_regression(self):
        mc = MetricsCollector()
        baseline = [
            BenchmarkResult(task_id="z", passed=True, score=1.0),
        ]
        experiment = [
            BenchmarkResult(task_id="z", passed=False, score=0.0),
        ]
        mc.add("baseline", baseline)
        mc.add("experiment", experiment)
        comp = mc.compare("baseline", "experiment")

        assert comp.pass_rate_delta == -1.0
        assert comp.passed_diff == []
        assert comp.failed_diff == ["z"]
        # per_task status
        assert comp.per_task[0]["status"] == "regressed"

    def test_compare_per_task_details(self):
        mc = MetricsCollector()
        mc.add("v1", [BenchmarkResult(task_id="t", passed=True, score=1.0, tool_steps=3)])
        mc.add("v2", [BenchmarkResult(task_id="t", passed=True, score=1.0, tool_steps=2)])
        comp = mc.compare("v1", "v2")
        assert comp.avg_steps_delta == -1.0
        task_detail = comp.per_task[0]
        assert task_detail["task_id"] == "t"
        assert task_detail["baseline_steps"] == 3
        assert task_detail["experiment_steps"] == 2
        assert task_detail["status"] == "same"

    def test_list_runs(self):
        mc = MetricsCollector()
        mc.add("v1", [])
        mc.add("v2", [])
        assert sorted(mc.list_runs()) == ["v1", "v2"]

    def test_get_missing_returns_none(self):
        mc = MetricsCollector()
        assert mc.get("nonexistent") is None

    def test_summary_missing_returns_empty(self):
        mc = MetricsCollector()
        assert mc.summary("nonexistent") == {}


# ═══════════════════════════════════════════════════
# 集成测试
# ═══════════════════════════════════════════════════

class TestWorkspaceIsolation:
    """确保 task 之间不互相污染。"""

    def test_different_tasks_have_different_workspaces(self):
        evaluator = Evaluator()
        task1 = BenchmarkTask(
            id="task-1",
            prompt="Write f1.txt",
            model_outputs=[
                '<tool name="write_file" path="f1.txt" content="one"/>',
                "Done.",
            ],
            expected_artifact="f1.txt",
            verifier="file_contains",
            verifier_params={"content": "one"},
        )
        task2 = BenchmarkTask(
            id="task-2",
            prompt="Write f2.txt",
            model_outputs=[
                '<tool name="write_file" path="f2.txt" content="two"/>',
                "Done.",
            ],
            expected_artifact="f2.txt",
            verifier="file_contains",
            verifier_params={"content": "two"},
        )
        r1 = evaluator.run_one(task1)
        r2 = evaluator.run_one(task2)
        assert r1.passed
        assert r2.passed
        # trace path 不同 → workspace 不同
        assert r1.trace_path != r2.trace_path

    def test_isolation_no_cross_contamination(self):
        """Task A 写文件不会影响 Task B。"""
        evaluator = Evaluator()
        task_a = BenchmarkTask(
            id="a",
            prompt="Write shared.txt",
            model_outputs=[
                '<tool name="write_file" path="shared.txt" content="from-A"/>',
                "Done.",
            ],
            expected_artifact="shared.txt",
            verifier="file_contains",
            verifier_params={"content": "from-A"},
        )
        task_b = BenchmarkTask(
            id="b",
            prompt="Write shared.txt",
            model_outputs=[
                '<tool name="write_file" path="shared.txt" content="from-B"/>',
                "Done.",
            ],
            expected_artifact="shared.txt",
            verifier="file_contains",
            verifier_params={"content": "from-B"},
        )
        assert evaluator.run_one(task_a).passed
        assert evaluator.run_one(task_b).passed


class TestTraceAndReport:
    """trace / report 生成。"""

    def test_trace_generated(self):
        evaluator = Evaluator()
        task = BenchmarkTask(
            id="trace-test",
            prompt="Do something.",
            model_outputs=["Done."],
        )
        result = evaluator.run_one(task)
        assert result.trace_path
        trace_file = Path(result.trace_path)
        assert trace_file.exists()
        events = read_jsonl(trace_file)
        assert len(events) > 0
        # 至少包含 run_started（log）和 run_finished 相关的 tool_executed log
        event_types = [e.get("event") for e in events]
        assert "run_started" in event_types

    def test_report_generated(self):
        """AgentLoop 结束写入 report.json。"""
        evaluator = Evaluator()
        task = BenchmarkTask(
            id="report-test",
            prompt="Just answer.",
            model_outputs=["Hello."],
        )
        result = evaluator.run_one(task)
        # report 在 run_store._run_dir / "report.json"
        trace_file = Path(result.trace_path)
        run_dir = trace_file.parent
        report_path = run_dir / "report.json"
        assert report_path.exists()
        report = json.loads(report_path.read_text(encoding="utf-8"))
        assert report["run_id"]
        assert report["status"] in ("completed", "stopped")


class TestFixtureCopy:
    """fixture 文件复制。"""

    def test_fixture_files_copied_to_workspace(self):
        with tempfile.TemporaryDirectory() as fixture_dir:
            fixture = Path(fixture_dir)
            (fixture / "src").mkdir(parents=True)
            (fixture / "src" / "main.py").write_text("print('hello')")
            (fixture / "data" / "input").mkdir(parents=True)
            (fixture / "data" / "input" / "config.yaml").write_text("key: value")

            evaluator = Evaluator()
            task = BenchmarkTask(
                id="fixture-test",
                prompt="Read and verify main.py exists.",
                fixture_repo=str(fixture),
                model_outputs=[
                    '<tool name="read_file" path="src/main.py"/>',
                    "Read the file.",
                ],
                verifier="no_errors",
            )
            result = evaluator.run_one(task)
            assert result.passed
