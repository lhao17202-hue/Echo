"""标杆评估器 —— 运行 BenchmarkSuite 并收集结果。

每个 task 在独立临时 workspace 中执行，用 FakeLLMClient 做确定性测试。
支持 7 个内置 verifier，可插拔扩展。

使用方式：
  suite = BenchmarkSuite.from_json("benchmarks/coding_tasks.json")
  evaluator = Evaluator()
  results = evaluator.run(suite)
  evaluator.save_results(results, "results.json")
"""

import json
import re
import shutil
import tempfile
import time
import uuid
from pathlib import Path
from typing import Callable

from echo.evaluation.benchmark import BenchmarkTask, BenchmarkResult, BenchmarkSuite
from echo.providers.fake_client import FakeLLMClient
from echo.tools.registry import ToolRegistry
from echo.tools.executor import ToolExecutor
from echo.tools.sandbox import Sandbox, ShellExecutor
from echo.memory.base import MemoryManager
from echo.memory.default import KeywordMemory
from echo.hooks.base import HookManager
from echo.hooks.builtin import PermissionHook, LogHook, PostLogHook, StatsHook
from echo.core.context_manager import ContextManager, ContextConfig
from echo.core.agent_loop import AgentLoop
from echo.core.task_state import TaskState
from echo.persistence.session_store import Session, SessionStore
from echo.persistence.run_store import RunStore


# ═══════════════════════════════════════════════════════
# Verifier 函数签名
# ═══════════════════════════════════════════════════════
# (task: BenchmarkTask, result: BenchmarkResult, workspace: Path)
#     → tuple[bool, float, dict]
#            passed  score  metrics


# ═══════════════════════════════════════════════════════
# VerifierRegistry — 内置 7 个验证器
# ═══════════════════════════════════════════════════════

class VerifierRegistry:
    """验证器注册表 —— 按名称查找验证函数。

    验证器函数签名：
      def verify(task: BenchmarkTask, result: BenchmarkResult,
                workspace: Path) -> tuple[bool, float, dict]:
          return (passed: bool, score: float, metrics: dict)

    内置验证器：
      - file_exists:      检查 expected_artifact 是否存在
      - file_contains:    检查文件内容是否包含指定文本
      - file_equals:      检查文件内容是否精确匹配
      - output_contains:  检查 final_answer 是否包含子串
      - output_matches:   检查 final_answer 是否匹配正则
      - no_errors:        检查 errors 列表是否为空
      - trace_has_event:  检查 trace 中是否存在指定事件类型
    """

    def __init__(self):
        self._verifiers: dict[str, Callable] = {}
        self._register_builtins()

    def _register_builtins(self) -> None:
        """注册 7 个内置验证器。"""
        self.register("file_exists", self._verify_file_exists)
        self.register("file_contains", self._verify_file_contains)
        self.register("file_equals", self._verify_file_equals)
        self.register("output_contains", self._verify_output_contains)
        self.register("output_matches", self._verify_output_matches)
        self.register("no_errors", self._verify_no_errors)
        self.register("trace_has_event", self._verify_trace_has_event)

    def register(self, name: str, fn: Callable) -> "VerifierRegistry":
        """注册一个验证函数。"""
        self._verifiers[name] = fn
        return self

    def get(self, name: str) -> Callable | None:
        """按名获取验证函数。"""
        return self._verifiers.get(name)

    def verify(self, name: str, task: BenchmarkTask, result: BenchmarkResult,
               workspace: Path) -> tuple[bool, float, dict]:
        """执行验证。"""
        fn = self._verifiers.get(name)
        if fn is None:
            raise NotImplementedError(f"验证器 '{name}' 未注册。"
                                      f"可用: {', '.join(self._verifiers.keys())}")
        return fn(task, result, workspace)

    def list_names(self) -> list[str]:
        """列出所有已注册的验证器名称。"""
        return list(self._verifiers.keys())

    # ── 内置验证器实现 ──────────────────────────────

    @staticmethod
    def _verify_file_exists(task: BenchmarkTask, result: BenchmarkResult,
                            workspace: Path) -> tuple[bool, float, dict]:
        path = workspace / task.expected_artifact
        passed = path.exists()
        return passed, 1.0 if passed else 0.0, {"file": str(path)}

    @staticmethod
    def _verify_file_contains(task: BenchmarkTask, result: BenchmarkResult,
                              workspace: Path) -> tuple[bool, float, dict]:
        path = workspace / task.expected_artifact
        expected = task.verifier_params.get("content", "")
        if not path.is_file():
            return False, 0.0, {"error": f"File not found: {path}"}
        text = path.read_text(encoding="utf-8")
        passed = expected in text
        return passed, 1.0 if passed else 0.0, {"file": str(path)}

    @staticmethod
    def _verify_file_equals(task: BenchmarkTask, result: BenchmarkResult,
                            workspace: Path) -> tuple[bool, float, dict]:
        path = workspace / task.expected_artifact
        expected = task.verifier_params.get("content", "")
        if not path.is_file():
            return False, 0.0, {"error": f"File not found: {path}"}
        text = path.read_text(encoding="utf-8")
        passed = text == expected
        return passed, 1.0 if passed else 0.0, {"file": str(path)}

    @staticmethod
    def _verify_output_contains(task: BenchmarkTask, result: BenchmarkResult,
                                workspace: Path) -> tuple[bool, float, dict]:
        expected = task.verifier_params.get("content", "")
        passed = expected in (result.final_answer or "")
        return passed, 1.0 if passed else 0.0, {}

    @staticmethod
    def _verify_output_matches(task: BenchmarkTask, result: BenchmarkResult,
                               workspace: Path) -> tuple[bool, float, dict]:
        pattern = task.verifier_params.get("pattern", "")
        try:
            passed = bool(re.search(pattern, result.final_answer or ""))
        except re.error:
            passed = False
        return passed, 1.0 if passed else 0.0, {"pattern": pattern}

    @staticmethod
    def _verify_no_errors(task: BenchmarkTask, result: BenchmarkResult,
                          workspace: Path) -> tuple[bool, float, dict]:
        # 1. result.errors 列表
        if result.errors:
            return False, 0.0, {"error_count": len(result.errors), "source": "result.errors"}

        # 2. final_answer 中出现 "Error:" 或 "Blocked:"（工具失败漏进回复）
        answer = (result.final_answer or "")
        if "Error:" in answer or "Blocked:" in answer:
            return False, 0.0, {"error_count": 1, "source": "final_answer"}

        return True, 1.0, {"error_count": 0}

    @staticmethod
    def _verify_trace_has_event(task: BenchmarkTask, result: BenchmarkResult,
                                workspace: Path) -> tuple[bool, float, dict]:
        event_type = task.verifier_params.get("event_type", "")
        if not result.trace_path or not Path(result.trace_path).exists():
            return False, 0.0, {"error": "Trace file not found"}
        from echo.persistence.trace import read_jsonl
        events = read_jsonl(result.trace_path)
        passed = any(e.get("event") == event_type for e in events)
        return passed, 1.0 if passed else 0.0, {
            "event_type": event_type,
            "total_events": len(events),
        }


# ═══════════════════════════════════════════════════════
# Evaluator — 标杆测试执行器
# ═══════════════════════════════════════════════════════

class Evaluator:
    """标杆测试执行器。

    设计要点：
      - 每个 task 在独立临时 workspace 中运行（目录隔离）
      - 使用 FakeLLMClient 做确定性测试（task.model_outputs 预设回复序列）
      - AgentLoop 直接组装（不经过 Echo 门面，避免耦合真实 provider）
      - 审批策略固定为 "auto"（safe + warn 自动放行，danger 拦截）
      - 结果通过 verifier 判定，支持 save/load
    """

    def __init__(self, workspace_root: str = ""):
        self.workspace_root = Path(workspace_root).resolve() if workspace_root else None
        self.verifiers = VerifierRegistry()

    @property
    def _bench_root(self) -> Path:
        """benchmark 工作区根目录（用户指定则用用户，否则系统临时）。"""
        if self.workspace_root:
            self.workspace_root.mkdir(parents=True, exist_ok=True)
        return self.workspace_root or Path(tempfile.gettempdir())

    # ── run / run_one ────────────────────────────────

    def run(self, suite: BenchmarkSuite) -> list[BenchmarkResult]:
        """运行整个测试套件，返回结果列表（顺序与 suite.tasks 一致）。"""
        results = []
        for task in suite.tasks:
            result = self.run_one(task)
            results.append(result)
        return results

    def run_one(self, task: BenchmarkTask) -> BenchmarkResult:
        """运行单个测试任务。

        1. 创建独立 workspace（用户指定根目录或系统临时）
        2. fixture_repo 不存在 → 直接失败
        3. allowed_tools 生效 → 只注册指定工具
        4. 用 FakeLLMClient 组装 AgentLoop 并执行
        5. 收集 errors（含 tool 失败 scan）、final_answer、trace_path
        6. 调用 verifier 判定通过/失败（任何异常都转 task failed）
        """
        start_time = time.time()

        # 0. fixture 路径检查（不存在 → 直接失败，避免静默跑空）
        if task.fixture_repo:
            fixture_src = Path(task.fixture_repo)
            if not fixture_src.exists():
                return BenchmarkResult(
                    task_id=task.id,
                    passed=False,
                    score=0.0,
                    errors=[f"Fixture repo not found: {task.fixture_repo}"],
                )

        # 1. 创建独立 workspace
        workspace = Path(tempfile.mkdtemp(
            prefix=f"{task.id}-",
            dir=str(self._bench_root),
        ))
        (workspace / ".echo").mkdir(parents=True, exist_ok=True)

        # 2. 复制 fixture 文件
        if task.fixture_repo:
            self._copy_fixtures(fixture_src, workspace)

        # 3. 组装子系统
        llm = FakeLLMClient(outputs=task.model_outputs)
        sandbox = Sandbox(str(workspace))
        shell = ShellExecutor(workspace)
        memory = MemoryManager(KeywordMemory())          # 无 durable backend，task 间隔离

        # ── allowed_tools 生效 ──
        tool_registry = ToolRegistry()
        tool_registry.discover("echo.tools.builtin")
        if task.allowed_tools:
            allowed_set = set(task.allowed_tools)
            tool_registry._tools = {
                name: t for name, t in tool_registry._tools.items()
                if name in allowed_set
            }
        executor = ToolExecutor(tool_registry)

        hooks = HookManager()
        hooks.register(PermissionHook(), priority=0)
        hooks.register(LogHook(), priority=100)
        hooks.register(PostLogHook(), priority=100)
        hooks.register(StatsHook(), priority=200)

        echo_dir = workspace / ".echo"
        context_manager = ContextManager(ContextConfig(
            persist_dir=str(echo_dir / "tool_outputs"),
            transcript_dir=str(echo_dir / "transcripts"),
        ))

        session_store = SessionStore(str(workspace))
        session = Session(
            session_id=f"bench-{task.id}-{uuid.uuid4().hex[:6]}",
            workspace_root=str(workspace),
            model_config={"provider": "fake", "model": "FakeLLMClient"},
            security_config={"approval_policy": "auto"},
        )
        session_store.save(session)

        run_store = RunStore(str(workspace / ".echo" / "sessions" / session.session_id))

        loop = AgentLoop(
            llm=llm, memory=memory, tools=executor,
            hooks=hooks, context=context_manager,
            sandbox=sandbox, shell=shell,
            session_store=session_store, run_store=run_store,
            max_steps=task.step_budget,
            approval_policy="auto",
        )
        loop._session = session

        # 4. 执行
        errors: list[str] = []
        tool_steps = 0
        attempts = 0
        final_answer = ""
        trace_path = ""

        try:
            final_answer = loop.run(task.prompt)
        except Exception as e:
            errors.append(f"AgentLoop crashed: {e}")

        # 收集运行后数据
        try:
            state = self._load_task_state(run_store)
            if state:
                tool_steps = getattr(state, "tool_steps", 0)
                attempts = getattr(state, "attempts", 0)
                for err in getattr(state, "errors", []) or []:
                    if err and err not in errors:
                        errors.append(str(err))
        except Exception:
            pass

        # ── 扫描 messages 中 tool_result 的 Error: ──
        errors.extend(self._scan_tool_errors(loop.messages))

        # 获取 trace 路径
        try:
            trace_path = str(run_store._run_dir / "trace.jsonl") if run_store._run_dir else ""
        except Exception:
            pass

        duration = time.time() - start_time

        # 构造结果
        result = BenchmarkResult(
            task_id=task.id,
            passed=False,            # 下面由 verifier 决定
            score=0.0,
            tool_steps=tool_steps,
            attempts=attempts,
            duration_seconds=round(duration, 3),
            final_answer=final_answer[:500],
            errors=errors,
            trace_path=trace_path,
        )

        # 5. 验证（任何异常都转 task failed，不炸 suite）
        if task.verifier:
            try:
                passed, score, metrics = self.verifiers.verify(
                    task.verifier, task, result, workspace,
                )
                result.passed = passed
                result.score = score
                result.metrics.update(metrics)
            except Exception as e:
                result.errors.append(f"Verifier '{task.verifier}' error: {e}")
                result.passed = False
        else:
            # 无 verifier → 默认通过（只要没异常 + 没工具失败）
            result.passed = len(errors) == 0
            result.score = 1.0 if result.passed else 0.0

        return result

    # ── save / load ──────────────────────────────────

    def save_results(self, results: list[BenchmarkResult], path: str) -> None:
        """保存评估结果到 JSON 文件。"""
        data = [r.to_dict() for r in results]
        Path(path).write_text(
            json.dumps(data, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    def load_results(self, path: str) -> list[BenchmarkResult]:
        """从 JSON 文件加载评估结果。"""
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return [_result_from_dict(d) for d in data]

    # ── helpers ──────────────────────────────────────

    @staticmethod
    def _copy_fixtures(src: Path, dst: Path) -> None:
        """将 fixture 目录中的文件递归复制到 workspace。

        只复制文件（跳过目录），不做覆盖检查。
        """
        for item in src.rglob("*"):
            if item.is_file():
                rel = item.relative_to(src)
                target = dst / rel
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(item, target)

    @staticmethod
    def _scan_tool_errors(messages: list[dict]) -> list[str]:
        """扫描 messages 中 tool_result 的 Error: 前缀。

        工具失败不一定进入 TaskState.errors（如工具不存在、沙箱逃逸），
        但 tool_result content 会以 "Error: ..." 开头。
        这个方法捕获那些 TaskState.errors 漏掉的错误。
        """
        errors: list[str] = []
        for msg in messages:
            content = msg.get("content", [])
            if not isinstance(content, list):
                continue
            for block in content:
                if isinstance(block, dict) and block.get("type") == "tool_result":
                    text = str(block.get("content", ""))
                    if text.startswith("Error:") or text.startswith("Blocked:"):
                        # 取第一行作为摘要
                        line = text.split("\n")[0][:200]
                        errors.append(line)
        return errors

    @staticmethod
    def _load_task_state(run_store: RunStore) -> TaskState | None:
        """从 run_store 加载 task_state（AgentLoop.run 完成后写入）。"""
        run_dir = getattr(run_store, "_run_dir", None)
        if not run_dir:
            return None
        state_path = run_dir / "task_state.json"
        if not state_path.exists():
            return None
        return TaskState.from_dict(json.loads(state_path.read_text(encoding="utf-8")))


# ── 辅助 ──────────────────────────────────────────────

def _result_from_dict(data: dict) -> BenchmarkResult:
    """从 dict 反序列化 BenchmarkResult。"""
    return BenchmarkResult(
        task_id=str(data.get("task_id", "")),
        passed=bool(data.get("passed", False)),
        score=float(data.get("score", 0.0)),
        tool_steps=int(data.get("tool_steps", 0)),
        attempts=int(data.get("attempts", 0)),
        duration_seconds=float(data.get("duration_seconds", 0.0)),
        final_answer=str(data.get("final_answer", "")),
        errors=list(data.get("errors", [])),
        metrics=dict(data.get("metrics", {})),
        trace_path=str(data.get("trace_path", "")),
    )
