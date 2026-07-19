"""
标杆测试数据模型 —— BenchmarkTask, BenchmarkResult, BenchmarkSuite。

Echo benchmark task and result schema.
目前为预留接口，字段和验证逻辑已定义，待后续实现完整的评估流水线。

预期使用流程：
  1. 加载 benchmark JSON → BenchmarkSuite
  2. 对每个 BenchmarkTask 创建 Echo Agent 实例
  3. 运行 task → 收集结果 → BenchmarkResult
  4. MetricsCollector 聚合多个 result 生成报告
"""

from dataclasses import dataclass, field
from typing import Any


# ═══════════════════════════════════════════════════════
# BenchmarkTask — 单个标杆测试任务
# ═══════════════════════════════════════════════════════

@dataclass
class BenchmarkTask:
    """单个标杆测试任务。

    包含必需任务字段和可选控制字段。

    Attributes:
        id: 任务唯一标识（如 "readme_intro_locked"）。
        prompt: 发给 Agent 的测试指令。
        fixture_repo: 测试用的 git repo 路径（将复制到临时目录）。
        allowed_tools: 允许使用的工具列表（如 ["read_file", "write_file"]）。
        step_budget: 最大工具调用步数。
        expected_artifact: 期望产出的文件路径。
        verifier: 验证器名称（决定如何判断任务是否通过）。
        category: 任务分类（如 "file_ops", "safety", "context"）。
        description: 任务描述（可选，用于报告）。
        setup: 前置场景配置 dict（可选）。
            kind: "context_reduction" | "freshness_mismatch" | "workspace_mismatch"
        model_outputs: 预设的模型回复列表（可选，用于 FakeLLMClient 确定性测试）。
        tags: 标签列表（可选，用于过滤）。
    """

    id: str = ""
    prompt: str = ""
    fixture_repo: str = ""
    allowed_tools: list[str] = field(default_factory=list)
    step_budget: int = 25
    expected_artifact: str = ""
    verifier: str = ""              # "file_equals" | "file_contains" | "output_matches"
    category: str = ""              # "file_ops" | "safety" | "context" | "recovery" | "multi_agent"
    description: str = ""
    setup: dict[str, Any] = field(default_factory=dict)
    model_outputs: list[str] = field(default_factory=list)
    verifier_params: dict[str, Any] = field(default_factory=dict)
    tags: list[str] = field(default_factory=list)

    @classmethod
    def from_dict(cls, data: dict) -> "BenchmarkTask":
        """从字典反序列化。"""
        return cls(
            id=str(data.get("id", "")),
            prompt=str(data.get("prompt", "")),
            fixture_repo=str(data.get("fixture_repo", "")),
            allowed_tools=list(data.get("allowed_tools", [])),
            step_budget=int(data.get("step_budget", 25)),
            expected_artifact=str(data.get("expected_artifact", "")),
            verifier=str(data.get("verifier", "")),
            category=str(data.get("category", "")),
            description=str(data.get("description", "")),
            setup=dict(data.get("setup", {})),
            model_outputs=list(data.get("model_outputs", [])),
            verifier_params=dict(data.get("verifier_params", {})),
            tags=list(data.get("tags", [])),
        )

    def to_dict(self) -> dict:
        return {
            "id": self.id, "prompt": self.prompt,
            "fixture_repo": self.fixture_repo, "allowed_tools": self.allowed_tools,
            "step_budget": self.step_budget, "expected_artifact": self.expected_artifact,
            "verifier": self.verifier, "category": self.category,
            "description": self.description, "setup": self.setup,
            "model_outputs": self.model_outputs,
            "verifier_params": self.verifier_params,
            "tags": self.tags,
        }


# ═══════════════════════════════════════════════════════
# BenchmarkResult — 单个任务的执行结果
# ═══════════════════════════════════════════════════════

@dataclass
class BenchmarkResult:
    """单个标杆任务的执行结果。

    Attributes:
        task_id: 对应 BenchmarkTask.id。
        passed: 是否通过测试。
        score: 得分（0.0 ~ 1.0）。
        tool_steps: 实际执行的工具步数。
        attempts: 模型调用次数。
        duration_seconds: 执行耗时。
        final_answer: Agent 的最终回复（截断到 500 字符）。
        errors: 错误信息列表。
        metrics: 额外指标 dict（由 verifier 填充）。
        trace_path: trace.jsonl 路径（用于事后分析）。
    """

    task_id: str = ""
    passed: bool = False
    score: float = 0.0
    tool_steps: int = 0
    attempts: int = 0
    duration_seconds: float = 0.0
    final_answer: str = ""
    errors: list[str] = field(default_factory=list)
    metrics: dict[str, Any] = field(default_factory=dict)
    trace_path: str = ""

    def to_dict(self) -> dict:
        return {
            "task_id": self.task_id, "passed": self.passed, "score": self.score,
            "tool_steps": self.tool_steps, "attempts": self.attempts,
            "duration_seconds": self.duration_seconds,
            "final_answer": self.final_answer[:500],
            "errors": self.errors, "metrics": self.metrics,
            "trace_path": self.trace_path,
        }


# ═══════════════════════════════════════════════════════
# BenchmarkSuite — 一组标杆测试任务
# ═══════════════════════════════════════════════════════

@dataclass
class BenchmarkSuite:
    """一组标杆测试任务。

    Echo benchmark suite schema。

    Attributes:
        schema_version: schema 版本号。
        tasks: 测试任务列表。
        name: 测试套件名称。
        description: 测试套件描述。
    """

    schema_version: int = 1
    tasks: list[BenchmarkTask] = field(default_factory=list)
    name: str = ""
    description: str = ""

    @classmethod
    def from_dict(cls, data: dict) -> "BenchmarkSuite":
        return cls(
            schema_version=int(data.get("schema_version", 1)),
            tasks=[BenchmarkTask.from_dict(t) for t in data.get("tasks", [])],
            name=str(data.get("name", "")),
            description=str(data.get("description", "")),
        )

    @classmethod
    def from_json(cls, path: str) -> "BenchmarkSuite":
        """从 JSON 文件加载测试套件。"""
        import json
        with open(path, encoding="utf-8") as f:
            return cls.from_dict(json.load(f))

    def to_dict(self) -> dict:
        return {
            "schema_version": self.schema_version,
            "name": self.name, "description": self.description,
            "tasks": [t.to_dict() for t in self.tasks],
        }

    def __len__(self) -> int:
        return len(self.tasks)
