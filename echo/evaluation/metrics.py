"""
评估指标收集与比较 —— MetricsCollector, RunComparison。

Evaluation metrics and run comparison utilities.
目前为预留接口，待后续实现完整的指标聚合和对比功能。

预期使用方式：
  collector = MetricsCollector()
  collector.add_run("baseline", results)
  collector.add_run("experiment", results2)
  comparison = collector.compare("baseline", "experiment")
"""

from dataclasses import dataclass, field
from typing import Any
from echo.evaluation.benchmark import BenchmarkResult


# ═══════════════════════════════════════════════════════
# RunRecord — 一次完整测试运行
# ═══════════════════════════════════════════════════════

@dataclass
class RunRecord:
    """一次完整测试运行 —— 包含所有 task 的结果和元数据。

    Attributes:
        run_id: 运行 ID（用于区分 baseline / experiment）。
        model: 使用的模型名称。
        provider: 使用的 provider。
        results: BenchmarkResult 列表。
        total_tasks: 任务总数。
        passed_count: 通过数量。
        total_steps: 总工具步数。
        total_duration: 总耗时（秒）。
        metadata: 额外元数据（温度、top_p、timezone 等）。
    """

    run_id: str = ""
    model: str = ""
    provider: str = ""
    results: list[BenchmarkResult] = field(default_factory=list)
    total_tasks: int = 0
    passed_count: int = 0
    total_steps: int = 0
    total_duration: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def pass_rate(self) -> float:
        """通过率：0.0 ~ 1.0。"""
        if self.total_tasks == 0:
            return 0.0
        return self.passed_count / self.total_tasks

    @property
    def avg_steps(self) -> float:
        """平均工具步数。"""
        if self.total_tasks == 0:
            return 0.0
        return self.total_steps / self.total_tasks

    @classmethod
    def from_results(cls, run_id: str, results: list[BenchmarkResult],
                     model: str = "", provider: str = "",
                     **metadata) -> "RunRecord":
        """从 BenchmarkResult 列表构建 RunRecord。"""
        return cls(
            run_id=run_id,
            model=model,
            provider=provider,
            results=results,
            total_tasks=len(results),
            passed_count=sum(1 for r in results if r.passed),
            total_steps=sum(r.tool_steps for r in results),
            total_duration=sum(r.duration_seconds for r in results),
            metadata=metadata,
        )


# ═══════════════════════════════════════════════════════
# RunComparison — 两次运行的对比结果
# ═══════════════════════════════════════════════════════

@dataclass
class RunComparison:
    """两次运行的对比结果。

    Attributes:
        baseline_id: 基准运行 ID。
        experiment_id: 实验运行 ID。
        pass_rate_delta: 通过率变化（正值 = 改善）。
        avg_steps_delta: 平均步数变化（负值 = 改善）。
        passed_diff: 新通过的 task ID 列表。
        failed_diff: 新失败的 task ID 列表。
        per_task: 每个 task 的对比详情 dict 列表。
    """

    baseline_id: str = ""
    experiment_id: str = ""
    pass_rate_delta: float = 0.0
    avg_steps_delta: float = 0.0
    passed_diff: list[str] = field(default_factory=list)
    failed_diff: list[str] = field(default_factory=list)
    per_task: list[dict] = field(default_factory=list)


# ═══════════════════════════════════════════════════════
# MetricsCollector — 指标收集器
# ═══════════════════════════════════════════════════════

class MetricsCollector:
    """指标收集器 —— 管理多次运行并进行对比。

    用于管理多次评估运行并比较结果。

    使用方式：
      collector = MetricsCollector()
      collector.add("v1", results)
      collector.add("v2", results2)
      comp = collector.compare("v1", "v2")
    """

    def __init__(self):
        self._runs: dict[str, RunRecord] = {}

    def add(self, run_id: str, results: list[BenchmarkResult],
            model: str = "", provider: str = "", **metadata) -> RunRecord:
        """添加一次运行记录。

        Args:
            run_id: 运行标识。
            results: BenchmarkResult 列表。
            model: 模型名称。
            provider: Provider 名称。
            **metadata: 额外元数据。

        Returns:
            构建的 RunRecord 实例。
        """
        record = RunRecord.from_results(run_id, results, model, provider, **metadata)
        self._runs[run_id] = record
        return record

    def get(self, run_id: str) -> RunRecord | None:
        """获取某次运行的记录。"""
        return self._runs.get(run_id)

    def compare(self, baseline_id: str, experiment_id: str) -> RunComparison:
        """比较两次运行（baseline vs experiment）。

        Args:
            baseline_id: 基准运行 ID。
            experiment_id: 实验运行 ID。

        Returns:
            RunComparison 实例，包含 pass_rate_delta、avg_steps_delta、
            新通过/新失败 task、per-task 详情。

        Raises:
            KeyError: 如果 baseline_id 或 experiment_id 不存在。
        """
        baseline = self._runs[baseline_id]
        experiment = self._runs[experiment_id]

        # 构建 per-task 对比
        baseline_map: dict[str, BenchmarkResult] = {}
        for r in baseline.results:
            baseline_map[r.task_id] = r
        experiment_map: dict[str, BenchmarkResult] = {}
        for r in experiment.results:
            experiment_map[r.task_id] = r

        all_task_ids = sorted(set(baseline_map.keys()) | set(experiment_map.keys()))

        per_task = []
        passed_diff: list[str] = []
        failed_diff: list[str] = []

        for task_id in all_task_ids:
            b = baseline_map.get(task_id)
            e = experiment_map.get(task_id)
            b_passed = b.passed if b else False
            e_passed = e.passed if e else False
            if b_passed and not e_passed:
                failed_diff.append(task_id)
            elif not b_passed and e_passed:
                passed_diff.append(task_id)

            per_task.append({
                "task_id": task_id,
                "baseline_passed": b_passed,
                "experiment_passed": e_passed,
                "baseline_steps": b.tool_steps if b else 0,
                "experiment_steps": e.tool_steps if e else 0,
                "baseline_score": b.score if b else 0.0,
                "experiment_score": e.score if e else 0.0,
                "status": (
                    "same" if b_passed == e_passed
                    else "improved" if e_passed and not b_passed
                    else "regressed"
                ),
            })

        pass_rate_delta = round(experiment.pass_rate - baseline.pass_rate, 4)
        avg_steps_delta = round(experiment.avg_steps - baseline.avg_steps, 2)

        return RunComparison(
            baseline_id=baseline_id,
            experiment_id=experiment_id,
            pass_rate_delta=pass_rate_delta,
            avg_steps_delta=avg_steps_delta,
            passed_diff=passed_diff,
            failed_diff=failed_diff,
            per_task=per_task,
        )

    def list_runs(self) -> list[str]:
        """列出所有运行 ID。"""
        return list(self._runs.keys())

    def summary(self, run_id: str) -> dict:
        """生成单次运行的摘要。

        Args:
            run_id: 运行 ID。

        Returns:
            包含 pass_rate、avg_steps 等指标的 dict。
        """
        record = self._runs.get(run_id)
        if record is None:
            return {}
        return {
            "run_id": record.run_id, "model": record.model, "provider": record.provider,
            "total_tasks": record.total_tasks, "passed_count": record.passed_count,
            "pass_rate": record.pass_rate, "avg_steps": record.avg_steps,
            "total_duration": record.total_duration, "metadata": record.metadata,
        }
