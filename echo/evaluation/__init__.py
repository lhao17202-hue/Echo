"""Evaluation module — benchmark harness, metrics collection, comparison.

Status: 预留接口。Evaluator / MetricsCollector / FakeLLMClient 的核心
方法签名已定义，具体实现待后续完成。

Evaluation helpers for Echo benchmarks.

使用示例：
  from echo.evaluation import BenchmarkTask, BenchmarkResult, BenchmarkSuite
  from echo.evaluation import Evaluator, MetricsCollector
"""

from echo.evaluation.benchmark import (
    BenchmarkTask, BenchmarkResult, BenchmarkSuite,
)
from echo.evaluation.evaluator import (
    Evaluator, FakeLLMClient, VerifierRegistry,
)
from echo.evaluation.metrics import (
    RunRecord, RunComparison, MetricsCollector,
)

__all__ = [
    "BenchmarkTask", "BenchmarkResult", "BenchmarkSuite",
    "Evaluator", "FakeLLMClient", "VerifierRegistry",
    "RunRecord", "RunComparison", "MetricsCollector",
]
