"""Cron scheduler — CronJob data model + CronScheduler."""

import threading
import time
import logging
from dataclasses import dataclass, field
import uuid

logger = logging.getLogger("echo.cron")


@dataclass
class CronJob:
    """定时任务。"""
    job_id: str = field(default_factory=lambda: uuid.uuid4().hex[:8])
    cron_expr: str = ""                # 5 字段 cron: "min hour dom month dow"
    prompt: str = ""
    recurring: bool = True
    durable: bool = True
    last_fire_timestamp: float = 0.0


class CronScheduler:
    """Cron 定时调度器。daemon 线程轮询。

    防重复：用时间戳判断时间差，不用字符串标记。
    时间跳变：检测回拨 >60s 则跳过本轮。
    """

    def __init__(self):
        self._jobs: dict[str, CronJob] = {}
        self._lock = threading.Lock()
        self._queue: list[CronJob] = []  # 待消费队列
        self._running = False
        self._thread: threading.Thread | None = None

    def start(self):
        """启动后台轮询线程。"""
        self._running = True
        self._thread = threading.Thread(target=self._poll_loop, daemon=True)
        self._thread.start()

    def stop(self):
        self._running = False

    def schedule(self, cron_expr: str, prompt: str,
                 recurring: bool = True, durable: bool = True) -> str:
        """添加定时任务，返回 job_id。"""
        job = CronJob(cron_expr=cron_expr, prompt=prompt,
                      recurring=recurring, durable=durable)
        with self._lock:
            self._jobs[job.job_id] = job
        return job.job_id

    def cancel(self, job_id: str) -> bool:
        with self._lock:
            if job_id in self._jobs:
                del self._jobs[job_id]
                return True
            return False

    def consume(self) -> list[CronJob]:
        """获取到期的 job（主循环每轮调用）。非阻塞。"""
        with self._lock:
            ready = list(self._queue)
            self._queue.clear()
        return ready

    # ── Internal ───────────────────────────────────

    def _poll_loop(self):
        """后台轮询（每秒检查一次）。"""
        while self._running:
            self._check_and_fire()
            time.sleep(1)

    def _check_and_fire(self):
        now = time.time()
        with self._lock:
            for job in list(self._jobs.values()):
                last = job.last_fire_timestamp
                if last == 0:
                    job.last_fire_timestamp = now
                    continue

                interval = self._next_interval(job)
                if interval <= 0:
                    continue

                if now - last >= interval:
                    # 时间跳变检测
                    if now < last - 60:
                        logger.warning(f"Time jumped backward, skipping {job.job_id}")
                        job.last_fire_timestamp = now
                        continue

                    self._queue.append(job)
                    job.last_fire_timestamp = now

                    if not job.recurring:
                        del self._jobs[job.job_id]

    @staticmethod
    def _next_interval(job: CronJob) -> float:
        """解析 cron 表达式，返回下次触发的秒数。

        简化实现：支持 5 字段，* 和 */N 语法。
        """
        try:
            parts = job.cron_expr.strip().split()
            if len(parts) != 5:
                return 600  # 默认 10 分钟，避免错误配置
            minute_part = parts[0]
            if minute_part == "*":
                return 60
            elif minute_part.startswith("*/"):
                try:
                    return int(minute_part[2:]) * 60
                except ValueError:
                    return 60
            return 60  # 默认每分钟检查
        except Exception:
            return 600
