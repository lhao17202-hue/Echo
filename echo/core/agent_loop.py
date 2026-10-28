"""Agent loop — perceive-decide-act-record main loop (synchronous).

The core of the Echo single-agent kernel.
"""

import time
import logging
from echo.core.task_state import TaskState
from echo.tools.base import ToolContext, ToolResult
from echo.tools.executor import ToolExecutor
from echo.providers.base import BaseLLMClient, TextBlock, ToolUseBlock
from echo.core.context_manager import ContextManager
from echo.hooks.base import HookManager, HookEvent
from echo.memory.base import MemoryManager
from echo.persistence.run_store import RunStore
from echo.persistence.checkpoint import CheckpointManager
from echo.runtime.events import RuntimeEvent, render_runtime_events

logger = logging.getLogger("echo.loop")

MODEL_MAX_TOKENS = {
    "claude-sonnet-4-6": 16000, "claude-opus-4-8": 32000,
    "gpt-4o-mini": 16000, "gpt-5.4": 16000,
    "deepseek-v4-pro": 16000, "qwen3:4b": 8000,
    "FakeLLMClient": 8000, "default": 8000,
}


class AgentLoop:
    """主循环（全同步）。

    每轮：
      1. COMPACT:  上下文压缩（被动 + 主动）
      2. PERCEIVE: 组装 system prompt
      3. DECIDE:   调用 LLM
      4. ACT:      遍历 tool_use → PermissionHook → 执行 → PostHook
      5. RECORD:   持久化 + 检查点 + 记忆更新
      6. 终止判断
    """

    def __init__(self, llm: BaseLLMClient, memory: MemoryManager,
                 tools: ToolExecutor, hooks: HookManager,
                 context: ContextManager, sandbox, shell,
                 session_store, run_store: RunStore,
                 max_steps: int = 25, max_retries: int = 3,
                 max_attempts: int | None = None,
                 approval_policy: str = "ask", approval_handler=None,
                 message_bus=None, teammate_manager=None, global_tasks=None,
                 llm_lock=None, skill_registry=None, scheduler=None,
                 background_manager=None, protocol_manager=None, mcp_manager=None,
                 max_tokens: int = 8000):
        self.llm = llm
        self.memory = memory
        self.tools = tools
        self.hooks = hooks
        self.context = context
        self.sandbox = sandbox
        self.shell = shell
        self.session_store = session_store
        self.run_store = run_store
        self.checkpoints = CheckpointManager(str(sandbox.root))
        self.max_steps = max(0, int(max_steps))
        self.max_retries = max(0, int(max_retries))
        self.max_attempts = (
            max(0, int(max_attempts))
            if max_attempts is not None
            else max(self.max_steps * 2, self.max_steps + 5)
        )
        self.max_tokens = max(1, int(max_tokens))
        self.approval_policy = approval_policy   # "ask" | "auto" | "never" | "danger"
        self.approval_handler = approval_handler
        self.message_bus = message_bus
        self.teammate_manager = teammate_manager
        self.global_tasks = global_tasks
        self.skill_registry = skill_registry
        self.scheduler = scheduler
        self.background_manager = background_manager
        self.protocol_manager = protocol_manager
        self.mcp_manager = mcp_manager
        self._llm_lock = llm_lock  # shared lock for lead+teammate llm.chat() serialisation
        if self._llm_lock is not None and getattr(self.context, "_llm_lock", None) is None:
            self.context._llm_lock = self._llm_lock

        self.messages: list[dict] = []
        self._tracked_files: list[str] = []
        self._compact_requested: bool = False
        known_model_max = MODEL_MAX_TOKENS.get(llm.model)
        if known_model_max is None:
            # For a custom/unknown model, the configured value is the only
            # reliable capability signal available to the loop.
            self._model_max = max(self.max_tokens, MODEL_MAX_TOKENS["default"])
        else:
            self._model_max = known_model_max
            self.max_tokens = min(self.max_tokens, self._model_max)
        self._last_llm_failure = ""
        self._last_llm_error = ""
        self._last_partial_response = None

    @staticmethod
    def _summarize_value(value, limit: int = 300) -> str:
        text = str(value)
        text = text.replace("\n", " ").strip()
        return text[:limit]

    @staticmethod
    def _output_summary(result, limit: int = 300) -> str:
        output = getattr(result, "output", "") or ""
        return str(output).replace("\n", " ").strip()[:limit]

    @staticmethod
    def _error_preview(result, limit: int = 300) -> str:
        error = getattr(result, "error", "") or ""
        return str(error).replace("\n", " ").strip()[:limit]

    # ── Public ─────────────────────────────────────

    def run(self, user_request: str, resume_messages: list[dict] | None = None) -> str:
        state = TaskState.create(user_request)
        self._last_state = state
        state.resume_status = getattr(self, "resume_status", "")
        self._tracked_files = []
        self._compact_requested = False
        self._last_llm_failure = ""
        self._last_llm_error = ""
        self._last_partial_response = None

        # 恢复 todos（跨 session）
        if hasattr(self, "_resume_todos") and self._resume_todos:
            state.todos = list(self._resume_todos)

        self.run_store.start_run(state)
        self.run_store.log("run_started", run_id=state.run_id, request=user_request[:300])

        # resume_messages 含历史 + checkpoint 注入，新请求由这里追加
        self.messages = list(resume_messages) if resume_messages else []
        # 统一在这里追加当前用户请求（TextBlock 格式确保进入模型上下文）
        self.memory.observe_user_message(user_request)
        self.messages.append({
            "role": "user", "content": [TextBlock(text=user_request)],
        })
        # A user prompt is a run-level event, not an event for every model turn.
        self.hooks.trigger(HookEvent.USER_PROMPT, request=user_request)

        while (
            state.is_running
            and state.tool_steps < self.max_steps
            and state.attempts < self.max_attempts
        ):
            # 0. RUNTIME EVENTS — inject external/runtime messages + sync snapshots
            runtime_events = self._ingest_runtime_events(state)
            if runtime_events:
                self.messages.append({
                    "role": "user",
                    "content": [TextBlock(text=render_runtime_events(runtime_events))],
                })

            # 1. COMPACT — 每轮 LLM 调用前压缩上下文
            self.messages = self.context.compact(self.messages, self.llm)

            # 2. PERCEIVE — 组装 system prompt + 工作记忆
            system = self.context.build_system(
                state, self.tools.registry, self.memory, self.sandbox,
                skill_registry=self.skill_registry,
                global_tasks=self.global_tasks,
                background_manager=self.background_manager,
                protocol_manager=self.protocol_manager,
                mcp_manager=self.mcp_manager,
            )
            # 3. DECIDE
            self.run_store.log("model_requested", run_id=state.run_id,
                               attempts=state.attempts, tool_steps=state.tool_steps,
                               message_count=len(self.messages),
                               max_tokens=self.max_tokens)

            response, self.messages = self._call_llm_with_retry(
                self.messages, self.tools.registry.list_schemas(), system,
                run_id=state.run_id,
            )
            # 同步 compact_count（passive/reactive/active 都覆盖）
            state.compact_count = self.context.compact_count
            # 记录模型返回的元信息
            if response:
                self.run_store.log("model_response", run_id=state.run_id,
                                   stop_reason=response.stop_reason,
                                   model=response.model,
                                   input_tokens=response.usage.input_tokens if response.usage else 0,
                                   output_tokens=response.usage.output_tokens if response.usage else 0)

            if response is None:
                self._stop_after_llm_failure(state)
                break

            state.record_attempt()
            self.messages.append({"role": "assistant", "content": response.content})

            # 4. ACT
            tool_blocks = [b for b in response.content if isinstance(b, ToolUseBlock)]

            if not tool_blocks:
                texts = [b.text for b in response.content if isinstance(b, TextBlock)]
                final_text = " ".join(texts).strip()
                if final_text:
                    state.finish_success(final_text)
                else:
                    state.stop_empty_response()
                break

            ctx = ToolContext(
                workspace_root=str(self.sandbox.root),
                sandbox=self.sandbox, shell=self.shell, memory=self.memory,
                task_state=state,
                llm=self.llm,
                tool_registry=self.tools.registry,
                skill_registry=self.skill_registry,
                message_bus=self.message_bus,
                teammate_manager=self.teammate_manager,
                global_tasks=self.global_tasks,
                background_manager=self.background_manager,
                agent_name="lead",
                run_id=state.run_id,
                trace_logger=self.run_store,
                depth=0, max_depth=1,
                unrestricted_paths=self.approval_policy == "danger",
            )

            tool_results = []
            for block in tool_blocks:
                tool = self.tools.registry.get(block.name)
                tool_call_id = getattr(block, "id", "")
                input_summary = self._summarize_value(block.input)
                budget_exhausted = state.tool_steps >= self.max_steps
                deny = None

                # One model response may contain several tool calls. Enforce the
                # budget per call so a batch cannot overshoot max_steps.
                if budget_exhausted:
                    result = ToolResult.fail(
                        f"Skipped: tool step limit ({self.max_steps}) reached"
                    )
                    self.run_store.log("tool_skipped", run_id=state.run_id, payload={
                        "tool": block.name,
                        "tool_call_id": tool_call_id,
                        "input_summary": input_summary,
                        "success": False,
                        "reason": "step_limit_reached",
                    })
                else:
                    # 权限检查
                    deny = self.hooks.trigger(
                        HookEvent.PRE_TOOL_USE,
                        tool=tool,
                        tool_input=block.input,
                        approval_policy=self.approval_policy,
                        approval_handler=self.approval_handler,
                    )

                if not budget_exhausted and deny:
                    result = ToolResult.fail(f"Blocked: {deny}")
                    self.run_store.log("tool_blocked", run_id=state.run_id, payload={
                        "tool": block.name,
                        "tool_call_id": tool_call_id,
                        "input_summary": input_summary,
                        "success": False,
                        "approval_policy": self.approval_policy,
                        "reason": str(deny),
                    })
                elif not budget_exhausted:
                    self.run_store.log("tool_started", run_id=state.run_id, payload={
                        "tool": block.name,
                        "tool_call_id": tool_call_id,
                        "input_summary": input_summary,
                    })
                    started_at = time.perf_counter()
                    result = self.tools.execute(block.name, block.input, ctx)
                    duration_ms = int((time.perf_counter() - started_at) * 1000)
                    self.hooks.trigger(HookEvent.POST_TOOL_USE, tool=tool, result=result)
                    if result.success:
                        self.run_store.log("tool_finished", run_id=state.run_id, payload={
                            "tool": block.name,
                            "tool_call_id": tool_call_id,
                            "input_summary": input_summary,
                            "success": True,
                            "output_summary": self._output_summary(result),
                            "duration_ms": duration_ms,
                            "files_touched": result.files_touched,
                            "files_read": result.files_read,
                            "files_written": result.files_written,
                            "files_deleted": result.files_deleted,
                        })
                    else:
                        self.run_store.log("tool_failed", run_id=state.run_id, payload={
                            "tool": block.name,
                            "tool_call_id": tool_call_id,
                            "input_summary": input_summary,
                            "success": False,
                            "error_preview": self._error_preview(result),
                            "output_summary": self._output_summary(result),
                            "duration_ms": duration_ms,
                            "files_touched": result.files_touched,
                            "files_read": result.files_read,
                            "files_written": result.files_written,
                            "files_deleted": result.files_deleted,
                        })

                    # 只有成功执行了才计入 step budget
                    # Hook 拦截或工具内部失败（沙箱逃逸等）不计步
                    if result.success:
                        state.record_tool(block.name)

                        for note in result.memory_notes:
                            self.memory.add(note, {"tags": [block.name], "source": block.name})
                        for fpath in result.files_touched:
                            self._tracked_files.append(fpath)

                        # 记忆观察
                        self.memory.observe_tool_result(block.name, block.input, result, ctx)

                        # compact 工具
                        if block.name == "compact":
                            self._compact_requested = True

                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "tool_name": block.name,
                    "tool_input": block.input,   # dedupe/compact 用
                    "content": result.output if not result.error
                               else f"Error: {result.error}\n{result.output}",
                })

            # 5. RECORD
            # role="user" 是 Anthropic 原生格式（tool_result 在 user 消息中）。
            # OpenAI/Ollama adapter 分别在自己的 _build_input/_build_messages 中
            # 检测 content 里的 tool_result 块并转为对应格式。
            self.messages.append({"role": "user", "content": tool_results})

            # 主动 compact: 模型调用了 compact 工具 → 触发 force_compact
            if self._compact_requested:
                self.messages = self.context.force_compact(
                    self.messages, self.llm, reason="tool_requested"
                )
                state.compact_count = self.context.compact_count  # 同步
                self.run_store.log("compaction_triggered", run_id=state.run_id,
                                   trigger="tool_requested",
                                   compact_count=state.compact_count)
                self._compact_requested = False

            self._sync_multi_agent_state(state)
            ckpt = self.checkpoints.create(
                state,
                recent_files=self._tracked_files,
                snapshot_teammates=state.active_teammates,
            )
            state.checkpoint_id = ckpt.checkpoint_id
            self._sync_session(state, ckpt)
            self.run_store.update_state(state)
            self.run_store.log("tool_executed", run_id=state.run_id,
                               step=state.tool_steps,
                               tools=[{"name": b.name, "input_summary": str(b.input)[:200]}
                                      for b in tool_blocks],
                               file_changes=self._tracked_files[-10:])

        if state.is_running and state.attempts >= self.max_attempts:
            state.stop_attempt_limit()
        elif state.is_running:
            state.stop_step_limit()

        # Persist and announce every terminal path exactly once, including limits
        # and model failures.
        self._sync_session(state)
        self.run_store.update_state(state)
        self.run_store.write_report(state)
        self.run_store.log(
            "run_finished",
            run_id=state.run_id,
            status=state.status.value,
            stop_reason=state.stop_reason,
            attempts=state.attempts,
            tool_steps=state.tool_steps,
        )
        self.hooks.trigger(HookEvent.RUN_STOP, state=state)

        return state.final_answer or f"Stopped: {state.stop_reason}"

    # ── LLM with retry ─────────────────────────────

    def _ingest_runtime_events(self, state: TaskState) -> list[RuntimeEvent]:
        """Collect runtime events from external subsystems and refresh state snapshots."""
        events: list[RuntimeEvent] = []
        events.extend(self._collect_teammate_events(state))
        events.extend(self._collect_cron_events(state))
        events.extend(self._collect_background_events(state))
        events.extend(self._collect_protocol_events(state))
        self._sync_multi_agent_state(state)
        return events

    def _collect_cron_events(self, state: TaskState) -> list[RuntimeEvent]:
        if not self.scheduler:
            return []
        try:
            jobs = self.scheduler.consume()
        except Exception as exc:
            self.run_store.log("cron_consume_failed", run_id=state.run_id, error=str(exc)[:300])
            return [RuntimeEvent(
                source="cron",
                event_type="error",
                content=f"Cron scheduler failed while collecting due jobs: {exc}",
            )]

        events = []
        for job in jobs:
            event = RuntimeEvent.from_cron_job(job)
            events.append(event)
            self.run_store.log(
                "cron_fired",
                run_id=state.run_id,
                job_id=event.metadata.get("job_id", ""),
                prompt_preview=event.content[:200],
            )
        return events

    def _collect_background_events(self, state: TaskState) -> list[RuntimeEvent]:
        if not self.background_manager:
            return []
        try:
            events = self.background_manager.poll_completed()
        except Exception as exc:
            self.run_store.log("background_poll_failed", run_id=state.run_id, error=str(exc)[:300])
            return [RuntimeEvent(
                source="background",
                event_type="error",
                content=f"Background manager failed while collecting completed tasks: {exc}",
            )]

        for event in events:
            bg_id = event.metadata.get("bg_id", "")
            if bg_id:
                state.remove_background_task(bg_id)
            self.run_store.log(
                "background_completed" if event.event_type == "completed" else "background_failed",
                run_id=state.run_id,
                bg_id=bg_id,
                background_event_type=event.event_type,
            )
        return events

    def _collect_protocol_events(self, state: TaskState) -> list[RuntimeEvent]:
        if not self.protocol_manager:
            return []
        try:
            events = self.protocol_manager.poll_events()
        except Exception as exc:
            self.run_store.log("protocol_poll_failed", run_id=state.run_id, error=str(exc)[:300])
            return [RuntimeEvent(
                source="protocol",
                event_type="error",
                content=f"Protocol manager failed while collecting events: {exc}",
            )]

        for event in events:
            request_id = event.metadata.get("request_id", "")
            if request_id in state.pending_protocols:
                state.pending_protocols.remove(request_id)
            self.run_store.log(
                "protocol_resolved",
                run_id=state.run_id,
                request_id=request_id,
                protocol_type=event.metadata.get("protocol_type", ""),
            )
        return events

    def _collect_teammate_events(self, state: TaskState) -> list[RuntimeEvent]:
        if not self.message_bus:
            return []
        messages = self.message_bus.receive("lead")
        if not messages:
            state.unprocessed_messages = []
            return []

        events: list[RuntimeEvent] = []
        for msg in messages:
            events.append(RuntimeEvent(
                source="teammate",
                event_type=msg.msg_type,
                content=msg.content,
                metadata={"from": msg.from_agent, "to": msg.to_agent},
            ))
            self.run_store.log(
                "message_received",
                run_id=state.run_id,
                from_agent=msg.from_agent,
                to_agent=msg.to_agent,
                msg_type=msg.msg_type,
            )
        state.unprocessed_messages = []
        return events

    def _inject_inbox_messages(self, state: TaskState) -> None:
        events = self._collect_teammate_events(state)
        if events:
            lines = ["## Teammate Messages"]
            for event in events:
                lines.append(
                    f"- From {event.metadata.get('from', '')} "
                    f"[{event.event_type}]: {event.content}"
                )
            self.messages.append({"role": "user", "content": [TextBlock(text="\n".join(lines))]})

    def _sync_multi_agent_state(self, state: TaskState) -> None:
        if self.teammate_manager:
            state.active_teammates = self.teammate_manager.snapshot()
        if self.global_tasks:
            state.global_task_ids = [task.task_id for task in self.global_tasks.list_all()]

    def _call_llm_with_retry(self, messages, tools, system, retries=0,
                              run_id: str = ""):
        """Return (response_or_None, messages). messages may be compacted on retry."""
        current_messages = messages
        retry_count = retries

        while True:
            try:
                if self._llm_lock:
                    with self._llm_lock:
                        resp = self.llm.chat(
                            current_messages, tools, system,
                            max_tokens=self.max_tokens,
                        )
                else:
                    resp = self.llm.chat(
                        current_messages, tools, system,
                        max_tokens=self.max_tokens,
                    )
            except Exception as exc:
                self._last_llm_error = str(exc)
                is_rate_limit = "429" in str(exc) or "529" in str(exc)
                is_context_overflow = self._is_prompt_too_long(exc)

                if retry_count >= self.max_retries:
                    self._last_llm_failure = (
                        "context_overflow" if is_context_overflow
                        else "retry_limit" if is_rate_limit
                        else "model_error"
                    )
                    logger.error("LLM error after retries: %s", exc)
                    return None, current_messages

                if is_rate_limit:
                    time.sleep(2 ** retry_count)
                    retry_count += 1
                    continue

                if is_context_overflow:
                    logger.warning("Prompt too long, triggering reactive compact")
                    compacted = self.context.reactive_compact(
                        current_messages, self.llm
                    )
                    self.run_store.log(
                        "compaction_triggered",
                        run_id=run_id,
                        trigger="prompt_too_long",
                        compact_count=self.context.compact_count,
                    )
                    current_messages = compacted
                    retry_count += 1
                    continue

                self._last_llm_failure = "model_error"
                logger.error("LLM error: %s", exc)
                return None, current_messages

            if resp.stop_reason == "max_tokens":
                if self.max_tokens < self._model_max:
                    # Retry the same request with a larger output allowance. Do
                    # not append a bare "Continue" without the truncated reply.
                    self.max_tokens = min(self.max_tokens * 2, self._model_max)
                    continue
                self._last_llm_failure = "max_tokens"
                self._last_partial_response = resp
                return None, current_messages

            return resp, current_messages

    def _stop_after_llm_failure(self, state: TaskState) -> None:
        """Map the last model-call failure to a precise terminal state."""
        if self._last_llm_failure == "retry_limit":
            state.stop_retry_limit()
            if self._last_llm_error:
                state.errors.append(self._last_llm_error)
            return
        if self._last_llm_failure == "context_overflow":
            state.stop_context_overflow(self._last_llm_error)
            return
        if self._last_llm_failure == "max_tokens":
            partial = ""
            if self._last_partial_response is not None:
                partial = " ".join(
                    block.text
                    for block in self._last_partial_response.content
                    if isinstance(block, TextBlock)
                ).strip()
            message = "Model output reached the maximum token limit."
            if partial:
                message += f"\n\nPartial response:\n{partial}"
            state.stop_max_tokens(message)
            return
        state.stop_model_error(
            self._last_llm_error or "LLM returned None after retries"
        )

    @staticmethod
    def _messages_to_json(messages: list[dict]) -> list[dict]:
        """Convert internal messages (TextBlock/ToolUseBlock) to JSON-serializable dicts."""
        result = []
        for msg in messages:
            content = []
            for block in (msg.get("content") or []):
                if isinstance(block, TextBlock):
                    content.append({"type": "text", "text": block.text})
                elif isinstance(block, ToolUseBlock):
                    content.append({
                        "type": "tool_use",
                        "id": block.id,
                        "name": block.name,
                        "input": block.input,
                    })
                elif isinstance(block, dict):
                    content.append(block)
            result.append({"role": msg.get("role", "user"), "content": content})
        return result

    def _sync_session(self, state, ckpt=None) -> None:
        """同步 session：history + memory + todos + checkpoint（立即落盘）。"""
        if not hasattr(self, '_session') or self._session is None:
            return
        mem = self.memory.to_dict()
        mem["todos"] = list(state.todos)
        self._session.history = self._messages_to_json(self.messages)
        self._session.short_term_memory = mem
        self._session.checkpoints["current_id"] = state.checkpoint_id
        if ckpt:
            self.checkpoints.save_to_session(ckpt, self._session)
        self.session_store.save(self._session)

    @staticmethod
    def _is_prompt_too_long(error: Exception) -> bool:
        msg = str(error).lower()
        return any(kw in msg for kw in (
            "prompt too long", "context length", "token limit",
            "maximum context", "too many tokens", "context_length_exceeded",
        ))
