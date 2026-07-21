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
                 approval_policy: str = "ask",
                 message_bus=None, teammate_manager=None, global_tasks=None):
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
        self.max_steps = max_steps
        self.max_retries = max_retries
        self.max_tokens = 8000
        self.approval_policy = approval_policy   # "ask" | "auto" | "never"
        self.message_bus = message_bus
        self.teammate_manager = teammate_manager
        self.global_tasks = global_tasks

        self.messages: list[dict] = []
        self._tracked_files: list[str] = []
        self._compact_requested: bool = False
        self._model_max = MODEL_MAX_TOKENS.get(llm.model, MODEL_MAX_TOKENS["default"])

    # ── Public ─────────────────────────────────────

    def run(self, user_request: str, resume_messages: list[dict] | None = None) -> str:
        state = TaskState.create(user_request)
        state.resume_status = getattr(self, "resume_status", "")

        # 恢复 todos（跨 session）
        if hasattr(self, "_resume_todos") and self._resume_todos:
            state.todos = list(self._resume_todos)

        self.run_store.start_run(state)
        self.run_store.log("run_started", run_id=state.run_id, request=user_request[:300])

        # resume_messages 含历史 + checkpoint 注入，新请求由这里追加
        if resume_messages:
            self.messages = list(resume_messages)
        # 统一在这里追加当前用户请求（TextBlock 格式确保进入模型上下文）
        self.memory.observe_user_message(user_request)
        self.messages.append({
            "role": "user", "content": [TextBlock(text=user_request)],
        })

        while state.is_running and state.tool_steps < self.max_steps:
            # 0. MULTI-AGENT — inject teammate messages + sync snapshots
            self._inject_inbox_messages(state)
            self._sync_multi_agent_state(state)

            # 1. COMPACT — 每轮 LLM 调用前压缩上下文
            self.messages = self.context.compact(self.messages, self.llm)

            # 2. PERCEIVE — 组装 system prompt + 工作记忆
            system = self.context.build_system(
                state, self.tools.registry, self.memory, self.sandbox,
            )
            self.hooks.trigger(HookEvent.USER_PROMPT, request=user_request)

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
                state.stop_model_error("LLM returned None after retries")
                self.hooks.trigger(HookEvent.RUN_STOP, state=state)
                break

            state.record_attempt()
            self.messages.append({"role": "assistant", "content": response.content})

            # 4. ACT
            tool_blocks = [b for b in response.content if isinstance(b, ToolUseBlock)]

            if not tool_blocks:
                texts = [b.text for b in response.content if isinstance(b, TextBlock)]
                state.finish_success(" ".join(texts))
                self._sync_session(state)
                self.hooks.trigger(HookEvent.RUN_STOP, state=state)
                break

            # 检查是否触发了 compact 工具（主动压缩）
            compact_tool = next((b for b in tool_blocks if b.name == "compact"), None)

            ctx = ToolContext(
                workspace_root=str(self.sandbox.root),
                sandbox=self.sandbox, shell=self.shell, memory=self.memory,
                task_state=state,
                llm=self.llm,
                tool_registry=self.tools.registry,
                message_bus=self.message_bus,
                teammate_manager=self.teammate_manager,
                global_tasks=self.global_tasks,
                agent_name="lead",
                run_id=state.run_id,
                trace_logger=self.run_store,
                depth=0, max_depth=1,
            )

            tool_results = []
            for block in tool_blocks:
                tool = self.tools.registry.get(block.name)

                # 权限检查
                deny = self.hooks.trigger(
                    HookEvent.PRE_TOOL_USE,
                    tool=tool,
                    tool_input=block.input,
                    approval_policy=self.approval_policy,
                )
                if deny:
                    result = ToolResult.fail(f"Blocked: {deny}")
                else:
                    result = self.tools.execute(block.name, block.input, ctx)
                    self.hooks.trigger(HookEvent.POST_TOOL_USE, tool=tool, result=result)

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

        if state.is_running:
            state.stop_step_limit()

        # 持久化最终状态（task_state.json + report.json + session）
        self.run_store.update_state(state)
        self.run_store.write_report(state)
        if hasattr(self, '_session') and self._session is not None:
            self.session_store.save(self._session)

        return state.final_answer or f"Stopped: {state.stop_reason}"

    # ── LLM with retry ─────────────────────────────

    def _inject_inbox_messages(self, state: TaskState) -> None:
        if not self.message_bus:
            return
        messages = self.message_bus.receive("lead")
        if not messages:
            state.unprocessed_messages = []
            return

        lines = ["## Teammate Messages"]
        for msg in messages:
            lines.append(f"- From {msg.from_agent} [{msg.msg_type}]: {msg.content}")
            self.run_store.log(
                "message_received",
                run_id=state.run_id,
                from_agent=msg.from_agent,
                to_agent=msg.to_agent,
                msg_type=msg.msg_type,
            )
        state.unprocessed_messages = []
        self.messages.append({"role": "user", "content": [TextBlock(text="\n".join(lines))]})

    def _sync_multi_agent_state(self, state: TaskState) -> None:
        if self.teammate_manager:
            state.active_teammates = self.teammate_manager.snapshot()
        if self.global_tasks:
            state.global_task_ids = [task.task_id for task in self.global_tasks.list_all()]

    def _call_llm_with_retry(self, messages, tools, system, retries=0,
                              run_id: str = ""):
        """Return (response_or_None, messages). messages may be compacted on retry."""
        try:
            resp = self.llm.chat(messages, tools, system, max_tokens=self.max_tokens)
        except Exception as e:
            if "429" in str(e) or "529" in str(e):
                if retries < self.max_retries:
                    time.sleep(2 ** retries)
                    return self._call_llm_with_retry(messages, tools, system, retries + 1, run_id=run_id)
            if self._is_prompt_too_long(e):
                logger.warning("Prompt too long, triggering reactive compact")
                compacted = self.context.reactive_compact(messages, self.llm)
                if compacted is not messages:
                    self.run_store.log("compaction_triggered",
                                       run_id=run_id,
                                       trigger="prompt_too_long",
                                       compact_count=self.context.compact_count)
                    resp, final_msgs = self._call_llm_with_retry(compacted, tools, system, retries + 1, run_id=run_id)
                    return resp, final_msgs
            logger.error(f"LLM error: {e}")
            return None, messages

        if resp.stop_reason == "max_tokens":
            if self.max_tokens < self._model_max:
                self.max_tokens = min(self.max_tokens * 2, self._model_max)
                messages.append({"role": "user", "content": [TextBlock(text="Continue.")]})
                return self._call_llm_with_retry(messages, tools, system, retries + 1, run_id=run_id)
        return resp, messages

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
