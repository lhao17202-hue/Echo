"""Echo facade — assemble all subsystems, provide simple ask() interface."""

from pathlib import Path
from echo.config import EchoConfig
from echo.providers.anthropic_client import AnthropicClient
from echo.providers.openai_client import OpenAIClient
from echo.providers.ollama_client import OllamaClient
from echo.tools.registry import ToolRegistry
from echo.tools.executor import ToolExecutor
from echo.tools.sandbox import Sandbox, ShellExecutor
from echo.memory.base import MemoryManager
from echo.memory.default import KeywordMemory
from echo.memory.durable import JsonDurableMemoryBackend
from echo.hooks.base import HookManager
from echo.hooks.builtin import PermissionHook, LogHook, PostLogHook, StatsHook
from echo.core.context_manager import ContextManager, ContextConfig
from echo.core.agent_loop import AgentLoop
from echo.persistence.session_store import Session, SessionStore
from echo.persistence.run_store import RunStore


class Echo:
    """Echo Agent 门面类。

    使用方式：
      echo = Echo(workspace_root="/path/to/project")
      answer = echo.ask("帮我看看这个项目")
    """

    def __init__(self, workspace_root: str = "", config: EchoConfig | None = None):
        self.config = config or EchoConfig.from_env()
        self.workspace_root = Path(workspace_root or ".").resolve()

        # ── Provider ────────────────────────────────
        provider = self.config.provider
        if provider == "openai":
            self.llm = OpenAIClient(
                model=self.config.model,
                api_key=self.config.api_key,
                base_url=self.config.base_url,
            )
        elif provider == "ollama":
            self.llm = OllamaClient(
                model=self.config.model,
                base_url=self.config.base_url,
            )
        else:
            # deepseek 和 anthropic 都走 Anthropic SDK（DeepSeek 有 Anthropic 兼容端点）
            self.llm = AnthropicClient(
                model=self.config.model,
                api_key=self.config.api_key,
                base_url=self.config.base_url,
            )

        # ── Infrastructure ──────────────────────────
        self.sandbox = Sandbox(str(self.workspace_root))
        self.shell = ShellExecutor(self.workspace_root)
        durable_path = str(self.workspace_root / ".echo" / "memory" / "durable.json")
        self.memory = MemoryManager(
            KeywordMemory(),
            durable_backend=JsonDurableMemoryBackend(durable_path),
        )
        self.session_store = SessionStore(str(self.workspace_root))

        # ── Tools ───────────────────────────────────
        self.tool_registry = ToolRegistry()
        self.tool_registry.discover("echo.tools.builtin")

        # ── Hooks ───────────────────────────────────
        self.hooks = HookManager()
        self.hooks.register(PermissionHook(), priority=0)
        self.hooks.register(LogHook(), priority=100)
        self.hooks.register(PostLogHook(), priority=100)
        self.hooks.register(StatsHook(), priority=200)

        # ── Context ─────────────────────────────────
        echo_dir = self.workspace_root / ".echo"
        self.context_manager = ContextManager(ContextConfig(
            enable_memory=self.config.enable_memory,
            enable_compaction=self.config.enable_compaction,
            persist_dir=str(echo_dir / "tool_outputs"),
            transcript_dir=str(echo_dir / "transcripts"),
        ))

        # ── Executor ────────────────────────────────
        self.executor = ToolExecutor(self.tool_registry)

    def ask(self, user_request: str, max_steps: int | None = None) -> str:
        """执行用户请求，返回最终回复。"""
        session = Session(
            session_id=self._generate_session_id(),
            workspace_root=str(self.workspace_root),
            model_config={
                "provider": self.config.provider,
                "model": self.config.model,
                "max_tokens": self.config.max_tokens,
                "temperature": self.config.temperature,
            },
            security_config={
                "approval_policy": self.config.approval_policy,
                "shell_env_allowlist": self.config.shell_env_allowlist,
            },
        )
        self.session_store.save(session)

        run_store = RunStore(
            str(self.workspace_root / ".echo" / "sessions" / session.session_id)
        )

        loop = AgentLoop(
            llm=self.llm,
            memory=self.memory,
            tools=self.executor,
            hooks=self.hooks,
            context=self.context_manager,
            sandbox=self.sandbox,
            shell=self.shell,
            session_store=self.session_store,
            run_store=run_store,
            max_steps=max_steps or self.config.max_steps,
            max_retries=self.config.max_retries,
            approval_policy=self.config.approval_policy,
        )
        # 注入当前 session，供 agent_loop 在检查点创建后持久化
        loop._session = session
        return loop.run(user_request)

    # ── Resume ──────────────────────────────────────

    def resume(self, session_id: str = "", user_request: str = "") -> str:
        """从已有 session 恢复，可选追加新用户请求。"""
        # 1. 加载 session
        sid = session_id or self.session_store.latest()
        if not sid:
            return "没有可恢复的 session。"
        session = self.session_store.load(sid)

        # 2. 恢复 memory
        if session.short_term_memory:
            self.memory.load_dict(session.short_term_memory)

        # 3. 恢复 messages（resume 信息用 TextBlock 格式确保进入模型上下文）
        from echo.providers.base import TextBlock
        resume_messages = list(session.history or [])

        # 4. 加载 checkpoint 并评估
        from echo.persistence.checkpoint import CheckpointManager, CHECKPOINT_NONE
        cm = CheckpointManager(str(self.workspace_root))
        ckpt = cm.load_from_session(session)
        resume_status = CHECKPOINT_NONE
        if ckpt:
            result = cm.evaluate(ckpt)
            resume_status = result["status"]
            if result["status"] != "schema-mismatch":
                checkpoint_text = cm.render(ckpt)
                if checkpoint_text:
                    resume_messages = [
                        {"role": "user",
                         "content": [TextBlock(
                             text=f"[Resumed session]\n\n{checkpoint_text}\n\nResume status: {resume_status}"
                         )]},
                        *resume_messages,
                    ]

        # 5. 如果没有新请求，返回就绪摘要
        if not user_request:
            parts = [f"Session {sid} 恢复就绪。"]
            if ckpt:
                parts.append(f"上次目标: {ckpt.current_goal or '-'}")
                parts.append(f"下一步: {ckpt.next_step or '-'}")
                parts.append(f"恢复状态: {resume_status}")
            return "\n".join(parts)

        # 6. 构建 AgentLoop 继续
        #    不在这里追加 user_request —— AgentLoop.run 在有 resume_messages 时
        #    会自动追加（用正确的 user_request 作为 TaskState 标记）。
        run_store = RunStore(str(self.workspace_root / ".echo" / "sessions" / sid))

        loop = AgentLoop(
            llm=self.llm, memory=self.memory, tools=self.executor,
            hooks=self.hooks, context=self.context_manager,
            sandbox=self.sandbox, shell=self.shell,
            session_store=self.session_store, run_store=run_store,
            max_steps=self.config.max_steps, max_retries=self.config.max_retries,
            approval_policy=self.config.approval_policy,
        )
        loop._session = session
        loop.resume_status = resume_status
        # 恢复 todos
        if session.short_term_memory and session.short_term_memory.get("todos"):
            loop._resume_todos = session.short_term_memory["todos"]

        return loop.run(user_request, resume_messages=resume_messages)

    def list_sessions(self, limit: int = 10) -> list[dict]:
        """列出最近会话。"""
        return self.session_store.list_sessions(limit)

    # ── Helpers ────────────────────────────────────

    @staticmethod
    def _generate_session_id() -> str:
        import uuid
        from datetime import datetime
        now = datetime.now()
        return f"{now.strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:6]}"
