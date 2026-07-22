"""Configuration — .env loader, env var resolution, EchoConfig."""

import os
from pathlib import Path
from dataclasses import dataclass, field


# ── Provider defaults ──────────────────────────────

DEFAULT_PROVIDER = "deepseek"
DEFAULT_DEEPSEEK_MODEL = "deepseek-v4-pro"
DEFAULT_DEEPSEEK_BASE_URL = "https://api.deepseek.com/anthropic"
DEFAULT_ANTHROPIC_MODEL = "claude-sonnet-4-6"
DEFAULT_OPENAI_MODEL = "gpt-4o-mini"
DEFAULT_OLLAMA_MODEL = "qwen3:4b"
DEFAULT_OLLAMA_HOST = "http://localhost:11434"
PROVIDER_CHOICES = ("anthropic", "openai", "deepseek", "ollama")


# ── Dotenv loader ──────────────────────────────────

def load_dotenv(workspace_root: str | None = None) -> None:
    """加载 .env 文件（从工作区根目录或当前目录向上查找）。"""
    try:
        from dotenv import load_dotenv as _load
        search_path = Path(workspace_root) if workspace_root else Path.cwd()
        for parent in [search_path] + list(search_path.parents):
            env_file = parent / ".env"
            if env_file.exists():
                _load(env_file)
                return
    except ImportError:
        pass


def get_env(key: str, default: str = "") -> str:
    """读取环境变量。"""
    return os.environ.get(key, default)


def provider_env(*keys: str, default: str = "") -> str:
    """读取多个候选环境变量，返回第一个存在的值。

    例如：provider_env("DEEPSEEK_API_KEY", "ANTHROPIC_API_KEY", default="")
    先查 DEEPSEEK_API_KEY，没有再查 ANTHROPIC_API_KEY。
    """
    for key in keys:
        val = os.environ.get(key, "")
        if val:
            return val
    return default


# ── EchoConfig ─────────────────────────────────────

@dataclass
class EchoConfig:
    """Echo Agent 全局配置。"""

    # Provider
    provider: str = DEFAULT_PROVIDER
    model: str = ""
    api_key: str = ""
    base_url: str | None = None
    max_tokens: int = 8000
    temperature: float = 0.0

    # Workspace
    workspace_root: str = ""

    # Agent
    max_steps: int = 25
    max_attempts: int | None = None
    max_retries: int = 3

    # Approval
    approval_policy: str = "ask"

    # Context
    context_budget_total: int = 12000
    compact_threshold_chars: int = 30000

    # Feature flags
    enable_memory: bool = True
    enable_compaction: bool = True
    enable_hooks: bool = True
    enable_cron: bool = False

    # Shell
    shell_env_allowlist: list[str] = field(default_factory=lambda: [
        "HOME", "LANG", "PATH", "USER", "VIRTUAL_ENV",
    ])

    @classmethod
    def from_env(cls, cli_provider: str = "", cli_model: str = "",
                 cli_base_url: str = "") -> "EchoConfig":
        """从环境变量构建配置。

        优先级：CLI 参数 > 专属环境变量 > 通用环境变量 > 默认值
        """
        load_dotenv()

        # Provider: CLI > ECHO_PROVIDER > default
        provider = cli_provider or get_env("ECHO_PROVIDER", DEFAULT_PROVIDER)
        if provider not in PROVIDER_CHOICES:
            provider = DEFAULT_PROVIDER

        # Model / API key / Base URL —— 按 provider 解析
        if provider == "deepseek":
            model = cli_model or provider_env("DEEPSEEK_MODEL", default=DEFAULT_DEEPSEEK_MODEL)
            api_key = provider_env("DEEPSEEK_API_KEY")
            base_url = cli_base_url or provider_env("DEEPSEEK_API_BASE", default=DEFAULT_DEEPSEEK_BASE_URL)
        elif provider == "openai":
            model = cli_model or provider_env("OPENAI_MODEL", default=DEFAULT_OPENAI_MODEL)
            api_key = provider_env("OPENAI_API_KEY")
            base_url = cli_base_url or provider_env("OPENAI_API_BASE") or None
        elif provider == "ollama":
            model = cli_model or provider_env("OLLAMA_MODEL", default=DEFAULT_OLLAMA_MODEL)
            api_key = ""
            base_url = cli_base_url or provider_env("OLLAMA_HOST", default=DEFAULT_OLLAMA_HOST)
        else:  # anthropic
            model = cli_model or provider_env("ANTHROPIC_MODEL", default=DEFAULT_ANTHROPIC_MODEL)
            api_key = provider_env("ANTHROPIC_API_KEY")
            base_url = cli_base_url or provider_env("ANTHROPIC_BASE_URL") or None

        return cls(
            provider=provider,
            model=model,
            api_key=api_key,
            base_url=base_url,
            max_tokens=int(get_env("ECHO_MAX_TOKENS", "8000")),
            max_steps=int(get_env("ECHO_MAX_STEPS", "25")),
            max_attempts=(
                int(get_env("ECHO_MAX_ATTEMPTS"))
                if get_env("ECHO_MAX_ATTEMPTS")
                else None
            ),
            approval_policy=get_env("ECHO_APPROVAL", "ask"),
            enable_cron=get_env("ECHO_CRON", "0") == "1",
        )
