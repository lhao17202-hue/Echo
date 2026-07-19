"""Security module — redaction, sandbox, permission, environment filtering.

Usage:
    from echo.security import Sandbox, redact_text, PermissionGuard, ShellExecutor
"""

from echo.security.redaction import (
    REDACTED_VALUE,
    SENSITIVE_ENV_NAME_MARKERS,
    DEFAULT_SECRET_ENV_NAMES,
    SECRET_SHAPED_TEXT_PATTERN,
    looks_sensitive_env_name,
    is_secret_env_name,
    configured_secret_items,
    detected_secret_items,
    secret_env_summary,
    redact_text,
    redact_artifact,
    is_secret_shaped,
)

from echo.security.sandbox import (
    Sandbox,
    PathEscapedError,
)

from echo.security.permission import (
    PermissionGuard,
)

from echo.security.env_filter import (
    ShellResult,
    ShellExecutor,
    build_shell_env,
    DEFAULT_SHELL_ENV_ALLOWLIST,
)

__all__ = [
    # Redaction
    "REDACTED_VALUE",
    "SENSITIVE_ENV_NAME_MARKERS",
    "DEFAULT_SECRET_ENV_NAMES",
    "SECRET_SHAPED_TEXT_PATTERN",
    "looks_sensitive_env_name",
    "is_secret_env_name",
    "configured_secret_items",
    "detected_secret_items",
    "secret_env_summary",
    "redact_text",
    "redact_artifact",
    "is_secret_shaped",
    # Sandbox
    "Sandbox",
    "PathEscapedError",
    # Permission
    "PermissionGuard",
    # Env filter
    "ShellResult",
    "ShellExecutor",
    "build_shell_env",
    "DEFAULT_SHELL_ENV_ALLOWLIST",
]
