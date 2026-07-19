"""Secret detection and redaction engine.

Two-tier architecture:
  1. Auto-detection: env var names matching markers (API_KEY, TOKEN, SECRET, PASSWORD)
  2. Explicit config: user-supplied --secret-env-name list

Redaction pipeline:
  - redact_text():  string-level substring replacement (sorted by value length, longest first)
  - redact_artifact(): recursive structural redaction (dict/list/tuple/str)
  - SECRET_SHAPED_TEXT_PATTERN: regex for detecting secret-shaped text in memory promotion
"""

import os
import re
from typing import Any, Iterable


# ── Constants ──────────────────────────────────────

REDACTED_VALUE = "<redacted>"

# Substrings that indicate a sensitive env var name (case-insensitive)
SENSITIVE_ENV_NAME_MARKERS = ("API_KEY", "TOKEN", "SECRET", "PASSWORD")

# Default secret env var names (always tracked, even without --secret-env-name)
DEFAULT_SECRET_ENV_NAMES = (
    "ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN",
    "OPENAI_API_KEY", "OPENAI_API_TOKEN",
    "DEEPSEEK_API_KEY",
    "GITHUB_TOKEN", "GITHUB_PAT", "GH_TOKEN", "GH_PAT",
    "ECHO_API_KEY",
)

# Regex for detecting secret-shaped text in memory / durable promotion
# Catches: api_key, token, secret, password keywords + sk- prefixed keys
SECRET_SHAPED_TEXT_PATTERN = re.compile(
    r"(?i)(\b(api[_ -]?key|token|secret|password)\b|sk-[A-Za-z0-9_-]{6,})"
)


# ── Env var name detection ─────────────────────────

def _normalize_names(names: Iterable[str] | None) -> set[str]:
    """Normalize an iterable of env var names to an uppercase-only set.

    Args:
        names: Iterable of string names, or None.

    Returns:
        Uppercase set of names. Never returns None — empty set for falsy input.
        Filters out None entries silently.
    """
    return {str(n).upper() for n in (names or ()) if n is not None}


def looks_sensitive_env_name(name: str) -> bool:
    """Auto-detect whether an env var name looks sensitive.

    Matches if the name (case-insensitively):
      - Is exactly a marker (e.g., "API_KEY")
      - Ends with a marker (e.g., "OPENAI_API_KEY")
      - Ends with "_<marker>" (e.g., "MY_SECRET_TOKEN")

    Examples that match:
      API_KEY, OPENAI_API_KEY, DB_PASSWORD, SERVICE_TOKEN,
      MY_SECRET, ANTHROPIC_AUTH_TOKEN

    Examples that do NOT match:
      PATH, HOME, USER, LANG

    Args:
        name: Environment variable name to check.

    Returns:
        True if the name matches any sensitive marker pattern.
    """
    upper = str(name).upper()
    return any(
        upper == marker
        or upper.endswith(marker)
        or upper.endswith(f"_{marker}")
        for marker in SENSITIVE_ENV_NAME_MARKERS
    )


def is_secret_env_name(name: str, secret_env_names: Iterable[str] | None = None) -> bool:
    """Combined check: explicit config OR auto-detection OR default list.

    Returns True if the name is in the explicitly configured set,
    passes the auto-detection heuristic, OR is in DEFAULT_SECRET_ENV_NAMES.

    Args:
        name: Environment variable name.
        secret_env_names: Explicitly configured secret names (from --secret-env-name).

    Returns:
        True if this name should be treated as secret.
    """
    upper = str(name).upper()
    return (
        upper in _normalize_names(secret_env_names)
        or upper in _normalize_names(DEFAULT_SECRET_ENV_NAMES)
        or looks_sensitive_env_name(upper)
    )


# ── Env var value collection ───────────────────────

def configured_secret_items(
    env: dict[str, str] | None = None,
    secret_env_names: Iterable[str] | None = None,
) -> list[tuple[str, str]]:
    """Return (name, value) pairs for explicitly configured secret env vars.

    Only includes names in the explicitly configured set (not auto-detected).
    Sorted by name for deterministic output.

    Args:
        env: Environment dict. Defaults to os.environ.
        secret_env_names: Explicitly configured secret names.

    Returns:
        Sorted list of (name, value) tuples for configured secrets with truthy values.
    """
    env = os.environ if env is None else env
    configured = _normalize_names(secret_env_names)
    items = [
        (name, value)
        for name, value in env.items()
        if str(name).upper() in configured and value
    ]
    items.sort(key=lambda item: item[0])
    return items


def detected_secret_items(
    env: dict[str, str] | None = None,
    secret_env_names: Iterable[str] | None = None,
) -> list[tuple[str, str]]:
    """Return (name, value) pairs for ALL detected secret env vars.

    Includes both explicitly configured AND auto-detected secrets.
    Sorted by name for deterministic output.

    This is the primary function used for redaction — it catches everything
    that needs to be scrubbed from output.

    Args:
        env: Environment dict. Defaults to os.environ.
        secret_env_names: Explicitly configured secret names (merged with auto-detected).

    Returns:
        Sorted list of (name, value) tuples for all detected secrets with truthy values.
    """
    env = os.environ if env is None else env
    items = [
        (name, value)
        for name, value in env.items()
        if is_secret_env_name(name, secret_env_names=secret_env_names) and value
    ]
    items.sort(key=lambda item: item[0])
    return items


def secret_env_summary(
    env: dict[str, str] | None = None,
    secret_env_names: Iterable[str] | None = None,
) -> dict:
    """Return a summary of secret env vars (names only, never values).

    Used for embedding in prompt metadata and reports. Values are never included.

    Args:
        env: Environment dict. Defaults to os.environ.
        secret_env_names: Explicitly configured secret names.

    Returns:
        Dict with "secret_env_count" (int) and "secret_env_names" (list[str]).
    """
    names = [name for name, _ in detected_secret_items(env=env, secret_env_names=secret_env_names)]
    return {"secret_env_count": len(names), "secret_env_names": names}


# ── Redaction ──────────────────────────────────────

def redact_text(
    text: str,
    env: dict[str, str] | None = None,
    secret_env_names: Iterable[str] | None = None,
) -> str:
    """String-level redaction: replace all detected secret values with <redacted>.

    Secret values are sorted by length (longest first) before replacement
    to prevent partial matches from interfering with full matches.

    Example:
      If env has KEY_A="abc" and KEY_B="abcdef", then "abcdef" is replaced
      first, preventing "abc" from partially matching within "abcdef".

    Args:
        text: Text to redact.
        env: Environment dict. Defaults to os.environ.
        secret_env_names: Explicitly configured secret names.

    Returns:
        Text with all detected secret values replaced by "<redacted>".
    """
    text = str(text)
    items = detected_secret_items(env=env, secret_env_names=secret_env_names)
    # Sort by value length descending — longest first to prevent substring collisions
    items.sort(key=lambda item: len(item[1]), reverse=True)
    for _name, value in items:
        text = text.replace(value, REDACTED_VALUE)
    return text


def redact_artifact(
    value: Any,
    key: str | None = None,
    env: dict[str, str] | None = None,
    secret_env_names: Iterable[str] | None = None,
) -> Any:
    """Recursive structural redaction for traces, reports, and logs.

    Redaction strategy:
      1. Key-based: If the current dict key is a secret env name, replace the
         ENTIRE value with <redacted> immediately. This catches cases where
         the key signals a secret but the value format differs from the env var.
      2. Dict recursion: Recurse into each key-value pair, passing the key name
         down so nested structures can also benefit from key-based redaction.
      3. List/tuple recursion: Recurse into each element. Tuples are returned
         as lists (type coercion for safety — original tuple order is preserved).
      4. String values: Apply redact_text() for substring-level redaction.
      5. Non-string, non-collection values (int, bool, None): Returned unchanged.

    Args:
        value: Any JSON-serializable value to redact.
        key: The dict key associated with this value (if nested in a dict).
        env: Environment dict. Defaults to os.environ.
        secret_env_names: Explicitly configured secret names.

    Returns:
        Redacted copy of the input. Original is never mutated.
    """
    # Key-based redaction: if the key itself indicates a secret
    if key is not None and is_secret_env_name(key, secret_env_names=secret_env_names):
        return REDACTED_VALUE

    # Dict: recurse into each value, passing the key name as context
    if isinstance(value, dict):
        return {
            str(item_key): redact_artifact(item_value, key=item_key, env=env, secret_env_names=secret_env_names)
            for item_key, item_value in value.items()
        }

    # List: recurse into each element, preserving list type
    if isinstance(value, list):
        return [redact_artifact(item, key=key, env=env, secret_env_names=secret_env_names) for item in value]

    # Tuple: recurse into each element, returned as list
    if isinstance(value, tuple):
        return [redact_artifact(item, key=key, env=env, secret_env_names=secret_env_names) for item in value]

    # String: substring-level redaction
    if isinstance(value, str):
        return redact_text(value, env=env, secret_env_names=secret_env_names)

    # Non-string, non-collection (int, bool, None, float, ...) — pass through
    return value


# ── Durable memory gate ────────────────────────────

def is_secret_shaped(text: str) -> bool:
    """Check if text looks like it contains a secret value.

    Used as a gate before promoting ephemeral memory to durable storage.
    Rejects text matching SECRET_SHAPED_TEXT_PATTERN (API key keywords,
    sk- prefixed tokens) or already-redacted text containing <redacted>.

    Args:
        text: The text to check.

    Returns:
        True if the text appears to contain secret information.
    """
    text = str(text or "")
    if REDACTED_VALUE in text:
        return True
    return bool(SECRET_SHAPED_TEXT_PATTERN.search(text))
