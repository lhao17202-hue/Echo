"""Unit tests for echo.security module.

Covers: redaction, sandbox, permission, env_filter.
Run: python -m pytest tests/test_security.py -v
"""

import os
import sys
import tempfile
import pytest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from echo.security.redaction import (
    REDACTED_VALUE,
    looks_sensitive_env_name,
    is_secret_env_name,
    configured_secret_items,
    detected_secret_items,
    secret_env_summary,
    redact_text,
    redact_artifact,
    is_secret_shaped,
    SECRET_SHAPED_TEXT_PATTERN,
    DEFAULT_SECRET_ENV_NAMES,
    SENSITIVE_ENV_NAME_MARKERS,
)

from echo.security.sandbox import Sandbox, PathEscapedError
from echo.security.permission import PermissionGuard
from echo.security.env_filter import (
    ShellExecutor,
    ShellResult,
    build_shell_env,
    DEFAULT_SHELL_ENV_ALLOWLIST,
)


# ═══════════════════════════════════════════════════
# Redaction — env var name detection
# ═══════════════════════════════════════════════════

class TestLooksSensitiveEnvName:
    """Auto-detection of sensitive env var names via marker matching."""

    def test_exact_marker_match(self):
        assert looks_sensitive_env_name("API_KEY") is True
        assert looks_sensitive_env_name("TOKEN") is True
        assert looks_sensitive_env_name("SECRET") is True
        assert looks_sensitive_env_name("PASSWORD") is True

    def test_case_insensitive(self):
        assert looks_sensitive_env_name("api_key") is True
        assert looks_sensitive_env_name("Api_Key") is True
        assert looks_sensitive_env_name("API_key") is True

    def test_prefix_with_underscore(self):
        assert looks_sensitive_env_name("OPENAI_API_KEY") is True
        assert looks_sensitive_env_name("GITHUB_TOKEN") is True
        assert looks_sensitive_env_name("DB_PASSWORD") is True
        assert looks_sensitive_env_name("MY_SECRET_TOKEN") is True

    def test_prefix_without_underscore(self):
        # "endswith(marker)" catches even without underscore prefix
        assert looks_sensitive_env_name("MYTOKEN") is True

    def test_non_sensitive_names(self):
        assert looks_sensitive_env_name("PATH") is False
        assert looks_sensitive_env_name("HOME") is False
        assert looks_sensitive_env_name("USER") is False
        assert looks_sensitive_env_name("LANG") is False
        assert looks_sensitive_env_name("PYTHONPATH") is False


class TestIsSecretEnvName:
    """Combined check: explicit config + auto-detection."""

    def test_auto_detected(self):
        # Even without explicit config, auto-detection catches API_KEY
        assert is_secret_env_name("OPENAI_API_KEY") is True

    def test_explicitly_configured(self):
        # Names in the explicit set are always secret
        assert is_secret_env_name("MY_CUSTOM_VAR", secret_env_names=["MY_CUSTOM_VAR"]) is True

    def test_not_secret(self):
        assert is_secret_env_name("HOME") is False
        assert is_secret_env_name("EDITOR") is False


# ═══════════════════════════════════════════════════
# Redaction — env var item collection
# ═══════════════════════════════════════════════════

class TestConfiguredSecretItems:
    """Explicitly configured secrets only."""

    def test_config_only(self):
        env = {"OPENAI_API_KEY": "sk-abc", "HOME": "/home"}
        items = configured_secret_items(env=env, secret_env_names=["OPENAI_API_KEY"])
        names = [n for n, _ in items]
        assert "OPENAI_API_KEY" in names
        assert "HOME" not in names  # auto-detection NOT applied

    def test_sorted_by_name(self):
        env = {"Z_KEY": "z", "A_KEY": "a"}
        items = configured_secret_items(env=env, secret_env_names=["Z_KEY", "A_KEY"])
        assert items[0][0] == "A_KEY"
        assert items[1][0] == "Z_KEY"


class TestDetectedSecretItems:
    """Explicit + auto-detected secrets."""

    def test_includes_auto_detected(self):
        env = {"OPENAI_API_KEY": "sk-abc", "HOME": "/home", "USER": "test"}
        items = detected_secret_items(env=env)
        names = [n for n, _ in items]
        assert "OPENAI_API_KEY" in names
        assert "HOME" not in names
        assert "USER" not in names


class TestSecretEnvSummary:
    """Summary never leaks values."""

    def test_names_only_no_values(self):
        env = {"OPENAI_API_KEY": "sk-super-secret-value"}
        summary = secret_env_summary(env=env)
        assert summary["secret_env_count"] == 1
        assert "OPENAI_API_KEY" in summary["secret_env_names"]
        assert "sk-super-secret-value" not in str(summary)


# ═══════════════════════════════════════════════════
# Redaction — text and artifact redaction
# ═══════════════════════════════════════════════════

class TestRedactText:
    """String-level secret value replacement."""

    def test_single_secret(self):
        env = {"OPENAI_API_KEY": "sk-my-secret-key"}
        result = redact_text("my key is sk-my-secret-key", env=env)
        assert "sk-my-secret-key" not in result
        assert REDACTED_VALUE in result

    def test_normal_text_untouched(self):
        env = {"OPENAI_API_KEY": "sk-secret"}
        result = redact_text("hello world", env=env)
        assert result == "hello world"

    def test_longest_value_first(self):
        # If KEY_A="abc" and KEY_B="abcdef", "abcdef" must be replaced first
        # so "abc" doesn't partially match inside "abcdef"
        env = {
            "SHORT_KEY": "abc",
            "LONG_KEY": "abcdef",
        }
        result = redact_text("test abc abcdef", env=env,
                             secret_env_names=["SHORT_KEY", "LONG_KEY"])
        assert "abcdef" not in result
        assert "abc" not in result


class TestRedactArtifact:
    """Recursive structural redaction."""

    def test_key_based_redaction(self):
        """When a dict key IS a secret name, the entire value is replaced."""
        env = {"API_KEY": "sk-123"}
        data = {"API_KEY": "sk-different-format-xxx"}
        result = redact_artifact(data, env=env)
        assert result["API_KEY"] == REDACTED_VALUE

    def test_nested_dict(self):
        env = {"SECRET": "my-secret"}
        data = {"config": {"auth": {"SECRET": "my-secret"}}}
        result = redact_artifact(data, env=env)
        assert result["config"]["auth"]["SECRET"] == REDACTED_VALUE

    def test_list_recursion(self):
        env = {"TOKEN": "tok-123"}
        data = ["hello", "tok-123", "world"]
        result = redact_artifact(data, env=env)
        assert REDACTED_VALUE in result
        assert "tok-123" not in result

    def test_tuple_to_list(self):
        """Tuples become lists after redaction."""
        env = {"API_KEY": "secret"}
        data = ("secret", "normal")
        result = redact_artifact(data, env=env)
        assert isinstance(result, list)
        assert REDACTED_VALUE in result

    def test_non_string_passthrough(self):
        result = redact_artifact(42)
        assert result == 42
        result = redact_artifact(None)
        assert result is None
        result = redact_artifact(True)
        assert result is True


class TestIsSecretShaped:
    """Gate for durable memory promotion."""

    def test_sk_prefix(self):
        assert is_secret_shaped("sk-live-secret-abc123") is True

    def test_api_key_keyword(self):
        assert is_secret_shaped("my api_key is compromised") is True

    def test_already_redacted(self):
        assert is_secret_shaped(f"value is {REDACTED_VALUE}") is True

    def test_safe_text(self):
        assert is_secret_shaped("normal memory entry") is False
        assert is_secret_shaped("") is False

    def test_none(self):
        assert is_secret_shaped(None) is False


class TestSecretShapedRegex:
    """SECRET_SHAPED_TEXT_PATTERN edge cases."""

    def test_sk_with_dashes(self):
        assert SECRET_SHAPED_TEXT_PATTERN.search("sk-ant-api03-abc123xyz") is not None

    def test_apikey_variants(self):
        assert SECRET_SHAPED_TEXT_PATTERN.search("apikey") is not None
        assert SECRET_SHAPED_TEXT_PATTERN.search("api_key") is not None
        assert SECRET_SHAPED_TEXT_PATTERN.search("api-key") is not None
        assert SECRET_SHAPED_TEXT_PATTERN.search("api key") is not None


# ═══════════════════════════════════════════════════
# Sandbox — path containment
# ═══════════════════════════════════════════════════

class TestSandbox:
    """File system sandbox tests."""

    def test_relative_path_ok(self):
        with tempfile.TemporaryDirectory() as d:
            s = Sandbox(d)
            resolved = s.resolve("subdir/file.py")
            assert resolved == Path(d).resolve() / "subdir" / "file.py"

    def test_dot_path(self):
        with tempfile.TemporaryDirectory() as d:
            s = Sandbox(d)
            resolved = s.resolve(".")
            assert str(resolved) == d

    def test_parent_traversal_blocked(self):
        with tempfile.TemporaryDirectory() as d:
            s = Sandbox(d)
            with pytest.raises(PathEscapedError):
                s.resolve("../etc/passwd")

    def test_absolute_outside_blocked(self):
        with tempfile.TemporaryDirectory() as d:
            s = Sandbox(d)
            with pytest.raises(PathEscapedError):
                s.resolve("/etc/passwd")

    def test_is_safe(self):
        with tempfile.TemporaryDirectory() as d:
            s = Sandbox(d)
            assert s.is_safe("file.txt") is True
            assert s.is_safe("../outside") is False

    def test_symlink_escape_blocked(self):
        """Symlinks pointing outside the workspace are caught by resolve()."""
        with tempfile.TemporaryDirectory() as d:
            s = Sandbox(d)
            symlink = Path(d) / "escape_link"
            # Create a symlink to /etc (or Windows equivalent)
            target = "/etc" if os.name != "nt" else "C:/Windows"
            try:
                os.symlink(target, symlink)
            except OSError:
                pytest.skip("Symlink creation requires privileges on this platform")
            with pytest.raises(PathEscapedError):
                s.resolve("escape_link")

    def test_snapshot_detects_changes(self):
        with tempfile.TemporaryDirectory() as d:
            s = Sandbox(d)
            f = Path(d) / "test.txt"
            f.write_text("v1")
            snap = s.snapshot(paths=[str(f)])
            f.write_text("v2")
            changed = s.diff(snap)
            assert str(f) in changed

    def test_snapshot_detects_deletion(self):
        with tempfile.TemporaryDirectory() as d:
            s = Sandbox(d)
            f = Path(d) / "test.txt"
            f.write_text("v1")
            snap = s.snapshot(paths=[str(f)])
            f.unlink()
            changed = s.diff(snap)
            assert any("deleted" in c for c in changed)

    def test_git_branch(self):
        with tempfile.TemporaryDirectory() as d:
            s = Sandbox(d)
            branch = s.git_branch
            assert isinstance(branch, str)
            # Not a git repo by default, so should be "unknown"
            assert branch == "unknown"


# ═══════════════════════════════════════════════════
# Permission — risk levels + command checks
# ═══════════════════════════════════════════════════

class TestPermissionGuardRiskLevels:
    """Risk-level-based authorization."""

    def test_safe_auto_approved(self):
        allowed, msg = PermissionGuard.check("safe", "read_file")
        assert allowed is True
        assert msg == ""

    def test_warn_requires_confirmation(self):
        allowed, msg = PermissionGuard.check("warn", "write_file")
        assert allowed is False
        assert "Approve" in msg

    def test_danger_default_deny(self):
        allowed, msg = PermissionGuard.check("danger", "run_shell")
        assert allowed is False
        assert "DANGER" in msg


class TestPermissionGuardShell:
    """Shell command deny and destructive lists."""

    def test_deny_rm_rf_root(self):
        allowed, msg = PermissionGuard.check_shell_command("rm -rf /")
        assert allowed is False
        assert "deny list" in msg.lower()

    def test_deny_sudo_rm(self):
        allowed, _ = PermissionGuard.check_shell_command("sudo rm -rf /var/log")
        assert allowed is False

    def test_destructive_rm(self):
        allowed, msg = PermissionGuard.check_shell_command("rm important_file.txt")
        assert allowed is False
        assert "Destructive" in msg

    def test_safe_command(self):
        allowed, msg = PermissionGuard.check_shell_command("ls -la")
        assert allowed is True
        assert msg == ""

    def test_is_denied(self):
        assert PermissionGuard.is_denied("rm -rf /") is True
        assert PermissionGuard.is_denied("ls -la") is False


class TestPermissionGuardPath:
    """Path boundary checks."""

    def test_path_in_workspace(self):
        with tempfile.TemporaryDirectory() as d:
            s = Sandbox(d)
            allowed, msg = PermissionGuard.check_path("file.txt", s)
            assert allowed is True

    def test_path_escape(self):
        with tempfile.TemporaryDirectory() as d:
            s = Sandbox(d)
            allowed, msg = PermissionGuard.check_path("../etc/passwd", s)
            assert allowed is False
            assert "path escapes workspace" in msg.lower()


# ═══════════════════════════════════════════════════
# Env filter — shell environment isolation
# ═══════════════════════════════════════════════════

class TestBuildShellEnv:
    """Environment allowlist filtering."""

    def test_pwd_overwritten(self):
        env = build_shell_env("/my/workspace")
        assert env["PWD"] == "/my/workspace"

    def test_path_preserved(self):
        os.environ["PATH"] = os.environ.get("PATH", "/usr/bin")
        env = build_shell_env("/tmp")
        assert "PATH" in env

    def test_secrets_filtered_out(self):
        """API keys in the parent env must NOT appear in the shell env."""
        env = build_shell_env("/tmp", env={"API_KEY": "secret", "HOME": "/home", "PATH": "/bin"})
        assert "API_KEY" not in env
        assert "HOME" in env

    def test_extra_allowlist(self):
        env = build_shell_env("/tmp", extra_allowlist=["MY_TOOL"], env={"MY_TOOL": "val"})
        assert env["MY_TOOL"] == "val"

    def test_default_allowlist_covers_essentials(self):
        """Verify the default allowlist has basic vars."""
        assert "HOME" in DEFAULT_SHELL_ENV_ALLOWLIST
        assert "PATH" in DEFAULT_SHELL_ENV_ALLOWLIST
        assert "USER" in DEFAULT_SHELL_ENV_ALLOWLIST or "USERNAME" in DEFAULT_SHELL_ENV_ALLOWLIST


class TestShellExecutor:
    """Shell command execution."""

    def test_basic_execution(self):
        with tempfile.TemporaryDirectory() as d:
            executor = ShellExecutor(d)
            result = executor.run("echo hello")
            assert result.success
            assert "hello" in result.output

    def test_timeout(self):
        with tempfile.TemporaryDirectory() as d:
            executor = ShellExecutor(d)
            import sys
            result = executor.run(f"{sys.executable} -c \"import time; time.sleep(10)\"", timeout=1)
            assert result.failed
            assert "timed out" in (result.error or "").lower()

    def test_error_command(self):
        with tempfile.TemporaryDirectory() as d:
            executor = ShellExecutor(d)
            result = executor.run("nonexistent_command_xyz")
            assert result.exit_code != 0

    def test_env_isolation(self):
        """ShellExecutor must not leak API keys to child processes."""
        with tempfile.TemporaryDirectory() as d:
            # Set a fake secret in os.environ
            os.environ["TEST_API_KEY"] = "sk-test-secret"
            try:
                executor = ShellExecutor(d)
                result = executor.run("echo $TEST_API_KEY")
                # The child should NOT see the secret
                assert "sk-test-secret" not in result.output
            finally:
                del os.environ["TEST_API_KEY"]

    def test_cwd_defaults_to_workspace(self):
        with tempfile.TemporaryDirectory() as d:
            executor = ShellExecutor(d)
            result = executor.run("pwd" if os.name != "nt" else "cd")
            assert d in result.output


# ═══════════════════════════════════════════════════
# Default secret env names
# ═══════════════════════════════════════════════════

class TestDefaultSecretEnvNames:
    """Verify default secret tracking covers major providers."""

    def test_anthropic_covered(self):
        assert "ANTHROPIC_API_KEY" in DEFAULT_SECRET_ENV_NAMES

    def test_openai_covered(self):
        assert "OPENAI_API_KEY" in DEFAULT_SECRET_ENV_NAMES

    def test_deepseek_covered(self):
        assert "DEEPSEEK_API_KEY" in DEFAULT_SECRET_ENV_NAMES

    def test_github_covered(self):
        assert "GITHUB_PAT" in DEFAULT_SECRET_ENV_NAMES
