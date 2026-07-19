"""Shell environment filter — allowlist-based isolation for subprocess execution.

Every shell command runs in a sanitized environment where only allowlisted
variables are passed through. This prevents secrets (API keys, tokens) from
leaking into tool output via environment variable dumps or child processes.

Design:
  1. Start from an allowlist of safe variable names
  2. Copy only those variables from the parent process environment
  3. Force PWD to the workspace root (not the process cwd)
  4. Preserve PATH even if omitted from the allowlist (safety fallback)
  5. All other variables are silently dropped

Combined with the ShellExecutor class, this forms the complete shell
execution security boundary.
"""

import os
import subprocess
from dataclasses import dataclass
from pathlib import Path


# ── Default allowlist ──────────────────────────────

# Variables that are safe to pass to subprocesses.
# PATH is handled specially — always preserved even if omitted from allowlist.
DEFAULT_SHELL_ENV_ALLOWLIST: tuple[str, ...] = (
    "HOME", "LANG", "LC_ALL", "LC_CTYPE",
    "PATH", "PWD", "SHELL", "TERM",
    "TMPDIR", "TMP", "TEMP", "USER", "USERNAME",
    "SYSTEMROOT",           # Windows
    "VIRTUAL_ENV",          # Python venv
    "CONDA_PREFIX",         # Conda
    "NODE_PATH",            # Node.js
    "PYTHONPATH",
)


# ── Shell environment builder ──────────────────────

def build_shell_env(
    workspace_root: str,
    extra_allowlist: list[str] | None = None,
    env: dict[str, str] | None = None,
) -> dict[str, str]:
    """Build a sanitized environment dict for shell command execution.

    Only variables in the allowlist are copied from the parent process.
    PWD is forced to the workspace root. PATH is preserved as a fallback
    even if omitted from the allowlist.

    Args:
        workspace_root: Absolute path to set as PWD.
        extra_allowlist: Additional variable names to allow beyond the defaults.
        env: Source environment dict. Defaults to os.environ.

    Returns:
        Sanitized environment dict suitable for subprocess.run(env=...).
    """
    env = os.environ if env is None else env

    # Build full allowlist
    allowed = set(DEFAULT_SHELL_ENV_ALLOWLIST)
    if extra_allowlist:
        allowed.update(str(n) for n in extra_allowlist)

    # Copy only allowlisted variables
    filtered: dict[str, str] = {
        name: env[name]
        for name in allowed
        if name in env
    }

    # Force PWD to workspace root
    filtered["PWD"] = str(workspace_root)

    # PATH fallback: always preserve PATH so commands can find binaries
    if "PATH" not in filtered and env.get("PATH"):
        filtered["PATH"] = env["PATH"]

    return filtered


# ── ShellResult ────────────────────────────────────

@dataclass
class ShellResult:
    """Result of a shell command execution."""
    output: str = ""
    error: str | None = None
    exit_code: int = 0

    @property
    def success(self) -> bool:
        """True if the command exited with code 0 and no error."""
        return self.exit_code == 0 and self.error is None

    @property
    def failed(self) -> bool:
        """True if the command failed (non-zero exit or error)."""
        return not self.success


# ── ShellExecutor ──────────────────────────────────

class ShellExecutor:
    """Execute shell commands in a sanitized environment.

    Every command runs:
      - With allowlisted environment variables only
      - With PWD set to the workspace root
      - With a configurable timeout

    Usage:
        executor = ShellExecutor(workspace_root, extra_env=["MY_TOOL_VAR"])
        result = executor.run("ls -la", timeout=10)
        print(result.output)
    """

    def __init__(
        self,
        workspace_root: Path | str,
        extra_env: list[str] | None = None,
    ):
        """Initialize the shell executor.

        Args:
            workspace_root: Workspace root directory (set as PWD).
            extra_env: Additional env var names to allow through the filter.
        """
        self.root = Path(workspace_root).resolve()  # 与 Sandbox 一致，规范化路径
        self.extra_env = extra_env or []

    def build_env(self) -> dict[str, str]:
        """Build the current sanitized environment.

        Returns:
            Filtered env dict with only allowlisted variables and PWD set.
        """
        return build_shell_env(
            workspace_root=str(self.root),
            extra_allowlist=self.extra_env,
        )

    def run(
        self,
        command: str,
        cwd: str = "",
        timeout: int = 20,
    ) -> ShellResult:
        """Execute a shell command synchronously.

        使用 shell=True。对于需要安全参数传递的场景，
        使用 run_list()（shell=False）。

        Args:
            command: Shell command string to execute.
            cwd: Working directory for the command (defaults to workspace root).
            timeout: Maximum execution time in seconds.

        Returns:
            ShellResult with output, error, and exit code.
        """
        cwd = cwd or str(self.root)

        try:
            proc = subprocess.run(
                command,
                shell=True,
                cwd=cwd,
                env=self.build_env(),
                capture_output=True,
                text=True,
                timeout=timeout,
            )
            return ShellResult(
                output=proc.stdout,
                error=proc.stderr if (proc.returncode != 0 and proc.stderr) else None,
                exit_code=proc.returncode,
            )
        except subprocess.TimeoutExpired as e:
            # 保留超时前的部分输出（对调试有价值）
            stdout_raw = e.stdout if e.stdout else None
            stderr_raw = e.stderr if e.stderr else None
            # text=True 时 stdout 已是 str；bytes 时需 decode
            if isinstance(stdout_raw, bytes):
                stdout_raw = stdout_raw.decode("utf-8", errors="replace")
            if isinstance(stderr_raw, bytes):
                stderr_raw = stderr_raw.decode("utf-8", errors="replace")
            partial_stdout = stdout_raw or ""
            partial_stderr = stderr_raw or ""
            return ShellResult(
                output=partial_stdout,
                error=f"Command timed out after {timeout}s"
                      + (f"\nPartial stderr:\n{partial_stderr}" if partial_stderr else ""),
                exit_code=-1,
            )
        except Exception as e:
            return ShellResult(
                error=f"Command failed: {e}",
                exit_code=-1,
            )

    def run_list(
        self,
        args: list[str],
        cwd: str = "",
        timeout: int = 20,
    ) -> ShellResult:
        """Execute a command from a list of arguments (shell=False)。

        与 run() 的区别：不使用 shell 解释器，避免注入和引号问题。
        跨平台安全：Windows/Linux 均可正确处理含空格/特殊字符的参数。

        Args:
            args: 命令参数列表，如 ["rg", "--line-number", "pattern", "path"]。
            cwd: 工作目录。
            timeout: 超时秒数。

        Returns:
            ShellResult with output, error, and exit code.
        """
        cwd = cwd or str(self.root)

        try:
            proc = subprocess.run(
                args,
                shell=False,
                cwd=cwd,
                env=self.build_env(),
                capture_output=True,
                text=True,
                timeout=timeout,
            )
            return ShellResult(
                output=proc.stdout,
                error=proc.stderr if (proc.returncode != 0 and proc.stderr) else None,
                exit_code=proc.returncode,
            )
        except subprocess.TimeoutExpired as e:
            stdout_raw = e.stdout if e.stdout else None
            stderr_raw = e.stderr if e.stderr else None
            if isinstance(stdout_raw, bytes):
                stdout_raw = stdout_raw.decode("utf-8", errors="replace")
            if isinstance(stderr_raw, bytes):
                stderr_raw = stderr_raw.decode("utf-8", errors="replace")
            return ShellResult(
                output=stdout_raw or "",
                error=f"Command timed out after {timeout}s",
                exit_code=-1,
            )
        except FileNotFoundError:
            return ShellResult(
                error=f"Command not found: {args[0]}",
                exit_code=-1,
            )
        except Exception as e:
            return ShellResult(
                error=f"Command failed: {e}",
                exit_code=-1,
            )

    def __repr__(self) -> str:
        return f"ShellExecutor(root={self.root})"
