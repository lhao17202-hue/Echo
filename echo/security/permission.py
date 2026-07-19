"""Permission guard — risk-level-based tool authorization + command deny lists.

Three authorization layers:
  Layer A — Risk level on tool definition (safe/warn/danger):
    safe   → auto-approve (read_file, glob, grep)
    warn   → interactive confirmation (write_file, patch_file)
    danger → default deny unless explicitly authorized (run_shell)

  Layer B — DENY_LIST for shell commands:
    Hard-blocked patterns that are never allowed, regardless of risk level.
    Based on substring matching of the entire command string.

  Layer C — DESTRUCTIVE list for shell commands:
    Interactive-prompt patterns for potentially destructive operations.
    User must type 'y' or 'yes' to proceed.

Design note (from v0.3 feedback):
  PermissionGuard is the SINGLE entry point for authorization.
  BaseTool.pre_hook() does NOT duplicate permission checks.
  Permission checks happen exclusively through HookManager("pre_tool_use").
"""

from echo.security.sandbox import PathEscapedError
from echo.security.sandbox import Sandbox, PathEscapedError

# ── Layer A: Risk-level strategies ─────────────────

class PermissionGuard:
    """Unified permission check entry point.

    Called from PermissionHook (pre_tool_use event) — the sole authorization
    gate in the agent loop. BaseTool does not call this independently.
    """

    # ── Risk-level resolution ──────────────────────

    @classmethod
    def check(cls, risk_level: str, tool_name: str) -> tuple[bool, str]:
        """Check whether a tool is authorized based on its risk level.

        Args:
            risk_level: One of "safe", "warn", "danger".
            tool_name: The tool's name (for the confirmation message).

        Returns:
            (allowed: bool, message: str)
            - (True, "") for safe tools — proceed without prompt.
            - (False, "Approve...") for warn tools — caller should prompt user.
            - (False, "DANGER...") for danger tools — caller should prompt user
              with elevated warning.

        Raises:
            ValueError: risk_level 不是 "safe" / "warn" / "danger"。
        """
        if risk_level == "safe":
            return True, ""
        elif risk_level == "warn":
            return False, f"Approve running '{tool_name}'?"
        elif risk_level == "danger":
            return False, f"DANGER: '{tool_name}' requires explicit authorization."
        else:
            raise ValueError(f"无效的 risk_level: '{risk_level}'（应为 safe/warn/danger）")

    # ── Shell command safety ───────────────────────

    # Commands / patterns that are NEVER allowed (hard block, no override).
    # These use substring containment — the pattern appearing ANYWHERE in the
    # command string triggers a denial.
    #
    # Note on substring matching: this is intentionally broad. A pattern
    # like "sudo" will match "pseudo" — false positives on deny are
    # safer than false negatives on allow.
    DENY_LIST: tuple[str, ...] = (
        "rm -rf /",
        "rm -rf --no-preserve-root",
        "sudo rm",
        "shutdown",
        "reboot",
        "mkfs",
        "dd if=",
        ":(){ :|:& };:",       # fork bomb
        "> /dev/sda",
        "chmod 777 /",
        "chmod -R 777 /",
    )

    DESTRUCTIVE: tuple[str, ...] = (
        "rm ",
        "rm -rf",
        "> /etc/",
        "chmod 777",
        "chown ",
        "mv /",
        "curl ",
        "wget ",
        " | sh",
    )

    @classmethod
    def check_shell_command(cls, command: str | None) -> tuple[bool, str]:
        """Check a shell command against deny and destructive lists.

        Called BEFORE shell execution. This is defense-in-depth on top
        of the risk-level check — even if the tool is authorized at the
        risk level, individual dangerous commands may still be blocked.

        Args:
            command: The full shell command string to execute.

        Returns:
            (allowed: bool, message: str)
            - (False, "Permission denied: ...") for deny-listed patterns.
            - (False, "Destructive command: ...") for destructive patterns
              requiring user confirmation.
            - (True, "") if the command passes all checks.
        """
        if not command:
            return True, ""
        cmd_lower = command.lower()

        # Layer B: DENY_LIST — hard block, no override
        for pattern in cls.DENY_LIST:
            if pattern in cmd_lower:
                return False, f"Permission denied: '{pattern}' is on the deny list."

        # Layer C: DESTRUCTIVE — interactive prompt
        for pattern in cls.DESTRUCTIVE:
            if pattern in cmd_lower:
                return False, f"Destructive command detected ('{pattern}'). Approve '{command[:120]}'?"

        return True, ""

    @classmethod
    def is_denied(cls, command: str | None) -> bool:
        """快速检查：命令是否在任何 DENY_LIST 模式中（硬阻止，无覆盖）。"""
        if not command:
            return False
        cmd_lower = command.lower()
        return any(pattern in cmd_lower for pattern in cls.DENY_LIST)

    # ── Path safety ────────────────────────────────

    @classmethod
    def check_path(cls, raw_path: str, sandbox:Sandbox) -> tuple[bool, str]:
        """Verify a file path is within the workspace sandbox.

        This is used as an additional safety check in tools that accept
        user-supplied paths, providing a clear error message when a path
        would escape the workspace.

        Args:
            raw_path: The raw path string to check.
            sandbox: The Sandbox instance defining the workspace boundary.

        Returns:
            (True, "") if the path is within the workspace.
            (False, "Permission denied: ...") if the path escapes.
        """
        try:
            sandbox.resolve(raw_path)
            return True, ""
        except PathEscapedError as e:
            return False, f"Permission denied: path escapes workspace: {raw_path} — {e}"
        except Exception as e:
            return False, f"Path validation error: {e}"
