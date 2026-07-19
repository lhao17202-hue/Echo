"""File system sandbox — path containment, snapshot, diff.

Core guarantee: all file-system operations are confined to the workspace root.

Two-layer defense:
  1. Path.resolve() — resolve relative paths, normalize "..", follow symlinks
  2. os.path.commonpath() — verify the resolved path is inside the workspace root

Implementation note: Echo uses Path.resolve() plus os.path.commonpath() for cross-version path containment.

Both approaches handle: relative paths, "../" escapes, absolute paths, symlinks.
Neither approach prevents TOCTOU races (filesystem state can change between check and use).
"""

import os
import hashlib
from pathlib import Path
from typing import Iterable


class PathEscapedError(ValueError):
    """Raised when a path attempts to escape the workspace boundary.

    This is a ValueError subclass so existing except ValueError handlers
    continue to work while allowing specific catch of path escape events.
    """
    pass


class Sandbox:
    """File system sandbox confining all path operations to the workspace root.

    Usage:
        sandbox = Sandbox("/path/to/project")
        safe_path = sandbox.resolve("src/main.py")       # OK
        safe_path = sandbox.resolve("../etc/passwd")     # raises PathEscapedError
        safe_path = sandbox.resolve("/absolute/outside") # raises PathEscapedError

    The sandbox also provides file-change detection via SHA-256 snapshots
    for tracking which files a tool modified.
    """

    def __init__(self, workspace_root: str):
        """Initialize the sandbox with an absolute workspace root.

        Args:
            workspace_root: Absolute or relative path to the workspace root.
                           Will be resolved to absolute on initialization.
        """
        self.root = Path(workspace_root).resolve()

    # ── Path resolution ───────────────────────────

    def resolve(self, raw_path: str) -> Path:
        """Resolve and validate a user-supplied path against workspace boundary.

        This is THE security-critical method. Every file-system tool MUST
        call this before reading, writing, or listing any path.

        Resolution steps:
          1. If relative, join with workspace root
          2. Call Path.resolve() — normalizes "../", follows symlinks
          3. Use os.path.commonpath() to verify containment

        Args:
            raw_path: Raw path string from user or LLM input.

        Returns:
            Resolved absolute Path (guaranteed to be within workspace).

        Raises:
            PathEscapedError: The resolved path lies outside the workspace root.
        """
        path = Path(raw_path)
        if not path.is_absolute():
            path = self.root / path

        # resolve() follows symlinks and normalizes ".." — this is the first
        # line of defense against path traversal attacks
        resolved = path.resolve()

        # commonpath() returns the longest common prefix path.
        # If the common prefix is NOT exactly the workspace root,
        # the resolved path has escaped outside the sandbox.
        try:
            common = os.path.commonpath([str(self.root), str(resolved)])
        except ValueError:
            # Windows: different drive letters → ValueError
            raise PathEscapedError(f"Path escapes workspace: {raw_path}")

        if common != str(self.root):
            raise PathEscapedError(
                f"Path escapes workspace: {raw_path} -> {resolved}"
            )

        return resolved

    def is_safe(self, raw_path: str) -> bool:
        """Check whether a path is within the workspace without raising.

        Args:
            raw_path: Path to check.

        Returns:
            True if the path is safe, False if it would escape.
        """
        try:
            self.resolve(raw_path)
            return True
        except PathEscapedError:
            return False

    # ── File change detection ──────────────────────

    def snapshot(self, paths: Iterable[str] | None = None) -> dict[str, str]:
        """Take a SHA-256 snapshot of specified files (or entire workspace).

        Used before executing a risky tool to detect what changed.

        安全：显式 paths 也会通过 resolve() 校验，防止路径逃逸。
        性能：跳过大于 10MB 的文件，避免内存问题。

        Args:
            paths: Specific file paths to snapshot. If None, snapshots ALL
                   files in the workspace (excluding .git/).

        Returns:
            Dict mapping absolute path string → SHA-256 hex digest.
        """
        snap: dict[str, str] = {}
        MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB

        if paths:
            targets = [self.resolve(p) for p in paths]  # 走沙箱校验
        else:
            targets = list(self.root.rglob("*"))

        for f in targets:
            if not f.is_file():
                continue
            if ".git" in f.parts:
                continue
            # 跳过过大的文件
            try:
                if f.stat().st_size > MAX_FILE_SIZE:
                    continue
            except (OSError, PermissionError):
                continue
            try:
                snap[str(f)] = hashlib.sha256(f.read_bytes()).hexdigest()
            except (OSError, PermissionError):
                # File might be locked, deleted between iteration and read, etc.
                continue

        return snap

    def diff(self, before: dict[str, str]) -> list[str]:
        """Compare current file hashes against a prior snapshot.

        Detects: modified files, deleted files, AND newly created files.

        Args:
            before: A snapshot dict from a previous Sandbox.snapshot() call.

        Returns:
            List of paths that were modified, created, or deleted since the snapshot.
            Format: "path" for modified, "path (created)" for new, "path (deleted)" for gone.
        """
        changed: list[str] = []
        # 检测修改和删除
        for path_str, old_hash in before.items():
            p = Path(path_str)
            if p.exists() and p.is_file():
                try:
                    new_hash = hashlib.sha256(p.read_bytes()).hexdigest()
                    if new_hash != old_hash:
                        changed.append(path_str)
                except (OSError, PermissionError):
                    changed.append(f"{path_str} (unreadable)")
            else:
                changed.append(f"{path_str} (deleted)")
        # 检测新建文件
        after = self.snapshot()
        for path_str in after:
            if path_str not in before:
                changed.append(f"{path_str} (created)")
        return changed

    # ── Workspace info ─────────────────────────────

    @property
    def git_branch(self) -> str:
        """Get the current git branch name.

        Returns:
            Branch name string, or "unknown" if git is not available.
        """
        import subprocess
        try:
            result = subprocess.run(
                ["git", "branch", "--show-current"],
                cwd=self.root,
                capture_output=True,
                text=True,
                timeout=5,
            )
            return result.stdout.strip() or "unknown"
        except Exception:
            return "unknown"

    @property
    def git_root(self) -> str | None:
        """Get the git repository root, if available.

        Returns:
            Absolute path to git root, or None if not in a git repo.
        """
        import subprocess
        try:
            result = subprocess.run(
                ["git", "rev-parse", "--show-toplevel"],
                cwd=self.root,
                capture_output=True,
                text=True,
                timeout=5,
            )
            return result.stdout.strip() or None
        except Exception:
            return None

    def __repr__(self) -> str:
        return f"Sandbox(root={self.root})"
