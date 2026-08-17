"""Synchronous stdio MCP client sessions."""

from __future__ import annotations

import json
import os
import queue
import subprocess
import threading
import time
from dataclasses import dataclass
from typing import Any

from echo.mcp.config import McpServerConfig


@dataclass(frozen=True)
class McpToolDefinition:
    """Remote MCP tool metadata needed by Echo adapters."""

    name: str
    description: str
    input_schema: dict[str, Any]


class McpClientSession:
    """Persistent stdio MCP session with synchronous public methods."""

    def __init__(self, config: McpServerConfig, request_timeout: float = 30.0):
        self.config = config
        self.request_timeout = request_timeout
        self._process: subprocess.Popen[str] | None = None
        self._next_id = 1
        self._lock = threading.Lock()
        self._responses: dict[int, dict[str, Any]] = {}
        self._stderr_lines: list[str] = []
        self._reader_errors: queue.Queue[BaseException] = queue.Queue()
        self._stdout_thread: threading.Thread | None = None
        self._stderr_thread: threading.Thread | None = None
        self._closed = False

    def initialize(self) -> None:
        """Start the subprocess and initialize the MCP session."""
        self._ensure_process()
        self._request("initialize", {
            "protocolVersion": "2024-11-05",
            "capabilities": {},
            "clientInfo": {"name": "echo", "version": "0.1.0"},
        })
        self._notify("notifications/initialized", {})

    def list_tools(self) -> list[McpToolDefinition]:
        """Return tools exposed by the remote MCP server."""
        result = self._request("tools/list", {})
        tools = result.get("tools", [])
        definitions: list[McpToolDefinition] = []
        for tool in tools:
            if not isinstance(tool, dict):
                continue
            input_schema = tool.get("inputSchema") or tool.get("input_schema")
            definitions.append(McpToolDefinition(
                name=str(tool.get("name", "")),
                description=str(tool.get("description", "") or ""),
                input_schema=input_schema if isinstance(input_schema, dict) else {"type": "object", "properties": {}},
            ))
        return definitions

    def call_tool(self, tool_name: str, arguments: dict[str, Any]) -> Any:
        """Call a remote MCP tool by its original MCP name."""
        return self._request("tools/call", {"name": tool_name, "arguments": arguments})

    def close(self) -> None:
        """Close the MCP subprocess."""
        self._closed = True
        process = self._process
        if process is None:
            return
        if process.stdin:
            try:
                process.stdin.close()
            except OSError:
                pass
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=5)
        self._process = None

    def _ensure_process(self) -> None:
        if self._closed:
            raise RuntimeError(f"MCP server {self.config.name!r} is unavailable")
        if self._process is not None:
            return
        self._process = subprocess.Popen(
            [self.config.command, *self.config.args],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            env=_minimal_env(self.config.env),
        )
        self._stdout_thread = threading.Thread(
            target=self._read_stdout,
            name=f"mcp-{self.config.name}-stdout",
            daemon=True,
        )
        self._stderr_thread = threading.Thread(
            target=self._read_stderr,
            name=f"mcp-{self.config.name}-stderr",
            daemon=True,
        )
        self._stdout_thread.start()
        self._stderr_thread.start()

    def _request(self, method: str, params: dict[str, Any]) -> dict[str, Any]:
        with self._lock:
            request_id = self._next_id
            self._next_id += 1
            self._write({"jsonrpc": "2.0", "id": request_id, "method": method, "params": params})
            response = self._wait_for_response(request_id)
        if "error" in response:
            message = response["error"].get("message", response["error"])
            raise RuntimeError(str(message))
        result = response.get("result", {})
        return result if isinstance(result, dict) else {"content": [{"type": "text", "text": str(result)}]}

    def _notify(self, method: str, params: dict[str, Any]) -> None:
        self._write({"jsonrpc": "2.0", "method": method, "params": params})

    def _write(self, payload: dict[str, Any]) -> None:
        process = self._process
        if process is None or process.stdin is None or process.poll() is not None:
            raise RuntimeError(f"MCP server {self.config.name!r} is unavailable")
        process.stdin.write(json.dumps(payload) + "\n")
        process.stdin.flush()

    def _wait_for_response(self, request_id: int) -> dict[str, Any]:
        deadline = time.monotonic() + self.request_timeout
        while time.monotonic() < deadline:
            if request_id in self._responses:
                return self._responses.pop(request_id)
            self._raise_reader_error_if_any()
            process = self._process
            if process is not None and process.poll() is not None:
                stderr = "\n".join(self._stderr_lines[-10:])
                raise RuntimeError(f"MCP server {self.config.name!r} exited with code {process.returncode}: {stderr}")
            time.sleep(0.01)
        raise TimeoutError(f"MCP request to {self.config.name!r} timed out")

    def _read_stdout(self) -> None:
        assert self._process is not None and self._process.stdout is not None
        try:
            for line in self._process.stdout:
                if not line.strip():
                    continue
                message = json.loads(line)
                message_id = message.get("id")
                if isinstance(message_id, int):
                    self._responses[message_id] = message
        except BaseException as exc:
            if not self._closed:
                self._reader_errors.put(exc)

    def _read_stderr(self) -> None:
        assert self._process is not None and self._process.stderr is not None
        for line in self._process.stderr:
            self._stderr_lines.append(line.rstrip())

    def _raise_reader_error_if_any(self) -> None:
        try:
            exc = self._reader_errors.get_nowait()
        except queue.Empty:
            return
        raise RuntimeError(f"MCP server {self.config.name!r} reader failed: {exc}") from exc


def _minimal_env(explicit_env: dict[str, str]) -> dict[str, str]:
    env: dict[str, str] = {}
    for key in ("PATH", "PATHEXT", "SYSTEMROOT", "COMSPEC"):
        value = os.environ.get(key)
        if value:
            env[key] = value
    env.update(explicit_env)
    return env
