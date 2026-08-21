"""Dependency providers for Echo Web API."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path
from threading import Lock
from uuid import uuid4

from echo.config import EchoConfig
from echo.server.approvals import WebApprovalManager
from echo.server.schemas import (
    ApprovalDecisionResponse,
    ApprovalPolicyUpdateRequest,
    ApprovalRequestDTO,
    ChatResponse,
    ConfigSummary,
    GitStatus,
    MessageDTO,
    RuntimeStatus,
    SessionDetail,
    SessionSummary,
    ToolCallSummary,
    TraceEventDTO,
    WorkspaceInfo,
    RunFileDiff,
    RunFileSummary,
)


class EchoService:
    """Thin service boundary around the Echo runtime."""

    def chat(self, message: str, session_id: str | None = None) -> ChatResponse:
        raise NotImplementedError

    def list_sessions(self, query: str | None = None) -> list[SessionSummary]:
        raise NotImplementedError

    def get_session(self, session_id: str) -> SessionDetail:
        raise NotImplementedError

    def rename_session(self, session_id: str, title: str) -> SessionSummary:
        raise NotImplementedError

    def delete_session(self, session_id: str) -> None:
        raise NotImplementedError

    def get_run_trace(self, run_id: str) -> list[TraceEventDTO]:
        raise NotImplementedError

    def get_run_files(self, run_id: str) -> list[RunFileSummary]:
        raise NotImplementedError

    def get_run_file_diff(self, run_id: str, file_path: str) -> RunFileDiff:
        raise NotImplementedError

    def get_workspace_info(self) -> WorkspaceInfo:
        raise NotImplementedError

    def get_git_status(self) -> GitStatus:
        raise NotImplementedError

    def get_config_summary(self) -> ConfigSummary:
        raise NotImplementedError

    def get_runtime_status(self) -> RuntimeStatus:
        raise NotImplementedError

    def update_approval_policy(self, approval_policy: str) -> ConfigSummary:
        raise NotImplementedError

    def pending_approvals(self) -> list[ApprovalRequestDTO]:
        raise NotImplementedError

    def decide_approval(self, request_id: str, approved: bool) -> ApprovalDecisionResponse:
        raise NotImplementedError


class DefaultEchoService(EchoService):
    def __init__(self, runtime=None, approval_manager: WebApprovalManager | None = None):
        self.runtime = runtime
        self.approval_manager = approval_manager or WebApprovalManager()
        self._lock = Lock()

    def _runtime(self):
        if self.runtime is None:
            from echo.core.echo import Echo

            workspace = Path.cwd().resolve()
            self.runtime = Echo(workspace_root=str(workspace), config=EchoConfig.from_env())
        return self.runtime

    def chat(self, message: str, session_id: str | None = None) -> ChatResponse:
        with self._lock:
            runtime = self._runtime()
            self._install_approval_handler(runtime)
            try:
                if session_id:
                    answer = runtime.resume(session_id, message)
                    resolved_session_id = session_id
                else:
                    answer = runtime.ask(message)
                    resolved_session_id = self._latest_session_id(runtime) or f"session_{uuid4().hex[:8]}"
            except FileNotFoundError:
                resolved_session_id = session_id or self._latest_session_id(runtime) or f"session_{uuid4().hex[:8]}"
                return self._failed_response(
                    resolved_session_id,
                    "会话不存在或已失效，请新建对话后重试。",
                    runtime,
                )
            except Exception as exc:
                resolved_session_id = session_id or self._latest_session_id(runtime) or f"session_{uuid4().hex[:8]}"
                return self._failed_response(
                    resolved_session_id,
                    f"Echo 后端运行失败：{exc}",
                    runtime,
                )

            run_id = self._current_run_id(runtime)
            trace = self.get_run_trace(run_id)
            answer_text = str(answer)
            status = "failed" if answer_text.startswith("Stopped:") else "completed"
            return ChatResponse(
                session_id=resolved_session_id,
                run_id=run_id,
                answer=answer_text,
                status=status,
                trace=trace,
                tools=self._tools_from_trace(trace),
                files_touched=self._files_from_trace(trace),
            )

    @staticmethod
    def _latest_session_id(runtime) -> str:
        try:
            return str(runtime.session_store.latest() or "")
        except (AttributeError, FileNotFoundError, OSError):
            return ""

    @staticmethod
    def _current_run_id(runtime) -> str:
        return getattr(getattr(runtime, "run_store", None), "current_run_id", "") or f"run_{uuid4().hex[:8]}"

    def _install_approval_handler(self, runtime) -> None:
        setattr(runtime, "approval_handler", self.approval_manager.request_approval)

    def _failed_response(self, session_id: str, answer: str, runtime) -> ChatResponse:
        return ChatResponse(
            session_id=session_id,
            run_id=self._current_run_id(runtime),
            answer=answer,
            status="failed",
            trace=[],
            tools=[],
            files_touched=[],
        )

    def list_sessions(self, query: str | None = None) -> list[SessionSummary]:
        runtime = self._runtime()
        sessions = runtime.list_sessions(limit=20)
        titles = self._session_title_overrides()
        summaries: list[SessionSummary] = []
        for item in sessions:
            session_id = str(item.get("session_id", ""))
            if not session_id:
                continue
            title = str(titles.get(session_id) or item.get("title") or self._session_title(session_id))
            summary = SessionSummary(
                session_id=session_id,
                title=title,
                updated_at=item.get("modified_at") or item.get("updated_at") or item.get("created_at") or None,
                run_count=int(item.get("run_count", 0) or 0),
            )
            if query:
                needle = query.lower()
                if needle not in summary.title.lower() and needle not in summary.session_id.lower():
                    continue
            summaries.append(summary)
        return summaries

    def get_session(self, session_id: str) -> SessionDetail:
        runtime = self._runtime()
        try:
            session = runtime.session_store.load(session_id)
        except FileNotFoundError:
            return SessionDetail(session_id=session_id, title="会话不存在或已失效", messages=[])
        messages = self._session_messages(session.history)
        title = messages[0].content[:30] if messages else session_id
        return SessionDetail(session_id=session_id, title=title, messages=messages)

    def _session_title(self, session_id: str) -> str:
        try:
            detail = self.get_session(session_id)
        except (FileNotFoundError, OSError, AttributeError):
            return session_id
        return detail.title or session_id

    def _web_metadata_dir(self) -> Path:
        return self._workspace_path() / ".echo" / "web"

    def _session_titles_path(self) -> Path:
        return self._web_metadata_dir() / "session_titles.json"

    def _session_title_overrides(self) -> dict[str, str]:
        path = self._session_titles_path()
        if not path.exists():
            return {}
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return {}
        if not isinstance(data, dict):
            return {}
        return {str(key): str(value) for key, value in data.items()}

    def _write_session_title_overrides(self, titles: dict[str, str]) -> None:
        path = self._session_titles_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(titles, ensure_ascii=False, indent=2), encoding="utf-8")

    def rename_session(self, session_id: str, title: str) -> SessionSummary:
        titles = self._session_title_overrides()
        titles[session_id] = title
        self._write_session_title_overrides(titles)
        for summary in self.list_sessions():
            if summary.session_id == session_id:
                return SessionSummary(
                    session_id=summary.session_id,
                    title=title,
                    updated_at=summary.updated_at,
                    run_count=summary.run_count,
                )
        return SessionSummary(session_id=session_id, title=title)

    def delete_session(self, session_id: str) -> None:
        sessions_root = (self._workspace_path() / ".echo" / "sessions").resolve()
        target = (sessions_root / session_id).resolve()
        if sessions_root not in target.parents and target != sessions_root:
            raise ValueError("invalid session path")
        if target.exists():
            shutil.rmtree(target)
        titles = self._session_title_overrides()
        if session_id in titles:
            titles.pop(session_id, None)
            self._write_session_title_overrides(titles)

    @staticmethod
    def _session_messages(history: list) -> list[MessageDTO]:
        messages: list[MessageDTO] = []
        for item in history:
            role = str(item.get("role", ""))
            text_parts: list[str] = []
            for block in item.get("content") or []:
                if isinstance(block, dict) and block.get("type") == "text":
                    text = str(block.get("text", "")).strip()
                    if text:
                        text_parts.append(text)
            if role in {"user", "assistant"} and text_parts:
                messages.append(MessageDTO(role=role, content="\n".join(text_parts)))
        return messages

    @staticmethod
    def _tools_from_trace(trace: list[TraceEventDTO]) -> list[ToolCallSummary]:
        tools: list[ToolCallSummary] = []
        for event in trace:
            for item in event.payload.get("tools", []):
                if not isinstance(item, dict):
                    continue
                tools.append(
                    ToolCallSummary(
                        name=str(item.get("name", "unknown")),
                        input_summary=str(item.get("input_summary", "")),
                        success=item.get("success") if isinstance(item.get("success"), bool) else None,
                        output_summary=str(item.get("output_summary", "")),
                    )
                )
            if event.event not in {"tool_finished", "tool_failed", "tool_blocked"}:
                continue
            tool_name = event.payload.get("tool") or event.payload.get("name")
            if not tool_name:
                continue
            tools.append(
                ToolCallSummary(
                    name=str(tool_name),
                    input_summary=str(event.payload.get("input_summary", "")),
                    success=event.payload.get("success") if isinstance(event.payload.get("success"), bool) else None,
                    output_summary=str(event.payload.get("output_summary") or event.payload.get("error_preview") or ""),
                )
            )
        return tools

    @staticmethod
    def _files_from_trace(trace: list[TraceEventDTO]) -> list[str]:
        files: list[str] = []
        for event in trace:
            candidates: list[object] = []
            candidates.extend(event.payload.get("file_changes", []))
            candidates.extend(event.payload.get("files_touched", []))
            candidates.extend(event.payload.get("files_written", []))
            candidates.extend(event.payload.get("files_deleted", []))
            for item in candidates:
                file = str(item)
                if file and file not in files:
                    files.append(file)
        return files

    def get_run_trace(self, run_id: str) -> list[TraceEventDTO]:
        runtime = self._runtime()
        workspace_value = getattr(runtime, "workspace_root", None) or Path.cwd()
        workspace = Path(workspace_value).resolve()
        trace_paths = sorted((workspace / ".echo" / "sessions").glob(f"*/runs/{run_id}/trace.jsonl"))
        if not trace_paths:
            return []

        events: list[TraceEventDTO] = []
        for line in trace_paths[0].read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                data = json.loads(line)
            except json.JSONDecodeError:
                continue
            event = str(data.pop("event", ""))
            if not event:
                continue
            run_id_value = data.pop("run_id", None)
            event_id = data.pop("event_id", None)
            created_at = data.pop("created_at", None)
            timestamp = data.pop("timestamp", None)
            nested_payload = data.pop("payload", None)
            payload = nested_payload if isinstance(nested_payload, dict) else data
            events.append(
                TraceEventDTO(
                    event=event,
                    run_id=run_id_value,
                    event_id=event_id,
                    created_at=created_at,
                    timestamp=timestamp,
                    payload=payload,
                )
            )
        return events

    def get_run_files(self, run_id: str) -> list[RunFileSummary]:
        files: list[RunFileSummary] = []
        for file_path in self._files_from_trace(self.get_run_trace(run_id)):
            files.append(RunFileSummary(path=file_path, status="modified"))
        return files

    def get_run_file_diff(self, run_id: str, file_path: str) -> RunFileDiff:
        workspace = self._workspace_path()
        target = (workspace / file_path).resolve()
        if workspace not in target.parents and target != workspace:
            raise ValueError("invalid file path")
        diff = self._run_git(["diff", "--", file_path], workspace)
        if diff:
            return RunFileDiff(path=file_path, status="modified", diff=diff)
        if target.exists() and target.is_file():
            try:
                return RunFileDiff(path=file_path, status="current", diff=target.read_text(encoding="utf-8"))
            except UnicodeDecodeError:
                return RunFileDiff(path=file_path, status="current", diff="<binary file>")
        return RunFileDiff(path=file_path, status="missing", diff="")

    def _workspace_path(self) -> Path:
        runtime = self._runtime()
        workspace_value = getattr(runtime, "workspace_root", None) or Path.cwd()
        return Path(workspace_value).resolve()

    def get_workspace_info(self) -> WorkspaceInfo:
        workspace = self._workspace_path()
        return WorkspaceInfo(name=workspace.name, root=str(workspace))

    def get_git_status(self) -> GitStatus:
        workspace = self._workspace_path()
        branch = self._run_git(["branch", "--show-current"], workspace).strip() or "unknown"
        status = self._run_git(["status", "--short"], workspace)
        changed_files = []
        for line in status.splitlines():
            if not line.strip():
                continue
            changed_files.append(line[3:].strip() if len(line) > 3 else line.strip())
        return GitStatus(branch=branch, dirty=bool(changed_files), changed_files=changed_files)

    @staticmethod
    def _run_git(args: list[str], cwd: Path) -> str:
        try:
            result = subprocess.run(
                ["git", *args],
                cwd=cwd,
                check=False,
                capture_output=True,
                text=True,
                timeout=5,
            )
        except (OSError, subprocess.SubprocessError):
            return ""
        if result.returncode != 0:
            return ""
        return result.stdout

    def get_config_summary(self) -> ConfigSummary:
        runtime = self._runtime()
        config = getattr(runtime, "config", None) or EchoConfig.from_env()
        provider = str(getattr(config, "provider", ""))
        api_key_name = f"{provider.upper()}_API_KEY" if provider else "API_KEY"
        api_key_configured = bool(os.environ.get(api_key_name) or getattr(config, "api_key", ""))
        return ConfigSummary(
            provider=provider,
            model=str(getattr(config, "model", "")),
            base_url=str(getattr(config, "base_url", "")),
            approval_policy=str(getattr(config, "approval_policy", "")),
            api_key_configured=api_key_configured,
        )

    def get_runtime_status(self) -> RuntimeStatus:
        runtime = self._runtime()
        config = getattr(runtime, "config", None) or EchoConfig.from_env()
        return RuntimeStatus(
            background_tasks=self._safe_len(getattr(runtime, "background_tasks", [])),
            cron_tasks=self._safe_len(getattr(runtime, "cron_tasks", [])),
            mcp_servers=self._safe_len(getattr(runtime, "mcp_servers", [])),
            tools=self._tool_count(runtime),
            approval_policy=str(getattr(config, "approval_policy", "")),
        )

    def update_approval_policy(self, approval_policy: str) -> ConfigSummary:
        allowed = {"ask", "auto", "never", "danger"}
        if approval_policy not in allowed:
            raise ValueError(f"unsupported approval policy: {approval_policy}")
        runtime = self._runtime()
        config = getattr(runtime, "config", None)
        if config is None:
            config = EchoConfig.from_env()
            setattr(runtime, "config", config)
        setattr(config, "approval_policy", approval_policy)
        return self.get_config_summary()

    def pending_approvals(self) -> list[ApprovalRequestDTO]:
        return [
            ApprovalRequestDTO(
                request_id=request.request_id,
                tool_name=request.tool_name,
                risk_level=request.risk_level,
                tool_input=request.tool_input,
                command=request.command,
                status=request.status,
            )
            for request in self.approval_manager.pending()
        ]

    def decide_approval(self, request_id: str, approved: bool) -> ApprovalDecisionResponse:
        request = self.approval_manager.decide(request_id, approved)
        return ApprovalDecisionResponse(request_id=request.request_id, status=request.status)

    @staticmethod
    def _tool_count(runtime) -> int:
        candidates = [
            getattr(runtime, "tool_registry", None),
            getattr(getattr(runtime, "executor", None), "registry", None),
            getattr(runtime, "tools", None),
        ]
        for registry in candidates:
            if registry is None:
                continue
            try:
                return len(registry)
            except TypeError:
                pass
            tools = getattr(registry, "tools", None)
            if tools is not None:
                return DefaultEchoService._safe_len(tools)
            private_tools = getattr(registry, "_tools", None)
            if private_tools is not None:
                return DefaultEchoService._safe_len(private_tools)
        return 0

    @staticmethod
    def _safe_len(value) -> int:
        try:
            if isinstance(value, dict):
                return len(value)
            return len(list(value))
        except TypeError:
            return 0


_service: DefaultEchoService | None = None


def get_echo_service() -> EchoService:
    global _service
    if _service is None:
        _service = DefaultEchoService()
    return _service
