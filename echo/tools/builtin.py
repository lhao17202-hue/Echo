"""内置工具 — read_file, write_file, run_shell, glob, grep, patch_file,
              delegate, todo_write, list_files, search_memory, save_memory, compact。

每个工具遵循"定义 pydantic Params → 继承 BaseTool → 实现 execute() → 返回 ToolResult"模式。
"""

from pathlib import Path
from pydantic import BaseModel, Field

from echo.tools.base import BaseTool, ToolContext, ToolResult


# ═══════════════════════════════════════════════════════
# read_file
# ═══════════════════════════════════════════════════════

class ReadFileParams(BaseModel):
    path: str = Field(..., description="File path relative to workspace root")
    start: int = Field(default=1, description="Starting line (1-indexed)")
    end: int = Field(default=200, description="Ending line (inclusive)")


class ReadFileTool(BaseTool):
    name = "read_file"
    description = "Read a file from the workspace. Returns content with line numbers."
    risk_level = "safe"
    is_read_only = True
    params_model = ReadFileParams

    def execute(self, ctx: ToolContext, params: dict) -> ToolResult:
        path = ctx.resolve_path(params["path"])
        if not path.is_file():
            return ToolResult.fail(f"Not a file: {params['path']}")

        lines = path.read_text(encoding="utf-8").splitlines()
        total = len(lines)
        start = max(1, params.get("start", 1))
        end = min(params.get("end", 200), total)
        selected = lines[start - 1:end]

        output = "\n".join(f"{i + start:6}\t{line}" for i, line in enumerate(selected))
        header = f"--- {params['path']} (lines {start}-{end} of {total}) ---\n"

        return ToolResult.ok(
            header + output,
            files_touched=[str(path)],
            memory_notes=[f"Read {params['path']} ({end - start + 1} lines)"],
        )


# ═══════════════════════════════════════════════════════
# write_file
# ═══════════════════════════════════════════════════════

class WriteFileParams(BaseModel):
    path: str = Field(..., description="File path relative to workspace root")
    content: str = Field(..., description="Full file content to write")


class WriteFileTool(BaseTool):
    name = "write_file"
    description = "Write content to a file. Creates parent directories if needed."
    risk_level = "warn"
    is_read_only = False
    params_model = WriteFileParams

    def execute(self, ctx: ToolContext, params: dict) -> ToolResult:
        path = ctx.resolve_path(params["path"])
        if path.is_dir():
            return ToolResult.fail(f"Path is a directory: {params['path']}")

        path.parent.mkdir(parents=True, exist_ok=True)
        content = params["content"]
        path.write_text(content, encoding="utf-8")

        return ToolResult.ok(
            f"Wrote {len(content)} chars to {params['path']}",
            files_touched=[str(path)],
            memory_notes=[f"Wrote {params['path']} ({len(content)} chars)"],
        )


# ═══════════════════════════════════════════════════════
# run_shell
# ═══════════════════════════════════════════════════════

class RunShellParams(BaseModel):
    command: str = Field(..., description="Shell command to execute")
    timeout: int = Field(default=20, description="Timeout in seconds (1-120)")
    cwd: str = Field(default=".", description="Working directory relative to workspace")


class RunShellTool(BaseTool):
    name = "run_shell"
    description = "Run a shell command in the workspace root."
    risk_level = "danger"
    is_read_only = False
    max_timeout = 120
    max_output_chars = 100_000
    params_model = RunShellParams

    def execute(self, ctx: ToolContext, params: dict) -> ToolResult:
        cwd_path = ctx.resolve_path(params.get("cwd", "."))
        result = ctx.shell.run(
            params["command"],
            cwd=str(cwd_path),
            timeout=params.get("timeout", 20),
        )

        if result.error:
            return ToolResult.partial(
                f"Exit {result.exit_code}\nSTDOUT:\n{result.output}\nSTDERR:\n{result.error}",
                error=result.error,
                partial_success=True,
            )
        return ToolResult.ok(result.output)


# ═══════════════════════════════════════════════════════
# glob
# ═══════════════════════════════════════════════════════

class GlobParams(BaseModel):
    pattern: str = Field(..., description="Glob pattern, e.g. 'src/**/*.py'")
    path: str = Field(default=".", description="Search root directory")


class GlobTool(BaseTool):
    name = "glob"
    description = "Find files matching a glob pattern."
    risk_level = "safe"
    is_read_only = True
    params_model = GlobParams

    def execute(self, ctx: ToolContext, params: dict) -> ToolResult:
        search_root = ctx.resolve_path(params.get("path", "."))
        matches = sorted(search_root.glob(params["pattern"]))

        output_lines = []
        for m in matches[:200]:
            try:
                rel = m.relative_to(ctx.workspace_root)
            except ValueError:
                rel = m
            output_lines.append(str(rel))

        output = "\n".join(output_lines) if output_lines else "No files matched."
        if len(matches) > 200:
            output += f"\n... and {len(matches) - 200} more files"

        return ToolResult.ok(output)


# ═══════════════════════════════════════════════════════
# grep
# ═══════════════════════════════════════════════════════

class GrepParams(BaseModel):
    pattern: str = Field(..., description="Regular expression to search for")
    path: str = Field(default=".", description="Search directory")
    glob: str = Field(default="", description="File filter, e.g. '*.py'")


class GrepTool(BaseTool):
    name = "grep"
    description = "Search file contents using regex via ripgrep."
    risk_level = "safe"
    is_read_only = True
    params_model = GrepParams

    def execute(self, ctx: ToolContext, params: dict) -> ToolResult:
        search_root = ctx.resolve_path(params.get("path", "."))
        pattern = params["pattern"]

        # 使用 run_list（shell=False）避免 Windows/Linux 引号兼容问题
        args = ["rg", "--line-number", "--color=never", pattern, str(search_root)]
        if params.get("glob"):
            args.extend(["--glob", params["glob"]])

        result = ctx.shell.run_list(args, timeout=10)
        output = result.output[:self.max_output_chars] if result.output else "No matches found."

        if result.error and not result.output:
            return ToolResult.fail(result.error, output=output)
        return ToolResult.ok(output)


# ═══════════════════════════════════════════════════════
# patch_file
# ═══════════════════════════════════════════════════════

class PatchFileParams(BaseModel):
    path: str = Field(..., description="File path")
    old_text: str = Field(..., description="Exact text to replace")
    new_text: str = Field(..., description="Replacement text")


class PatchFileTool(BaseTool):
    name = "patch_file"
    description = "Replace one exact text block in a file. old_text must be unique."
    risk_level = "warn"
    is_read_only = False
    params_model = PatchFileParams

    def execute(self, ctx: ToolContext, params: dict) -> ToolResult:
        path = ctx.resolve_path(params["path"])
        if not path.is_file():
            return ToolResult.fail(f"Not a file: {params['path']}")

        content = path.read_text(encoding="utf-8")
        old = params["old_text"]
        count = content.count(old)
        if count == 0:
            return ToolResult.fail(f"old_text not found in {params['path']}")
        if count > 1:
            return ToolResult.fail(
                f"old_text found {count} times in {params['path']} — must be unique"
            )

        new_content = content.replace(old, params["new_text"], 1)
        path.write_text(new_content, encoding="utf-8")

        return ToolResult.ok(
            f"Patched {params['path']}: 1 replacement "
            f"({len(old)} -> {len(params['new_text'])} chars)",
            files_touched=[str(path)],
        )


# ═══════════════════════════════════════════════════════
# delegate
# ═══════════════════════════════════════════════════════

class DelegateParams(BaseModel):
    task: str = Field(..., description="Task for the sub-agent")
    max_steps: int = Field(default=5, description="Max steps the sub-agent can take")


class DelegateTool(BaseTool):
    name = "delegate"
    description = "Spawn a read-only sub-agent to research a subtask independently."
    risk_level = "safe"
    is_read_only = True
    params_model = DelegateParams

    def execute(self, ctx: ToolContext, params: dict) -> ToolResult:
        if ctx.depth >= ctx.max_depth:
            return ToolResult.fail("Max delegation depth reached")

        if ctx.llm is None or ctx.tool_registry is None:
            return ToolResult.fail("Delegate: llm or tool_registry not available in ToolContext")

        from echo.multi_agent.sub_agent import SubAgent

        # 筛选只读工具 + 排除 delegate（防递归）
        all_tools = ctx.tool_registry.get_all()
        read_only = [
            t for t in all_tools
            if t.is_read_only and t.name != "delegate"
        ]

        if not read_only:
            return ToolResult.fail("Delegate: no read-only tools available")

        child_ctx = ToolContext(
            workspace_root=ctx.workspace_root,
            sandbox=ctx.sandbox,
            shell=ctx.shell,
            memory=ctx.memory,
            run_id=ctx.run_id,
            trace_logger=ctx.trace_logger,
            depth=ctx.depth + 1,
            max_depth=ctx.max_depth,
        )

        try:
            max_steps = max(1, min(int(params.get("max_steps", 5)), 30))

            # trace: delegate_started
            if ctx.trace_logger:
                ctx.trace_logger.log(
                    "delegate_started",
                    run_id=ctx.run_id,
                    task=params["task"][:300],
                    depth=ctx.depth,
                    max_steps=max_steps,
                    allowed_tools=[t.name for t in read_only],
                )

            sub = SubAgent(llm=ctx.llm, tools=read_only, ctx=child_ctx)
            summary = sub.run(params["task"], max_steps=max_steps)

            # trace: delegate_finished (success)
            if ctx.trace_logger:
                ctx.trace_logger.log(
                    "delegate_finished",
                    run_id=ctx.run_id,
                    task=params["task"][:300],
                    success=True,
                    summary_preview=summary[:300],
                )

            return ToolResult.ok(
                summary,
                memory_notes=[f"Delegate: {params['task'][:200]}"],
            )
        except Exception as e:
            # trace: delegate_finished (failure)
            if ctx.trace_logger:
                ctx.trace_logger.log(
                    "delegate_finished",
                    run_id=ctx.run_id,
                    task=params.get("task", "")[:300],
                    success=False,
                    error=str(e)[:300],
                )
            return ToolResult.fail(f"Sub-agent failed: {e}")


# ═══════════════════════════════════════════════════════
# todo_write
# ═══════════════════════════════════════════════════════

class TodoWriteParams(BaseModel):
    todos: list[dict] = Field(
        ...,
        description="Task list: [{content, status, activeForm}]",
    )


class TodoWriteTool(BaseTool):
    name = "todo_write"
    description = "Create and update a structured task list."
    risk_level = "safe"
    is_read_only = False
    params_model = TodoWriteParams

    VALID_STATUSES = {"pending", "in_progress", "completed"}

    def execute(self, ctx: ToolContext, params: dict) -> ToolResult:
        todos = params["todos"]
        errors = []
        for i, t in enumerate(todos):
            if not isinstance(t, dict):
                errors.append(f"todo[{i}] is not a dict")
                continue
            if "content" not in t:
                errors.append(f"todo[{i}] missing 'content'")
            status = t.get("status", "pending")
            if status not in self.VALID_STATUSES:
                errors.append(f"todo[{i}] invalid status '{status}'")
        if errors:
            return ToolResult.fail("; ".join(errors))

        # 写入 TaskState（若 ctx 有引用）
        if ctx.task_state is not None and hasattr(ctx.task_state, "todos"):
            ctx.task_state.todos = list(todos)

        lines = [f"- [{t.get('status', 'pending')}] {t.get('content', '')}" for t in todos]
        return ToolResult.ok("\n".join(lines))


# ═══════════════════════════════════════════════════════
# list_files — 列出目录内容
# ═══════════════════════════════════════════════════════

class ListFilesParams(BaseModel):
    path: str = Field(default=".", description="Directory path relative to workspace root")


class ListFilesTool(BaseTool):
    name = "list_files"
    description = "List files and directories in a given path."
    risk_level = "safe"
    is_read_only = True
    params_model = ListFilesParams

    def execute(self, ctx: ToolContext, params: dict) -> ToolResult:
        dir_path = ctx.resolve_path(params.get("path", "."))
        if not dir_path.is_dir():
            return ToolResult.fail(f"Not a directory: {params['path']}")

        items = sorted(dir_path.iterdir(), key=lambda p: (not p.is_dir(), p.name.lower()))
        lines = []
        for item in items[:500]:
            kind = "/" if item.is_dir() else ""
            try:
                size = item.stat().st_size if item.is_file() else 0
            except OSError:
                size = 0
            lines.append(f"{kind:1} {size:>10}  {item.name}")

        output = f"--- {params['path']} ({len(items)} entries) ---\n" + "\n".join(lines)
        if len(items) > 500:
            output += f"\n... and {len(items) - 500} more entries"
        return ToolResult.ok(output)


# ═══════════════════════════════════════════════════════
# search_memory — 搜索记忆
# ═══════════════════════════════════════════════════════

class SearchMemoryParams(BaseModel):
    query: str = Field(..., description="Search query for memory")
    top_k: int = Field(default=5, description="Number of results")


class SearchMemoryTool(BaseTool):
    name = "search_memory"
    description = "Search through Agent memory for relevant information."
    risk_level = "safe"
    is_read_only = True
    params_model = SearchMemoryParams

    def execute(self, ctx: ToolContext, params: dict) -> ToolResult:
        if ctx.memory is None:
            return ToolResult.fail("记忆后端不可用")
        results = ctx.memory.search(params["query"], params.get("top_k", 5))
        if not results:
            return ToolResult.ok("没有匹配的记忆。")
        lines = [f"找到 {len(results)} 条记忆:"]
        for e in results:
            kind_mark = "[durable]" if e.kind == "durable" else "[working]"
            tag_str = ", ".join(e.tags) if e.tags else "无标签"
            lines.append(f"- {kind_mark} [{tag_str}] {e.text[:300]}")
        return ToolResult.ok("\n".join(lines))


# ═══════════════════════════════════════════════════════
# save_memory — 手动保存记忆
# ═══════════════════════════════════════════════════════

class SaveMemoryParams(BaseModel):
    content: str = Field(..., description="Memory content to save")
    tags: list[str] = Field(default_factory=list, description="Tags for categorization")


class SaveMemoryTool(BaseTool):
    name = "save_memory"
    description = "Manually save a piece of information to Agent memory."
    risk_level = "warn"     # 持久写入属状态变更
    is_read_only = False
    params_model = SaveMemoryParams

    def execute(self, ctx: ToolContext, params: dict) -> ToolResult:
        if ctx.memory is None:
            return ToolResult.fail("记忆后端不可用")
        try:
            entry_id = ctx.memory.save_durable(
                params["content"],
                {"tags": params.get("tags", []), "source": "manual"},
            )
            # 不重复写 working memory（memory_notes 会导致主 loop 再写入一次）
            return ToolResult.ok(f"Durable memory saved: {entry_id}")
        except ValueError as e:
            return ToolResult.fail(str(e))


# ═══════════════════════════════════════════════════════
# compact — 触发上下文压缩
# ═══════════════════════════════════════════════════════

class CompactParams(BaseModel):
    """compact 工具不需要参数。"""
    pass


class CompactTool(BaseTool):
    name = "compact"
    description = (
        "Trigger context compaction. Use when the conversation becomes too long. "
        "Previous conversation will be summarized and archived."
    )
    risk_level = "safe"
    is_read_only = False
    params_model = CompactParams

    def execute(self, ctx: ToolContext, params: dict) -> ToolResult:
        # 压缩由 AgentLoop 处理——此工具只是触发信号
        # AgentLoop 检测到 compact 工具被调用后执行 compact_history
        return ToolResult.ok(
            "Compact triggered. Context will be compressed before the next turn.",
            memory_notes=["Context compaction triggered"],
        )


# ═══════════════════════════════════════════════════════
# persistent teammate tools
# ═══════════════════════════════════════════════════════

class SpawnTeammateParams(BaseModel):
    name: str = Field(..., description="Unique teammate name")
    role: str = Field(default="assistant", description="Teammate role")
    prompt: str = Field(default="", description="Additional teammate instructions")


class SpawnTeammateTool(BaseTool):
    name = "spawn_teammate"
    description = "Spawn a long-lived read-only teammate agent for background research tasks."
    risk_level = "warn"
    is_read_only = False
    params_model = SpawnTeammateParams

    def execute(self, ctx: ToolContext, params: dict) -> ToolResult:
        if ctx.teammate_manager is None:
            return ToolResult.fail("Teammate manager unavailable")
        info = ctx.teammate_manager.spawn(
            name=params["name"],
            role=params.get("role", "assistant"),
            prompt=params.get("prompt", ""),
            run_id=ctx.run_id,
            trace_logger=ctx.trace_logger,
        )
        if not info.get("success"):
            return ToolResult.fail(info.get("error", "failed to spawn teammate"))
        return ToolResult.ok(f"Spawned teammate: {info['teammate']}")


class AssignTaskParams(BaseModel):
    teammate: str = Field(..., description="Target teammate name")
    subject: str = Field(..., description="Short task subject")
    description: str = Field(default="", description="Detailed task description")


class AssignTaskTool(BaseTool):
    name = "assign_task"
    description = "Assign a global task to an existing teammate."
    risk_level = "warn"
    is_read_only = False
    params_model = AssignTaskParams

    def execute(self, ctx: ToolContext, params: dict) -> ToolResult:
        if ctx.teammate_manager is None:
            return ToolResult.fail("Teammate manager unavailable")
        try:
            task_id = ctx.teammate_manager.assign_task(
                teammate=params["teammate"],
                subject=params["subject"],
                description=params.get("description", ""),
                run_id=ctx.run_id,
                trace_logger=ctx.trace_logger,
            )
        except Exception as e:
            return ToolResult.fail(str(e))
        return ToolResult.ok(f"Assigned task {task_id} to {params['teammate']}")


class ListTeammatesParams(BaseModel):
    pass


class ListTeammatesTool(BaseTool):
    name = "list_teammates"
    description = "List persistent teammate agents and their current status."
    risk_level = "safe"
    is_read_only = True
    params_model = ListTeammatesParams

    def execute(self, ctx: ToolContext, params: dict) -> ToolResult:
        if ctx.teammate_manager is None:
            return ToolResult.fail("Teammate manager unavailable")
        items = ctx.teammate_manager.list()
        if not items:
            return ToolResult.ok("No teammates running.")
        return ToolResult.ok("\n".join(str(item) for item in items))


class StopTeammateParams(BaseModel):
    name: str = Field(..., description="Teammate name to stop")


class StopTeammateTool(BaseTool):
    name = "stop_teammate"
    description = "Stop a persistent teammate agent after its current task boundary."
    risk_level = "warn"
    is_read_only = False
    params_model = StopTeammateParams

    def execute(self, ctx: ToolContext, params: dict) -> ToolResult:
        if ctx.teammate_manager is None:
            return ToolResult.fail("Teammate manager unavailable")
        ok = ctx.teammate_manager.stop(params["name"])
        if not ok:
            return ToolResult.fail(f"Unknown teammate: {params['name']}")
        return ToolResult.ok(f"Teammate {params['name']} stopped")


class ListGlobalTasksParams(BaseModel):
    pass


class ListGlobalTasksTool(BaseTool):
    name = "list_global_tasks"
    description = "List global tasks shared by lead and teammates."
    risk_level = "safe"
    is_read_only = True
    params_model = ListGlobalTasksParams

    def execute(self, ctx: ToolContext, params: dict) -> ToolResult:
        if ctx.global_tasks is None:
            return ToolResult.fail("Global task manager unavailable")
        tasks = ctx.global_tasks.list_all()
        if not tasks:
            return ToolResult.ok("No global tasks.")
        lines = []
        for task in tasks:
            result_preview = (task.result or "")[:120]
            lines.append(
                f"- {task.task_id} [{task.status}] owner={task.owner_agent or '-'} "
                f"subject={task.subject} result={result_preview}"
            )
        return ToolResult.ok("\n".join(lines))


class WaitGlobalTaskParams(BaseModel):
    task_id: str = Field(..., description="Global task id to wait for")
    timeout_seconds: float = Field(default=10, description="Maximum seconds to wait (0-30)")


class WaitGlobalTaskTool(BaseTool):
    name = "wait_global_task"
    description = "Wait for a global teammate task to complete or fail, then return its result."
    risk_level = "safe"
    is_read_only = True
    max_timeout = 35
    params_model = WaitGlobalTaskParams

    def execute(self, ctx: ToolContext, params: dict) -> ToolResult:
        if ctx.global_tasks is None:
            return ToolResult.fail("Global task manager unavailable")

        task_id = str(params.get("task_id", "")).strip()
        if not task_id:
            return ToolResult.fail("task_id is required")

        try:
            timeout = float(params.get("timeout_seconds", 10))
        except (TypeError, ValueError):
            timeout = 10.0
        timeout = max(0.0, min(timeout, 30.0))

        task = ctx.global_tasks.wait(task_id, timeout=timeout)
        if task is None:
            return ToolResult.fail(f"Unknown global task: {task_id}")

        header = (
            f"Task {task.task_id} [{task.status}] "
            f"owner={task.owner_agent or '-'} subject={task.subject}"
        )
        result = task.result or ""

        if task.status == "completed":
            return ToolResult.ok(f"{header}\n\n{result}".strip())
        if task.status == "failed":
            return ToolResult.fail(f"Global task failed: {task_id}",
                                   output=f"{header}\n\n{result}".strip())

        return ToolResult.partial(
            f"{header}\n\nTask is not complete after {timeout:.1f}s.",
            error=f"Global task not complete: {task.status}",
        )
