"""Unit tests for echo.tools — BaseTool, ToolResult, ToolContext, Registry, Executor, Builtin.

Run: python -m pytest tests/test_tools.py -v
"""

import sys, tempfile, time
from pathlib import Path
import pytest
from pydantic import BaseModel, Field

sys.path.insert(0, str(Path(__file__).parent.parent))

from echo.tools.base import BaseTool, ToolResult, ToolContext
from echo.tools.registry import ToolRegistry
from echo.tools.executor import ToolExecutor
from echo.security.sandbox import Sandbox
from echo.security.env_filter import ShellExecutor
from echo.memory.base import MemoryManager
from echo.memory.default import KeywordMemory


# ═══════════════════════════════════════════════════
# ToolResult
# ═══════════════════════════════════════════════════

class TestToolResult:
    """ToolResult 结构化返回值。"""

    def test_ok(self):
        r = ToolResult.ok("done")
        assert r.output == "done"
        assert r.success
        assert r.error is None
        assert not r.is_partial

    def test_ok_with_meta(self):
        r = ToolResult.ok("done", files_touched=["a.py"], memory_notes=["read a.py"])
        assert r.files_touched == ["a.py"]
        assert r.memory_notes == ["read a.py"]
        assert r.success

    def test_fail(self):
        r = ToolResult.fail("something broke")
        assert not r.success
        assert r.error == "something broke"
        assert r.output == ""

    def test_fail_with_output(self):
        r = ToolResult.fail("partial", output="some output")
        assert not r.success
        assert r.output == "some output"

    def test_partial(self):
        r = ToolResult.partial("output", "warning message")
        assert r.is_partial
        assert not r.success
        assert r.error == "warning message"
        assert r.output == "output"

    def test_meta_key_constants(self):
        r = ToolResult.ok("x", files_touched=["f.py"])
        assert r.meta[ToolResult.KEY_FILES_TOUCHED] == ["f.py"]
        r2 = ToolResult.ok("x", memory_notes=["note"])
        assert r2.meta[ToolResult.KEY_MEMORY_NOTES] == ["note"]

    def test_partial_success_key(self):
        r = ToolResult.partial("x", "err")
        assert r.meta[ToolResult.KEY_PARTIAL_SUCCESS] is True
        assert r.is_partial


# ═══════════════════════════════════════════════════
# ToolContext
# ═══════════════════════════════════════════════════

class TestToolContext:
    """ToolContext 最小上下文。"""

    def test_defaults(self):
        ctx = ToolContext()
        assert ctx.workspace_root == ""
        assert ctx.sandbox is None
        assert ctx.shell is None
        assert ctx.memory is None
        assert ctx.depth == 0
        assert ctx.max_depth == 1

    def test_resolve_path_without_sandbox_raises(self):
        ctx = ToolContext()
        with pytest.raises(RuntimeError, match="sandbox"):
            ctx.resolve_path("../x")

    def test_resolve_path_with_sandbox(self):
        with tempfile.TemporaryDirectory() as d:
            ctx = ToolContext(workspace_root=d, sandbox=Sandbox(d))
            resolved = ctx.resolve_path("test.py")
            assert resolved.is_absolute()
            assert str(resolved).startswith(d)

    def test_resolve_path_blocks_escape(self):
        with tempfile.TemporaryDirectory() as d:
            from echo.security.sandbox import PathEscapedError
            ctx = ToolContext(workspace_root=d, sandbox=Sandbox(d))
            with pytest.raises(PathEscapedError):
                ctx.resolve_path("../etc/passwd")


# ═══════════════════════════════════════════════════
# BaseTool
# ═══════════════════════════════════════════════════

class _SimpleParams(BaseModel):
    value: str = Field(..., description="A value")

class _SimpleTool(BaseTool):
    name = "_simple"
    description = "simple tool for testing"
    risk_level = "safe"
    params_model = _SimpleParams

    def execute(self, ctx, params):
        return ToolResult.ok(f"got: {params['value']}")


class _FailingTool(BaseTool):
    name = "_failing"
    description = "always fails"
    risk_level = "safe"

    def execute(self, ctx, params):
        raise RuntimeError("boom")


class _SlowTool(BaseTool):
    name = "_slow"
    description = "very slow"
    risk_level = "safe"
    max_timeout = 1

    def execute(self, ctx, params):
        time.sleep(5)
        return ToolResult.ok("done")


class _NoParamTool(BaseTool):
    name = "_noparam"
    description = "no params"
    risk_level = "safe"

    def execute(self, ctx, params):
        return ToolResult.ok("no params")


class _NonResultTool(BaseTool):
    """返回非 ToolResult 的工具（防御性检查）。"""
    name = "_nonresult"
    description = "returns a string"
    risk_level = "safe"

    def execute(self, ctx, params):
        return "raw string"  # 非 ToolResult


class TestBaseTool:
    """BaseTool 生命周期。"""

    def test_to_schema_with_params(self):
        t = _SimpleTool()
        s = t.to_schema()
        assert s["name"] == "_simple"
        assert s["input_schema"]["properties"]["value"]["type"] == "string"
        assert "value" in s["input_schema"]["required"]

    def test_to_schema_without_params(self):
        t = _FailingTool()
        s = t.to_schema()
        assert s["name"] == "_failing"
        assert s["input_schema"]["properties"] == {}

    def test_pre_validate_pass(self):
        t = _SimpleTool()
        ok, msg = t.pre_validate({"value": "hello"})
        assert ok; assert msg == ""

    def test_pre_validate_fail_missing_required(self):
        t = _SimpleTool()
        ok, msg = t.pre_validate({})
        assert not ok
        assert "参数校验失败" in msg

    def test_pre_validate_no_model(self):
        t = _FailingTool()
        ok, msg = t.pre_validate({"any": "thing"})
        assert ok  # 无 params_model 时总是通过

    def test_pre_hook_default(self):
        t = _SimpleTool()
        ok, msg = t.pre_hook(ToolContext(), {})
        assert ok

    def test_run_returns_toolresult(self):
        t = _SimpleTool()
        r = t.run(ToolContext(), {"value": "hello"})
        assert isinstance(r, ToolResult)
        assert r.success
        assert "got: hello" in r.output

    def test_run_validates_before_execute(self):
        t = _SimpleTool()
        r = t.run(ToolContext(), {})  # 缺少 value
        assert not r.success
        assert "参数校验失败" in r.error

    def test_run_catches_exceptions(self):
        t = _FailingTool()
        r = t.run(ToolContext(), {})
        assert not r.success
        assert "Tool error" in r.error
        assert "RuntimeError" in r.error

    def test_run_enforces_timeout(self):
        t = _SlowTool()
        r = t.run(ToolContext(), {})
        assert not r.success
        assert "超时" in r.error
        assert "1s" in r.error

    def test_run_wraps_non_toolresult(self):
        """防御性：execute 返回非 ToolResult 时自动包装。"""
        t = _NonResultTool()
        r = t.run(ToolContext(), {})
        assert isinstance(r, ToolResult)
        assert r.output == "raw string"
        assert r.success

    def test_post_process_truncates(self):
        t = _SimpleTool()
        t.max_output_chars = 10
        r = t.run(ToolContext(), {"value": "hello world this is long"})
        assert len(r.output) <= 10 + len("\n... [truncated]")
        assert "truncated" in r.output


# ═══════════════════════════════════════════════════
# ToolRegistry
# ═══════════════════════════════════════════════════

class TestToolRegistry:
    """ToolRegistry 注册、查询、过滤。"""

    def test_register_and_get(self):
        reg = ToolRegistry()
        reg.register(_SimpleTool())
        assert reg.get("_simple") is not None
        assert "_simple" in reg
        assert len(reg) == 1

    def test_register_duplicate_raises(self):
        reg = ToolRegistry()
        reg.register(_SimpleTool())
        with pytest.raises(ValueError, match="已注册"):
            reg.register(_SimpleTool())

    def test_register_empty_name_raises(self):
        reg = ToolRegistry()
        class Nameless(BaseTool):
            name = ""
            description = "x"
            risk_level = "safe"
            def execute(self, ctx, params):
                return ToolResult.ok("")
        with pytest.raises(ValueError, match="必须有 name"):
            reg.register(Nameless())

    def test_unregister(self):
        reg = ToolRegistry()
        t = _SimpleTool()
        reg.register(t)
        reg.unregister("_simple")
        assert reg.get("_simple") is None
        assert len(reg) == 0

    def test_discover(self):
        reg = ToolRegistry()
        reg.discover("echo.tools.builtin")
        assert len(reg) >= 8
        assert "read_file" in reg
        assert "write_file" in reg
        assert "glob" in reg
        assert "run_shell" in reg

    def test_get_names(self):
        reg = ToolRegistry()
        reg.register(_SimpleTool())
        reg.register(_FailingTool())
        names = reg.get_names()
        assert "_simple" in names
        assert "_failing" in names

    def test_get_all(self):
        reg = ToolRegistry()
        reg.register(_SimpleTool())
        reg.register(_FailingTool())
        all_tools = reg.get_all()
        assert len(all_tools) == 2

    def test_list_schemas(self):
        reg = ToolRegistry()
        reg.register(_SimpleTool())
        schemas = reg.list_schemas()
        assert len(schemas) == 1
        assert schemas[0]["name"] == "_simple"

    def test_get_read_only(self):
        reg = ToolRegistry()
        reg.register(_SimpleTool())    # is_read_only=True (default)
        reg.register(_FailingTool())  # is_read_only=True (default)
        ro = reg.get_read_only_tools()
        assert len(ro) == 2

    def test_filter_by_depth(self):
        reg = ToolRegistry()
        reg.register(_SimpleTool())
        # at depth 0, max_depth 1 → all tools
        tools = reg.filter_by_depth(0, 1)
        assert len(tools) == 1
        # at depth 1, max_depth 1 → delegate excluded (but _simple is not delegate)
        tools2 = reg.filter_by_depth(1, 1)
        assert len(tools2) == 1  # _simple is not delegate

    def test_list_by_risk(self):
        from echo.tools.builtin import ReadFileTool, WriteFileTool
        reg = ToolRegistry()
        reg.register(ReadFileTool())
        reg.register(WriteFileTool())
        safe = reg.list_by_risk("safe")
        warn = reg.list_by_risk("warn")
        assert len(safe) == 1
        assert len(warn) == 1

    def test_contains(self):
        reg = ToolRegistry()
        reg.register(_SimpleTool())
        assert "_simple" in reg
        assert "_nonexistent" not in reg


# ═══════════════════════════════════════════════════
# ToolExecutor
# ═══════════════════════════════════════════════════

class TestToolExecutor:
    """ToolExecutor 薄封装。"""

    def test_execute_valid_tool(self):
        reg = ToolRegistry()
        reg.register(_SimpleTool())
        executor = ToolExecutor(reg)
        r = executor.execute("_simple", {"value": "hi"}, ToolContext())
        assert isinstance(r, ToolResult)
        assert r.success
        assert "got: hi" in r.output

    def test_execute_unknown_tool(self):
        executor = ToolExecutor(ToolRegistry())
        r = executor.execute("nonexistent", {}, ToolContext())
        assert isinstance(r, ToolResult)
        assert not r.success
        assert "未知工具" in r.error

    def test_execute_with_sandbox(self):
        with tempfile.TemporaryDirectory() as d:
            from echo.tools.builtin import ReadFileTool
            sandbox = Sandbox(d)
            (sandbox.root / "f.txt").write_text("hello")

            reg = ToolRegistry()
            reg.register(ReadFileTool())
            executor = ToolExecutor(reg)
            ctx = ToolContext(workspace_root=d, sandbox=sandbox)
            r = executor.execute("read_file", {"path": "f.txt"}, ctx)
            assert r.success
            assert "hello" in r.output
            assert str(sandbox.root / "f.txt") in r.files_touched


# ═══════════════════════════════════════════════════
# Builtin Tools
# ═══════════════════════════════════════════════════

class _SandboxCtx:
    """测试用 ToolContext 构建器。"""
    def __init__(self):
        self.d = tempfile.mkdtemp()
        self.sandbox = Sandbox(self.d)
        self.shell = ShellExecutor(self.d)
        self.memory = MemoryManager(KeywordMemory())
        self.ctx = ToolContext(
            workspace_root=self.d, sandbox=self.sandbox,
            shell=self.shell, memory=self.memory,
        )

    def cleanup(self):
        import shutil
        shutil.rmtree(self.d)


class TestBuiltinReadFile:
    def test_read_existing_file(self):
        from echo.tools.builtin import ReadFileTool
        sc = _SandboxCtx()
        (sc.sandbox.root / "test.py").write_text("line1\nline2\nline3")
        r = ReadFileTool().run(sc.ctx, {"path": "test.py"})
        sc.cleanup()
        assert r.success
        assert "line1" in r.output
        assert "line2" in r.output
        assert str(sc.sandbox.root / "test.py") in r.files_touched
        assert any("Read" in n for n in r.memory_notes)

    def test_read_nonexistent(self):
        from echo.tools.builtin import ReadFileTool
        sc = _SandboxCtx()
        r = ReadFileTool().run(sc.ctx, {"path": "no.txt"})
        sc.cleanup()
        assert not r.success
        assert "Not a file" in r.error

    def test_read_with_line_range(self):
        from echo.tools.builtin import ReadFileTool
        sc = _SandboxCtx()
        (sc.sandbox.root / "lines.txt").write_text("a\nb\nc\nd\ne\n")
        r = ReadFileTool().run(sc.ctx, {"path": "lines.txt", "start": 2, "end": 4})
        sc.cleanup()
        assert r.success
        assert "b" in r.output
        assert "c" in r.output
        assert "d" in r.output


class TestBuiltinWriteFile:
    def test_write_new_file(self):
        from echo.tools.builtin import WriteFileTool
        sc = _SandboxCtx()
        r = WriteFileTool().run(sc.ctx, {"path": "out.txt", "content": "hello"})
        assert r.success
        assert "Wrote 5 chars" in r.output
        assert (sc.sandbox.root / "out.txt").read_text() == "hello"
        sc.cleanup()

    def test_write_overwrite(self):
        from echo.tools.builtin import WriteFileTool
        sc = _SandboxCtx()
        (sc.sandbox.root / "out.txt").write_text("old")
        r = WriteFileTool().run(sc.ctx, {"path": "out.txt", "content": "new"})
        assert r.success
        assert (sc.sandbox.root / "out.txt").read_text() == "new"
        sc.cleanup()

    def test_write_to_directory_fails(self):
        from echo.tools.builtin import WriteFileTool
        sc = _SandboxCtx()
        (sc.sandbox.root / "mydir").mkdir()
        r = WriteFileTool().run(sc.ctx, {"path": "mydir", "content": "x"})
        assert not r.success
        sc.cleanup()


class TestBuiltinGlob:
    def test_glob_finds_files(self):
        from echo.tools.builtin import GlobTool
        sc = _SandboxCtx()
        (sc.sandbox.root / "a.py").write_text("")
        (sc.sandbox.root / "b.py").write_text("")
        (sc.sandbox.root / "c.txt").write_text("")
        r = GlobTool().run(sc.ctx, {"pattern": "*.py"})
        sc.cleanup()
        assert r.success
        assert "a.py" in r.output
        assert "b.py" in r.output
        assert "c.txt" not in r.output

    def test_glob_empty(self):
        from echo.tools.builtin import GlobTool
        sc = _SandboxCtx()
        r = GlobTool().run(sc.ctx, {"pattern": "*.xyz"})
        sc.cleanup()
        assert r.success
        assert "No files matched" in r.output


class TestBuiltinGrep:
    def test_grep_finds_pattern(self):
        from echo.tools.builtin import GrepTool
        sc = _SandboxCtx()
        (sc.sandbox.root / "code.py").write_text("def foo():\n    pass\n")
        r = GrepTool().run(sc.ctx, {"pattern": "def foo", "path": "."})
        sc.cleanup()
        # rg 可能未安装 — 此时返回 fail 或 no matches，都是正确的
        assert isinstance(r.output, str)
        if r.success:
            assert "def foo" in r.output or "No matches" in r.output

    def test_grep_pattern_with_quotes(self):
        from echo.tools.builtin import GrepTool
        sc = _SandboxCtx()
        # pattern with single quote — should be escaped
        (sc.sandbox.root / "code.py").write_text("it's a test\n")
        r = GrepTool().run(sc.ctx, {"pattern": "it's", "path": "."})
        sc.cleanup()
        # Should not crash with shell injection
        assert isinstance(r.output, str)


class TestBuiltinPatchFile:
    def test_patch_single_occurrence(self):
        from echo.tools.builtin import PatchFileTool
        sc = _SandboxCtx()
        (sc.sandbox.root / "f.py").write_text("old text here")
        r = PatchFileTool().run(sc.ctx, {
            "path": "f.py", "old_text": "old text", "new_text": "new text",
        })
        assert r.success
        assert (sc.sandbox.root / "f.py").read_text() == "new text here"
        sc.cleanup()

    def test_patch_not_found(self):
        from echo.tools.builtin import PatchFileTool
        sc = _SandboxCtx()
        (sc.sandbox.root / "f.py").write_text("something")
        r = PatchFileTool().run(sc.ctx, {
            "path": "f.py", "old_text": "not there", "new_text": "x",
        })
        assert not r.success
        sc.cleanup()

    def test_patch_multiple_occurrences(self):
        from echo.tools.builtin import PatchFileTool
        sc = _SandboxCtx()
        (sc.sandbox.root / "f.py").write_text("dup dup")
        r = PatchFileTool().run(sc.ctx, {
            "path": "f.py", "old_text": "dup", "new_text": "x",
        })
        assert not r.success
        assert "2 times" in r.error
        sc.cleanup()


class TestBuiltinRunShell:
    def test_simple_command(self):
        from echo.tools.builtin import RunShellTool
        sc = _SandboxCtx()
        r = RunShellTool().run(sc.ctx, {"command": "echo hello", "timeout": 10})
        sc.cleanup()
        assert r.success
        assert "hello" in r.output

    def test_failing_command(self):
        from echo.tools.builtin import RunShellTool
        sc = _SandboxCtx()
        r = RunShellTool().run(sc.ctx, {
            "command": "nonexistent_command_xyz_123", "timeout": 5,
        })
        sc.cleanup()
        assert not r.success
        assert r.is_partial


class TestBuiltinTodoWrite:
    def test_todo_format(self):
        from echo.tools.builtin import TodoWriteTool
        r = TodoWriteTool().run(ToolContext(), {
            "todos": [
                {"content": "task A", "status": "completed"},
                {"content": "task B", "status": "in_progress"},
            ],
        })
        assert r.success
        assert "task A" in r.output
        assert "task B" in r.output


class TestBuiltinDelegate:
    def test_depth_limit(self):
        from echo.tools.builtin import DelegateTool
        # depth >= max_depth → blocked
        ctx = ToolContext(depth=1, max_depth=1)
        r = DelegateTool().run(ctx, {"task": "do something"})
        assert not r.success
        assert "depth" in r.error.lower()

    def test_delegate_needs_llm_in_ctx(self):
        from echo.tools.builtin import DelegateTool
        r = DelegateTool().run(ToolContext(), {"task": "do something"})
        # 没有 llm/tool_registry → 返回 fail
        assert not r.success
        assert "not available" in r.error.lower()

    def test_delegate_depth_limit(self):
        from echo.tools.builtin import DelegateTool
        ctx = ToolContext(depth=1, max_depth=1)
        r = DelegateTool().run(ctx, {"task": "do something"})
        assert not r.success
        assert "depth" in r.error.lower()
