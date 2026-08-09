"""Tests for workspace-local skill loading."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from echo.skills.registry import SkillRegistry


def test_skill_registry_scans_skill_with_frontmatter(tmp_path):
    skill_dir = tmp_path / "skills" / "review"
    skill_dir.mkdir(parents=True)
    (skill_dir / "SKILL.md").write_text(
        "---\n"
        "name: code-review\n"
        "description: Review changed code.\n"
        "---\n\n"
        "# Code Review\n\nFull skill body",
        encoding="utf-8",
    )

    registry = SkillRegistry(tmp_path / "skills").scan()

    skill = registry.get("code-review")
    assert skill is not None
    assert skill.name == "code-review"
    assert skill.description == "Review changed code."
    assert skill.path == skill_dir / "SKILL.md"
    assert "Full skill body" in skill.content
    assert registry.load_content("code-review") == skill.content
    assert registry.names() == ["code-review"]


def test_skill_registry_uses_directory_and_heading_fallback(tmp_path):
    skill_dir = tmp_path / "skills" / "docs-writer"
    skill_dir.mkdir(parents=True)
    (skill_dir / "SKILL.md").write_text(
        "# Documentation Writer\n\nWrite concise docs.",
        encoding="utf-8",
    )

    registry = SkillRegistry(tmp_path / "skills").scan()

    skill = registry.get("docs-writer")
    assert skill is not None
    assert skill.name == "docs-writer"
    assert skill.description == "Documentation Writer"


def test_skill_registry_catalog_is_lightweight(tmp_path):
    skill_dir = tmp_path / "skills" / "demo"
    skill_dir.mkdir(parents=True)
    (skill_dir / "SKILL.md").write_text(
        "---\nname: demo\ndescription: Demo skill\n---\n\nSecret full body details",
        encoding="utf-8",
    )

    catalog = SkillRegistry(tmp_path / "skills").scan().list_catalog()

    assert "- **demo**: Demo skill" in catalog
    assert "Secret full body details" not in catalog


def test_skill_registry_can_disable_and_reenable_skills(tmp_path):
    skill_dir = tmp_path / "skills" / "demo"
    skill_dir.mkdir(parents=True)
    (skill_dir / "SKILL.md").write_text(
        "---\nname: demo\ndescription: Demo skill\n---\n\nFull skill body",
        encoding="utf-8",
    )

    registry = SkillRegistry(tmp_path / "skills").scan()

    assert registry.is_enabled("demo")
    assert registry.names() == ["demo"]
    assert registry.load_content("demo") is not None

    assert registry.disable("demo")
    assert not registry.is_enabled("demo")
    assert registry.names() == []
    assert registry.names(include_disabled=True) == ["demo"]
    assert registry.get("demo") is None
    assert registry.get("demo", include_disabled=True).enabled is False
    assert registry.load_content("demo") is None
    assert registry.list_catalog() == ""

    registry.scan()

    assert not registry.is_enabled("demo")
    assert registry.names() == []

    assert registry.enable("demo")
    assert registry.is_enabled("demo")
    assert registry.names() == ["demo"]
    assert "Full skill body" in registry.load_content("demo")


def test_skill_registry_unknown_and_path_like_names_are_not_loaded(tmp_path):
    skill_dir = tmp_path / "skills" / "safe"
    skill_dir.mkdir(parents=True)
    (skill_dir / "SKILL.md").write_text(
        "---\nname: safe\ndescription: Safe skill\n---\nSafe body",
        encoding="utf-8",
    )

    registry = SkillRegistry(tmp_path / "skills").scan()

    assert registry.load_content("missing") is None
    assert registry.load_content("../safe") is None
    assert registry.load_content("safe/SKILL.md") is None
    assert registry.load_content("C:\\safe") is None


def test_skill_registry_skips_invalid_skill_names_and_validates_them(tmp_path):
    bad_dir = tmp_path / "skills" / "bad"
    bad_dir.mkdir(parents=True)
    (bad_dir / "SKILL.md").write_text(
        "---\nname: ../bad\ndescription: Bad skill\n---\nBad body",
        encoding="utf-8",
    )

    registry = SkillRegistry(tmp_path / "skills").scan()
    issues = registry.validate()

    assert registry.names() == []
    assert any(issue.level == "error" and "Invalid skill name" in issue.message for issue in issues)


def test_skill_registry_reports_duplicate_names(tmp_path):
    first = tmp_path / "skills" / "first"
    second = tmp_path / "skills" / "second"
    first.mkdir(parents=True)
    second.mkdir(parents=True)
    (first / "SKILL.md").write_text(
        "---\nname: duplicate\ndescription: First\n---\nFirst body",
        encoding="utf-8",
    )
    (second / "SKILL.md").write_text(
        "---\nname: duplicate\ndescription: Second\n---\nSecond body",
        encoding="utf-8",
    )

    registry = SkillRegistry(tmp_path / "skills").scan()
    issues = registry.validate()

    assert registry.names() == ["duplicate"]
    assert registry.get("duplicate").description == "First"
    assert any(issue.level == "error" and "Duplicate skill name" in issue.message for issue in issues)


def test_skill_registry_rejects_symlink_manifest(tmp_path):
    outside = tmp_path / "outside.md"
    outside.write_text(
        "---\nname: leaked\ndescription: Leaked skill\n---\nOutside content",
        encoding="utf-8",
    )
    skill_dir = tmp_path / "skills" / "leaked"
    skill_dir.mkdir(parents=True)
    manifest = skill_dir / "SKILL.md"
    try:
        manifest.symlink_to(outside)
    except (OSError, NotImplementedError):
        return

    registry = SkillRegistry(tmp_path / "skills").scan()
    issues = registry.validate()

    assert registry.names() == []
    assert registry.load_content("leaked") is None
    assert any(issue.level == "error" and "Symlink SKILL.md" in issue.message for issue in issues)


from echo.config import EchoConfig
from echo.core.agent_loop import AgentLoop
from echo.core.context_manager import ContextManager
from echo.core.echo import Echo
from echo.core.task_state import TaskState
from echo.hooks.base import HookManager
from echo.hooks.builtin import LogHook, PermissionHook, PostLogHook, StatsHook
from echo.memory.base import MemoryManager
from echo.memory.default import KeywordMemory
from echo.persistence.run_store import RunStore
from echo.persistence.session_store import SessionStore
from echo.providers.fake_client import FakeLLMClient
from echo.security.env_filter import ShellExecutor
from echo.security.sandbox import Sandbox
from echo.tools.base import ToolContext
from echo.tools.builtin import LoadSkillTool
from echo.tools.executor import ToolExecutor
from echo.tools.registry import ToolRegistry


def _write_demo_skill(root: Path, name: str = "demo", body: str = "Full skill body"):
    skill_dir = root / "skills" / name
    skill_dir.mkdir(parents=True)
    (skill_dir / "SKILL.md").write_text(
        f"---\nname: {name}\ndescription: Demo skill\n---\n\n{body}",
        encoding="utf-8",
    )


def test_load_skill_tool_loads_registered_skill(tmp_path):
    _write_demo_skill(tmp_path)
    registry = SkillRegistry(tmp_path / "skills").scan()
    ctx = ToolContext(skill_registry=registry)

    result = LoadSkillTool().run(ctx, {"name": "demo"})

    assert result.success
    assert "Full skill body" in result.output
    assert "Loaded skill demo" in result.memory_notes


def test_load_skill_tool_rejects_unknown_skill(tmp_path):
    _write_demo_skill(tmp_path)
    registry = SkillRegistry(tmp_path / "skills").scan()
    ctx = ToolContext(skill_registry=registry)

    result = LoadSkillTool().run(ctx, {"name": "missing"})

    assert not result.success
    assert "Skill not found" in result.error
    assert "demo" in result.error


def test_load_skill_tool_fails_without_registry():
    result = LoadSkillTool().run(ToolContext(), {"name": "demo"})

    assert not result.success
    assert "Skill registry is not available" in result.error


def test_context_system_includes_skill_catalog_not_body(tmp_path):
    _write_demo_skill(tmp_path, body="Secret full body details")
    skill_registry = SkillRegistry(tmp_path / "skills").scan()
    tool_registry = ToolRegistry()
    tool_registry.register(LoadSkillTool())

    class _Memory:
        def render_working(self):
            return ""

        def retrieve(self, *args, **kwargs):
            _ = (args, kwargs)
            return []

        def relevant_for_prompt(self, *args, **kwargs):
            _ = (args, kwargs)
            return ""

    class _Sandbox:
        root = tmp_path
        git_branch = "test"

    system = ContextManager().build_system(
        TaskState.create("use skills"),
        tool_registry,
        _Memory(),
        _Sandbox(),
        skill_registry=skill_registry,
    )

    assert "## Available Skills" in system
    assert "**demo**: Demo skill" in system
    assert "Secret full body details" not in system
    assert "load_skill" in system


def test_agent_loop_passes_skill_registry_to_prompt_and_tool(tmp_path):
    _write_demo_skill(tmp_path)
    skill_registry = SkillRegistry(tmp_path / "skills").scan()
    registry = ToolRegistry()
    registry.discover("echo.tools.builtin")
    hooks = HookManager()
    hooks.register(PermissionHook(), priority=0)
    hooks.register(LogHook(), priority=100)
    hooks.register(PostLogHook(), priority=100)
    hooks.register(StatsHook(), priority=200)

    loop = AgentLoop(
        llm=FakeLLMClient(outputs=[
            '<tool name="load_skill" name="demo" />',
            "Loaded demo skill.",
        ]),
        memory=MemoryManager(KeywordMemory()),
        tools=ToolExecutor(registry),
        hooks=hooks,
        context=ContextManager(),
        sandbox=Sandbox(str(tmp_path)),
        shell=ShellExecutor(str(tmp_path)),
        session_store=SessionStore(str(tmp_path)),
        run_store=RunStore(str(tmp_path / ".echo" / "sessions" / "test-session")),
        max_steps=5,
        approval_policy="auto",
        skill_registry=skill_registry,
    )

    answer = loop.run("load demo skill")

    assert "Loaded demo skill" in answer
    assert "## Available Skills" in loop.llm.last_system
    assert "**demo**: Demo skill" in loop.llm.last_system


def test_echo_rescans_skills_before_each_ask(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "echo.core.echo.OllamaClient",
        lambda *args, **kwargs: FakeLLMClient(outputs=["first", "second"]),
    )
    echo = Echo(workspace_root=str(tmp_path), config=EchoConfig(provider="ollama"))

    first = echo.ask("first", max_steps=1)
    assert "first" in first
    assert echo.skill_registry.names() == []

    _write_demo_skill(tmp_path, name="new-skill")
    second = echo.ask("second", max_steps=1)

    assert "second" in second
    assert echo.skill_registry.names() == ["new-skill"]


def test_cli_skills_list_prints_catalog(tmp_path, capsys):
    _write_demo_skill(tmp_path, name="cli-demo")

    from echo.cli import main

    code = main(["--workspace", str(tmp_path), "skills", "list"])

    captured = capsys.readouterr()
    assert code == 0
    assert "cli-demo" in captured.out
    assert "Demo skill" in captured.out


def test_cli_skills_validate_reports_errors(tmp_path, capsys):
    bad_dir = tmp_path / "skills" / "bad"
    bad_dir.mkdir(parents=True)
    (bad_dir / "SKILL.md").write_text(
        "---\nname: ../bad\ndescription: Bad skill\n---\nBad body",
        encoding="utf-8",
    )

    from echo.cli import main

    code = main(["--workspace", str(tmp_path), "skills", "validate"])

    captured = capsys.readouterr()
    assert code == 1
    assert "error" in captured.out
    assert "Invalid skill name" in captured.out
