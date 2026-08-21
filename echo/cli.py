"""CLI entry point — argument parsing, REPL, single-shot mode, resume."""

import argparse
from pathlib import Path
from echo.core.echo import Echo
from echo.config import PROVIDER_CHOICES, DEFAULT_PROVIDER
from echo.multi_agent.task_manager import GlobalTaskManager
from echo.skills import SkillRegistry


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="echo-agent",
        description="Echo — lightweight local coding agent",
    )
    p.add_argument("request", nargs="*", help="User request (single-shot mode)")
    p.add_argument("-w", "--workspace", default=".", help="Workspace root directory")
    p.add_argument("-p", "--provider", default=None,
                   choices=PROVIDER_CHOICES,
                   help=f"Model backend: {', '.join(PROVIDER_CHOICES)} (default: {DEFAULT_PROVIDER})")
    p.add_argument("-m", "--model", default=None, help="Model name override")
    p.add_argument("--base-url", default=None, help="Provider API base URL override")
    p.add_argument("--max-steps", type=int, default=25, help="Max tool steps per request")
    p.add_argument("--approval", default=None, choices=["ask", "auto", "never", "danger"],
                   help="Approval policy")
    # Resume
    p.add_argument("--resume", nargs="?", const="__latest__", default=None,
                   metavar="SESSION_ID",
                   help="Resume from a session (no value = latest)")
    p.add_argument("--list-sessions", action="store_true",
                   help="List recent sessions and exit")
    return p


def _handle_skills_command(args) -> int:
    registry = SkillRegistry(Path(args.workspace).resolve() / "skills").scan()
    command = args.request[1] if len(args.request) > 1 else ""
    if command == "list":
        catalog = registry.list_catalog()
        print(catalog or "No skills found.")
        return 0

    if command == "validate":
        issues = registry.validate()
        if not issues:
            print("Skills validation passed.")
            return 0
        for issue in issues:
            print(f"{issue.level}: {issue.path}: {issue.message}")
        return 1 if any(issue.level == "error" for issue in issues) else 0

    print("Expected 'skills list' or 'skills validate'.")
    return 2


def _task_manager_for_workspace(workspace: str) -> GlobalTaskManager:
    return GlobalTaskManager(str(Path(workspace).resolve() / ".echo" / "global" / "tasks.json"))


def _handle_tasks_command(args) -> int:
    tasks = _task_manager_for_workspace(args.workspace)
    parts = list(args.request or [])
    command = parts[1] if len(parts) > 1 else ""

    def value_after(flag: str, default: str = "") -> str:
        if flag not in parts:
            return default
        index = parts.index(flag)
        if index + 1 >= len(parts):
            return default
        return parts[index + 1]

    def values_after(flag: str) -> list[str]:
        if flag not in parts:
            return []
        index = parts.index(flag) + 1
        values = []
        while index < len(parts) and not parts[index].startswith("--"):
            values.append(parts[index])
            index += 1
        return values

    if command == "list":
        items = tasks.list_all()
        if not items:
            print("No global tasks.")
            return 0
        for task in items:
            print(tasks.format_task(task))
        return 0

    if command == "create":
        if len(parts) < 3:
            print("Usage: tasks create <subject> [--description text] [--blocked-by dep ...]")
            return 2
        subject = parts[2]
        description = value_after("--description")
        blocked_by = values_after("--blocked-by")
        task_id = tasks.create(subject, description, blocked_by=blocked_by)
        print(f"Created task {task_id}: {subject}")
        return 0

    if command == "get":
        if len(parts) < 3:
            print("Usage: tasks get <task_id>")
            return 2
        task = tasks.get(parts[2])
        if task is None:
            print(f"Unknown global task: {parts[2]}")
            return 1
        import json
        print(json.dumps(task.to_dict(), indent=2, ensure_ascii=False))
        return 0

    if command == "claim":
        if len(parts) < 3:
            print("Usage: tasks claim <task_id> [--owner name]")
            return 2
        task_id = parts[2]
        owner = value_after("--owner", "lead") or "lead"
        ok, message = tasks.claim_task(task_id, owner)
        print(message)
        return 0 if ok else 1

    if command == "complete":
        if len(parts) < 3:
            print("Usage: tasks complete <task_id> [--result text]")
            return 2
        task_id = parts[2]
        ok, message = tasks.complete_task(task_id, value_after("--result"))
        print(message)
        return 0 if ok else 1

    if command == "fail":
        if len(parts) < 3:
            print("Usage: tasks fail <task_id> [--error text]")
            return 2
        task_id = parts[2]
        ok, message = tasks.fail_task(task_id, value_after("--error"))
        print(message)
        return 0 if ok else 1

    if command == "validate":
        issues = tasks.validate()
        if not issues:
            print("Task validation passed.")
            return 0
        for issue in issues:
            print(f"{issue.level}: {issue.task_id}: {issue.message}")
        return 1 if any(issue.level == "error" for issue in issues) else 0

    print("Expected tasks list|get|create|claim|complete|fail|validate.")
    return 2


def _split_command_argv(argv) -> tuple[list[str] | None, list[str]]:
    """Split known subcommand tails before argparse validates global options."""
    if argv is None:
        return None, []
    items = list(argv)
    for index, item in enumerate(items):
        if item in ("skills", "tasks"):
            return items[:index + 2], items[index + 2:]
    return items, []


def main(argv=None) -> int:
    parser = build_parser()
    parse_argv, command_tail = _split_command_argv(argv)
    args = parser.parse_args(parse_argv)
    if command_tail:
        args.request.extend(command_tail)

    if args.request and args.request[0] == "skills":
        return _handle_skills_command(args)
    if args.request and args.request[0] == "tasks":
        return _handle_tasks_command(args)

    workspace = str(Path(args.workspace).resolve())

    from echo.config import EchoConfig
    config = EchoConfig.from_env(
        cli_provider=args.provider or "",
        cli_model=args.model or "",
        cli_base_url=args.base_url or "",
    )
    if args.approval:
        config.approval_policy = args.approval

    echo = Echo(workspace_root=workspace, config=config)

    # --list-sessions
    if args.list_sessions:
        sessions = echo.list_sessions()
        if not sessions:
            print("No sessions found.")
        else:
            print(f"{'SESSION ID':<30} {'CREATED':<22} {'WORKSPACE'}")
            print("-" * 80)
            for s in sessions:
                sid = s["session_id"]
                created = s.get("created_at", "")[:19]
                w = s.get("workspace_root", "")[:40]
                print(f"{sid:<30} {created:<22} {w}")
        return 0

    # --resume
    if args.resume is not None:
        session_id = "" if args.resume == "__latest__" else args.resume
        query = " ".join(args.request) if args.request else ""
        if query:
            print(f"Echo (resume {session_id or 'latest'})> {query}")
        else:
            print(f"Echo — resuming {session_id or 'latest'}")
        answer = echo.resume(session_id=session_id, user_request=query)
        print(f"\n{answer}")
        return 0

    # Normal mode
    if args.request:
        query = " ".join(args.request)
        print(f"Echo ({echo.llm.model})> {query}")
        answer = echo.ask(query, max_steps=args.max_steps)
        print(f"\n{answer}")
        return 0
    else:
        print(f"Echo Agent — {echo.llm.model} @ {echo.config.base_url or 'default'}")
        print(f"Workspace: {workspace}")
        print('Type "exit" to quit.\n')

        while True:
            try:
                query = input(">> ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\nBye.")
                break
            if not query:
                continue
            if query.lower() in ("exit", "quit", "q"):
                break
            answer = echo.ask(query, max_steps=args.max_steps)
            print(f"\n{answer}\n")
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
