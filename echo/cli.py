"""CLI entry point — argument parsing, REPL, single-shot mode, resume."""

import sys
import argparse
from pathlib import Path
from echo.core.echo import Echo
from echo.config import PROVIDER_CHOICES, DEFAULT_PROVIDER


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
    p.add_argument("--approval", default=None, choices=["ask", "auto", "never"],
                   help="Approval policy")
    # Resume
    p.add_argument("--resume", nargs="?", const="__latest__", default=None,
                   metavar="SESSION_ID",
                   help="Resume from a session (no value = latest)")
    p.add_argument("--list-sessions", action="store_true",
                   help="List recent sessions and exit")
    return p


def main():
    parser = build_parser()
    args = parser.parse_args()

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
        return

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
        return

    # Normal mode
    if args.request:
        query = " ".join(args.request)
        print(f"Echo ({echo.llm.model})> {query}")
        answer = echo.ask(query, max_steps=args.max_steps)
        print(f"\n{answer}")
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


if __name__ == "__main__":
    main()
