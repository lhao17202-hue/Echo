"""CLI tests for persistent global task commands."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


def test_cli_tasks_list_empty(tmp_path, capsys):
    from echo.cli import main

    code = main(["--workspace", str(tmp_path), "tasks", "list"])

    captured = capsys.readouterr()
    assert code == 0
    assert "No global tasks" in captured.out


def test_cli_tasks_create_and_get(tmp_path, capsys):
    from echo.cli import main

    create_code = main([
        "--workspace", str(tmp_path),
        "tasks", "create", "Write docs",
        "--description", "Document task system",
    ])
    created = capsys.readouterr().out
    task_id = created.split()[2].rstrip(":")

    get_code = main(["--workspace", str(tmp_path), "tasks", "get", task_id])
    detail = capsys.readouterr().out

    assert create_code == 0
    assert get_code == 0
    assert "Write docs" in detail
    assert "Document task system" in detail


def test_cli_tasks_blocked_claim_exits_nonzero(tmp_path, capsys):
    from echo.cli import main

    main([
        "--workspace", str(tmp_path),
        "tasks", "create", "Blocked",
        "--blocked-by", "missing",
    ])
    created = capsys.readouterr().out
    task_id = created.split()[2].rstrip(":")

    code = main(["--workspace", str(tmp_path), "tasks", "claim", task_id])
    output = capsys.readouterr().out

    assert code == 1
    assert "Blocked by" in output
    assert "missing" in output


def test_cli_tasks_complete_unblocks_downstream(tmp_path, capsys):
    from echo.cli import main

    main(["--workspace", str(tmp_path), "tasks", "create", "First"])
    first_id = capsys.readouterr().out.split()[2].rstrip(":")
    main([
        "--workspace", str(tmp_path),
        "tasks", "create", "Second",
        "--blocked-by", first_id,
    ])
    second_id = capsys.readouterr().out.split()[2].rstrip(":")

    assert main(["--workspace", str(tmp_path), "tasks", "claim", first_id]) == 0
    complete_code = main([
        "--workspace", str(tmp_path),
        "tasks", "complete", first_id,
        "--result", "done",
    ])
    output = capsys.readouterr().out

    assert complete_code == 0
    assert "Unblocked" in output
    assert second_id in output


def test_cli_tasks_validate_reports_missing_dependency(tmp_path, capsys):
    from echo.cli import main

    main([
        "--workspace", str(tmp_path),
        "tasks", "create", "Blocked",
        "--blocked-by", "missing",
    ])
    capsys.readouterr()

    code = main(["--workspace", str(tmp_path), "tasks", "validate"])
    output = capsys.readouterr().out

    assert code == 1
    assert "Unknown dependency" in output


def test_cli_rejects_misspelled_global_option_before_agent_start(capsys):
    from echo.cli import main

    try:
        main(["--workspce", ".", "tasks", "list"])
    except SystemExit as exc:
        code = exc.code
    else:
        code = 0

    captured = capsys.readouterr()
    assert code == 2
    assert "unrecognized arguments" in captured.err
