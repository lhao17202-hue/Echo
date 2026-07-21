"""Tests for Persistent Teammates V1 — task manager, teammate agent, manager, tools, state, loop wiring, and e2e."""

import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from echo.multi_agent.task_manager import GlobalTaskManager


class TestGlobalTaskManagerTeammateV1:
    def test_assign_keeps_task_pending_and_sets_owner(self):
        tasks = GlobalTaskManager()
        task_id = tasks.create("Inspect README", "Find the project name")

        assert tasks.assign(task_id, "researcher") is True

        task = tasks.get(task_id)
        assert task is not None
        assert task.status == "pending"
        assert task.owner_agent == "researcher"

    def test_assign_rejects_missing_or_non_pending_task(self):
        tasks = GlobalTaskManager()
        task_id = tasks.create("Inspect README")
        assert tasks.assign("missing", "researcher") is False

        assert tasks.claim(task_id, "researcher") is True
        assert tasks.assign(task_id, "other") is False
        assert tasks.get(task_id).owner_agent == "researcher"

    def test_list_all_returns_all_tasks(self):
        tasks = GlobalTaskManager()
        first = tasks.create("First")
        second = tasks.create("Second")
        tasks.complete(second, "done")

        all_tasks = tasks.list_all()
        assert [t.task_id for t in all_tasks] == [first, second]
        assert [t.status for t in all_tasks] == ["pending", "completed"]

    def test_list_available_filters_by_owner_when_agent_name_given(self):
        tasks = GlobalTaskManager()
        unowned = tasks.create("Unowned")
        owned_by_a = tasks.create("Owned by A")
        owned_by_b = tasks.create("Owned by B")
        tasks.assign(owned_by_a, "agent-a")
        tasks.assign(owned_by_b, "agent-b")

        visible_to_a = tasks.list_available("agent-a")
        assert [t.task_id for t in visible_to_a] == [unowned, owned_by_a]
