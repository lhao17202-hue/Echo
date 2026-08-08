# Task System Design

Date: 2026-08-08

## Goal

Complete Echo's task system by turning the existing global teammate task pool into a general persistent task graph, while keeping `TaskState.todos` as the per-run planning view. The result should support explicit task creation, dependency blocking, claiming, completion/failure, CLI management, prompt visibility, and deterministic tests.

This design references `C:\Users\VK\Desktop\learn-claude-code-main\s12_task_system\code.py`, but adapts the idea to Echo's existing architecture instead of introducing the reference script's standalone `.tasks/` directory or duplicate loop.

## Scope

This design implements both layers requested by the user:

- A workspace-persistent global task graph based on `echo.multi_agent.task_manager.GlobalTaskManager`.
- A per-run task view based on `echo.core.task_state.TaskState.todos` and `TaskState.global_task_ids`.
- Built-in tools for creating, reading, claiming, completing, failing, listing, assigning, and waiting on global tasks.
- Prompt injection for current todos and relevant global tasks.
- CLI commands for task inspection and mutation.
- Validation for dependency references and cycles.
- Tests for manager behavior, tool behavior, CLI behavior, prompt integration, persistence, and teammate compatibility.
- README documentation for todos versus global tasks.

Out of scope:

- Process or worktree isolation for teammates.
- Teammate restart/recovery after process exit.
- A separate `.tasks/` storage tree; Echo will continue using `.echo/global/tasks.json`.
- Automatic conversion of every `todo_write` item into a global task.
- Distributed locking across multiple OS processes. The current manager remains thread-safe inside one Echo process and uses atomic file replacement for writes.

## Current State

Echo already has several task-related pieces:

- `echo/core/task_state.py`
  - `TaskState` tracks a single `ask()` run.
  - It includes `todos`, `global_task_ids`, `bound_global_task_id`, and runtime status fields.
- `echo/multi_agent/task_manager.py`
  - `GlobalTaskManager` stores `GlobalTask` objects in memory and optionally persists them to `.echo/global/tasks.json`.
  - It supports `create`, `assign`, `claim`, `complete`, `fail`, `wait`, `list_all`, `list_available`, and `get`.
  - It already models `blocked_by`, `owner_agent`, `status`, `result`, timestamps, and `run_id`.
- `echo/tools/builtin.py`
  - Existing teammate/global task tools include `assign_task`, `list_global_tasks`, and `wait_global_task`.
  - `todo_write` updates `TaskState.todos` only.
- `echo/core/context_manager.py`
  - The system prompt already renders current unfinished todos.
- `echo/core/agent_loop.py`
  - `ToolContext` receives `task_state` and `global_tasks`, so task tools can update both.
- `echo/core/echo.py`
  - The facade constructs `GlobalTaskManager` at `.echo/global/tasks.json` and passes it into the loop.

The missing piece is that these are not yet a coherent user-facing task system. Global tasks are mostly teammate assignment plumbing, while todos are local run planning only.

## Reference Mapping

The reference implementation in `s12_task_system/code.py` contains:

- `Task(id, subject, description, status, owner, blockedBy)`
- Persistent JSON task storage.
- `create_task`, `save_task`, `load_task`, `list_tasks`, `get_task`.
- `can_start`, where missing dependencies are blocked.
- `claim_task`, which moves pending tasks to `in_progress` only when dependencies are completed.
- `complete_task`, which reports downstream tasks unblocked by completion.
- Tools: `create_task`, `list_tasks`, `get_task`, `claim_task`, `complete_task`.

Echo should map those concepts to existing names:

| Reference concept | Echo concept |
| --- | --- |
| `.tasks/*.json` | `.echo/global/tasks.json` |
| `Task.id` | `GlobalTask.task_id` |
| `owner` | `owner_agent` |
| `blockedBy` | `blocked_by` |
| standalone functions | `GlobalTaskManager` methods |
| reference task tools | Echo built-in `BaseTool` subclasses |

## Architecture

### Two-Layer Model

Echo will use two task layers with separate responsibilities.

#### Global Task Graph

The global task graph is the durable source of truth for work items that should survive runs or coordinate multiple agents. It lives in `GlobalTaskManager` and persists to `.echo/global/tasks.json`.

It owns:

- Task identity.
- Subject and description.
- Status.
- Owner or assigned teammate.
- Dependency edges.
- Creation/completion timestamps.
- Result/error text.
- Run association.

#### Run Task View

The run task view is `TaskState.todos`, updated by `todo_write`. It remains a short-lived planning aid for the model's current `ask()` run.

It owns:

- Current local plan steps.
- In-progress and pending todo visibility in the prompt.
- Session short-term memory persistence through existing Echo mechanisms.

It does not automatically create durable global tasks. This avoids polluting the global task graph every time the model revises its local plan.

### Relationship Between Layers

Global task tools will append touched task IDs to `TaskState.global_task_ids`, avoiding duplicates. This links durable tasks to the current run without conflating them with todos.

`ContextManager` will render both:

- Existing `## Current Todos` from `TaskState.todos`.
- New `## Relevant Global Tasks` from `TaskState.global_task_ids` plus claimable pending tasks from `GlobalTaskManager`.

The global task prompt section should stay compact: task ID, status, owner, subject, blocked_by, and short result preview only.

## Data Model

### GlobalTask

Keep `GlobalTask` in `echo/multi_agent/task_manager.py` and strengthen its status semantics.

Statuses:

- `pending`: created but not started.
- `in_progress`: claimed by an owner.
- `completed`: finished successfully.
- `failed`: terminal failure.

Fields stay compatible with current persisted JSON:

```python
@dataclass
class GlobalTask:
    task_id: str
    subject: str
    description: str
    status: str
    owner_agent: str | None
    blocked_by: list[str]
    worktree: str | None
    created_at: str
    completed_at: str | None
    result: str
    run_id: str
```

Do not rename persisted fields. Existing `.echo/global/tasks.json` files should continue to load.

### Validation

Add validation that reports structured issues without mutating tasks:

- Unknown dependency IDs.
- Dependency cycles.
- Unknown status values.
- Empty subject.
- Duplicate IDs are impossible in the current JSON object shape, but malformed files should be handled defensively.

Validation issues should include `level`, `task_id`, and `message`. Use `Literal["warning", "error"]` or an enum for severity.

## GlobalTaskManager API

Enhance `echo/multi_agent/task_manager.py` with:

- `can_start(task_id: str) -> bool`
  - Returns `True` only when the task exists, is pending, and all dependencies exist and are completed.
- `blocked_dependencies(task_id: str) -> list[str]`
  - Returns dependency IDs that are missing or not completed.
- `list_unblocked_after(task_id: str) -> list[GlobalTask]`
  - Returns pending downstream tasks that are now claimable.
- `validate() -> list[TaskValidationIssue]`
  - Returns dependency/status/shape issues.
- Optional read helpers for formatting or exact lookup if useful.

Update existing methods:

- `claim(task_id, agent_name)`
  - Keep the existing bool return for backward compatibility.
  - Treat missing dependencies as blocked.
  - Treat failed dependencies as blocked.
  - Keep owner guard: unowned tasks or tasks assigned to the same agent can be claimed; tasks assigned to other agents cannot.
- `complete(task_id, result="")`
  - Only transition `in_progress` tasks to `completed`.
  - Return enough information for tools to report unblocked downstream tasks, or expose that through `list_unblocked_after` immediately after completion.
  - Keep current callers compatible if they ignore the return value.
- `fail(task_id, error="")`
  - Mark existing tasks as failed and write result.
- `_load()`
  - Continue tolerating corrupt files by leaving an empty in-memory task map, but validation should surface malformed task data when possible if loaded through a validation path.

## Tools

Add or extend built-in task tools in `echo/tools/builtin.py`.

### New Tools

#### `create_task`

Input:

- `subject: str`
- `description: str = ""`
- `blocked_by: list[str] = []`

Behavior:

- Creates a global task with `run_id=ctx.run_id`.
- Adds task ID to `ctx.task_state.global_task_ids` when available.
- Returns created task ID and dependency summary.

#### `get_task`

Input:

- `task_id: str`

Behavior:

- Returns full task detail in readable JSON or structured text.
- Fails if unknown.

#### `claim_task`

Input:

- `task_id: str`
- `owner: str = ctx.agent_name`

Behavior:

- Claims a pending task for the owner if dependencies are complete.
- On blocked tasks, returns the blocking dependency IDs.
- Adds task ID to `TaskState.global_task_ids` on success.

#### `complete_task`

Input:

- `task_id: str`
- `result: str = ""`

Behavior:

- Completes an in-progress task.
- Adds task ID to `TaskState.global_task_ids`.
- Reports downstream tasks that became unblocked.

#### `fail_task`

Input:

- `task_id: str`
- `error: str = ""`

Behavior:

- Marks the task failed.
- Adds task ID to `TaskState.global_task_ids`.
- Dependent tasks remain blocked.

### Extended Tools

#### `assign_task`

Add optional `blocked_by: list[str] = []`. It continues to create a task assigned to a teammate, but now supports dependencies.

#### `list_global_tasks`

Add optional filters:

- `status: str = ""`
- `owner: str = ""`
- `available_only: bool = False`

Show:

- task ID
- status
- owner
- subject
- blocked_by
- result preview

#### `wait_global_task`

Keep current behavior. If the task is still blocked/pending after timeout, include dependency/owner/status summary in the partial result.

### Tool State Tracking

Create a small helper in `builtin.py`, for example `_remember_global_task(ctx, task_id)`, that appends to `ctx.task_state.global_task_ids` only when present and not already included. Reuse it across task tools.

## Prompt Integration

Update `ContextManager.build_system()` to optionally accept `global_tasks=None`, or pass it through another existing context object if cleaner.

Render a compact `## Relevant Global Tasks` section when `global_tasks` is available and either:

- `state.global_task_ids` is non-empty, or
- `global_tasks.list_available("lead")` returns pending claimable tasks.

Example prompt section:

```text
## Relevant Global Tasks
- task_ab12cd34 [in_progress] owner=lead subject=Implement parser blocked_by=[]
- task_ef56ab78 [pending] owner=- subject=Write docs blocked_by=[task_ab12cd34]
```

Do not include full task result bodies unless short. Long results should remain accessible through `get_task`.

## CLI

Extend `echo/cli.py` with `tasks` commands. The CLI should operate on the workspace path and not instantiate provider SDK clients when only managing tasks.

Commands:

- `echo-agent tasks list`
- `echo-agent tasks get <task_id>`
- `echo-agent tasks create "Subject" --description "..." --blocked-by dep1 dep2`
- `echo-agent tasks claim <task_id> --owner lead`
- `echo-agent tasks complete <task_id> --result "..."`
- `echo-agent tasks fail <task_id> --error "..."`
- `echo-agent tasks validate`

Exit codes:

- `0` for successful list/get/create/claim/complete/fail/valid validation.
- `1` when validation finds errors or a task mutation cannot be applied.
- `2` for invalid CLI usage.

## Documentation

Update `README.md`:

- Project layout: mention global task system under `multi_agent/` or a new task manager line.
- Built-in tools: list task tools.
- Runtime data: mention `.echo/global/tasks.json`.
- Add `## Task System` explaining:
  - `todo_write` is per-run planning.
  - Global tasks are persistent cross-run/cross-agent work items.
  - Dependencies use `blocked_by`.
  - CLI examples.
  - Teammate assignment flow.

## Testing Plan

### Manager Tests

Add or extend tests for `GlobalTaskManager`:

- create/list/get persistence roundtrip.
- missing dependency blocks claim.
- pending dependency blocks claim.
- completed dependency allows claim.
- failed dependency blocks claim.
- complete reports or enables downstream unblocked tasks.
- validate catches missing dependency.
- validate catches dependency cycle.
- owner guard still rejects claiming another teammate's assigned task.

### Tool Tests

Add tests for built-in task tools:

- `create_task` creates and remembers task ID in `TaskState.global_task_ids`.
- `get_task` returns details.
- `claim_task` succeeds only when unblocked.
- `complete_task` returns unblocked downstream.
- `fail_task` marks failed.
- `assign_task` accepts `blocked_by` and still works with `TeammateManager`.
- `list_global_tasks` filters status/owner/available tasks.

### Prompt Tests

Add a `ContextManager` test showing:

- current todos still render.
- relevant global tasks render.
- long task results are truncated or omitted from prompt.

### CLI Tests

Add tests for:

- `tasks list` on empty task store.
- create then get.
- blocked claim exits non-zero.
- complete dependency unblocks downstream.
- validate catches missing dependency and cycle.

### Regression Tests

Run:

```bash
python -m pytest tests/test_teammates.py tests/test_tools.py tests/test_persistence.py -q
python -m pytest -q
```

## Migration and Compatibility

Existing `.echo/global/tasks.json` files should continue to load because field names are preserved.

Existing teammate tests should remain valid:

- `assign_task` still assigns teammate tasks.
- `TeammateAgent` can still claim assigned pending tasks.
- `wait_global_task` remains compatible.

Existing `todo_write` behavior should remain valid:

- It still updates `TaskState.todos`.
- It does not create global tasks implicitly.

## Risks and Mitigations

- Risk: Global task tools may overlap confusingly with `todo_write`.
  - Mitigation: Document the distinction and keep automatic conversion out of scope.
- Risk: Prompt grows too large with global task details.
  - Mitigation: render compact summaries only; use `get_task` for details.
- Risk: Dependency cycles can make tasks permanently blocked.
  - Mitigation: add `validate()` and CLI validation.
- Risk: Changing `claim` or `complete` return values may break existing teammate code.
  - Mitigation: keep existing public return types where tests depend on them; expose extra details via helper methods.

## Implementation Sequence

1. Add manager tests for dependency semantics and validation.
2. Enhance `GlobalTaskManager` with helper methods and validation.
3. Add task tool tests.
4. Implement new task tools and extend existing task tools.
5. Add prompt integration tests.
6. Render relevant global tasks in `ContextManager` and pass `global_tasks` from `AgentLoop`.
7. Add CLI tests and implement `tasks` commands without provider initialization.
8. Update README.
9. Run targeted tests and full regression suite.
