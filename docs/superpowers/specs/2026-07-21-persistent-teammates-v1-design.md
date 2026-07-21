# Persistent Teammates V1 Design

Date: 2026-07-21
Project: Echo Agent
Status: approved for implementation planning

## Summary

Echo currently has a synchronous single-agent loop and a one-shot read-only `delegate` tool. Persistent Teammates V1 upgrades this to a lightweight multi-agent runtime where the lead agent can spawn long-lived teammate threads, assign tasks to them, receive their results through the existing `MessageBus`, and see those results injected into the lead loop on the next model turn.

V1 intentionally stays conservative: daemon threads, same-process message queues, file-backed global task metadata, read-only teammate tools, trace/audit coverage, and deterministic tests. It does not introduce multi-process execution, worktree isolation, automatic teammate thread recovery, RAG, MCP, or complex planning approval.

## Goals

V1 must provide these lead-agent tools:

- `spawn_teammate`
- `assign_task`
- `list_teammates`
- `stop_teammate`
- `list_global_tasks`

V1 must also provide:

- Lead inbox injection from teammate messages.
- Trace/audit events for teammate lifecycle, task lifecycle, and message delivery.
- Tests covering the core teammate runtime path.

The product outcome is to move Echo from a one-shot `delegate` model to a first persistent-teammate model that can run background research tasks while the lead agent continues operating.

## Non-goals

V1 will not implement:

- Real multi-process teammates.
- Git worktree isolation per teammate.
- Automatic recovery of teammate threads after process restart.
- RAG/vector memory.
- MCP tool integration.
- Complex task-plan approval workflows.
- Writable or shell-capable teammates by default.
- Persistent mailbox replay. JSONL mailboxes remain audit records in V1; same-process queues are the delivery path.

## Recommended approach

Use the conservative thread-based V1.

Alternatives considered:

1. Allow teammate write/shell tools in V1.
   - Rejected for this iteration because concurrent writes, shell side effects, and approval policy need more design.
2. Add teammate recovery and persisted mailbox consumption in V1.
   - Rejected for this iteration because offset tracking, idempotent message handling, and lifecycle recovery would expand scope.

## Runtime architecture

```text
Lead AgentLoop
  -> ToolContext exposes teammate_manager / message_bus / global_tasks
  -> spawn_teammate tool
  -> TeammateManager.spawn()
  -> TeammateAgent daemon thread
  -> GlobalTaskManager claim task
  -> TeammateAgent uses read-only tools to execute task
  -> MessageBus send(teammate -> lead)
  -> AgentLoop next iteration _inject_inbox_messages()
  -> lead sees teammate result in model context
```

New files:

- `echo/multi_agent/teammate.py`
  - `TeammateState`
  - `TeammateAgent`
- `echo/multi_agent/teammate_manager.py`
  - `TeammateManager`

Modified files:

- `echo/core/echo.py`
  - Initialize `MessageBus`, `GlobalTaskManager`, and `TeammateManager`.
  - Register the lead mailbox.
  - Pass the multi-agent objects into `AgentLoop`.
- `echo/core/agent_loop.py`
  - Accept `message_bus`, `teammate_manager`, and `global_tasks` parameters.
  - Inject lead inbox messages before compaction each loop iteration.
  - Pass multi-agent handles into `ToolContext`.
  - Update `TaskState` with teammate/task snapshots.
- `echo/tools/base.py`
  - Extend `ToolContext` with `message_bus`, `teammate_manager`, `global_tasks`, and `agent_name`.
- `echo/tools/builtin.py`
  - Add the five teammate/global task tools.
- `echo/multi_agent/task_manager.py`
  - Add `list_all()` and `assign()`.
  - Ensure assigned-but-unclaimed tasks stay `pending` until claimed.
- `echo/core/task_state.py`
  - Add teammate/task snapshot fields and serialization support.
- `tests/test_teammates.py`
  - Add deterministic teammate runtime tests using `FakeLLMClient`.

## TeammateAgent

`TeammateAgent` is not `SubAgent`.

`SubAgent` is a one-shot synchronous read-only helper invoked by `delegate`. `TeammateAgent` is a long-lived daemon thread that repeatedly wakes, checks inbox, claims assigned global tasks, performs read-only work, and reports back to the lead.

### State

`TeammateState` fields:

- `name: str`
- `role: str`
- `status: str` with values `idle | running | stopped | failed`
- `current_task_id: str`
- `session_id: str`
- `last_error: str`
- `started_at: str`
- `stopped_at: str`

### Methods

`TeammateAgent` should expose:

- `start()`
- `stop()`
- `snapshot()`
- `run_loop()`
- `_tick()`
- `_handle_inbox()`
- `_claim_available_task()`
- `_run_task(task)`

### Loop behavior

While stop has not been requested:

1. Read teammate inbox.
2. Handle `stop`, message, or task instructions.
3. If no blocking inbox command exists, scan available global tasks.
4. Claim a task if it is unowned or assigned to this teammate.
5. Execute the task with read-only tools.
6. Complete or fail the task.
7. Send a `task_completed` or `task_failed` message to `lead`.
8. Sleep briefly before the next tick.

## TeammateManager

`TeammateManager` owns all teammate instances in the process.

Core API:

- `spawn(name: str, role: str, prompt: str = "") -> dict`
- `stop(name: str) -> bool`
- `list() -> list[dict]`
- `assign_task(teammate: str, subject: str, description: str = "") -> str`
- `snapshot() -> dict`

### Spawn behavior

`spawn()` must:

1. Reject duplicate teammate names.
2. Build a read-only tool list.
3. Exclude recursive or lead-control tools.
4. Create `TeammateAgent`.
5. Register the teammate mailbox in `MessageBus`.
6. Start the daemon thread.
7. Write `teammate_spawned` trace.
8. Return the teammate snapshot.

### Tool filtering

V1 teammate tools are read-only only.

Allowed by default:

- `read_file`
- `glob`
- `grep`
- `list_files`
- `search_memory`

Blocked:

- `delegate`
- `compact`
- `spawn_teammate`
- `assign_task`
- `list_teammates`
- `stop_teammate`
- `list_global_tasks`
- `write_file`
- `patch_file`
- `run_shell`
- Any tool where `is_read_only` is false.

This preserves safety and avoids concurrent file mutation in the first implementation.

## Global tasks

`GlobalTaskManager` already supports create, claim, complete, fail, list_available, and get.

Add:

- `list_all() -> list[GlobalTask]`
- `assign(task_id: str, agent_name: str) -> bool`

Semantics:

- `assign()` sets `owner_agent` but keeps `status == "pending"`.
- `claim()` changes `status` to `in_progress`.
- A teammate should claim only tasks where `owner_agent is None` or `owner_agent == teammate_name`.
- Existing dependency checks still apply.

This makes assignment advisory until the teammate actually starts work.

## Lead inbox injection

`AgentLoop` must inject teammate messages before context compaction each loop iteration.

Behavior:

1. Call `message_bus.receive("lead")`.
2. If no messages exist, do nothing.
3. Build a user message with a heading such as `## Teammate Messages`.
4. Include one bullet per message with sender, type, and content.
5. Append it to `self.messages` as a `TextBlock` user message.
6. Log `message_received` for each message.

This step is critical: without it, teammate results exist in the bus but never enter the lead model context.

## ToolContext extension

Add fields:

- `message_bus: Any = None`
- `teammate_manager: Any = None`
- `global_tasks: Any = None`
- `agent_name: str = "lead"`

These handles are optional so existing tests and isolated tools continue to work.

## Built-in tools

### `spawn_teammate`

Params:

- `name: str`
- `role: str = "assistant"`
- `prompt: str = ""`

Behavior:

- Fails if `ctx.teammate_manager` is missing.
- Calls `ctx.teammate_manager.spawn(...)`.
- Returns a structured summary of the teammate snapshot.

### `assign_task`

Params:

- `teammate: str`
- `subject: str`
- `description: str = ""`

Behavior:

- Fails if teammate manager is missing.
- Fails if teammate does not exist.
- Creates a global task assigned to the teammate.
- Returns the task ID.

### `list_teammates`

Params: none.

Behavior:

- Returns manager list output.
- Fails if manager is unavailable.

### `stop_teammate`

Params:

- `name: str`

Behavior:

- Calls manager stop.
- Returns success/failure.

### `list_global_tasks`

Params: none.

Behavior:

- Returns all global tasks with task id, subject, status, owner, and result summary.
- Fails if global task manager is unavailable.

## Trace and audit

Trace event names:

- `teammate_spawned`
- `teammate_started`
- `teammate_stopped`
- `teammate_task_claimed`
- `teammate_task_completed`
- `teammate_task_failed`
- `global_task_created`
- `global_task_assigned`
- `message_sent`
- `message_received`

Implementation can continue using `run_store.log(event_name, ...)` directly. Adding constants in `echo/persistence/trace.py` is optional but useful for consistency.

## TaskState snapshot

Add fields:

- `active_teammates: dict = field(default_factory=dict)`
- `global_task_ids: list[str] = field(default_factory=list)`
- `unprocessed_messages: list = field(default_factory=list)`

Update `to_dict()` and `from_dict()`.

`AgentLoop` should update `active_teammates` from `teammate_manager.snapshot()` each loop. V1 stores metadata only; it does not automatically restore teammate threads from this snapshot.

## Error handling

- Duplicate teammate name: return a failed tool result and do not start a thread.
- Missing manager/task bus: return a failed tool result.
- Unknown teammate on assignment/stop: return a failed tool result.
- Teammate task exception: mark task failed, set teammate `last_error`, send `task_failed` to lead, log `teammate_task_failed`.
- LLM exhaustion or empty output in tests: treat as a failed/completed task according to the fake response behavior; never crash the thread.
- Stop request during idle: set status to `stopped` and exit loop.
- Stop request during task execution: stop after the current task boundary in V1.

## Tests

Add `tests/test_teammates.py` with deterministic tests using `FakeLLMClient`.

Minimum coverage:

1. `TeammateManager.spawn()` creates a teammate.
2. Duplicate names are rejected.
3. `stop()` transitions status to `stopped`.
4. `assign_task()` creates a global task with owner set.
5. Teammate claims only pending tasks assigned to it or unowned tasks.
6. Teammate completes a task and sends a message to lead.
7. `AgentLoop._inject_inbox_messages()` appends teammate messages into `messages`.
8. Trace contains teammate lifecycle and message events.
9. Teammate tool list excludes `delegate` and write/shell tools.
10. `TaskState` snapshot includes teammate metadata.

Run targeted tests first, then the existing deterministic suite:

```bash
python -B -m pytest tests/test_teammates.py tests/test_agent_loop.py tests/test_persistence.py -p no:cacheprovider
python -B -m pytest tests --ignore=tests/test_providers.py -p no:cacheprovider
```

## Provider compatibility

`echo/providers/anthropic_client.py` has separate current-model compatibility concerns noted in ProjectMemo, especially around always passing `temperature` to newer Claude models. This design intentionally does not address that provider/API migration. Treat it as a separate follow-up task after Persistent Teammates V1.

## Acceptance criteria

The implementation is complete when:

- The lead agent can call `spawn_teammate`.
- The lead agent can call `assign_task`.
- A teammate daemon thread can claim an assigned pending task.
- The teammate executes the task with read-only tools.
- The task is completed or failed in `GlobalTaskManager`.
- The teammate sends a result message to `lead` through `MessageBus`.
- `AgentLoop` injects teammate messages into the lead conversation on the next iteration.
- Trace/audit records teammate lifecycle, task lifecycle, and message receipt.
- Tests cover the core path and pass locally.
