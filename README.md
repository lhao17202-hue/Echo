# Echo Agent

Echo is a local-first coding agent framework focused on a reliable synchronous agent loop, structured tools, safe workspace operations, context compaction, session recovery, durable memory, delegation, and deterministic evaluation.

## Current Status

Echo currently provides a usable single-agent core with several production-oriented supporting systems:

- Synchronous AgentLoop with native tool-call handling
- Structured tool system with pydantic validation
- Workspace path sandbox and permission hooks
- Environment filtering and secret redaction
- Multi-level context compaction with transcript archival
- Todo, session persistence, checkpoint, and resume support
- Read-only one-shot delegate subagent
- Working memory plus JSON durable memory V1
- Benchmark and evaluation V1 using FakeLLMClient
- Persistent teammate V1 (same-process daemon threads with global task pool and lead inbox injection)

Planned follow-ups: teammate process/worktree isolation, teammate recovery on restart, vector/RAG memory, MCP integration, deeper scheduler/background-task integration.

## Quick Start

``bash
pip install -e .
cp .env.example .env
``

Set the provider API key in .env, then run:

``bash
echo-agent "Inspect this project"
``

You can also run the module directly:

``bash
python -m echo "Inspect this project"
``

## Project Layout

``text
echo/
  core/          AgentLoop, TaskState, ContextManager, Echo facade
  tools/         BaseTool, registry, executor, built-in tools
  skills/        workspace-local skill discovery and validation
  providers/     Anthropic, OpenAI, Ollama, and fake clients
  memory/        working memory and durable JSON memory
  persistence/   sessions, runs, trace, reports, checkpoints
  security/      sandboxing, permissions, env filtering, redaction
  hooks/         hook manager and built-in hooks
  multi_agent/   read-only SubAgent plus collaboration primitives
  scheduler/     scheduler primitives
  evaluation/    benchmark tasks, evaluator, metrics

skills/          workspace-local SKILL.md instructions loaded on demand
tests/           unit, integration, regression, and evaluation tests
``

## Built-in Tools

Echo includes file tools, shell execution, search/list helpers, todo management, context compaction, memory tools, skill loading, and read-only delegation:

- read_file, write_file, patch_file
- glob, grep, list_files
- run_shell
- todo_write, compact
- save_memory, search_memory
- load_skill
- delegate

## Workspace Skills

Workspace-local skills live in `skills/<skill-name>/SKILL.md`. Echo scans this directory before each active `ask()` or resumed run, injects only a lightweight catalog into the system prompt, and loads full skill instructions on demand through the safe `load_skill` tool.

Each `SKILL.md` may define `name` and `description` in YAML frontmatter:

``markdown
---
name: code-review
description: Review changed code for correctness, maintainability, and test coverage.
---

# Code Review

Full task-specific instructions go here.
``

Skill names are exact registered identifiers, not paths. Names containing `/`, `\\`, `..`, drive prefixes, or empty values are rejected. Symlink skill directories and symlink `SKILL.md` manifests are rejected during scanning and validation.

Manage skills from the CLI:

``bash
python -m echo.cli skills list
python -m echo.cli skills validate
``

## Runtime Data

Runtime state is written under .echo/ inside the workspace. This directory contains sessions, runs, traces, reports, checkpoints, transcript archives, large tool outputs, and durable memory data. It should not be committed.

## Evaluation

Run the deterministic test and benchmark suite with:

``bash
python -B -m pytest tests --ignore=tests/test_providers.py -p no:cacheprovider
``

Provider adapter tests require the optional provider SDK packages to be installed in the local environment.

## Roadmap

1. Keep public documentation aligned with the current implementation
2. Harden persistent teammate runtime: shared LLM locking, restart recovery, process/worktree isolation
3. Add vector/RAG memory V1
4. Add MCP tool integration
5. Expand scheduler/background-task runtime support
