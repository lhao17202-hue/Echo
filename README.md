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

Planned work includes vector/RAG memory, persistent teammate agents, MCP integration, and deeper scheduler/background-task integration.

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
  providers/     Anthropic, OpenAI, Ollama, and fake clients
  memory/        working memory and durable JSON memory
  persistence/   sessions, runs, trace, reports, checkpoints
  security/      sandboxing, permissions, env filtering, redaction
  hooks/         hook manager and built-in hooks
  multi_agent/   read-only SubAgent plus collaboration primitives
  scheduler/     scheduler primitives
  evaluation/    benchmark tasks, evaluator, metrics

tests/           unit, integration, regression, and evaluation tests
``

## Built-in Tools

Echo includes file tools, shell execution, search/list helpers, todo management, context compaction, memory tools, and read-only delegation:

-
ead_file, write_file, patch_file
- glob, grep, list_files
-
un_shell
- 	odo_write, compact
- save_memory, search_memory
- delegate

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
2. Add vector/RAG memory V1
3. Add persistent teammate agents and real message-bus runtime integration
4. Add MCP tool integration
5. Expand scheduler/background-task runtime support
