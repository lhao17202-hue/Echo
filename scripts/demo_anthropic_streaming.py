#!/usr/bin/env python3
"""Manual Anthropic streaming demo.

Run:
    python scripts/demo_anthropic_streaming.py

This is not an automated regression test. It is a playground for seeing the
typewriter effect and early tool execution in a terminal.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from echo.config import DEFAULT_ANTHROPIC_MODEL, load_dotenv, provider_env
from echo.providers.anthropic_client import AnthropicClient
from echo.providers.base import TextBlock


def probe_tool(name: str, tool_input: dict) -> str:
    if name != "echo_probe":
        raise ValueError(f"unknown tool: {name}")
    value = tool_input.get("value", "")
    return f"probe:{value}"


def probe_tool_schema() -> dict:
    return {
        "name": "echo_probe",
        "description": "Return a probe value for streaming demo.",
        "input_schema": {
            "type": "object",
            "properties": {
                "value": {
                    "type": "string",
                    "description": "The exact probe value.",
                },
            },
            "required": ["value"],
            "additionalProperties": False,
        },
    }


def anthropic_client() -> AnthropicClient:
    load_dotenv(str(ROOT))
    api_key = provider_env("ANTHROPIC_API_KEY")
    if not api_key:
        raise RuntimeError("ANTHROPIC_API_KEY is not set in .env or environment")

    return AnthropicClient(
        model=provider_env("ANTHROPIC_MODEL", default=DEFAULT_ANTHROPIC_MODEL),
        api_key=api_key,
        base_url=provider_env("ANTHROPIC_BASE_URL") or None,
    )


def demo_typewriter(client: AnthropicClient) -> None:
    print("\n[typewriter] streaming text:")
    started_at = time.perf_counter()
    first_delta_at = None
    chunks = 0

    stream = client.chat_stream(
        [{
            "role": "user",
            "content": [TextBlock(
                text=(
                    "Just output one simple sentence with more than twelve words, never call any tools, only output plain text without extra symbols."
                ),
            )],
        }],
        max_tokens=96,
        temperature=0.0,
    )

    for event in stream:
        # 调试打印所有流式事件
        text_val = getattr(event, "text", "")
        print(f"[DEBUG] event_type={event.type}, text_content={repr(text_val)}", flush=True)

        if event.type == "text_delta" and text_val:
            if first_delta_at is None:
                first_delta_at = time.perf_counter()
            chunks += 1
            print(text_val, end="", flush=True)
        elif event.type == "done":
            break

    finished_at = time.perf_counter()
    print()
    if first_delta_at is not None:
        print(f"[metrics] first_delta_ms={(first_delta_at - started_at) * 1000:.0f}")
    else:
        print("[metrics] first_delta_ms=none")
    print(f"[metrics] chunks={chunks}")
    print(f"[metrics] total_ms={(finished_at - started_at) * 1000:.0f}")


def demo_early_tool_execution(client: AnthropicClient) -> None:
    print("\n[tool] waiting for streamed tool call:")
    started_at = time.perf_counter()
    current_tool = None
    input_chunks = []
    executed_at = None

    stream = client.chat_stream(
        [{
            "role": "user",
            "content": [TextBlock(
                text=(
                    "Call the echo_probe tool once with value set exactly "
                    "to ping. Do not answer in plain text."
                ),
            )],
        }],
        tools=[probe_tool_schema()],
        max_tokens=80,
        temperature=0.0,
    )

    for event in stream:
        if event.type == "tool_use_start":
            current_tool = event
            print(f"[tool-start] name={event.tool_name} id={event.tool_id}", flush=True)
        elif event.type == "tool_use_delta":
            input_chunks.append(event.tool_input_json)
            print(event.tool_input_json, end="", flush=True)
        elif event.type == "tool_use_end" and current_tool is not None:
            tool_input = json.loads("".join(input_chunks))
            result = probe_tool(current_tool.tool_name, tool_input)
            executed_at = time.perf_counter()
            print()
            print(f"[tool-execute] input={tool_input} result={result}", flush=True)
        elif event.type == "done":
            done_at = time.perf_counter()
            print("[stream-done]", flush=True)
            if executed_at is not None:
                print("[metrics] tool_executed_before_done=True")
                print(f"[metrics] tool_ready_ms={(executed_at - started_at) * 1000:.0f}")
                print(f"[metrics] done_ms={(done_at - started_at) * 1000:.0f}")
            break


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        choices=("all", "typewriter", "tool"),
        default="all",
    )
    args = parser.parse_args()

    client = anthropic_client()
    if args.mode in ("all", "typewriter"):
        demo_typewriter(client)
    if args.mode in ("all", "tool"):
        demo_early_tool_execution(client)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
