"""Tiny JSON-RPC stdio MCP server for Echo tests.

It intentionally implements only initialize, tools/list, tools/call, and notifications/initialized.
"""

from __future__ import annotations

import json
import os
import sys


def send(payload: dict) -> None:
    sys.stdout.write(json.dumps(payload) + "\n")
    sys.stdout.flush()


def make_result(request_id, result):
    return {"jsonrpc": "2.0", "id": request_id, "result": result}


def make_error(request_id, code, message):
    return {"jsonrpc": "2.0", "id": request_id, "error": {"code": code, "message": message}}


def handle(request: dict) -> None:
    method = request.get("method")
    request_id = request.get("id")

    if method == "notifications/initialized":
        return

    if method == "initialize":
        send(make_result(request_id, {
            "protocolVersion": "2024-11-05",
            "capabilities": {"tools": {}},
            "serverInfo": {"name": "echo-demo", "version": "0.1.0"},
        }))
        return

    if method == "tools/list":
        send(make_result(request_id, {
            "tools": [
                {
                    "name": "get_status",
                    "description": "Return demo status.",
                    "inputSchema": {
                        "type": "object",
                        "properties": {"verbose": {"type": "boolean"}},
                    },
                },
                {
                    "name": "echo_structured",
                    "description": "Return structured content.",
                    "inputSchema": {"type": "object", "properties": {}},
                },
                {
                    "name": "fail_tool",
                    "description": "Return an MCP error result.",
                    "inputSchema": {"type": "object", "properties": {}},
                },
                {
                    "name": "read_env",
                    "description": "Return selected environment variables.",
                    "inputSchema": {"type": "object", "properties": {}},
                },
            ]
        }))
        return

    if method == "tools/call":
        params = request.get("params", {})
        name = params.get("name")
        arguments = params.get("arguments", {})
        if name == "get_status":
            suffix = " verbose" if arguments.get("verbose") in (True, "true", "yes", "1") else ""
            send(make_result(request_id, {"content": [{"type": "text", "text": f"status ok{suffix}"}]}))
            return
        if name == "echo_structured":
            send(make_result(request_id, {"structuredContent": {"ok": True, "answer": 42}}))
            return
        if name == "fail_tool":
            send(make_result(request_id, {"isError": True, "content": [{"type": "text", "text": "demo failure"}]}))
            return
        if name == "read_env":
            send(make_result(request_id, {
                "content": [{
                    "type": "text",
                    "text": json.dumps({
                        "DEMO_TOKEN": os.environ.get("DEMO_TOKEN", ""),
                        "HOST_SECRET": os.environ.get("HOST_SECRET", ""),
                        "PATH_PRESENT": bool(os.environ.get("PATH")),
                    }, sort_keys=True),
                }]
            }))
            return
        send(make_error(request_id, -32602, f"Unknown tool: {name}"))
        return

    send(make_error(request_id, -32601, f"Unknown method: {method}"))


def main() -> None:
    for line in sys.stdin:
        if not line.strip():
            continue
        handle(json.loads(line))


if __name__ == "__main__":
    main()
