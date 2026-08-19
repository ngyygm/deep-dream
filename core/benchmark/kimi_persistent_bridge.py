"""Long-lived Kimi 1.49 bridge with a fresh Soul/context for every request.

This module is executed by the Python interpreter inside the pinned Kimi
runtime.  Its stdout is a deliberately tiny JSON-lines protocol; Kimi logs
remain on stderr.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
from pathlib import Path
import sys
import time
from typing import Any

import acp
from kaos.path import KaosPath
from kimi_cli.acp.session import ACPSession
from kimi_cli.app import KimiCLI
from kimi_cli.session import Session


class _Collector:
    def __init__(self) -> None:
        self.updates: list[dict[str, Any]] = []

    async def session_update(self, session_id: str, update: Any, **_: Any) -> None:
        data = update.model_dump(by_alias=True, exclude_none=True)
        # Hidden reasoning is neither persisted nor returned to the parent.
        if data.get("sessionUpdate") != "agent_thought_chunk":
            self.updates.append(data)

    async def request_permission(self, **_: Any) -> Any:
        return acp.schema.RequestPermissionResponse(outcome={"outcome": "selected", "optionId": "allow_once"})

    def on_connect(self, _conn: Any) -> None:
        return None


async def _run_one(args: argparse.Namespace, request: dict[str, Any]) -> dict[str, Any]:
    collector = _Collector()
    session = await Session.create(
        KaosPath.unsafe_from_local_path(Path(args.project_root).resolve())
    )
    # Session.create always allocates a new context/wire file.  Never load,
    # resume, fork, or continue a previous session here.
    mcp_config = {
        "mcpServers": {
            "deep-dream": {
                "command": request["mcp_python"],
                "args": [
                    "-m", "core.benchmark.mcp_server",
                    "--run-dir", request["run_dir"],
                    "--scope-id", request["scope_id"],
                    "--config", request["service_config"],
                ],
                "env": {
                    "PYTHONPATH": args.project_root,
                    "DEEPDREAM_JSON_OUTPUT": "1",
                    "DEEPDREAM_LOG_LEVEL": "ERROR",
                },
            }
        }
    }
    cli: KimiCLI | None = None
    started = time.monotonic()
    try:
        cli = await KimiCLI.create(
            session,
            config=Path(args.config),
            model_name="deep-dream-qwen",
            thinking=args.thinking,
            yolo=True,
            afk=True,
            runtime_afk=True,
            agent_file=Path(args.agent_file),
            mcp_configs=[mcp_config],
            max_steps_per_turn=args.max_steps,
            max_retries_per_step=3,
            max_ralph_iterations=0,
            ui_mode="acp",
        )
        acp_session = ACPSession(session.id, cli, collector)
        response = await asyncio.wait_for(
            acp_session.prompt([
                acp.schema.TextContentBlock(type="text", text=request["prompt"])
            ]),
            timeout=float(request["timeout_seconds"]),
        )
        return {
            "ok": True,
            "request_id": request["request_id"],
            "session_id": session.id,
            "fresh_context": True,
            "stop_reason": response.stop_reason,
            "latency_seconds": round(time.monotonic() - started, 3),
            "updates": collector.updates,
            "bridge_pid": os.getpid(),
        }
    finally:
        if cli is not None:
            await cli.shutdown_background_tasks()


async def _main(args: argparse.Namespace) -> None:
    # Requests are intentionally sequential inside one bridge. Parallelism is
    # achieved by one bridge per benchmark worker.
    for line in sys.stdin:
        if not line.strip():
            continue
        try:
            request = json.loads(line)
            result = await _run_one(args, request)
        except BaseException as exc:
            result = {
                "ok": False,
                "request_id": request.get("request_id") if "request" in locals() else None,
                "error_type": type(exc).__name__,
                "error": str(exc),
                "bridge_pid": os.getpid(),
            }
        sys.stdout.write(json.dumps(result, ensure_ascii=False) + "\n")
        sys.stdout.flush()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--agent-file", required=True)
    parser.add_argument("--max-steps", required=True, type=int)
    parser.add_argument("--thinking", action="store_true")
    asyncio.run(_main(parser.parse_args()))


if __name__ == "__main__":
    main()
