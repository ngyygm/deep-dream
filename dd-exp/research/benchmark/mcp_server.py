"""Conversation-scoped, read-only Deep-Dream MCP server for benchmark agents."""
from __future__ import annotations

import argparse
import atexit
import copy
import json
import os
from pathlib import Path
from typing import Any

from .agentic import AgenticMemoryTools, AgenticMemoryRunner
from .datasets import group_by_scope, load_benchmark
from .retrieval import UnifiedRetriever


def _string_list(value: list[str] | str | None) -> list[str]:
    """Accept strict arrays plus the JSON-array strings emitted by some tool models."""
    if value is None:
        return []
    if isinstance(value, list):
        return list(dict.fromkeys(str(item) for item in value if str(item)))
    text = value.strip()
    if not text:
        return []
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        parsed = [text]
    if not isinstance(parsed, list):
        raise ValueError("Expected an array of strings")
    return list(dict.fromkeys(str(item) for item in parsed if str(item)))


class ScopedMemoryServer:
    """Open exactly one frozen benchmark scope and track surfaced evidence IDs."""

    def __init__(
        self,
        run_dir: Path,
        scope_id: str,
        config_path: Path,
        *,
        allow_unsurfaced_evidence: bool = False,
    ):
        from core.server.config import load_config
        from core.server.registry import GraphRegistry

        manifest = json.loads((run_dir / "run_manifest.json").read_text(encoding="utf-8"))
        scope_info = (manifest.get("scopes") or {}).get(scope_id)
        if not scope_info:
            raise KeyError(f"Unknown benchmark scope: {scope_id}")
        items, _ = load_benchmark(
            str(manifest["dataset"]), Path(str(manifest["dataset_path"])).parent,
        )
        scope_items = group_by_scope(items).get(scope_id) or []
        if not scope_items:
            raise KeyError(f"Dataset has no items for scope: {scope_id}")
        visible = set(scope_info.get("visible_sessions") or [])
        sessions = [row for row in scope_items[0].sessions if row.session_id in visible]
        documents = scope_info.get("documents") or {}
        document_to_session = {
            str(documents[row.session_id]["document_id"]): row.session_id
            for row in sessions
            if row.session_id in documents and documents[row.session_id].get("status") == "active"
        }
        if not document_to_session:
            raise RuntimeError(f"Scope {scope_id} has no active documents")
        scope_dir = run_dir / str(scope_info["library_dir"])
        config = copy.deepcopy(load_config(str(config_path)))
        config["storage_path"] = str(scope_dir)
        # Benchmark tools are read-only. Avoid GraphRegistry.__init__/get_processor here:
        # those compatibility paths rewrite library.json and their fixed temporary filename
        # is unsafe when multiple stdio MCP processes open the same frozen library.
        self.registry = GraphRegistry.__new__(GraphRegistry)
        self.registry._config = config
        self.registry._embedding_client = None
        processor = self.registry._build_processor(str(scope_dir), "library")
        self.processor = processor  # keep owner alive; its finalizer closes the storage
        retriever = UnifiedRetriever(
            processor.storage, document_to_session, sessions,
            allowed_document_ids=document_to_session,
        )
        self.tools = AgenticMemoryTools(
            processor.storage, retriever, document_to_session, sessions,
        )
        self.storage = processor.storage
        self.allow_unsurfaced_evidence = allow_unsurfaced_evidence
        self.surfaced_sessions: list[str] = []
        self.surfaced_episodes: list[str] = []
        self.surfaced_turns: list[str] = []

    def close(self) -> None:
        self.storage.close()

    def _track(self, value: Any) -> Any:
        AgenticMemoryRunner._collect_ids(
            value, self.surfaced_sessions, self.surfaced_episodes, self.surfaced_turns,
        )
        return value

    def execute(self, name: str, **arguments: Any) -> Any:
        return self._track(self.tools.execute(name, arguments))

    def read_session(
        self, session_id: str, turn_ids: list[str] | None = None,
        offset: int = 0, limit: int = 20, neighbor_turns: int = 1,
    ) -> dict[str, Any]:
        payload = self.tools.read_session({"session_id": session_id})
        lines = str(payload.get("text") or "").splitlines()
        all_ids = list(payload.get("turn_ids") or [])
        selected_indices: list[int] = []
        requested = set(turn_ids or [])
        if requested:
            for index, turn_id in enumerate(all_ids):
                if turn_id in requested:
                    selected_indices.extend(range(
                        max(0, index - max(0, neighbor_turns)),
                        min(len(lines), index + max(0, neighbor_turns) + 1),
                    ))
        else:
            start = max(0, offset)
            selected_indices = list(range(start, min(len(lines), start + max(1, min(limit, 50)))))
        selected_indices = list(dict.fromkeys(selected_indices))
        result = {
            "session_id": session_id,
            "timestamp": payload.get("timestamp"),
            "turn_ids": [all_ids[index] for index in selected_indices if index < len(all_ids)],
            "text": "\n".join(lines[index] for index in selected_indices if index < len(lines)),
            "offset": max(0, offset),
            "has_more": bool(selected_indices and selected_indices[-1] + 1 < len(lines)),
        }
        return self._track(result)

    def submit(
        self,
        session_ids: list[str],
        episode_ids: list[str],
        turn_ids: list[str],
        *,
        allow_unsurfaced_evidence: bool | None = None,
    ) -> dict[str, Any]:
        # X7 arm #6: when the caller explicitly opts in, skip the surfaced-set
        # gate so an answer may consume derived memory the agent never read.
        relaxed = self.allow_unsurfaced_evidence if allow_unsurfaced_evidence is None else allow_unsurfaced_evidence
        if not relaxed:
            unknown = (
                set(session_ids) - set(self.surfaced_sessions)
                or set(episode_ids) - set(self.surfaced_episodes)
                or set(turn_ids) - set(self.surfaced_turns)
            )
            if unknown:
                raise ValueError(f"Evidence was not surfaced by tools: {sorted(unknown)}")
        self.tools.validate_submission(session_ids, episode_ids, turn_ids)
        contexts, ranked_turns = self.tools.contexts_for_submission(
            session_ids, episode_ids, turn_ids, limit=5,
        )
        return {
            "accepted": True,
            "session_ids": [row["session_id"] for row in contexts],
            "episode_ids": list(dict.fromkeys(episode_ids)),
            "turn_ids": ranked_turns,
            "evidence_count": len(contexts) + len(episode_ids) + len(ranked_turns),
        }


def create_mcp_server(state: ScopedMemoryServer):
    try:
        from fastmcp import FastMCP
    except ImportError as exc:  # pragma: no cover - exercised by runtime preflight
        raise RuntimeError("Install the `benchmark-agent` optional dependencies") from exc

    mcp = FastMCP("deep-dream-benchmark-readonly")

    @mcp.tool
    def search_documents(query: str, terms: list[str] | str | None = None, limit: int = 10):
        return state.execute("search_documents", query=query, terms=_string_list(terms), limit=limit)

    @mcp.tool
    def explore_memory(query: str, terms: list[str] | str | None = None, limit: int = 10):
        return state.execute("explore_memory", query=query, terms=_string_list(terms), limit=limit)

    @mcp.tool
    def search_concepts(query: str, limit: int = 10):
        return state.execute("search_concepts", query=query, limit=limit)

    @mcp.tool
    def trace_concept(family_id: str, limit: int = 10):
        return state.execute("trace_concept", family_id=family_id, limit=limit)

    @mcp.tool
    def expand_neighbors(family_id: str, depth: int = 1, limit: int = 10):
        return state.execute("expand_neighbors", family_id=family_id, depth=depth, limit=limit)

    @mcp.tool
    def relation_evidence(concept_a: str, concept_b: str, limit: int = 10):
        return state.execute(
            "relation_evidence", concept_a=concept_a, concept_b=concept_b, limit=limit,
        )

    @mcp.tool
    def read_episode(episode_id: str):
        return state.execute("read_episode", episode_id=episode_id)

    @mcp.tool
    def read_session(
        session_id: str, turn_ids: list[str] | str | None = None, offset: int = 0,
        limit: int = 20, neighbor_turns: int = 1,
    ):
        return state.read_session(
            session_id, turn_ids=_string_list(turn_ids), offset=offset, limit=limit,
            neighbor_turns=neighbor_turns,
        )

    @mcp.tool
    def submit_evidence(
        session_ids: list[str] | str | None = None,
        episode_ids: list[str] | str | None = None,
        turn_ids: list[str] | str | None = None,
    ):
        return state.submit(
            _string_list(session_ids), _string_list(episode_ids), _string_list(turn_ids),
        )

    return mcp


def main(argv: list[str] | None = None) -> int:
    # MCP reserves stdout for JSON-RPC. Deep-Dream logging checks this env dynamically.
    os.environ["DEEPDREAM_JSON_OUTPUT"] = "1"
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--scope-id", required=True)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args(argv)
    state = ScopedMemoryServer(args.run_dir.resolve(), args.scope_id, args.config.resolve())
    atexit.register(state.close)
    create_mcp_server(state).run(transport="stdio", show_banner=False)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
