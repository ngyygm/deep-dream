"""Command line interface for the v1 Document-first concept graph."""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Any, Dict

from core.server.config import DEFAULTS, load_config
from core.server.registry import GraphRegistry
from core.storage.sqlite import SQLiteGraphStorageManager


def _load_config(path: str) -> Dict[str, Any]:
    try:
        return load_config(path)
    except Exception:
        return copy.deepcopy(DEFAULTS)


def _registry(config: Dict[str, Any]) -> GraphRegistry:
    return GraphRegistry(config.get("storage_path", "./graph"), config)


def _storage_for(config: Dict[str, Any], graph_id: str) -> SQLiteGraphStorageManager:
    registry = _registry(config)
    graph_dir = registry.graph_dir(graph_id)
    graph_dir.mkdir(parents=True, exist_ok=True)
    registry.set_graph_metadata(graph_id)
    vector_dim = (config.get("storage") or {}).get("vector_dim", 1024)
    return SQLiteGraphStorageManager(
        storage_path=str(graph_dir),
        graph_id=graph_id,
        vector_dim=vector_dim,
    )


def _registry_json_path(config: Dict[str, Any]) -> Path:
    return Path(config.get("storage_path", "./graph")) / "registry.json"


def _write_active_graph(config: Dict[str, Any], graph_id: str) -> None:
    path = _registry_json_path(config)
    data = {"graphs": {}}
    if path.exists():
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            data = {"graphs": {}}
    data.setdefault("graphs", {})
    data["active_graph_id"] = graph_id
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def _active_graph(config: Dict[str, Any], explicit: str | None = None) -> str:
    if explicit:
        return explicit
    path = _registry_json_path(config)
    if path.exists():
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            if data.get("active_graph_id"):
                return str(data["active_graph_id"])
        except json.JSONDecodeError:
            pass
    return "default"


def _print_json(data: Any) -> None:
    print(json.dumps(data, ensure_ascii=False, indent=2))


def _cmd_graph_create(args, config: Dict[str, Any]) -> int:
    with _storage_for(config, args.graph_id) as storage:
        stats = storage.get_stats()
    _print_json({"graph_id": args.graph_id, "created": True, "stats": stats})
    return 0


def _cmd_graph_use(args, config: Dict[str, Any]) -> int:
    registry = _registry(config)
    registry.set_graph_metadata(args.graph_id)
    _write_active_graph(config, args.graph_id)
    _print_json({"active_graph_id": args.graph_id})
    return 0


def _cmd_vault_index(args, config: Dict[str, Any]) -> int:
    graph_id = _active_graph(config, args.graph)
    with _storage_for(config, graph_id) as storage:
        result = storage.index_vault(args.path, force=args.force)
    _print_json({"graph_id": graph_id, **result})
    return 0


def _cmd_remember(args, config: Dict[str, Any]) -> int:
    graph_id = _active_graph(config, args.graph)
    file_path = Path(args.file)
    text = file_path.read_text(encoding=args.encoding)
    registry = _registry(config)
    processor = registry.get_processor(graph_id)
    result = processor.remember_text(
        text,
        doc_name=args.source or file_path.name,
        verbose=bool(args.verbose),
        source_document=str(file_path.resolve()),
    )
    _print_json({"graph_id": graph_id, "result": result})
    return 0


def _cmd_find(args, config: Dict[str, Any]) -> int:
    graph_id = _active_graph(config, args.graph)
    with _storage_for(config, graph_id) as storage:
        concepts = storage.search_concepts_by_bm25(
            args.query,
            role=args.role,
            limit=args.limit,
            time_point=args.time_point,
        )
    _print_json({"graph_id": graph_id, "concepts": concepts, "total": len(concepts)})
    return 0


def _cmd_trace(args, config: Dict[str, Any]) -> int:
    graph_id = _active_graph(config, args.graph)
    with _storage_for(config, graph_id) as storage:
        provenance = storage.get_concept_provenance(args.family_id, time_point=args.time_point)
    _print_json({"graph_id": graph_id, "family_id": args.family_id, "provenance": provenance})
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="deep-dream", description="Deep-Dream v1 concept graph CLI")
    parser.add_argument("--config", default="service_config.json", help="Path to service_config.json")
    sub = parser.add_subparsers(dest="command", required=True)

    graph = sub.add_parser("graph", help="Manage physically isolated graphs")
    graph_sub = graph.add_subparsers(dest="graph_command", required=True)
    graph_create = graph_sub.add_parser("create", help="Create a graph")
    graph_create.add_argument("graph_id")
    graph_create.set_defaults(func=_cmd_graph_create)
    graph_use = graph_sub.add_parser("use", help="Set the active graph")
    graph_use.add_argument("graph_id")
    graph_use.set_defaults(func=_cmd_graph_use)

    vault = sub.add_parser("vault", help="Index Markdown vaults")
    vault_sub = vault.add_subparsers(dest="vault_command", required=True)
    vault_index = vault_sub.add_parser("index", help="Index a Markdown/Obsidian vault")
    vault_index.add_argument("path")
    vault_index.add_argument("--graph")
    vault_index.add_argument("--force", action="store_true")
    vault_index.set_defaults(func=_cmd_vault_index)

    remember = sub.add_parser("remember", help="Run remember on a Markdown/text file")
    remember.add_argument("--file", required=True)
    remember.add_argument("--graph")
    remember.add_argument("--source")
    remember.add_argument("--encoding", default="utf-8")
    remember.add_argument("--verbose", action="store_true")
    remember.set_defaults(func=_cmd_remember)

    find = sub.add_parser("find", help="Search concepts")
    find.add_argument("query")
    find.add_argument("--graph")
    find.add_argument("--role", choices=["document", "episode", "entity", "relation"])
    find.add_argument("--limit", type=int, default=20)
    find.add_argument("--time-point")
    find.set_defaults(func=_cmd_find)

    trace = sub.add_parser("trace", help="Trace concept provenance")
    trace.add_argument("family_id")
    trace.add_argument("--graph")
    trace.add_argument("--time-point")
    trace.set_defaults(func=_cmd_trace)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    config = _load_config(args.config)
    return args.func(args, config)


if __name__ == "__main__":
    raise SystemExit(main())
