"""Command line interface for the v1 Document-first concept graph.

The CLI is the agent-facing local entrypoint. It keeps raw documents as the
first retrieval layer, then exposes episode/concept/relation mapping helpers
for deeper graph work.
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import re
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from core.server.config import DEFAULTS, load_config
from core.server.registry import LIBRARY_ID, GraphRegistry
from core.storage.sqlite import SQLiteGraphStorageManager
from core.library import migrate_legacy_graphs


def _load_config(path: str) -> Dict[str, Any]:
    try:
        config = load_config(path)
    except Exception:
        config = copy.deepcopy(DEFAULTS)
    if not config.get("storage_path"):
        config["storage_path"] = "./library"
    return config


def _registry(config: Dict[str, Any]) -> GraphRegistry:
    return GraphRegistry(config.get("storage_path", "./graph"), config)


def _storage_for(config: Dict[str, Any], graph_id: str, *, ensure: bool = False) -> SQLiteGraphStorageManager:
    graph_id = GraphRegistry.normalize_graph_id(graph_id)
    registry = _registry(config)
    graph_dir = registry.graph_dir(graph_id)
    if ensure:
        graph_dir.mkdir(parents=True, exist_ok=True)
        registry.set_graph_metadata(graph_id)
    elif not graph_dir.is_dir():
        raise FileNotFoundError(f"图谱不存在: {graph_id}")
    vector_dim = (config.get("storage") or {}).get("vector_dim", 1024)
    return SQLiteGraphStorageManager(
        storage_path=str(graph_dir),
        graph_id=graph_id,
        vector_dim=vector_dim,
    )


def _registry_json_path(config: Dict[str, Any]) -> Path:
    return Path(config.get("storage_path", "./library")) / "library.json"


def _write_active_graph(config: Dict[str, Any], graph_id: str) -> None:
    path = _registry_json_path(config)
    data = {"library": {"id": LIBRARY_ID, "graph_id": LIBRARY_ID}}
    if path.exists():
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            data = {"library": {"id": LIBRARY_ID, "graph_id": LIBRARY_ID}}
    data.setdefault("library", {"id": LIBRARY_ID, "graph_id": LIBRARY_ID})
    data["library"]["graph_id"] = LIBRARY_ID
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def _active_graph(config: Dict[str, Any], explicit: str | None = None) -> str:
    if explicit:
        return LIBRARY_ID
    path = _registry_json_path(config)
    if path.exists():
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            if data.get("active_graph_id"):
                return LIBRARY_ID
        except json.JSONDecodeError:
            pass
    return LIBRARY_ID


def _print_json(data: Any) -> None:
    print(json.dumps(data, ensure_ascii=False, indent=2))


def _ok(command: str, graph_id: Optional[str], data: Any, **extra: Any) -> Dict[str, Any]:
    payload = {
        "success": True,
        "command": command,
        "graph_id": graph_id,
        "data": data,
    }
    payload.update(extra)
    return payload


def _storage_root(config: Dict[str, Any]) -> Path:
    return Path(config.get("storage_path") or ".")


def _graph_dir(config: Dict[str, Any], graph_id: str) -> Path:
    return _registry(config).graph_dir(graph_id)


def _read_sql(storage: SQLiteGraphStorageManager, sql: str, params: Any = None, limit: int = 200) -> List[dict]:
    return storage.read_sql(sql, params=params or {}, limit=limit)["rows"]


def _resolve_storage_path(storage: SQLiteGraphStorageManager, path_value: str) -> Path:
    path = Path(path_value)
    if path.is_absolute():
        return path
    resolver = getattr(storage, "_resolve_storage_path", None)
    if resolver is not None:
        try:
            return resolver(path_value)
        except Exception:
            pass
    return Path(storage.storage_path) / path_value


def _readable_document_path(storage: SQLiteGraphStorageManager, doc: dict) -> tuple[Optional[Path], str]:
    candidates: list[tuple[str, str]] = []
    if doc.get("source_mode") == "external" and doc.get("absolute_path"):
        candidates.append((doc["absolute_path"], "raw_file"))
    for key, label in (
        ("read_path", "raw_file"),
        ("managed_path", "raw_file"),
        ("snapshot_path", "snapshot"),
        ("absolute_path", "raw_file"),
    ):
        value = doc.get(key) or ""
        if value:
            candidates.append((value, label))
    seen = set()
    for value, label in candidates:
        if value in seen:
            continue
        seen.add(value)
        path = _resolve_storage_path(storage, value)
        if path.is_file():
            return path, label
    return None, "missing"


def _document_file_payload(storage: SQLiteGraphStorageManager, doc: dict) -> dict:
    path, verification = _readable_document_path(storage, doc)
    item = dict(doc)
    item["resolved_path"] = str(path) if path else ""
    item["verification"] = verification
    return item


def _document_rows(storage: SQLiteGraphStorageManager, limit: int = 500) -> List[dict]:
    return _read_sql(
        storage,
        """
        SELECT document_version_id, document_family_id, title, source_mode,
               absolute_path, managed_path, snapshot_path, relative_path,
               vault_root, read_path, content_hash, byte_size, char_count,
               line_count, processed_time, complete_windows, total_windows,
               missing_windows
        FROM v_document_files
        ORDER BY processed_time DESC
        """,
        limit=limit,
    )


def _map_path_to_documents(storage: SQLiteGraphStorageManager, file_path: str, limit: int = 20) -> List[dict]:
    raw = str(file_path)
    resolved = str(Path(file_path).expanduser().resolve())
    rows = _read_sql(
        storage,
        """
        SELECT *
        FROM v_document_files
        WHERE absolute_path IN (:raw, :resolved)
           OR managed_path IN (:raw, :resolved)
           OR snapshot_path IN (:raw, :resolved)
           OR read_path IN (:raw, :resolved)
           OR relative_path = :raw
        ORDER BY processed_time DESC
        """,
        {"raw": raw, "resolved": resolved},
        limit=limit,
    )
    if rows:
        return rows
    matches = []
    for doc in _document_rows(storage, limit=5000):
        payload = _document_file_payload(storage, doc)
        if payload.get("resolved_path") and str(Path(payload["resolved_path"]).resolve()) == resolved:
            matches.append(doc)
            if len(matches) >= limit:
                break
    return matches


def _iter_searchable_documents(storage: SQLiteGraphStorageManager, limit: int = 1000) -> Iterable[dict]:
    for doc in _document_rows(storage, limit=limit):
        payload = _document_file_payload(storage, doc)
        if payload.get("resolved_path"):
            yield payload


def _search_document_files(storage: SQLiteGraphStorageManager, pattern: str, *, regex: bool, limit: int) -> list[dict]:
    if not pattern:
        raise ValueError("pattern 不能为空")
    matcher = re.compile(pattern, re.IGNORECASE) if regex else None
    hits: list[dict] = []
    for doc in _iter_searchable_documents(storage):
        path = Path(doc["resolved_path"])
        try:
            lines = path.read_text(encoding="utf-8").splitlines()
        except UnicodeDecodeError:
            try:
                lines = path.read_text(encoding="utf-8-sig").splitlines()
            except Exception:
                continue
        except OSError:
            continue
        for line_no, line in enumerate(lines, start=1):
            matched = bool(matcher.search(line)) if matcher else pattern.lower() in line.lower()
            if not matched:
                continue
            hits.append({
                "document": {
                    "document_version_id": doc.get("document_version_id", ""),
                    "title": doc.get("title", ""),
                    "read_path": doc.get("resolved_path") or doc.get("read_path", ""),
                    "source_mode": doc.get("source_mode", ""),
                    "line_start": line_no,
                    "line_end": line_no,
                },
                "episode": None,
                "concepts": [],
                "relations": [],
                "verification": doc.get("verification", "raw_file"),
                "text": line,
            })
            if len(hits) >= limit:
                return hits
    return hits


def _expand_query_terms(query: str, explicit_terms: Optional[str] = None) -> list[dict]:
    """Return user/agent-provided query terms without domain-specific defaults."""
    raw_terms = [query.strip()] if query and query.strip() else []
    if explicit_terms:
        raw_terms.extend(t.strip() for t in explicit_terms.split(",") if t.strip())

    seen = set()
    out = []
    for idx, term in enumerate(raw_terms):
        normalized = term.strip()
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        out.append({
            "term": normalized,
            "source": "original" if idx == 0 and normalized == query.strip() else "expanded",
        })
    return out


def _search_document_terms(
    storage: SQLiteGraphStorageManager,
    terms: list[dict],
    *,
    per_term_limit: int,
    total_limit: int,
) -> list[dict]:
    hits: list[dict] = []
    seen = set()
    for term_info in terms:
        term = term_info["term"]
        for hit in _search_document_files(storage, term, regex=False, limit=per_term_limit):
            doc = hit.get("document") or {}
            key = (doc.get("document_version_id"), doc.get("line_start"), hit.get("text"))
            if key in seen:
                continue
            seen.add(key)
            hit["matched_term"] = term
            hit["term_source"] = term_info.get("source", "expanded")
            hits.append(hit)
            if len(hits) >= total_limit:
                return hits
    return hits


def _compact_text(text: str, matched_terms: Iterable[str] = (), max_chars: int = 280) -> str:
    clean = re.sub(r"\s+", " ", text or "").strip()
    if len(clean) <= max_chars:
        return clean
    terms = [t for t in matched_terms if t]
    positions = [clean.find(t) for t in terms if clean.find(t) >= 0]
    if positions:
        center = min(positions)
        start = max(0, center - max_chars // 3)
    else:
        start = 0
    end = min(len(clean), start + max_chars)
    prefix = "..." if start > 0 else ""
    suffix = "..." if end < len(clean) else ""
    return f"{prefix}{clean[start:end]}{suffix}"


def _evidence_cards(file_hits: list[dict], source_evidence: list[dict], terms: list[dict], limit: int) -> list[dict]:
    term_values = [t["term"] for t in terms]
    cards: list[dict] = []
    seen = set()
    for hit in file_hits:
        doc = hit.get("document") or {}
        key = ("file", doc.get("document_version_id"), doc.get("line_start"), hit.get("text"))
        if key in seen:
            continue
        seen.add(key)
        cards.append({
            "claim_hint": "raw document match",
            "document": doc,
            "episode": None,
            "matched_terms": [hit.get("matched_term") or ""],
            "source_excerpt": _compact_text(hit.get("text", ""), [hit.get("matched_term", "")]),
            "verification": hit.get("verification", "raw_file"),
        })
        if len(cards) >= limit:
            return cards
    for ev in source_evidence:
        key = ("episode", ev.get("episode_version_id"), ev.get("target_family_id"))
        if key in seen:
            continue
        seen.add(key)
        matched = [t for t in term_values if t and t in (ev.get("source_text") or "")]
        cards.append({
            "claim_hint": ev.get("target_name") or ev.get("target_role") or "graph evidence",
            "document": {
                "document_version_id": ev.get("document_version_id", ""),
                "title": ev.get("title", ""),
                "read_path": ev.get("read_path", ""),
                "source_mode": ev.get("source_mode", ""),
                "line_start": ev.get("line_start"),
                "line_end": ev.get("line_end"),
            },
            "episode": {
                "episode_version_id": ev.get("episode_version_id", ""),
                "heading_path": ev.get("heading_path", ""),
            },
            "concepts": [{
                "family_id": ev.get("target_family_id", ""),
                "name": ev.get("target_name", ""),
                "role": ev.get("target_role", ""),
            }],
            "matched_terms": matched,
            "source_excerpt": _compact_text(ev.get("source_text", ""), matched),
            "verification": "source_text",
        })
        if len(cards) >= limit:
            return cards
    return cards


def _resolve_concept_id(storage: SQLiteGraphStorageManager, value: str) -> Optional[str]:
    concept = storage.get_concept_by_family_id(value)
    if concept:
        return concept["family_id"]
    matches = storage.search_concepts_by_bm25(value, limit=1)
    return matches[0]["family_id"] if matches else None


def _concept_source_evidence(storage: SQLiteGraphStorageManager, family_ids: Iterable[str], limit: int = 20) -> list[dict]:
    ids = [fid for fid in dict.fromkeys(family_ids) if fid]
    if not ids:
        return []
    placeholders = ",".join(f":id{i}" for i in range(len(ids)))
    params = {f"id{i}": fid for i, fid in enumerate(ids)}
    return _read_sql(
        storage,
        f"""
        SELECT d.title, d.read_path, d.source_mode, ep.version_id AS episode_version_id,
               ep.document_version_id, ep.heading_path, ep.line_start, ep.line_end,
               ep.source_text, m.target_family_id, m.target_name, m.target_role
        FROM v_mentions m
        JOIN v_episodes ep ON ep.version_id = m.episode_version_id
        LEFT JOIN v_document_files d ON d.document_version_id = ep.document_version_id
        WHERE m.target_family_id IN ({placeholders})
        ORDER BY d.processed_time DESC, ep.start_offset
        """,
        params,
        limit=limit,
    )


def _relation_evidence(storage: SQLiteGraphStorageManager, concept_a: str, concept_b: str, limit: int = 50) -> list[dict]:
    a = _resolve_concept_id(storage, concept_a)
    b = _resolve_concept_id(storage, concept_b)
    if not a or not b:
        return []
    return _read_sql(
        storage,
        """
        SELECT re.relation_family_id, re.relation_version_id,
               re.relation_name, re.relation_content,
               re.entity1_name, re.entity2_name,
               d.title, d.read_path, d.source_mode,
               ep.version_id AS episode_version_id,
               ep.line_start, ep.line_end, ep.source_text
        FROM v_relation_edges re
        JOIN v_episodes ep ON ep.version_id = re.episode_version_id
        LEFT JOIN v_document_files d ON d.document_version_id = re.document_version_id
        WHERE (re.entity1_family_id = :a AND re.entity2_family_id = :b)
           OR (re.entity1_family_id = :b AND re.entity2_family_id = :a)
        ORDER BY d.title, ep.start_offset
        """,
        {"a": a, "b": b},
        limit=limit,
    )


def _cmd_doctor(args, config: Dict[str, Any]) -> int:
    registry = _registry(config)
    api_base = args.api_base.rstrip("/")
    api_health: dict[str, Any] = {"available": False}
    try:
        with urllib.request.urlopen(f"{api_base}/health", timeout=2) as resp:
            api_health = {"available": True, "response": json.loads(resp.read().decode("utf-8"))}
    except (OSError, urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
        api_health = {"available": False, "error": str(exc)}
    graph_infos = registry.list_graphs_info()
    _print_json(_ok("doctor", None, {
        "storage_path": str(_storage_root(config).resolve()),
        "registry_path": str(_registry_json_path(config).resolve()),
        "graphs": graph_infos,
        "graph_count": len(graph_infos),
        "api_base": api_base,
        "api_health": api_health,
    }))
    return 0


def _cmd_library_migrate(args, config: Dict[str, Any]) -> int:
    config_path = Path(args.config).resolve()
    config_dir = config_path.parent if config_path.exists() else Path.cwd()
    legacy_root = Path(args.legacy_root).resolve() if args.legacy_root else config_dir
    target_root = (
        Path(args.target_root).resolve()
        if args.target_root
        else Path(config.get("storage_path") or "./library").resolve()
    )
    result = migrate_legacy_graphs(
        legacy_root=legacy_root,
        target_root=target_root,
        source_ids=args.source or None,
        backup=not args.no_backup,
        force=args.force,
    )
    _print_json(_ok("library migrate", LIBRARY_ID, result))
    return 0


def _cmd_graph_create(args, config: Dict[str, Any]) -> int:
    with _storage_for(config, args.graph_id, ensure=True) as storage:
        stats = storage.get_stats()
    _print_json(_ok("graph create", args.graph_id, {"created": True, "stats": stats}))
    return 0


def _cmd_graph_list(args, config: Dict[str, Any]) -> int:
    registry = _registry(config)
    _print_json(_ok("graph list", None, {
        "graphs": registry.list_graphs(),
        "graphs_info": registry.list_graphs_info(),
    }))
    return 0


def _cmd_graph_use(args, config: Dict[str, Any]) -> int:
    registry = _registry(config)
    registry.set_graph_metadata(args.graph_id)
    _write_active_graph(config, args.graph_id)
    _print_json(_ok("graph use", args.graph_id, {"active_graph_id": args.graph_id}))
    return 0


def _cmd_graph_stats(args, config: Dict[str, Any]) -> int:
    graph_id = _active_graph(config, args.graph)
    info = _registry(config).get_graph_info(graph_id)
    _print_json(_ok("graph stats", graph_id, info or {}))
    return 0


def _cmd_rebuild(args, config: Dict[str, Any]) -> int:
    graph_id = _active_graph(config, args.graph)
    with _storage_for(config, graph_id) as storage:
        before = _registry(config).get_graph_info(graph_id) or {}
        storage.clear_graph_data()
    _print_json(_ok("rebuild", graph_id, {
        "cleared": True,
        "previous_stats": {
            "families": before.get("concept_family_count", 0),
            "versions": before.get("concept_version_count", 0),
            "edges": before.get("concept_edge_count", 0),
            "documents": before.get("document_count", 0),
        },
        "message": "Graph data cleared. Re-run remember or vault index to rebuild.",
    }))
    return 0


def _cmd_vault_index(args, config: Dict[str, Any]) -> int:
    graph_id = _active_graph(config, args.graph)
    with _storage_for(config, graph_id, ensure=True) as storage:
        result = storage.index_vault(args.path, force=args.force)
    _print_json(_ok("vault index", graph_id, result))
    return 0


def _cmd_remember(args, config: Dict[str, Any]) -> int:
    graph_id = _active_graph(config, args.graph)
    if args.text is not None:
        text = args.text
        source = args.source or "cli:text"
        source_document = source
    else:
        file_path = Path(args.file)
        text = file_path.read_text(encoding=args.encoding)
        source = args.source or file_path.name
        source_document = str(file_path.resolve())
    registry = _registry(config)
    processor = registry.get_processor(graph_id)
    result = processor.remember_text(
        text,
        doc_name=source,
        verbose=bool(args.verbose),
        source_document=source_document,
    )
    _print_json(_ok("remember", graph_id, {"result": result}))
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
    _print_json(_ok("find", graph_id, {"concepts": concepts, "total": len(concepts)}))
    return 0


def _cmd_trace(args, config: Dict[str, Any]) -> int:
    graph_id = _active_graph(config, args.graph)
    with _storage_for(config, graph_id) as storage:
        provenance = storage.get_concept_provenance(args.family_id, time_point=args.time_point)
    _print_json(_ok("trace", graph_id, {"family_id": args.family_id, "provenance": provenance}))
    return 0


def _cmd_docs_roots(args, config: Dict[str, Any]) -> int:
    graph_id = _active_graph(config, args.graph)
    with _storage_for(config, graph_id) as storage:
        docs = _document_rows(storage, limit=5000)
        roots = set()
        for doc in docs:
            for key in ("vault_root",):
                if doc.get(key):
                    roots.add(str(Path(doc[key]).resolve()))
            if doc.get("source_mode") == "external" and doc.get("absolute_path"):
                roots.add(str(Path(doc["absolute_path"]).resolve().parent))
        roots.add(str(Path(storage.storage_path).resolve() / "content"))
        _print_json(_ok("docs roots", graph_id, {"roots": sorted(roots), "document_count": len(docs)}))
    return 0


def _cmd_docs_list(args, config: Dict[str, Any]) -> int:
    graph_id = _active_graph(config, args.graph)
    with _storage_for(config, graph_id) as storage:
        docs = [_document_file_payload(storage, d) for d in _document_rows(storage, limit=args.limit)]
    _print_json(_ok("docs list", graph_id, {"documents": docs, "total": len(docs)}))
    return 0


def _cmd_docs_path(args, config: Dict[str, Any]) -> int:
    graph_id = _active_graph(config, args.graph)
    with _storage_for(config, graph_id) as storage:
        info = storage.get_document_file_info(args.document_id)
        info = _document_file_payload(storage, info)
    _print_json(_ok("docs path", graph_id, info))
    return 0


def _cmd_docs_search(args, config: Dict[str, Any]) -> int:
    graph_id = _active_graph(config, args.graph)
    with _storage_for(config, graph_id) as storage:
        hits = _search_document_files(storage, args.pattern, regex=False, limit=args.limit)
    _print_json(_ok("docs search", graph_id, {"hits": hits, "total": len(hits)}, used={
        "raw_files": True,
        "sqlite": True,
        "semantic": False,
        "graph_traversal": False,
        "api": False,
    }))
    return 0


def _cmd_docs_grep(args, config: Dict[str, Any]) -> int:
    graph_id = _active_graph(config, args.graph)
    with _storage_for(config, graph_id) as storage:
        hits = _search_document_files(storage, args.pattern, regex=True, limit=args.limit)
    _print_json(_ok("docs grep", graph_id, {"hits": hits, "total": len(hits)}, used={
        "raw_files": True,
        "sqlite": True,
        "semantic": False,
        "graph_traversal": False,
        "api": False,
    }))
    return 0


def _cmd_docs_map(args, config: Dict[str, Any]) -> int:
    graph_id = _active_graph(config, args.graph)
    with _storage_for(config, graph_id) as storage:
        docs = [_document_file_payload(storage, d) for d in _map_path_to_documents(storage, args.path)]
    _print_json(_ok("docs map", graph_id, {"path": args.path, "documents": docs, "total": len(docs)}))
    return 0


def _cmd_episode_from_file(args, config: Dict[str, Any]) -> int:
    graph_id = _active_graph(config, args.graph)
    with _storage_for(config, graph_id) as storage:
        docs = [_document_file_payload(storage, d) for d in _map_path_to_documents(storage, args.path, limit=5)]
        episodes: list[dict] = []
        for doc in docs:
            params = {"doc": doc["document_version_id"]}
            line_clause = ""
            if args.line is not None:
                params["line"] = int(args.line)
                params["offset"] = -1
                resolved_path = doc.get("resolved_path")
                if resolved_path:
                    try:
                        content_lines = Path(resolved_path).read_text(encoding="utf-8").splitlines(keepends=True)
                        if 1 <= args.line <= len(content_lines):
                            params["offset"] = sum(len(line) for line in content_lines[: args.line - 1])
                    except OSError:
                        pass
                line_clause = """
                  AND (
                    (
                      line_start IS NOT NULL
                      AND COALESCE(CAST(line_start AS INTEGER), -1) <= :line
                      AND COALESCE(CAST(line_end AS INTEGER), 2147483647) >= :line
                    )
                    OR (
                      :offset >= 0
                      AND COALESCE(CAST(start_offset AS INTEGER), -1) <= :offset
                      AND COALESCE(CAST(end_offset AS INTEGER), -1) >= :offset
                    )
                  )
                """
            episodes.extend(_read_sql(
                storage,
                f"""
                SELECT version_id AS episode_version_id, document_version_id,
                       heading_path, start_offset, end_offset, line_start,
                       line_end, source_path, source_text
                FROM v_episodes
                WHERE document_version_id = :doc {line_clause}
                ORDER BY start_offset
                """,
                params,
                limit=args.limit,
            ))
        _print_json(_ok("episode from-file", graph_id, {"documents": docs, "episodes": episodes, "total": len(episodes)}))
    return 0


def _cmd_episode_concepts(args, config: Dict[str, Any]) -> int:
    graph_id = _active_graph(config, args.graph)
    with _storage_for(config, graph_id) as storage:
        concepts = _read_sql(
            storage,
            """
            SELECT m.target_family_id AS family_id, m.target_name AS name,
                   lc.role, lc.content, lc.confidence, m.provenance
            FROM v_mentions m
            JOIN v_latest_concept lc
              ON lc.graph_id = m.graph_id
             AND lc.family_id = m.target_family_id
            WHERE m.episode_version_id = :episode
            ORDER BY m.target_name
            """,
            {"episode": args.episode_id},
            limit=args.limit,
        )
    _print_json(_ok("episode concepts", graph_id, {"episode_id": args.episode_id, "concepts": concepts, "total": len(concepts)}))
    return 0


def _cmd_concept_search(args, config: Dict[str, Any]) -> int:
    graph_id = _active_graph(config, args.graph)
    with _storage_for(config, graph_id) as storage:
        if args.semantic:
            result = storage.agent_semantic_search(args.query, role=args.role, top_k=args.limit, threshold=args.threshold)
            concepts = result["results"]
        else:
            concepts = storage.search_concepts_by_bm25(args.query, role=args.role, limit=args.limit)
    _print_json(_ok("concept search", graph_id, {"concepts": concepts, "total": len(concepts)}, used={
        "raw_files": False,
        "sqlite": True,
        "semantic": bool(args.semantic),
        "graph_traversal": False,
        "api": False,
    }))
    return 0


def _cmd_concept_trace(args, config: Dict[str, Any]) -> int:
    return _cmd_trace(args, config)


def _cmd_concept_neighbors(args, config: Dict[str, Any]) -> int:
    graph_id = _active_graph(config, args.graph)
    with _storage_for(config, graph_id) as storage:
        fid = _resolve_concept_id(storage, args.family_id)
        neighbors = storage.get_concept_neighbors(fid, max_depth=args.depth, max_results=args.limit) if fid else []
    _print_json(_ok("concept neighbors", graph_id, {"family_id": fid, "neighbors": neighbors, "total": len(neighbors)}, used={
        "raw_files": False,
        "sqlite": True,
        "semantic": False,
        "graph_traversal": True,
        "api": False,
    }))
    return 0


def _cmd_relation_evidence(args, config: Dict[str, Any]) -> int:
    graph_id = _active_graph(config, args.graph)
    with _storage_for(config, graph_id) as storage:
        evidence = _relation_evidence(storage, args.concept_a, args.concept_b, limit=args.limit)
    _print_json(_ok("relation evidence", graph_id, {"evidence": evidence, "total": len(evidence)}, used={
        "raw_files": False,
        "sqlite": True,
        "semantic": False,
        "graph_traversal": True,
        "api": False,
    }))
    return 0


def _cmd_sql(args, config: Dict[str, Any]) -> int:
    graph_id = _active_graph(config, args.graph)
    with _storage_for(config, graph_id) as storage:
        result = storage.read_sql(args.query, limit=args.limit, include_query_plan=args.explain)
    _print_json(_ok("sql", graph_id, result, used={
        "raw_files": False,
        "sqlite": True,
        "semantic": False,
        "graph_traversal": False,
        "api": False,
    }))
    return 0


def _cmd_explore(args, config: Dict[str, Any]) -> int:
    graph_id = _active_graph(config, args.graph)
    with _storage_for(config, graph_id) as storage:
        query_terms = _expand_query_terms(args.question, args.terms)
        if not args.expand_query:
            query_terms = query_terms[:1]
        file_hits = _search_document_terms(
            storage,
            query_terms,
            per_term_limit=args.per_term_file_limit,
            total_limit=args.file_limit,
        )
        semantic_results: list[dict] = []
        semantic_seen = set()
        semantic_queries = query_terms if args.expand_query else query_terms[:1]
        for term_info in semantic_queries[: args.semantic_queries]:
            semantic = storage.agent_semantic_search(
                term_info["term"],
                role=args.role,
                top_k=args.limit,
                threshold=args.threshold,
            )
            for item in semantic.get("results", []):
                if item.get("score") is not None and float(item.get("score") or 0.0) < args.min_semantic_score:
                    continue
                fid = item.get("family_id", "")
                if not fid or fid in semantic_seen:
                    continue
                semantic_seen.add(fid)
                item = dict(item)
                item["matched_query"] = term_info["term"]
                item["query_source"] = term_info.get("source", "expanded")
                semantic_results.append(item)
                if len(semantic_results) >= args.limit:
                    break
            if len(semantic_results) >= args.limit:
                break
        semantic_results.sort(key=lambda x: float(x.get("score") or 0.0), reverse=True)
        concept_ids = [r.get("family_id", "") for r in semantic_results if r.get("family_id")]
        episode_ids = [r.get("episode_version_id", "") for r in semantic_results if r.get("episode_version_id")]
        source_evidence = _concept_source_evidence(storage, concept_ids, limit=args.limit)
        neighbors: list[dict] = []
        for fid in concept_ids[: args.neighbor_seeds]:
            try:
                neighbors.extend(storage.get_concept_neighbors(fid, max_depth=args.depth, max_results=args.neighbor_limit))
            except Exception:
                continue
        relation_samples: list[dict] = []
        relation_pairs = []
        for i, left in enumerate(concept_ids[: args.relation_seed_count]):
            for right in concept_ids[i + 1: args.relation_seed_count]:
                if left != right:
                    relation_pairs.append((left, right))
        for left, right in relation_pairs[: args.relation_pair_limit]:
            evidence = _relation_evidence(storage, left, right, limit=args.relation_evidence_limit)
            for item in evidence:
                item = dict(item)
                item["query_pair"] = [left, right]
                relation_samples.append(item)
            if len(relation_samples) >= args.relation_evidence_limit:
                relation_samples = relation_samples[: args.relation_evidence_limit]
                break
        cards = _evidence_cards(file_hits, source_evidence, query_terms, limit=args.evidence_limit)
        _print_json(_ok("explore", graph_id, {
            "question": args.question,
            "query_terms": query_terms,
            "file_hits": file_hits,
            "semantic_hits": semantic_results,
            "semantic_total": len(semantic_results),
            "episode_ids": episode_ids,
            "source_evidence": source_evidence,
            "evidence_cards": cards,
            "neighbors": neighbors[: args.neighbor_limit],
            "relation_evidence": relation_samples,
            "coverage": {
                "file_hits": len(file_hits),
                "semantic_hits": len(semantic_results),
                "source_evidence": len(source_evidence),
                "evidence_cards": len(cards),
                "neighbors": len(neighbors[: args.neighbor_limit]),
                "relation_evidence": len(relation_samples),
                "relation_pairs_checked": min(len(relation_pairs), args.relation_pair_limit),
            },
        }, used={
            "raw_files": True,
            "sqlite": True,
            "semantic": True,
            "graph_traversal": True,
            "api": False,
        }))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="deep-dream", description="Deep-Dream document-first library CLI")
    parser.add_argument("--config", default="service_config.json", help="Path to service_config.json")
    sub = parser.add_subparsers(dest="command", required=True)

    doctor = sub.add_parser("doctor", help="Inspect local Deep-Dream configuration and health")
    doctor.add_argument("--api-base", default="http://127.0.0.1:16200/api/v1")
    doctor.set_defaults(func=_cmd_doctor)

    library = sub.add_parser("library", help="Manage the single local library")
    library_sub = library.add_subparsers(dest="library_command", required=True)
    library_migrate = library_sub.add_parser("migrate", help="Migrate legacy graphs/* data into the single library")
    library_migrate.add_argument("--legacy-root", help="Directory containing the old graphs/ folder")
    library_migrate.add_argument("--target-root", help="Single-library storage directory")
    library_migrate.add_argument("--source", action="append", help="Legacy graph id to migrate; repeat to select multiple")
    library_migrate.add_argument("--force", action="store_true", help="Replace an existing target graph.db")
    library_migrate.add_argument("--no-backup", action="store_true", help="Leave the old graphs/ directory in place")
    library_migrate.set_defaults(func=_cmd_library_migrate)

    graph = sub.add_parser("graph", help="Compatibility aliases for the single library")
    graph_sub = graph.add_subparsers(dest="graph_command", required=True)
    graph_list = graph_sub.add_parser("list", help="List graphs")
    graph_list.set_defaults(func=_cmd_graph_list)
    graph_create = graph_sub.add_parser("create", help="Create a graph")
    graph_create.add_argument("graph_id")
    graph_create.set_defaults(func=_cmd_graph_create)
    graph_use = graph_sub.add_parser("use", help="Set the active graph")
    graph_use.add_argument("graph_id")
    graph_use.set_defaults(func=_cmd_graph_use)
    graph_stats = graph_sub.add_parser("stats", help="Show graph stats")
    graph_stats.add_argument("--graph")
    graph_stats.set_defaults(func=_cmd_graph_stats)
    graph_rebuild = graph_sub.add_parser("rebuild", help="Clear graph data for re-indexing")
    graph_rebuild.add_argument("--graph")
    graph_rebuild.set_defaults(func=_cmd_rebuild)

    vault = sub.add_parser("vault", help="Index Markdown vaults")
    vault_sub = vault.add_subparsers(dest="vault_command", required=True)
    vault_index = vault_sub.add_parser("index", help="Index a Markdown/Obsidian vault")
    vault_index.add_argument("path")
    vault_index.add_argument("--graph")
    vault_index.add_argument("--force", action="store_true")
    vault_index.set_defaults(func=_cmd_vault_index)

    remember = sub.add_parser("remember", help="Run remember on a Markdown/text file")
    remember_input = remember.add_mutually_exclusive_group(required=True)
    remember_input.add_argument("--file")
    remember_input.add_argument("--text")
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

    docs = sub.add_parser("docs", help="Document-first file discovery and search")
    docs_sub = docs.add_subparsers(dest="docs_command", required=True)
    docs_roots = docs_sub.add_parser("roots", help="List searchable document roots")
    docs_roots.add_argument("--graph")
    docs_roots.set_defaults(func=_cmd_docs_roots)
    docs_list = docs_sub.add_parser("list", help="List indexed documents")
    docs_list.add_argument("--graph")
    docs_list.add_argument("--limit", type=int, default=100)
    docs_list.set_defaults(func=_cmd_docs_list)
    docs_path = docs_sub.add_parser("path", help="Resolve a document version to a readable path")
    docs_path.add_argument("document_id")
    docs_path.add_argument("--graph")
    docs_path.set_defaults(func=_cmd_docs_path)
    docs_search = docs_sub.add_parser("search", help="Literal search over readable document files")
    docs_search.add_argument("pattern")
    docs_search.add_argument("--graph")
    docs_search.add_argument("--limit", type=int, default=50)
    docs_search.set_defaults(func=_cmd_docs_search)
    docs_grep = docs_sub.add_parser("grep", help="Regex search over readable document files")
    docs_grep.add_argument("pattern")
    docs_grep.add_argument("--graph")
    docs_grep.add_argument("--limit", type=int, default=50)
    docs_grep.set_defaults(func=_cmd_docs_grep)
    docs_map = docs_sub.add_parser("map", help="Map a file path to Deep-Dream documents")
    docs_map.add_argument("path")
    docs_map.add_argument("--graph")
    docs_map.set_defaults(func=_cmd_docs_map)

    episode = sub.add_parser("episode", help="Episode mapping helpers")
    episode_sub = episode.add_subparsers(dest="episode_command", required=True)
    ep_from_file = episode_sub.add_parser("from-file", help="Map a file path/line to episodes")
    ep_from_file.add_argument("path")
    ep_from_file.add_argument("--line", type=int)
    ep_from_file.add_argument("--graph")
    ep_from_file.add_argument("--limit", type=int, default=50)
    ep_from_file.set_defaults(func=_cmd_episode_from_file)
    ep_concepts = episode_sub.add_parser("concepts", help="List concepts mentioned by an episode")
    ep_concepts.add_argument("episode_id")
    ep_concepts.add_argument("--graph")
    ep_concepts.add_argument("--limit", type=int, default=100)
    ep_concepts.set_defaults(func=_cmd_episode_concepts)

    concept = sub.add_parser("concept", help="Concept search, trace, and neighbor expansion")
    concept_sub = concept.add_subparsers(dest="concept_command", required=True)
    concept_search = concept_sub.add_parser("search", help="Search concepts")
    concept_search.add_argument("query")
    concept_search.add_argument("--graph")
    concept_search.add_argument("--role", choices=["document", "episode", "entity", "relation"])
    concept_search.add_argument("--limit", type=int, default=20)
    concept_search.add_argument("--semantic", action="store_true")
    concept_search.add_argument("--threshold", type=float, default=0.3)
    concept_search.set_defaults(func=_cmd_concept_search)
    concept_trace = concept_sub.add_parser("trace", help="Trace concept provenance")
    concept_trace.add_argument("family_id")
    concept_trace.add_argument("--graph")
    concept_trace.add_argument("--time-point")
    concept_trace.set_defaults(func=_cmd_concept_trace)
    concept_neighbors = concept_sub.add_parser("neighbors", help="Expand concept graph neighbors")
    concept_neighbors.add_argument("family_id")
    concept_neighbors.add_argument("--graph")
    concept_neighbors.add_argument("--depth", type=int, default=1)
    concept_neighbors.add_argument("--limit", type=int, default=50)
    concept_neighbors.set_defaults(func=_cmd_concept_neighbors)

    relation = sub.add_parser("relation", help="Relation evidence helpers")
    relation_sub = relation.add_subparsers(dest="relation_command", required=True)
    rel_evidence = relation_sub.add_parser("evidence", help="Find evidence between two concepts")
    rel_evidence.add_argument("concept_a")
    rel_evidence.add_argument("concept_b")
    rel_evidence.add_argument("--graph")
    rel_evidence.add_argument("--limit", type=int, default=50)
    rel_evidence.set_defaults(func=_cmd_relation_evidence)

    sql = sub.add_parser("sql", help="Run graph-local read-only SQL")
    sql.add_argument("--query", required=True)
    sql.add_argument("--graph")
    sql.add_argument("--limit", type=int, default=200)
    sql.add_argument("--explain", action="store_true")
    sql.set_defaults(func=_cmd_sql)

    explore = sub.add_parser("explore", help="Document-first semantic and graph exploration")
    explore.add_argument("question")
    explore.add_argument("--graph")
    explore.add_argument("--role", choices=["document", "episode", "entity", "relation"])
    explore.add_argument("--limit", type=int, default=20)
    explore.add_argument("--threshold", type=float, default=0.2)
    explore.add_argument("--file-limit", type=int, default=20)
    explore.add_argument("--per-term-file-limit", type=int, default=5)
    explore.add_argument("--expand-query", action=argparse.BooleanOptionalAction, default=True)
    explore.add_argument("--terms", help="Comma-separated query expansion terms generated by the caller/agent")
    explore.add_argument("--semantic-queries", type=int, default=5)
    explore.add_argument("--min-semantic-score", type=float, default=0.0)
    explore.add_argument("--evidence-limit", type=int, default=12)
    explore.add_argument("--neighbor-seeds", type=int, default=3)
    explore.add_argument("--neighbor-limit", type=int, default=50)
    explore.add_argument("--depth", type=int, default=1)
    explore.add_argument("--relation-seed-count", type=int, default=5)
    explore.add_argument("--relation-pair-limit", type=int, default=8)
    explore.add_argument("--relation-evidence-limit", type=int, default=10)
    explore.set_defaults(func=_cmd_explore)

    # ── db sub-commands (V1.5) ──────────────────────────
    db = sub.add_parser("db", help="Database maintenance and V1.5 schema management")
    db_sub = db.add_subparsers(dest="db_command", required=True)

    db_init = db_sub.add_parser("init-v15", help="Initialize V1.5 schema on the current graph.db")
    db_init.add_argument("--smoke-test", action="store_true", help="Run smoke tests after init (default in CI)")
    db_init.set_defaults(func=_cmd_db_init_v15)

    db_reset = db_sub.add_parser("reset-v15", help="Backup old graph.db and create fresh V1.5 database")
    db_reset.add_argument("--backup-old", action="store_true", required=True, help="Required: confirm backup of old database")
    db_reset.set_defaults(func=_cmd_db_reset_v15)

    db_rebuild_fts = db_sub.add_parser("rebuild-fts", help="Full rebuild of episodes_fts")
    db_rebuild_fts.set_defaults(func=_cmd_db_rebuild_fts)

    db_validate = db_sub.add_parser("validate", help="Run integrity validation")
    db_validate.add_argument("--repair", action="store_true", help="Auto-repair fixable issues (content/current, FTS)")
    db_validate.set_defaults(func=_cmd_db_validate)

    db_rebuild_current = db_sub.add_parser("rebuild-current", help="Rebuild content/current/ from DB")
    db_rebuild_current.set_defaults(func=_cmd_db_rebuild_current)

    db_vacuum = db_sub.add_parser("vacuum-embeddings", help="Clean up orphaned embeddings")
    db_vacuum.add_argument("--inactive", action="store_true", help="Also clean superseded/stale owner embeddings")
    db_vacuum.add_argument("--dry-run", action="store_true", help="Only report count, do not delete")
    db_vacuum.set_defaults(func=_cmd_db_vacuum_embeddings)

    db_compact = db_sub.add_parser("compact", help="VACUUM the graph.db to reclaim space")
    db_compact.set_defaults(func=_cmd_db_compact)
    return parser


# ── V1.5 DB maintenance commands ───────────────────────────


def _get_storage_path(config: Dict[str, Any]) -> str:
    return config.get("storage_path", "./library")


def _open_db_conn(config: Dict[str, Any]) -> "sqlite3.Connection":
    import sqlite3 as _sqlite3
    storage_path = _get_storage_path(config)
    db_path = os.path.join(storage_path, "graph.db")
    if not os.path.exists(db_path):
        raise FileNotFoundError(f"graph.db not found at {db_path}")
    conn = _sqlite3.connect(db_path)
    conn.row_factory = _sqlite3.Row
    return conn


def _cmd_db_init_v15(args, config: Dict[str, Any]) -> int:
    import sqlite3
    from core.storage.sqlite.schema_v15 import init_schema_v15
    storage_path = _get_storage_path(config)
    db_path = os.path.join(storage_path, "graph.db")
    os.makedirs(os.path.dirname(db_path) or ".", exist_ok=True)
    conn = sqlite3.connect(db_path)
    try:
        result = init_schema_v15(conn)
        _print_json({"success": True, "action": "init-v15", **result})
        if args.smoke_test:
            from core.storage.sqlite.integrity import validate_all
            violations = validate_all(conn, library_path=storage_path, include_file_checks=False)
            _print_json({"smoke_test": len(violations) == 0, "violations": len(violations)})
        return 0
    except Exception as e:
        _print_json({"success": False, "error": str(e)})
        return 1
    finally:
        conn.close()


def _cmd_db_reset_v15(args, config: Dict[str, Any]) -> int:
    import sqlite3
    from core.storage.sqlite.schema_v15 import init_schema_v15
    storage_path = _get_storage_path(config)
    db_path = os.path.join(storage_path, "graph.db")

    if not os.path.exists(db_path):
        _print_json({"success": False, "error": "No existing graph.db found"})
        return 1

    # Backup using sqlite3 backup API
    from datetime import datetime, timezone
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    backup_path = os.path.join(storage_path, f"graph.legacy.{ts}.db")

    src_conn = sqlite3.connect(db_path)
    try:
        src_conn.execute("PRAGMA wal_checkpoint(FULL)")
        backup_conn = sqlite3.connect(backup_path)
        src_conn.backup(backup_conn)
        backup_conn.close()
    finally:
        src_conn.close()

    if not os.path.exists(backup_path):
        _print_json({"success": False, "error": "Backup failed"})
        return 1

    # Remove old DB and create new
    os.remove(db_path)
    for suffix in ("-wal", "-shm"):
        p = db_path + suffix
        if os.path.exists(p):
            os.remove(p)

    conn = sqlite3.connect(db_path)
    try:
        result = init_schema_v15(conn)
        from core.storage.sqlite.integrity import validate_all
        violations = validate_all(conn, library_path=storage_path, include_file_checks=False)
        _print_json({
            "success": True,
            "action": "reset-v15",
            "backup": backup_path,
            "violations": len(violations),
            **result,
        })
        return 0
    except Exception as e:
        _print_json({"success": False, "error": str(e), "backup": backup_path})
        return 1
    finally:
        conn.close()


def _cmd_db_rebuild_fts(args, config: Dict[str, Any]) -> int:
    conn = _open_db_conn(config)
    try:
        from core.storage.sqlite.repositories.episodes import rebuild_fts_all
        count = rebuild_fts_all(conn)
        _print_json({"success": True, "action": "rebuild-fts", "episodes_indexed": count})
        return 0
    except NotImplementedError:
        # Fallback: direct SQL rebuild
        conn.execute("DELETE FROM episodes_fts")
        conn.execute("""
            INSERT INTO episodes_fts (episode_id, document_id, document_version_id,
                                       name, heading_path, source_text, memory_text)
            SELECT episode_id, document_id, document_version_id,
                   name, heading_path, source_text, memory_text
            FROM episodes
            WHERE status = 'active'
        """)
        count = conn.execute("SELECT changes()").fetchone()[0]
        conn.commit()
        _print_json({"success": True, "action": "rebuild-fts", "episodes_indexed": count})
        return 0
    except Exception as e:
        _print_json({"success": False, "error": str(e)})
        return 1
    finally:
        conn.close()


def _cmd_db_validate(args, config: Dict[str, Any]) -> int:
    from core.storage.sqlite.integrity import validate_all
    storage_path = _get_storage_path(config)
    conn = _open_db_conn(config)
    try:
        violations = validate_all(conn, library_path=storage_path,
                                  include_file_checks=True)
        if args.repair:
            from core.storage.sqlite.content_fs import rebuild_current_files
            count = rebuild_current_files(conn, storage_path)
            _print_json({
                "success": True,
                "action": "validate --repair",
                "violations": len(violations),
                "current_files_rebuilt": count,
                "details": violations[:50],
            })
        else:
            _print_json({
                "success": len(violations) == 0,
                "action": "validate",
                "violations": len(violations),
                "details": violations[:50],
            })
        return 0 if len(violations) == 0 else 1
    except Exception as e:
        _print_json({"success": False, "error": str(e)})
        return 1
    finally:
        conn.close()


def _cmd_db_rebuild_current(args, config: Dict[str, Any]) -> int:
    from core.storage.sqlite.content_fs import rebuild_current_files
    storage_path = _get_storage_path(config)
    conn = _open_db_conn(config)
    try:
        count = rebuild_current_files(conn, storage_path)
        _print_json({"success": True, "action": "rebuild-current", "files_written": count})
        return 0
    except Exception as e:
        _print_json({"success": False, "error": str(e)})
        return 1
    finally:
        conn.close()


def _cmd_db_vacuum_embeddings(args, config: Dict[str, Any]) -> int:
    conn = _open_db_conn(config)
    try:
        from core.storage.sqlite.repositories.embeddings import (
            vacuum_orphaned, vacuum_deleted_documents, vacuum_inactive,
        )
        orphaned = vacuum_orphaned(conn)
        deleted = vacuum_deleted_documents(conn)
        inactive = 0
        if args.inactive:
            inactive = vacuum_inactive(conn, dry_run=args.dry_run)
        conn.commit()
        _print_json({
            "success": True,
            "action": "vacuum-embeddings",
            "orphaned_removed": orphaned,
            "deleted_doc_removed": deleted,
            "inactive_removed": inactive,
            "dry_run": args.dry_run,
        })
        return 0
    except NotImplementedError:
        # Fallback: basic orphan cleanup
        orphaned = conn.execute("""
            DELETE FROM embeddings WHERE owner_type = 'episode'
            AND owner_id NOT IN (SELECT episode_id FROM episodes)
        """)
        conn.commit()
        _print_json({"success": True, "action": "vacuum-embeddings", "orphaned_removed": orphaned.rowcount})
        return 0
    except Exception as e:
        _print_json({"success": False, "error": str(e)})
        return 1
    finally:
        conn.close()


def _cmd_db_compact(args, config: Dict[str, Any]) -> int:
    conn = _open_db_conn(config)
    try:
        conn.execute("VACUUM")
        _print_json({"success": True, "action": "compact"})
        return 0
    except Exception as e:
        _print_json({"success": False, "error": str(e)})
        return 1
    finally:
        conn.close()


def main(argv: list[str] | None = None) -> int:
    import os
    os.environ["DEEPDREAM_JSON_OUTPUT"] = "1"
    parser = build_parser()
    args = parser.parse_args(argv)
    config = _load_config(args.config)
    try:
        return args.func(args, config)
    except Exception as e:
        _print_json({"success": False, "error": str(e), "type": type(e).__name__})
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
