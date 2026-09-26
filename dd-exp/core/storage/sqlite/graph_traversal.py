"""Graph traversal and document graph rendering for V1.5 schema."""
from __future__ import annotations

import logging
import sqlite3
import time
from typing import List, Optional, Tuple

from .repositories import search as search_repo

logger = logging.getLogger(__name__)


def _enrich_names(conn: sqlite3.Connection, results: List[dict]) -> List[dict]:
    """Batch-resolve entity names for neighbor results."""
    if not results:
        return results
    fids = [r["family_id"] for r in results]
    # Batch lookup canonical_name from entity_families
    placeholders = ",".join("?" * len(fids))
    rows = conn.execute(
        f"SELECT entity_family_id, canonical_name "
        f"FROM entity_families WHERE entity_family_id IN ({placeholders})",
        fids,
    ).fetchall()
    name_map = {r[0]: r[1] for r in rows}
    for r in results:
        r["name"] = name_map.get(r["family_id"], "")
    return results


def get_concept_neighbors(conn: sqlite3.Connection, family_id: str,
                          max_depth: int = 1, max_results: int = 200,
                          edge_types: Optional[List[str]] = None) -> List[dict]:
    """BFS expansion of concept neighbors, annotating each result with its hop depth.

    Parameters
    ----------
    conn : sqlite3.Connection
    family_id : str
        Starting concept family ID.
    max_depth : int
        Maximum BFS depth (1 = direct neighbors only).
    max_results : int
        Cap on total neighbor results returned.
    edge_types : list[str] or None
        If given, only these edge types are followed.

    Returns
    -------
    list[dict]
        Each dict has keys: ``edge_type``, ``family_id``, ``depth``.
        Results are ordered by depth (closest first), with RELATES edges
        prioritised within the same depth level.
    """
    valid_types = {"RELATES", "MENTIONS", "ASSERTS"}
    if edge_types:
        valid_types = valid_types.intersection(edge_types)

    visited: set[str] = {family_id}
    results: list[dict] = []
    frontier: list[str] = [family_id]

    for current_depth in range(1, max_depth + 1):
        next_frontier: list[str] = []
        for fid in frontier:
            # Fetch more edges than needed because MENTIONS edges (with empty
            # source_family_id) are filtered out.
            fetch_limit = max(max_results * 10, 1000)
            edges = search_repo.get_graph_neighbors(conn, fid, limit=fetch_limit)
            # Put RELATES edges first — they have both source and target
            # family_ids and yield real concept-to-concept neighbors.
            edges.sort(key=lambda e: 0 if e.get("edge_type") == "RELATES" else 1)

            for e in edges:
                et = e.get("edge_type", "")
                if et not in valid_types:
                    continue
                target_fid = e.get("target_family_id") or ""
                source_fid = e.get("source_family_id") or ""
                # Determine the OTHER side of the edge
                if source_fid == fid:
                    neighbor_fid = target_fid
                elif target_fid == fid:
                    neighbor_fid = source_fid
                else:
                    # Edge not connected to current fid — skip
                    continue
                if not neighbor_fid or neighbor_fid == family_id or neighbor_fid in visited:
                    continue
                visited.add(neighbor_fid)
                next_frontier.append(neighbor_fid)
                results.append({
                    "edge_type": et,
                    "family_id": neighbor_fid,
                    "depth": current_depth,
                })
                if len(results) >= max_results:
                    break
            if len(results) >= max_results:
                return _enrich_names(conn, results)
        frontier = next_frontier
        if not frontier:
            break
    return _enrich_names(conn, results)


def traverse_concepts(conn: sqlite3.Connection,
                      start_ids: List[str], max_depth: int = 2,
                      max_results: int = 500,
                      edge_types: Optional[List[str]] = None,
                      timeout_seconds: float = 30.0) -> dict:
    deadline = time.monotonic() + min(max(float(timeout_seconds or 30.0), 0.1), 120.0)
    timed_out = False
    visited = set(start_ids)
    all_edges = []
    frontier = list(start_ids)
    for _ in range(max_depth):
        next_frontier = []
        for fid in frontier:
            if time.monotonic() >= deadline:
                timed_out = True
                break
            neighbors = search_repo.get_graph_neighbors(conn, fid, limit=max_results)
            for n in neighbors:
                if time.monotonic() >= deadline:
                    timed_out = True
                    break
                if edge_types and n.get("edge_type") not in edge_types:
                    continue
                all_edges.append(n)
                # Bidirectional traversal: determine which endpoint is the neighbor
                source_fid = n.get("source_family_id") or ''
                target_fid = n.get("target_family_id") or ''
                if source_fid == fid:
                    neighbor = target_fid
                elif target_fid == fid:
                    neighbor = source_fid
                else:
                    neighbor = target_fid
                if neighbor and neighbor not in visited:
                    visited.add(neighbor)
                    next_frontier.append(neighbor)
            if timed_out:
                break
        frontier = next_frontier
        if timed_out or not frontier or len(all_edges) >= max_results:
            break
    return {"edges": all_edges[:max_results], "visited": list(visited),
            "visited_count": len(visited),
            "truncated": timed_out or len(all_edges) > max_results,
            "timed_out": timed_out}


def batch_bfs_traverse(conn: sqlite3.Connection,
                       seed_ids: List[str], max_depth: int = 2,
                       max_nodes: int = 50) -> Tuple[list, list, dict]:
    result = traverse_concepts(conn, seed_ids, max_depth=max_depth, max_results=max_nodes)
    visited_family_ids = list(result.get("visited", []))
    edges = result.get("edges", [])

    # Build lightweight Entity-like objects from the visited family_ids so
    # that callers (GraphTraversalSearcher.bfs_expand_with_relations) receive
    # objects with .family_id and .name attributes instead of raw edge dicts.
    entities = _build_lightweight_entities(conn, visited_family_ids)

    return entities, edges, visited_family_ids


def _build_lightweight_entities(conn: sqlite3.Connection,
                                family_ids: List[str]) -> list:
    """Build lightweight Entity-like objects for the given family_ids."""
    if not family_ids:
        return []
    # Batch fetch entity info
    ph = ",".join("?" for _ in family_ids)
    rows = conn.execute(
        f"SELECT ef.entity_family_id, ef.canonical_name, ef.canonical_content "
        f"FROM entity_families ef "
        f"WHERE ef.entity_family_id IN ({ph})",
        family_ids,
    ).fetchall()
    entities = []
    from core.models import Entity
    from datetime import datetime as _dt
    for r in rows:
        entities.append(Entity(
            absolute_id=r[0],
            family_id=r[0],
            name=r[1] or "",
            content=r[2] or "",
            event_time=_dt.now(),
            processed_time=_dt.now(),
            episode_id="",
            source_document="",
        ))
    return entities


# ── Document graph ────────────────────────────────────

def _resolve_document_ids(conn, document_version_ids=None, document_family_ids=None):
    """Resolve document_version_ids to (doc_ids, doc_version_ids, doc_rows)."""
    doc_ids = set()
    doc_version_ids_resolved = set()

    if document_family_ids:
        doc_ids.update(document_family_ids)

    if document_version_ids:
        rows = conn.execute(
            "SELECT dv.document_id, dv.document_version_id FROM document_versions dv "
            "WHERE dv.document_version_id IN ({}) AND dv.status = 'active'".format(
                ",".join("?" for _ in document_version_ids)
            ),
            document_version_ids,
        ).fetchall()
        for r in rows:
            doc_ids.add(r[0])
            doc_version_ids_resolved.add(r[1])

    # Get document rows
    if not doc_ids:
        return [], [], []
    ph = ",".join("?" for _ in doc_ids)
    doc_rows = conn.execute(
        f"SELECT d.document_id, d.title, d.managed_path, d.relative_path, d.status, "
        f"dv.document_version_id, dv.byte_size, dv.processed_at "
        f"FROM documents d "
        f"JOIN document_versions dv ON dv.document_id = d.document_id AND dv.status = 'active' "
        f"WHERE d.document_id IN ({ph}) AND d.status = 'active' "
        f"ORDER BY d.document_id",
        list(doc_ids),
    ).fetchall()
    return list(doc_ids), list(doc_version_ids_resolved), doc_rows


def _build_document_nodes(doc_rows):
    """Build document nodes for the frontend."""
    documents = []
    for r in doc_rows:
        doc = {
            "document_id": r[0],
            "title": r[1],
            "managed_path": r[2],
            "relative_path": r[3],
            "status": r[4],
            "document_version_id": r[5],
            "size": r[6] or 0,
            "processed_time": r[7],
        }
        documents.append(doc)
    return documents


def _build_episode_nodes(conn, doc_version_ids):
    """Fetch episodes for given document version IDs."""
    if not doc_version_ids:
        return []
    ph = ",".join("?" for _ in doc_version_ids)
    rows = conn.execute(
        f"SELECT ep.episode_id, ep.episode_family_id, ep.name, "
        f"ep.source_text, ep.event_time, ep.processed_at, "
        f"ep.document_id, ep.document_version_id, "
        f"ep.heading_path, ep.chunk_index, "
        f"ep.start_offset, ep.end_offset, ep.memory_text "
        f"FROM episodes ep "
        f"WHERE ep.document_version_id IN ({ph}) AND ep.status = 'active' "
        f"ORDER BY ep.chunk_index",
        doc_version_ids,
    ).fetchall()
    episodes = []
    for r in rows:
        episodes.append({
            "version_id": r[0],
            "family_id": r[1],
            "name": r[2],
            "content": (r[3] or "")[:500],
            "event_time": r[4],
            "processed_time": r[5],
            "document_family_id": r[6],
            "document_version_id": r[7],
            "heading_path": r[8],
            "chunk_index": r[9],
            "start_offset": r[10],
            "end_offset": r[11],
            "memory_text": (r[12] or "")[:2000],
        })
    return episodes


def _build_entity_concepts(conn, episode_ids):
    """Fetch entity concepts mentioned in given episodes."""
    if not episode_ids:
        return []
    ph = ",".join("?" for _ in episode_ids)
    rows = conn.execute(
        f"SELECT DISTINCT eo.entity_family_id, eo.name, eo.content, "
        f"eo.processed_at, eo.entity_id, "
        f"COALESCE(NULLIF(eo.episode_id, ''), em.episode_id) "
        f"FROM entity_mentions em "
        f"JOIN entity_observations eo ON eo.entity_id = em.entity_id AND eo.status = 'active' "
        f"WHERE em.episode_id IN ({ph}) "
        f"ORDER BY eo.name",
        episode_ids,
    ).fetchall()
    concepts = []
    for r in rows:
        concepts.append({
            "family_id": r[0],
            "name": r[1],
            "content": r[2] or "",
            "role": "entity",
            "processed_time": r[3],
            "version_id": r[4],
            "episode_id": r[5],
            "metadata": {},
        })
    return concepts


def _build_relation_concepts(conn, episode_ids):
    """Fetch relation concepts asserted in given episodes."""
    if not episode_ids:
        return []
    ph = ",".join("?" for _ in episode_ids)
    rows = conn.execute(
        f"SELECT ra.relation_family_id, ra.content, "
        f"ra.subject_entity_family_id, ra.object_entity_family_id, "
        f"ra.processed_at, ra.relation_id, ra.episode_id "
        f"FROM relation_assertions ra "
        f"WHERE ra.episode_id IN ({ph}) AND ra.status = 'active' "
        f"ORDER BY ra.processed_at",
        episode_ids,
    ).fetchall()
    concepts = []
    seen = set()
    for r in rows:
        fid = r[0]
        if fid in seen:
            continue
        seen.add(fid)
        concepts.append({
            "family_id": fid,
            "name": "",
            "content": r[1] or "",
            "role": "relation",
            "processed_time": r[4],
            "version_id": r[5],
            "episode_version_id": r[6],
            "metadata": {
                "entity1_family_id": r[2],
                "entity2_family_id": r[3],
            },
        })
    return concepts


def _build_edges(documents, episodes, entities, relations):
    """Build edge list for the frontend graph.

    Returns edges with from/to in the format:
      doc:<version_id>, episode:<version_id>, concept:<family_id>
    """
    edges = []

    # doc_id -> document_version_id mapping
    doc_ver_by_doc_id = {}
    for d in documents:
        doc_ver_by_doc_id[d["document_id"]] = d["document_version_id"]

    # HAS_EPISODE: document -> episode
    for ep in episodes:
        doc_ver = ep.get("document_version_id")
        if not doc_ver:
            continue
        edges.append({
            "edge_id": f"he:{ep['version_id']}",
            "from": f"doc:{doc_ver}",
            "to": f"episode:{ep['version_id']}",
            "edge_type": "HAS_EPISODE",
            "document_version_id": doc_ver,
            "episode_version_id": ep["version_id"],
        })

    # Entity family_ids set for validation

    return edges, doc_ver_by_doc_id


def _build_mention_edges(conn, episode_ids, documents):
    """Build MENTIONS edges from entity_mentions."""
    if not episode_ids:
        return []
    ph = ",".join("?" for _ in episode_ids)

    doc_ver_by_doc_id = {}
    for d in documents:
        doc_ver_by_doc_id[d["document_id"]] = d["document_version_id"]

    rows = conn.execute(
        f"SELECT 'entity' AS target_role, em.episode_id, em.entity_family_id, em.entity_id "
        f"FROM entity_mentions em "
        f"JOIN entity_observations eo ON eo.entity_id=em.entity_id AND eo.status='active' "
        f"WHERE em.episode_id IN ({ph}) "
        f"UNION ALL "
        f"SELECT 'relation' AS target_role, rm.episode_id, rm.relation_family_id, rm.relation_id "
        f"FROM relation_mentions rm "
        f"JOIN relation_assertions ra ON ra.relation_id=rm.relation_id AND ra.status='active' "
        f"WHERE rm.episode_id IN ({ph})",
        [*episode_ids, *episode_ids],
    ).fetchall()
    edges = []
    for r in rows:
        role = r[0]
        edges.append({
            "edge_id": f"ment:{r[1]}:{r[2]}" if role == "entity" else f"rment:{r[1]}:{r[2]}",
            "from": f"episode:{r[1]}",
            "to": f"concept:{r[2]}",
            "edge_type": "MENTIONS",
            "target_family_id": r[2],
            "target_version_id": r[3],
            "target_role": role,
            "episode_version_id": r[1],
        })
    return edges


def _build_relation_edges(conn, episode_ids, relations):
    """Build CONNECTS edges for relations, tagged with originating episode."""
    if not episode_ids:
        return []

    ph = ",".join("?" for _ in episode_ids)
    rows = conn.execute(
        f"SELECT ra.relation_family_id, ra.subject_entity_family_id, "
        f"ra.object_entity_family_id, ra.episode_id "
        f"FROM relation_assertions ra "
        f"WHERE ra.episode_id IN ({ph}) AND ra.status = 'active'",
        episode_ids,
    ).fetchall()
    edges = []
    seen = set()
    for r in rows:
        rel_fid = r[0]
        sub_fid = r[1]
        obj_fid = r[2]
        ep_id = r[3]
        key = (rel_fid, sub_fid, obj_fid)
        if key in seen:
            continue
        seen.add(key)
        edges.append({
            "edge_id": f"conn:{rel_fid}",
            "from": f"concept:{sub_fid}",
            "to": f"concept:{obj_fid}",
            "edge_type": "CONNECTS",
            "relation_family_id": rel_fid,
            "source_family_id": sub_fid,
            "target_family_id": obj_fid,
            "episode_version_id": ep_id,
        })
    return edges


def _build_version_counts(conn, entity_families, relation_families):
    """Count observation/assertion versions per family."""
    versions = {}
    if entity_families:
        ph = ",".join("?" for _ in entity_families)
        rows = conn.execute(
            f"SELECT entity_family_id, COUNT(*) FROM entity_observations "
            f"WHERE entity_family_id IN ({ph}) AND status = 'active' "
            f"GROUP BY entity_family_id",
            list(entity_families),
        ).fetchall()
        for r in rows:
            versions[r[0]] = {"total": r[1]}
    if relation_families:
        ph = ",".join("?" for _ in relation_families)
        rows = conn.execute(
            f"SELECT relation_family_id, COUNT(*) FROM relation_assertions "
            f"WHERE relation_family_id IN ({ph}) AND status = 'active' "
            f"GROUP BY relation_family_id",
            list(relation_families),
        ).fetchall()
        for r in rows:
            versions[r[0]] = {"total": r[1]}
    return versions


def _build_has_episode_edges(episodes: List[dict]) -> List[dict]:
    edges = []
    for ep in episodes:
        doc_ver = ep.get("document_version_id")
        if doc_ver:
            edges.append({
                "edge_id": f"he:{ep['version_id']}",
                "from": f"doc:{doc_ver}",
                "to": f"episode:{ep['version_id']}",
                "edge_type": "HAS_EPISODE",
                "document_version_id": doc_ver,
                "episode_version_id": ep["version_id"],
            })
    return edges


def _document_graph_counts(conn: sqlite3.Connection, episode_ids: List[str]) -> dict:
    if not episode_ids:
        return {"episodes": 0, "concepts": 0, "relations": 0}
    ph = ",".join("?" for _ in episode_ids)
    entity_count = conn.execute(
        f"SELECT COUNT(DISTINCT entity_family_id) FROM entity_mentions "
        f"WHERE episode_id IN ({ph})",
        episode_ids,
    ).fetchone()[0] or 0
    relation_count = conn.execute(
        f"SELECT COUNT(DISTINCT relation_family_id) FROM relation_assertions "
        f"WHERE episode_id IN ({ph}) AND status = 'active'",
        episode_ids,
    ).fetchone()[0] or 0
    return {"episodes": len(episode_ids), "concepts": entity_count, "relations": relation_count}


def get_document_graph(conn: sqlite3.Connection,
                       document_version_ids: List[str] = None,
                       document_family_ids: List[str] = None,
                       max_episodes: int = 10000,
                       max_concepts: int = 50000,
                       include_relations: bool = True,
                       include_versions: bool = True) -> dict:
    doc_ids, resolved_ver_ids, doc_rows = _resolve_document_ids(
        conn, document_version_ids, document_family_ids)

    if not doc_ids:
        return {"documents": [], "episodes": [], "concepts": [], "edges": [],
                "versions": {}, "counts": {}}

    documents = _build_document_nodes(doc_rows)
    ver_ids = [r[5] for r in doc_rows]
    episodes = _build_episode_nodes(conn, ver_ids)
    if max_episodes and len(episodes) > max_episodes:
        episodes = episodes[:max_episodes]

    episode_ids = [ep["version_id"] for ep in episodes]

    entities = _build_entity_concepts(conn, episode_ids)
    relations = _build_relation_concepts(conn, episode_ids) if include_relations else []
    if max_concepts and len(entities) > max_concepts:
        kept_entity_fams = {c["family_id"] for c in entities[:max_concepts]}
        entities = entities[:max_concepts]
        relations = [r for r in relations
                     if r.get("metadata", {}).get("entity1_family_id") in kept_entity_fams
                     and r.get("metadata", {}).get("entity2_family_id") in kept_entity_fams]
    concepts = entities + relations

    has_ep_edges = _build_has_episode_edges(episodes)
    mention_edges = _build_mention_edges(conn, episode_ids, documents)
    relation_edges = _build_relation_edges(conn, episode_ids, relations) if include_relations else []
    all_edges = has_ep_edges + mention_edges + relation_edges

    entity_fams = {e["family_id"] for e in entities}
    relation_fams = {r["family_id"] for r in relations}
    versions = _build_version_counts(conn, entity_fams, relation_fams) if include_versions else {}

    return {
        "documents": documents,
        "episodes": episodes,
        "concepts": concepts,
        "edges": all_edges,
        "versions": versions,
        "counts": {
            "episodes": len(episodes),
            "concepts": len(entities),
            "relations": len(relations),
        },
    }


def get_document_graph_outline(conn: sqlite3.Connection,
                                document_version_ids: List[str] = None,
                                document_family_ids: List[str] = None,
                                max_episodes: int = 10000) -> dict:
    doc_ids, resolved_ver_ids, doc_rows = _resolve_document_ids(
        conn, document_version_ids, document_family_ids)
    if not doc_ids:
        return {"documents": [], "episodes": [], "concepts": [], "edges": [],
                "versions": {}, "counts": {}, "cursor": 0, "next_cursor": None}

    documents = _build_document_nodes(doc_rows)
    ver_ids = [r[5] for r in doc_rows]
    all_episodes = _build_episode_nodes(conn, ver_ids)
    counts = _document_graph_counts(conn, [ep["version_id"] for ep in all_episodes])
    episodes = all_episodes[:max_episodes] if max_episodes else all_episodes
    return {
        "documents": documents,
        "episodes": episodes,
        "concepts": [],
        "edges": _build_has_episode_edges(episodes),
        "versions": {},
        "counts": counts,
        "cursor": 0,
        "next_cursor": len(episodes) if episodes else None,
    }


def get_document_graph_chunk(conn: sqlite3.Connection,
                              document_version_ids: List[str] = None,
                              document_family_ids: List[str] = None,
                              cursor: int = 0, limit: int = 12,
                              include_relations: bool = True,
                              include_versions: bool = True,
                              max_concepts: int = 8000) -> dict:
    doc_ids, resolved_ver_ids, doc_rows = _resolve_document_ids(
        conn, document_version_ids, document_family_ids)
    if not doc_ids:
        return {"documents": [], "episodes": [], "concepts": [], "edges": [],
                "versions": {}, "counts": {}, "cursor": cursor, "next_cursor": None}

    documents = _build_document_nodes(doc_rows)
    ver_ids = [r[5] for r in doc_rows]
    all_episodes = _build_episode_nodes(conn, ver_ids)
    counts = _document_graph_counts(conn, [ep["version_id"] for ep in all_episodes])
    episodes = all_episodes

    # Paginate episodes by cursor (offset into the episode list)
    if cursor >= len(episodes):
        return {
            "documents": documents,
            "episodes": [],
            "concepts": [],
            "edges": [],
            "versions": {},
            "counts": counts,
            "cursor": cursor,
            "next_cursor": None,
        }
    if cursor > 0:
        episodes = episodes[cursor:]
    if limit and len(episodes) > limit:
        episodes = episodes[:limit]
        next_cursor = (cursor or 0) + len(episodes)
    else:
        next_cursor = None

    episode_ids = [ep["version_id"] for ep in episodes]
    entities = _build_entity_concepts(conn, episode_ids)
    if max_concepts and len(entities) > max_concepts:
        entities = entities[:max_concepts]
    entity_fams = {c["family_id"] for c in entities}
    relations = _build_relation_concepts(conn, episode_ids) if include_relations else []
    relations = [r for r in relations
                 if r.get("metadata", {}).get("entity1_family_id") in entity_fams
                 and r.get("metadata", {}).get("entity2_family_id") in entity_fams]
    chunk_concepts = entities + relations
    chunk_edges = (
        _build_has_episode_edges(episodes)
        + [e for e in _build_mention_edges(conn, episode_ids, documents)
           if e.get("target_family_id") in entity_fams]
        + (_build_relation_edges(conn, episode_ids, relations) if include_relations else [])
    )
    rel_fams = {c["family_id"] for c in relations}
    chunk_versions = _build_version_counts(conn, entity_fams, rel_fams) if include_versions else {}

    return {
        "documents": documents,
        "episodes": episodes,
        "concepts": chunk_concepts,
        "edges": chunk_edges,
        "versions": chunk_versions,
        "counts": counts,
        "cursor": cursor,
        "next_cursor": next_cursor,
    }
