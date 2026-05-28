"""Graph traversal and document graph rendering for V1.5 schema."""
from __future__ import annotations

import logging
import sqlite3
from typing import Dict, List, Optional, Tuple

from .repositories import search as search_repo

logger = logging.getLogger(__name__)


def get_concept_neighbors(conn: sqlite3.Connection, family_id: str,
                          max_depth: int = 1, max_results: int = 200,
                          edge_types: Optional[List[str]] = None) -> List[dict]:
    edges = search_repo.get_graph_neighbors(conn, family_id, limit=max_results)
    if edge_types:
        edges = [e for e in edges if e.get("edge_type") in edge_types]
    results = []
    for e in edges:
        target = e.get("target_family_id") or e.get("target_id", "")
        source = e.get("source_family_id") or e.get("source_id", "")
        # Return the OTHER side of the edge
        neighbor_fid = target if source == family_id else source
        if not neighbor_fid or neighbor_fid == family_id:
            neighbor_fid = target
        results.append({
            "edge_type": e.get("edge_type", ""),
            "target_id": e.get("target_id", ""),
            "family_id": neighbor_fid,
        })
    return results[:max_results]


def traverse_concepts(conn: sqlite3.Connection,
                      start_ids: List[str], max_depth: int = 2,
                      max_results: int = 500,
                      edge_types: Optional[List[str]] = None,
                      timeout_seconds: float = 30.0) -> dict:
    visited = set(start_ids)
    all_edges = []
    frontier = list(start_ids)
    for _ in range(max_depth):
        next_frontier = []
        for fid in frontier:
            neighbors = search_repo.get_graph_neighbors(conn, fid, limit=max_results)
            for n in neighbors:
                if edge_types and n.get("edge_type") not in edge_types:
                    continue
                all_edges.append(n)
                target = n.get("target_family_id") or n.get("target_id")
                if target and target not in visited:
                    visited.add(target)
                    next_frontier.append(target)
        frontier = next_frontier
        if not frontier or len(all_edges) >= max_results:
            break
    return {"edges": all_edges[:max_results], "visited": list(visited),
            "visited_count": len(visited), "truncated": len(all_edges) > max_results}


def batch_bfs_traverse(conn: sqlite3.Connection,
                       seed_ids: List[str], max_depth: int = 2,
                       max_nodes: int = 50) -> Tuple[list, list, dict]:
    result = traverse_concepts(conn, seed_ids, max_depth=max_depth, max_results=max_nodes)
    return [], [], {"hops": {}}


def batch_get_entity_degrees(conn: sqlite3.Connection,
                             family_ids: List[str]) -> Dict[str, int]:
    if not family_ids:
        return {}
    placeholders = ",".join("?" for _ in family_ids)
    rows = conn.execute(
        f"SELECT entity_family_id, COUNT(*) as cnt FROM ("
        f"  SELECT rf.subject_entity_family_id AS entity_family_id "
        f"  FROM relation_families rf "
        f"  WHERE rf.subject_entity_family_id IN ({placeholders}) "
        f"  UNION ALL "
        f"  SELECT rf.object_entity_family_id AS entity_family_id "
        f"  FROM relation_families rf "
        f"  WHERE rf.object_entity_family_id IN ({placeholders})"
        f") GROUP BY entity_family_id",
        family_ids + family_ids,
    ).fetchall()
    result = {fid: 0 for fid in family_ids}
    for row in rows:
        result[row[0]] = row[1]
    return result


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
        f"eo.processed_at, eo.entity_id "
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
    episode_set = set(ep["version_id"] for ep in episodes)
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
    entity_families = {e["family_id"] for e in entities}

    # MENTIONS: episode -> entity
    for ep in episodes:
        ep_id = ep["version_id"]
        doc_ver = ep.get("document_version_id", "")
        # We'll add MENTIONS edges after building concepts
        # by querying entity_mentions for this episode
        # Actually we need to do this in the main function with DB access

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
        f"SELECT em.episode_id, em.entity_family_id, em.entity_id "
        f"FROM entity_mentions em "
        f"WHERE em.episode_id IN ({ph})",
        episode_ids,
    ).fetchall()
    edges = []
    for r in rows:
        edges.append({
            "edge_id": f"ment:{r[0]}:{r[1]}",
            "from": f"episode:{r[0]}",
            "to": f"concept:{r[1]}",
            "edge_type": "MENTIONS",
            "target_family_id": r[1],
            "target_version_id": r[2],
            "episode_version_id": r[0],
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


def get_document_graph(conn: sqlite3.Connection,
                       document_version_ids: List[str] = None,
                       document_family_ids: List[str] = None) -> dict:
    doc_ids, resolved_ver_ids, doc_rows = _resolve_document_ids(
        conn, document_version_ids, document_family_ids)

    if not doc_ids:
        return {"documents": [], "episodes": [], "concepts": [], "edges": [],
                "versions": {}, "counts": {}}

    documents = _build_document_nodes(doc_rows)
    ver_ids = [r[5] for r in doc_rows]
    episodes = _build_episode_nodes(conn, ver_ids)

    episode_ids = [ep["version_id"] for ep in episodes]

    entities = _build_entity_concepts(conn, episode_ids)
    relations = _build_relation_concepts(conn, episode_ids)
    concepts = entities + relations

    # Build edges
    has_ep_edges = []
    for ep in episodes:
        doc_ver = ep.get("document_version_id")
        if doc_ver:
            has_ep_edges.append({
                "edge_id": f"he:{ep['version_id']}",
                "from": f"doc:{doc_ver}",
                "to": f"episode:{ep['version_id']}",
                "edge_type": "HAS_EPISODE",
                "document_version_id": doc_ver,
                "episode_version_id": ep["version_id"],
            })

    mention_edges = _build_mention_edges(conn, episode_ids, documents)
    relation_edges = _build_relation_edges(conn, episode_ids, relations)
    all_edges = has_ep_edges + mention_edges + relation_edges

    entity_fams = {e["family_id"] for e in entities}
    relation_fams = {r["family_id"] for r in relations}
    versions = _build_version_counts(conn, entity_fams, relation_fams)

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
    graph = get_document_graph(conn, document_version_ids, document_family_ids)
    episodes = graph["episodes"]
    graph["next_cursor"] = len(episodes) if episodes else 0
    # Outline includes edges for skeleton rendering (HAS_EPISODE only)
    # but keeps concepts empty for progressive loading
    graph["concepts"] = []
    # Only keep HAS_EPISODE and DOCUMENT_LINK edges in outline
    graph["edges"] = [e for e in graph["edges"] if e.get("edge_type") == "HAS_EPISODE"]
    graph["versions"] = {}
    return graph


def get_document_graph_chunk(conn: sqlite3.Connection,
                              document_version_ids: List[str] = None,
                              document_family_ids: List[str] = None,
                              cursor: int = 0, limit: int = 12,
                              include_relations: bool = True,
                              include_versions: bool = True,
                              max_concepts: int = 8000) -> dict:
    graph = get_document_graph(conn, document_version_ids, document_family_ids)
    episodes = graph["episodes"]

    # Paginate episodes by cursor (offset into the episode list)
    if cursor >= len(episodes):
        return {
            "documents": graph["documents"],
            "episodes": [],
            "concepts": [],
            "edges": [],
            "versions": {},
            "counts": {"episodes": 0, "concepts": 0, "relations": 0},
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

    episode_ids = set(ep["version_id"] for ep in episodes)

    # Filter edges to only include those for the current chunk's episodes
    chunk_edges = [e for e in graph["edges"]
                   if e.get("episode_version_id") in episode_ids
                   or e.get("edge_type") == "HAS_EPISODE"]
    # Re-add HAS_EPISODE for current chunk
    has_ep_ids = set()
    for ep in episodes:
        doc_ver = ep.get("document_version_id")
        if doc_ver:
            eid = f"he:{ep['version_id']}"
            has_ep_ids.add(eid)
    chunk_edges = [e for e in chunk_edges
                   if e.get("edge_id") in has_ep_ids or e.get("edge_type") != "HAS_EPISODE"]

    # Filter concepts to those referenced in current chunk
    entity_fams_in_chunk = set()
    rel_fams_in_chunk = set()
    for e in chunk_edges:
        if e.get("edge_type") == "MENTIONS" and e.get("target_family_id"):
            entity_fams_in_chunk.add(e["target_family_id"])
        if e.get("edge_type") == "CONNECTS":
            if e.get("source_family_id"):
                entity_fams_in_chunk.add(e["source_family_id"])
            if e.get("target_family_id"):
                entity_fams_in_chunk.add(e["target_family_id"])
            if e.get("relation_family_id"):
                rel_fams_in_chunk.add(e["relation_family_id"])

    chunk_concepts = [c for c in graph["concepts"]
                      if (c["role"] == "entity" and c["family_id"] in entity_fams_in_chunk)
                      or (c["role"] == "relation" and c["family_id"] in rel_fams_in_chunk)]

    chunk_versions = {fid: v for fid, v in graph["versions"].items()
                      if fid in entity_fams_in_chunk or fid in rel_fams_in_chunk}

    return {
        "documents": graph["documents"],
        "episodes": episodes,
        "concepts": chunk_concepts,
        "edges": chunk_edges,
        "versions": chunk_versions,
        "counts": {
            "episodes": len(episodes),
            "concepts": len([c for c in chunk_concepts if c["role"] == "entity"]),
            "relations": len([c for c in chunk_concepts if c["role"] == "relation"]),
        },
        "cursor": cursor,
        "next_cursor": next_cursor,
    }
