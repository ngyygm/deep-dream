"""Community detection mixin — Louvain, community queries."""

import logging
import time
from typing import Any, Dict, List, Optional, Tuple

try:
    import networkx as nx
    from networkx.algorithms.community import louvain_communities
except ImportError:
    nx = None
    louvain_communities = None

logger = logging.getLogger(__name__)


class _CommunityMixin:

    def _write_community_labels(self, assignment: Dict[str, int]):
        if not assignment:
            return
        items = list(assignment.items())
        batch_size = 5000
        conn = self._connect()
        try:
            for i in range(0, len(items), batch_size):
                batch = items[i:i + batch_size]
                conn.executemany(
                    "UPDATE entity SET community_id = ? WHERE uuid = ? AND graph_id = ?",
                    [(str(cid), uuid_val, self._graph_id) for uuid_val, cid in batch],
                )
            conn.commit()
        finally:
            conn.rollback()

    def clear_communities(self) -> int:
        conn = self._connect()
        try:
            cursor = conn.execute("UPDATE entity SET community_id = NULL WHERE community_id IS NOT NULL AND graph_id = ?", (self._graph_id,))
            conn.commit()
            count = cursor.rowcount
        finally:
            conn.rollback()
        self._cache.invalidate_keys(["communities"])
        return count

    def count_communities(self) -> int:
        conn = self._connect()
        try:
            row = conn.execute(
                "SELECT COUNT(DISTINCT community_id) AS cnt FROM entity WHERE community_id IS NOT NULL AND graph_id = ?",
                (self._graph_id,),
            ).fetchone()
        finally:
            conn.rollback()
        return row["cnt"] if row else 0

    def detect_communities(self, algorithm: str = 'louvain', resolution: float = 1.0) -> Dict:
        if nx is None:
            return {"error": "networkx not installed", "communities": 0}
        t0 = time.time()
        conn = self._connect()
        try:
            entity_rows = conn.execute("SELECT uuid, family_id, name FROM entity WHERE graph_id = ?", (self._graph_id,)).fetchall()
            entity_map = {r["uuid"]: r["family_id"] for r in entity_rows}
            edge_rows = conn.execute(
                "SELECT entity1_uuid AS src, entity2_uuid AS tgt FROM relates_to WHERE graph_id = ?",
                (self._graph_id,),
            ).fetchall()
        finally:
            conn.rollback()
        G = nx.Graph()
        for uuid_val in entity_map:
            G.add_node(uuid_val)
        for r in edge_rows:
            if r["src"] in G and r["tgt"] in G:
                G.add_edge(r["src"], r["tgt"])
        communities = louvain_communities(G, resolution=resolution, seed=42)
        assignment = {}
        for cid, community_set in enumerate(communities):
            for uuid_val in community_set:
                assignment[uuid_val] = cid
        self._write_community_labels(assignment)
        elapsed = time.time() - t0
        community_sizes = [len(c) for c in communities]
        return {"total_communities": len(communities), "community_sizes": sorted(community_sizes, reverse=True), "elapsed_seconds": round(elapsed, 3)}

    def get_communities(self, limit: int = 50, min_size: int = 3, offset: int = 0) -> Tuple[List[Dict], int]:
        conn = self._connect()
        try:
            # Get communities with size >= min_size
            rows = conn.execute(
                "SELECT community_id, COUNT(*) AS size FROM entity "
                "WHERE community_id IS NOT NULL AND graph_id = ? GROUP BY community_id HAVING COUNT(*) >= ? "
                "ORDER BY size DESC LIMIT ? OFFSET ?",
                (self._graph_id, min_size, limit, offset),
            ).fetchall()
            total_row = conn.execute(
                "SELECT COUNT(*) AS cnt FROM (SELECT community_id FROM entity "
                "WHERE community_id IS NOT NULL AND graph_id = ? GROUP BY community_id HAVING COUNT(*) >= ?)",
                (self._graph_id, min_size),
            ).fetchone()
            total = total_row["cnt"] if total_row else 0
        finally:
            conn.rollback()
        communities = []
        for r in rows:
            cid = r["community_id"]
            members = conn.execute(
                "SELECT uuid, family_id, name FROM entity WHERE community_id = ? AND graph_id = ?",
                (cid, self._graph_id),
            ).fetchall()
            communities.append({
                "community_id": cid,
                "size": r["size"],
                "members": [{"uuid": m["uuid"], "family_id": m["family_id"], "name": m["name"]} for m in members],
            })
        return communities, total

    def get_community(self, cid: int) -> Optional[Dict]:
        cache_key = f"community:{cid}"
        cached = self._cache.get(cache_key)
        if cached is not None:
            return cached
        conn = self._connect()
        try:
            cid_str = str(cid)
            member_rows = conn.execute(
                "SELECT uuid, family_id, name, content FROM entity WHERE community_id = ? AND graph_id = ? ORDER BY name LIMIT 500",
                (cid_str, self._graph_id),
            ).fetchall()
            if not member_rows:
                return None
            members = [{"uuid": r["uuid"], "family_id": r["family_id"], "name": r["name"], "content": r["content"] or ""} for r in member_rows]
            # Get edges within community
            member_uuids = [r["uuid"] for r in member_rows]
            ph = ",".join("?" * len(member_uuids))
            edge_rows = conn.execute(
                f"SELECT r.entity1_uuid, r.entity2_uuid, r.fact, r.relation_uuid "
                f"FROM relates_to r WHERE r.graph_id = ? "
                f"AND r.entity1_uuid IN ({ph}) AND r.entity2_uuid IN ({ph})",
                [self._graph_id] + member_uuids + member_uuids,
            ).fetchall()
            uuid_to_name = {r["uuid"]: r["name"] for r in member_rows}
            relations = []
            seen_edges = set()
            for r in edge_rows:
                e_key = (r["entity1_uuid"], r["entity2_uuid"])
                if e_key not in seen_edges:
                    seen_edges.add(e_key)
                    relations.append({
                        "source_uuid": r["entity1_uuid"], "source_name": uuid_to_name.get(r["entity1_uuid"], ""),
                        "target_uuid": r["entity2_uuid"], "target_name": uuid_to_name.get(r["entity2_uuid"], ""),
                        "content": r["fact"] or "", "relation_uuid": r["relation_uuid"],
                    })
        finally:
            conn.rollback()
        result = {"community_id": cid, "size": len(members), "members": members, "relations": relations}
        self._cache.set(cache_key, result, ttl=120)
        return result

    def get_community_graph(self, cid: int) -> Dict:
        cache_key = f"community_graph:{cid}"
        cached = self._cache.get(cache_key)
        if cached is not None:
            return cached
        conn = self._connect()
        try:
            cid_str = str(cid)
            member_rows = conn.execute(
                "SELECT uuid, family_id, name FROM entity WHERE community_id = ? AND graph_id = ? LIMIT 300",
                (cid_str, self._graph_id),
            ).fetchall()
            member_uuids = [r["uuid"] for r in member_rows]
            if not member_uuids:
                return {"nodes": [], "edges": []}
            ph = ",".join("?" * len(member_uuids))
            edge_rows = conn.execute(
                f"SELECT entity1_uuid, entity2_uuid, fact FROM relates_to WHERE graph_id = ? "
                f"AND entity1_uuid IN ({ph}) AND entity2_uuid IN ({ph})",
                [self._graph_id] + member_uuids + member_uuids,
            ).fetchall()
            nodes = [{"uuid": r["uuid"], "family_id": r["family_id"], "name": r["name"]} for r in member_rows]
            seen_edges = set()
            edges = []
            for r in edge_rows:
                e_key = (r["entity1_uuid"], r["entity2_uuid"])
                if e_key not in seen_edges:
                    seen_edges.add(e_key)
                    edges.append({"source_uuid": r["entity1_uuid"], "target_uuid": r["entity2_uuid"], "content": r["fact"] or ""})
        finally:
            conn.rollback()
        result = {"nodes": nodes, "edges": edges}
        self._cache.set(cache_key, result, ttl=120)
        return result
