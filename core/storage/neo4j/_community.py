"""Neo4j CommunityMixin — extracted from neo4j_store."""
import logging
import time
from typing import Any, Dict, List, Optional, Tuple

try:
    import networkx as nx
    from networkx.algorithms.community import louvain_communities
except ImportError:
    nx = None
    louvain_communities = None

from ...models import Entity

logger = logging.getLogger(__name__)


class CommunityMixin:
    """Community operations for Neo4j backend.
    Shared state contract (set by Neo4jStorageManager.__init__):
        self._session()              → Neo4j session factory
        self._run(session, cypher, **kw) → execute Cypher with graph_id injection
        self._graph_id: str          → active graph ID
    """


    def _write_community_labels(self, assignment: Dict[str, int]):
        """批量 UNWIND SET community_id。"""
        items = [{"uuid": uuid_val, "cid": cid} for uuid_val, cid in assignment.items()]
        if not items:
            return
        batch_size = 5000
        with self._session() as session:
            for i in range(0, len(items), batch_size):
                batch = items[i:i + batch_size]
                self._run(session,
                    "UNWIND $items AS item "
                    "MATCH (e:Entity {uuid: item.uuid}) "
                    "SET e.community_id = item.cid",
                    items=batch,
                )



    def clear_communities(self) -> int:
        """清除所有 community_id 属性，返回清除数量。"""
        with self._session() as session:
            result = self._run(session, 
                "MATCH (e:Entity) WHERE e.community_id IS NOT NULL "
                "REMOVE e.community_id RETURN COUNT(e) AS cleared"
            )
            record = result.single()
            count = record["cleared"] if record else 0
            self._cache.invalidate(pattern="communities")
            return count

    # ------------------------------------------------------------------
    # 时间旅行（Time Travel）功能
    # ------------------------------------------------------------------



    def count_communities(self) -> int:
        """统计社区数量（DISTINCT community_id）。"""
        with self._session() as session:
            result = self._run(session, 
                "MATCH (e:Entity) WHERE e.community_id IS NOT NULL "
                "RETURN count(DISTINCT e.community_id) AS cnt"
            )
            record = result.single()
            return record["cnt"] if record else 0


    # ------------------------------------------------------------------

    def detect_communities(self, algorithm: str = 'louvain', resolution: float = 1.0) -> Dict:
        """从 Neo4j 加载图 → networkx Louvain → 写回 community_id。"""
        if nx is None:
            return {"error": "networkx not installed", "communities": 0}

        t0 = time.time()

        # 加载所有 Entity + RELATES_TO 边
        with self._session() as session:
            result = self._run(session, 
                "MATCH (e:Entity) RETURN e.uuid AS uuid, e.family_id AS fid, e.name AS name"
            )
            entity_map = {}  # uuid -> fid
            for r in result:
                entity_map[r["uuid"]] = r["fid"]

            result = self._run(session, 
                "MATCH (a:Entity)-[r:RELATES_TO]->(b:Entity) "
                "RETURN a.uuid AS src, b.uuid AS tgt"
            )
            edges = [(r["src"], r["tgt"]) for r in result]

        # 构建 networkx Graph
        G = nx.Graph()
        for uuid_val in entity_map:
            G.add_node(uuid_val)
        for src, tgt in edges:
            if src in G and tgt in G:
                G.add_edge(src, tgt)

        # Louvain 社区检测
        communities = louvain_communities(G, resolution=resolution, seed=42)

        # 构建 assignment: uuid -> community_id
        assignment = {}
        for cid, community_set in enumerate(communities):
            for uuid_val in community_set:
                assignment[uuid_val] = cid

        # 写回 Neo4j
        self._write_community_labels(assignment)

        elapsed = time.time() - t0
        community_sizes = [len(c) for c in communities]
        return {
            "total_communities": len(communities),
            "community_sizes": sorted(community_sizes, reverse=True),
            "elapsed_seconds": round(elapsed, 3),
        }



    def get_communities(self, limit: int = 50, min_size: int = 3, offset: int = 0) -> Tuple[List[Dict], int]:
        """按社区分组，返回 members 列表 + 总数。"""
        with self._session() as session:
            # 先获取数据
            result = self._run(session, 
                "MATCH (e:Entity) WHERE e.community_id IS NOT NULL "
                "WITH e.community_id AS cid, collect(e) AS members "
                "WHERE size(members) >= $min_size "
                "RETURN cid, size(members) AS size, "
                "[m IN members | {uuid: m.uuid, family_id: m.family_id, name: m.name}] AS members "
                "ORDER BY size DESC SKIP $offset LIMIT $limit",
                min_size=min_size, offset=offset, limit=limit,
            )
            communities = []
            for r in result:
                communities.append({
                    "community_id": r["cid"],
                    "size": r["size"],
                    "members": r["members"],
                })

            # Count total (same session — result already fully consumed above)
            count_result = self._run(session,
                "MATCH (e:Entity) WHERE e.community_id IS NOT NULL "
                "WITH e.community_id AS cid, collect(e) AS members "
                "WHERE size(members) >= $min_size "
                "RETURN count(cid) AS total",
                min_size=min_size,
            )
            count_record = count_result.single()
            total = count_record["total"] if count_record else len(communities)

        return communities, total



    def get_community(self, cid: int) -> Optional[Dict]:
        """单社区详情 + 社区内关系（合并单条 Cypher，LIMIT 500）。"""
        cache_key = f"community:{cid}"
        cached = self._cache.get(cache_key)
        if cached is not None:
            return cached

        with self._session() as session:
            result = self._run(session, 
                "MATCH (e:Entity) WHERE e.community_id = $cid "
                "OPTIONAL MATCH (e)-[r:RELATES_TO]-(other:Entity) "
                "WHERE other.community_id = $cid "
                "WITH e, collect(DISTINCT {uuid: other.uuid, name: other.name, fact: r.fact, "
                "ruuid: r.relation_uuid}) AS rels "
                "RETURN e.uuid AS uuid, e.family_id AS family_id, e.name AS name, "
                "e.content AS content, rels "
                "ORDER BY size(rels) DESC LIMIT 500",
                cid=cid,
            )
            members = []
            all_relations = []
            seen_rels = set()
            for r in result:
                members.append({
                    "uuid": r["uuid"],
                    "family_id": r["family_id"],
                    "name": r["name"],
                    "content": r["content"] or "",
                })
                for rel in (r["rels"] or []):
                    if not rel.get("uuid"):
                        continue
                    rel_key = (r["uuid"], rel["uuid"]) if r["uuid"] <= rel["uuid"] else (rel["uuid"], r["uuid"])
                    if rel_key not in seen_rels:
                        seen_rels.add(rel_key)
                        all_relations.append({
                            "source_uuid": r["uuid"],
                            "source_name": r["name"],
                            "target_uuid": rel["uuid"],
                            "target_name": rel["name"],
                            "content": rel["fact"] or "",
                            "relation_uuid": rel["ruuid"],
                        })
            if not members:
                return None

            result = {
                "community_id": cid,
                "size": len(members),
                "members": members,
                "relations": all_relations,
            }
            self._cache.set(cache_key, result, ttl=120)
            return result



    def get_community_graph(self, cid: int) -> Dict:
        """社区子图的 nodes + edges（供 vis-network 渲染），LIMIT 300 节点。"""
        cache_key = f"community_graph:{cid}"
        cached = self._cache.get(cache_key)
        if cached is not None:
            return cached

        with self._session() as session:
            result = self._run(session, 
                "MATCH (e:Entity) WHERE e.community_id = $cid "
                "WITH e LIMIT 300 "
                "OPTIONAL MATCH (a:Entity)-[r:RELATES_TO]->(b:Entity) "
                "WHERE (a.uuid = e.uuid OR b.uuid = e.uuid) "
                "WITH e, collect(DISTINCT {src: a.uuid, tgt: b.uuid, fact: r.fact}) AS rels "
                "RETURN e.uuid AS uuid, e.family_id AS family_id, e.name AS name, rels",
                cid=cid,
            )
            nodes = []
            seen_edges = set()
            edges = []
            for r in result:
                nodes.append({
                    "uuid": r["uuid"],
                    "family_id": r["family_id"],
                    "name": r["name"],
                })
                for rel in (r["rels"] or []):
                    if rel.get("src") and rel.get("tgt"):
                        edge_key = (rel["src"], rel["tgt"])
                        if edge_key not in seen_edges:
                            seen_edges.add(edge_key)
                            edges.append({
                                "source_uuid": rel["src"],
                                "target_uuid": rel["tgt"],
                                "content": rel["fact"] or "",
                            })

            result = {"nodes": nodes, "edges": edges}
            self._cache.set(cache_key, result, ttl=120)
            return result

