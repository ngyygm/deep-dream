"""Neo4j GraphTraversalMixin — extracted from neo4j_store."""
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional, Set, Tuple

from ...models import Entity, Relation
from ._helpers import _ENTITY_RETURN_FIELDS, _neo4j_record_to_entity, _neo4j_record_to_relation, _q

logger = logging.getLogger(__name__)


class GraphTraversalMixin:
    """GraphTraversal operations for Neo4j backend.
    Shared state contract (set by Neo4jStorageManager.__init__):
        self._session()              → Neo4j session factory
        self._run(session, cypher, **kw) → execute Cypher with graph_id injection
        self._graph_id: str          → active graph ID
    """


    def batch_bfs_traverse(self, seed_family_ids: List[str], max_depth: int = 2, max_nodes: int = 50,
                           time_point: Optional[str] = None) -> Tuple[List[Entity], List[Relation], Dict[str, int]]:
        """批量 BFS 遍历：从种子实体出发，单次 Cypher 查询完成多跳扩展。

        Args:
            seed_family_ids: 种子实体的 family_id 列表
            max_depth: 最大扩展深度
            max_nodes: 最多返回的节点数
            time_point: ISO 8601 时间点，仅返回 valid_at <= time_point 且未失效的实体/关系

        Returns:
            (entities, relations, hop_map) 其中 hop_map[family_id] = hop 距离
        """
        if not seed_family_ids:
            return [], [], {}

        # 构建 time_point 过滤条件
        _tp_filter_seed_resolve = ""   # for seed resolution (variable 'e')
        _tp_filter_seed_bfs = ""       # for BFS seed match (variable 'seed')
        _tp_filter_neighbor = ""       # for BFS neighbor match (variable 'neighbor')
        _tp_param = {}
        if time_point:
            _tp_filter_seed_resolve = " AND (e.valid_at IS NULL OR e.valid_at <= datetime($tp))"
            _tp_filter_seed_bfs = " AND (seed.valid_at IS NULL OR seed.valid_at <= datetime($tp))"
            _tp_filter_neighbor = " AND (neighbor.valid_at IS NULL OR neighbor.valid_at <= datetime($tp))"
            _tp_param["tp"] = time_point

        with self._session() as session:
            # 第一步：解析种子 family_id → absolute_id
            seed_result = self._run(session, """
                MATCH (e:Entity)
                WHERE e.family_id IN $family_ids AND e.invalid_at IS NULL%s
                RETURN e.family_id AS family_id, e.uuid AS absolute_id
            """ % _tp_filter_seed_resolve, family_ids=seed_family_ids, **_tp_param)

            seed_abs_to_fid = {}
            seed_fids = []
            for rec in seed_result:
                fid = rec["family_id"]
                aid = rec["absolute_id"]
                seed_abs_to_fid[aid] = fid
                seed_fids.append(fid)

            if not seed_fids:
                return [], [], {}

            # 第二步：Cypher BFS — 从种子 absolute_id 出发，沿关系边扩展
            # 通过 Entity 节点的 RELATES_TO 边进行遍历（已有边类型）
            result = self._run(session, """
                MATCH (seed:Entity)
                WHERE seed.family_id IN $seed_fids AND seed.invalid_at IS NULL%s
                WITH collect(seed) AS seeds
                UNWIND seeds AS s
                MATCH path = (s)-[:RELATES_TO*1..%d]-(neighbor:Entity)
                WHERE neighbor.invalid_at IS NULL%s
                WITH DISTINCT neighbor, length(path) AS dist
                ORDER BY dist ASC
                LIMIT $max_nodes
                RETURN neighbor.uuid AS uuid,
                       neighbor.family_id AS family_id,
                       neighbor.name AS name,
                       neighbor.content AS content,
                       neighbor.event_time AS event_time,
                       neighbor.processed_time AS processed_time,
                       neighbor.episode_id AS episode_id,
                       neighbor.source_document AS source_document,
                       neighbor.summary AS summary,
                       neighbor.confidence AS confidence,
                       neighbor.attributes AS attributes,
                       neighbor.community_id AS community_id,
                       neighbor.valid_at AS valid_at,
                       neighbor.invalid_at AS invalid_at,
                       neighbor.content_format AS content_format,
                       dist AS dist
            """ % (_tp_filter_seed_bfs, max_depth, _tp_filter_neighbor),
                seed_fids=seed_fids, max_nodes=max_nodes, **_tp_param)

            entities = []
            hop_map = {}
            for rec in result:
                ent = _neo4j_record_to_entity(rec)
                if ent and ent.family_id not in hop_map:
                    entities.append(ent)
                    hop_map[ent.family_id] = rec["dist"]

            # 种子实体也加入（如果 BFS 没有返回）
            for fid in seed_fids:
                if fid not in hop_map:
                    entity = self.get_entity_by_family_id(fid)
                    if entity and entity not in entities:
                        entities.insert(0, entity)
                        hop_map[fid] = 0

            # 第三步：批量获取这些实体之间的关系（带 time_point 过滤）
            discovered_fids = list(hop_map.keys())
            if discovered_fids:
                relations = self.get_relations_by_family_ids(
                    discovered_fids, limit=max_nodes * 3, time_point=time_point)
            else:
                relations = []

            return entities, relations, hop_map



    def batch_get_entity_degrees(self, family_ids: List[str]) -> Dict[str, int]:
        """批量获取实体度数 — 单次 Cypher 查询替代 N 次 get_entity_degree。"""
        if not family_ids:
            return {}
        with self._session() as session:
            result = self._run(session, """
                UNWIND $fids AS fid
                MATCH (e:Entity) WHERE e.family_id = fid AND e.invalid_at IS NULL
                WITH fid, collect(DISTINCT e.uuid) AS abs_ids
                UNWIND abs_ids AS aid
                OPTIONAL MATCH (r:Relation)
                WHERE (r.entity1_absolute_id = aid OR r.entity2_absolute_id = aid)
                  AND r.invalid_at IS NULL
                RETURN fid, count(DISTINCT r) AS cnt
            """, fids=family_ids)
            degree_map = {}
            for record in result:
                degree_map[record["fid"]] = record["cnt"]
        for fid in family_ids:
            degree_map.setdefault(fid, 0)
        return degree_map


    # ------------------------------------------------------------------

    def find_shortest_path_cypher(self, source_family_id: str, target_family_id: str,
                                   max_depth: int = 6) -> List[List[str]]:
        """使用 Cypher shortestPath 查找单条最短路径（性能更优）。

        Returns:
            路径列表，每条路径为实体名称列表。
        """
        with self._session() as session:
            result = self._run(session, 
                """
                MATCH (a:Entity {family_id: $sid}), (b:Entity {family_id: $tid})
                MATCH path = shortestPath((a)-[:RELATES_TO*1..""" + str(max_depth) + """]-(b))
                RETURN [n IN nodes(path) | n.name] AS names
                """,
                sid=source_family_id,
                tid=target_family_id,
            )
            records = list(result)
            if not records:
                return []
            return [record["names"] for record in records]



    def find_shortest_paths(self, source_family_id: str, target_family_id: str,
                             max_depth: int = 6, max_paths: int = 10) -> Dict[str, Any]:
        """使用 Neo4j Cypher 查找最短路径。"""
        result_empty = {
            "source_entity": None,
            "target_entity": None,
            "path_length": -1,
            "total_shortest_paths": 0,
            "paths": [],
        }

        source_entity = self.get_entity_by_family_id(source_family_id)
        target_entity = self.get_entity_by_family_id(target_family_id)

        if not source_entity or not target_entity:
            result_empty["source_entity"] = source_entity
            result_empty["target_entity"] = target_entity
            return result_empty

        if source_family_id == target_family_id:
            return {
                "source_entity": source_entity,
                "target_entity": target_entity,
                "path_length": 0,
                "total_shortest_paths": 1,
                "paths": [{
                    "entities": [source_entity],
                    "relations": [],
                    "length": 0,
                }],
            }

        # 使用 Cypher allShortestPaths
        with self._session() as session:
            result = self._run(session, 
                """
                MATCH (source:Entity {family_id: $sid}),
                      (target:Entity {family_id: $tid})
                MATCH path = allShortestPaths((source)-[:RELATES_TO*1..""" + str(max_depth) + """]-(target))
                UNWIND [n IN nodes(path) | n.uuid] AS abs_ids
                UNWIND [r IN relationships(path) | r.relation_uuid] AS rel_uuids
                WITH path, COLLECT(DISTINCT abs_ids) AS abs_id_set, COLLECT(DISTINCT rel_uuids) AS rel_id_set
                RETURN abs_id_set, rel_id_set
                LIMIT $max_paths
                """,
                sid=source_family_id,
                tid=target_family_id,
                max_paths=max_paths,
            )

            paths_result = []
            abs_to_eid: Dict[str, str] = {}
            needed_abs_ids: Set[str] = set()
            needed_rel_ids: Set[str] = set()

            for record in result:
                abs_ids = record["abs_id_set"]
                rel_ids = record["rel_id_set"]
                needed_abs_ids.update(abs_ids)
                needed_rel_ids.update(rel_ids)

            # 批量获取实体和关系
            abs_entity_map: Dict[str, Entity] = {}
            if needed_abs_ids:
                res = self._run(session, 
                    f"""
                    MATCH (e:Entity)
                    WHERE e.uuid IN $uuids
                    RETURN {_ENTITY_RETURN_FIELDS}
                    """,
                    uuids=list(needed_abs_ids),
                )
                for r in res:
                    entity = _neo4j_record_to_entity(r)
                    abs_entity_map[entity.absolute_id] = entity
                    abs_to_eid[entity.absolute_id] = entity.family_id

            rel_map: Dict[str, Relation] = {}
            if needed_rel_ids:
                res = self._run(session, _q("""
                    MATCH (r:Relation)
                    WHERE r.uuid IN $uuids
                    RETURN __REL_FIELDS__
                    """),
                    uuids=list(needed_rel_ids),
                )
                for r in res:
                    relation = _neo4j_record_to_relation(r)
                    rel_map[relation.absolute_id] = relation

            # 构建路径结果
            result2 = self._run(session, 
                """
                MATCH (source:Entity {family_id: $sid}),
                      (target:Entity {family_id: $tid})
                MATCH path = allShortestPaths((source)-[:RELATES_TO*1..""" + str(max_depth) + """]-(target))
                UNWIND nodes(path) AS n
                UNWIND relationships(path) AS r
                WITH path, COLLECT(DISTINCT {uuid: n.uuid, family_id: n.family_id}) AS nodes,
                     COLLECT(DISTINCT {uuid: r.relation_uuid}) AS rels
                RETURN nodes, rels
                LIMIT $max_paths
                """,
                sid=source_family_id,
                tid=target_family_id,
                max_paths=max_paths,
            )

            for record in result2:
                path_entities = []
                seen_abs: Set[str] = set()
                for node_info in record["nodes"]:
                    abs_id = node_info["uuid"]
                    if abs_id not in seen_abs and abs_id in abs_entity_map:
                        path_entities.append(abs_entity_map[abs_id])
                        seen_abs.add(abs_id)

                path_relations = []
                for rel_info in record["rels"]:
                    rel_id = rel_info["uuid"]
                    if rel_id in rel_map:
                        path_relations.append(rel_map[rel_id])

                paths_result.append({
                    "entities": path_entities,
                    "relations": path_relations,
                    "length": len(path_entities) - 1,
                })

            path_length = paths_result[0]["length"] if paths_result else -1

            return {
                "source_entity": source_entity,
                "target_entity": target_entity,
                "path_length": path_length,
                "total_shortest_paths": len(paths_result),
                "paths": paths_result,
            }

    # ------------------------------------------------------------------
    # Neo4j 特有操作（新增能力）
    # ------------------------------------------------------------------



    def get_entity_neighbors(self, entity_uuid: str, depth: int = 1) -> Dict:
        """获取实体的邻居图，返回完整的 nodes + edges 结构。"""
        with self._session() as session:
            # 先获取中心节点
            center = self._run(session, 
                "MATCH (e:Entity {uuid: $uuid}) RETURN e.uuid AS uuid, e.name AS name, e.family_id AS family_id",
                uuid=entity_uuid,
            )
            center_records = list(center)
            center_node = None
            if center_records:
                r = center_records[0]
                center_node = {"uuid": r["uuid"], "name": r["name"], "family_id": r["family_id"]}

            # 获取所有邻居节点和边
            result = self._run(session, 
                f"""
                MATCH (e:Entity {{uuid: $uuid}})-[r:RELATES_TO*1..{depth}]-(neighbor:Entity)
                UNWIND r AS rel
                WITH DISTINCT neighbor, rel LIMIT 500
                RETURN neighbor.uuid AS uuid, neighbor.name AS name, neighbor.family_id AS family_id,
                       startNode(rel).uuid AS source_uuid, endNode(rel).uuid AS target_uuid,
                       rel.relation_uuid AS relation_uuid, rel.fact AS fact
                """,
                uuid=entity_uuid,
            )
            neighbors = {
                "entity": center_node,
                "nodes": [],
                "edges": [],
            }
            seen = set()
            seen_edges = set()
            for record in result:
                uuid_val = record["uuid"]
                if uuid_val and uuid_val not in seen:
                    neighbors["nodes"].append({
                        "uuid": uuid_val,
                        "name": record["name"],
                        "family_id": record["family_id"],
                    })
                    seen.add(uuid_val)
                edge_key = (record.get("source_uuid"), record.get("target_uuid"))
                if edge_key[0] and edge_key[1] and edge_key not in seen_edges:
                    neighbors["edges"].append({
                        "source_uuid": edge_key[0],
                        "target_uuid": edge_key[1],
                        "content": record["fact"],
                        "relation_uuid": record.get("relation_uuid"),
                    })
                    seen_edges.add(edge_key)
            return neighbors

    # ------------------------------------------------------------------

    def merge_entity_families(self, target_family_id: str, source_family_ids: List[str],
                              skip_name_check: bool = False) -> Dict[str, Any]:
        """合并多个 family_id 到目标 family_id。

        Args:
            target_family_id: 目标实体的 family_id。
            source_family_ids: 要合并的源实体 family_id 列表。
            skip_name_check: 如果为 True，跳过名称安全检查（用于已确认的合并）。
        """
        target_family_id = self.resolve_family_id(target_family_id)
        if not target_family_id or not source_family_ids:
            return {"entities_updated": 0, "relations_updated": 0}

        # 名称安全检查：拒绝名称完全不相关的合并
        if not skip_name_check:
            target_entity = self.get_entity_by_family_id(target_family_id)
            target_name = target_entity.name if target_entity else ""
            for source_id in source_family_ids:
                resolved_source = self.resolve_family_id(source_id)
                if not resolved_source:
                    continue
                source_entity = self.get_entity_by_family_id(resolved_source)
                if not source_entity:
                    continue
                source_name = source_entity.name
                if target_name and source_name:
                    shared = len(set(source_name) & set(target_name))
                    total = len(set(source_name) | set(target_name))
                    overlap = shared / total if total > 0 else 0
                    if overlap < 0.2:
                        logging.getLogger(__name__).warning(
                            f"拒绝合并：名称差异过大 — "
                            f"target={target_name}({target_family_id}) "
                            f"source={source_name}({resolved_source}) "
                            f"overlap={overlap:.2f}"
                        )
                        # 从列表中移除此 source
                        source_family_ids = [
                            s for s in source_family_ids
                            if self.resolve_family_id(s) != resolved_source
                        ]

        if not source_family_ids:
            return {"entities_updated": 0, "relations_updated": 0, "rejected": True}

        with self._write_lock:
            with self._session() as session:
                entities_updated = 0
                canonical_source_ids: List[str] = []

                for source_id in source_family_ids:
                    source_id = self._resolve_family_id_in_session(session, source_id)
                    if not source_id or source_id == target_family_id or source_id in canonical_source_ids:
                        continue
                    canonical_source_ids.append(source_id)

                    # 更新所有使用 source_id 的实体节点
                    result = self._run(session, 
                        """
                        MATCH (e:Entity {family_id: $sid})
                        SET e.family_id = $tid
                        RETURN COUNT(e) AS cnt
                        """,
                        sid=source_id,
                        tid=target_family_id,
                    )
                    record = result.single()
                    if record:
                        entities_updated += record["cnt"]

                    # 创建 redirect
                    now_iso = datetime.now().isoformat()
                    self._run(session, 
                        """
                        MERGE (red:EntityRedirect {source_id: $sid})
                        SET red.target_id = $tid, red.updated_at = $now
                        """,
                        sid=source_id,
                        tid=target_family_id,
                        now=now_iso,
                    )

                return {
                    "entities_updated": entities_updated,
                    "relations_updated": 0,
                    "target_family_id": target_family_id,
                    "merged_source_ids": canonical_source_ids,
                }

