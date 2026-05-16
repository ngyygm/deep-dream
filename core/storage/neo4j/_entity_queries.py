"""Neo4j EntityQueryMixin — all read-only entity query methods."""
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

from ...models import ContentPatch, Entity, Relation
from ...content_schema import parse_markdown_sections, compute_section_diff
from ._helpers import _ENTITY_RETURN_FIELDS, _ENTITY_RETURN_FIELDS_WITH_EMB, _fmt_dt, _neo4j_record_to_entity, _neo4j_record_to_relation, _parse_dt, _q

logger = logging.getLogger(__name__)


class EntityQueryMixin:
    """Read-only entity query methods.

    Shared state contract (set by Neo4jStorageManager.__init__):
        self._session()              -> Neo4j session factory
        self._streaming_session()    -> Neo4j streaming session factory
        self._run(session, cypher, **kw) -> execute Cypher with graph_id injection
        self._graph_id: str          -> active graph ID
        self._cache                  -> QueryCache
        self.resolve_family_id(fid)  -> resolve redirect
        self.resolve_family_ids(fids)-> batch resolve redirects
        self.entity_content_snippet_length -> content snippet length
    """

    def batch_get_entity_profiles(self, family_ids: List[str]) -> List[Dict[str, Any]]:
        """批量获取实体档案（entity + relations + version_count），一次查询。

        替代对每个 family_id 分别调用 get_entity_by_family_id +
        get_entity_relations_by_family_id + get_entity_version_count 的 N+1 模式。

        Returns:
            [{"family_id", "entity", "relations", "version_count"}, ...]
        """
        if not family_ids:
            return []

        # 去重 + 解析 canonical IDs（批量解析）
        resolved_map = self.resolve_family_ids(family_ids)
        canonical_map: Dict[str, str] = {}  # original -> canonical
        canonical_set: List[str] = []
        _seen_resolved: set = set()  # O(1) membership check
        for fid in family_ids:
            resolved = resolved_map.get(fid, fid)
            if resolved and resolved not in _seen_resolved:
                canonical_map[fid] = resolved
                canonical_set.append(resolved)
                _seen_resolved.add(resolved)

        if not canonical_set:
            return [{"family_id": fid, "entity": None, "relations": [], "version_count": 0} for fid in family_ids]

        # Session 1: 批量获取实体 + 版本数 + 所有 absolute_ids（单次查询）
        # Query 1a: Count total versions per family_id (including invalidated)
        with self._session() as session:
            vcnt_result = self._run(session,
                """
                MATCH (e:Entity)
                WHERE e.family_id IN $fids
                RETURN e.family_id AS fid, COUNT(e) AS total_vcnt
                """,
                fids=canonical_set,
            )
            vcnt_map = {rec["fid"]: rec["total_vcnt"] for rec in vcnt_result}

        # Query 1b: Get latest valid entity for display + ALL absolute_ids for relation matching
        with self._session() as session:
            result = self._run(session,
                f"""
                MATCH (e:Entity)
                WHERE e.family_id IN $fids
                WITH e.family_id AS fid, COLLECT(e.uuid) AS all_uuids
                // Collect ALL version UUIDs for relation lookup (including invalidated)
                WITH fid, all_uuids
                // Find latest valid entity for display
                MATCH (latest:Entity {{family_id: fid}})
                WHERE latest.invalid_at IS NULL
                WITH fid, all_uuids, latest
                ORDER BY latest.processed_time DESC
                WITH fid, all_uuids, HEAD(COLLECT(latest)) AS latest
                RETURN latest.uuid AS uuid, latest.family_id AS family_id, latest.name AS name,
                      latest.content AS content, latest.summary AS summary,
                      latest.attributes AS attributes, latest.confidence AS confidence,
                      latest.content_format AS content_format, latest.community_id AS community_id,
                      latest.valid_at AS valid_at, latest.invalid_at AS invalid_at,
                      latest.event_time AS event_time, latest.processed_time AS processed_time,
                      latest.episode_id AS episode_id, latest.source_document AS source_document,
                      latest.embedding AS embedding,
                      fid, all_uuids
                """,
                fids=canonical_set,
            )
            records = list(result)

        entity_map: Dict[str, tuple] = {}  # family_id -> (entity, version_count)
        fid_to_aids: Dict[str, List[str]] = {}
        all_aids = set()
        for record in records:
            entity = _neo4j_record_to_entity(record)
            vc = vcnt_map.get(entity.family_id, 1)
            entity_map[entity.family_id] = (entity, vc)
            aids = record.get("all_uuids", [])
            fid_to_aids[entity.family_id] = aids
            all_aids.update(aids)

        relations_map: Dict[str, List] = {fid: [] for fid in canonical_set}
        _fids_list = list(canonical_set)
        if all_aids:
            with self._session() as session:
                result = self._run(session, _q("""
                    MATCH (r:Relation)
                    WHERE (r.entity1_absolute_id IN $aids OR r.entity2_absolute_id IN $aids
                           OR r.entity1_family_id IN $fids OR r.entity2_family_id IN $fids)
                      AND r.invalid_at IS NULL
                    RETURN __REL_FIELDS__
                    """),
                    aids=list(all_aids), fids=_fids_list,
                )
                all_rels = [_neo4j_record_to_relation(rec) for rec in result]

            # Deduplicate by relation family_id (may match via both absolute_id and family_id)
            _seen_rel_fids = set()
            _deduped_rels = []
            for rel in all_rels:
                if rel.family_id and rel.family_id in _seen_rel_fids:
                    continue
                if rel.family_id:
                    _seen_rel_fids.add(rel.family_id)
                _deduped_rels.append(rel)

            # Assign relations to family_id (prefer absolute_id match, fall back to family_id)
            aid_to_fid = {}
            for fid, aids in fid_to_aids.items():
                for aid in aids:
                    aid_to_fid[aid] = fid
            for rel in _deduped_rels:
                fid1 = aid_to_fid.get(rel.entity1_absolute_id) or (
                    rel.entity1_family_id if rel.entity1_family_id in canonical_set else None
                )
                fid2 = aid_to_fid.get(rel.entity2_absolute_id) or (
                    rel.entity2_family_id if rel.entity2_family_id in canonical_set else None
                )
                if fid1:
                    relations_map[fid1].append(rel)
                if fid2 and fid2 != fid1:
                    relations_map[fid2].append(rel)

        # 组装结果
        results = []
        seen_fids = set()
        for fid in family_ids:
            canonical = canonical_map.get(fid, fid)
            if canonical in seen_fids:
                results.append({"family_id": fid, "entity": None, "relations": [], "version_count": 0})
                continue
            seen_fids.add(canonical)
            if canonical in entity_map:
                entity, vc = entity_map[canonical]
                results.append({
                    "family_id": canonical,
                    "entity": entity,
                    "relations": relations_map.get(canonical, []),
                    "version_count": vc,
                })
            else:
                results.append({"family_id": fid, "entity": None, "relations": [], "version_count": 0})

        return results

    def count_entities_since(self, since: str) -> int:
        """Count entities whose latest version has processed_time > since."""
        with self._session() as session:
            result = self._run(session, """
                MATCH (e:Entity)
                WITH e.family_id AS fid, COLLECT(e) AS ents
                UNWIND ents AS e
                WITH fid, e ORDER BY e.processed_time DESC
                WITH fid, HEAD(COLLECT(e)) AS e
                WHERE e.processed_time > datetime($since)
                RETURN COUNT(e) AS cnt
            """, since=since)
            rec = result.single()
            return rec["cnt"] if rec else 0

    def count_entity_relations_by_family_ids(self, family_ids: List[str]) -> Dict[str, int]:
        """批量获取多个 family_id 的关系数量（单个 Cypher 聚合查询）。"""
        if not family_ids:
            return {}
        resolved_map = self.resolve_family_ids(family_ids)
        canonical_ids = list({r for r in resolved_map.values() if r})
        if not canonical_ids:
            return {}
        with self._session() as session:
            result = self._run(session,
                """
                UNWIND $fids AS fid
                MATCH (e:Entity {family_id: fid})
                WITH fid, e.uuid AS aid
                OPTIONAL MATCH (r:Relation)
                WHERE (r.entity1_absolute_id = aid OR r.entity2_absolute_id = aid)
                  AND r.invalid_at IS NULL
                RETURN fid AS family_id, count(DISTINCT r) AS cnt
                """,
                fids=canonical_ids,
            )
            counts = {record["family_id"]: record["cnt"] for record in result}
            # Map back to original (pre-resolve) family_ids
            return {orig: counts.get(resolved, 0) for orig, resolved in resolved_map.items() if resolved}

    def count_isolated_entities(self) -> int:
        """统计孤立实体数量（基于 RELATES_TO 图边，与 get_isolated_entities 一致）。"""
        with self._session() as session:
            r = self._run(session, """
                MATCH (e:Entity)
                WHERE e.invalid_at IS NULL AND e.family_id IS NOT NULL
                  AND NOT EXISTS { MATCH (e)-[:RELATES_TO]-() }
                RETURN count(DISTINCT e.family_id) AS cnt
            """)
            row = r.single()
            return row["cnt"] if row else 0

    def count_unique_entities(self) -> int:
        """统计有效实体中不重复的 family_id 数量。"""
        with self._session() as session:
            result = self._run(session,
                "MATCH (e:Entity) WHERE e.invalid_at IS NULL RETURN COUNT(DISTINCT e.family_id) AS cnt"
            )
            record = result.single()
            return record["cnt"] if record else 0

    def find_entity_by_name_prefix(self, prefix: str, limit: int = 5) -> list:
        """查找名称以 prefix 开头的实体（处理消歧括号场景）。
        例如 prefix="Go语言" 可匹配 "Go语言（Golang）"。
        返回 Entity 对象列表，按 processed_time 倒序。
        """
        if not prefix:
            return []
        try:
            with self._session() as session:
                result = self._run(session,
                    """
                    MATCH (e:Entity)
                    WHERE (e.name STARTS WITH $prefix OR e.name = $prefix)
                      AND e.invalid_at IS NULL
                    RETURN e.uuid AS uuid, e.family_id AS family_id,
                           e.name AS name, e.content AS content,
                           e.summary AS summary,
                           e.attributes AS attributes, e.confidence AS confidence,
                           e.source_document AS source_document, e.episode_id AS episode_id,
                           e.doc_name AS doc_name,
                           e.processed_time AS processed_time, e.event_time AS event_time,
                           e.content_format AS content_format
                    ORDER BY e.processed_time DESC
                    LIMIT $limit
                    """,
                    prefix=prefix,
                    limit=limit,
                )
                entities = []
                seen_fids = set()
                for record in result:
                    fid = record.get("family_id")
                    if fid and fid not in seen_fids:
                        seen_fids.add(fid)
                        entities.append(_neo4j_record_to_entity(record))
                return entities
        except Exception as e:
            logger.debug("find_entity_by_name_prefix failed for '%s': %s", prefix, e)
            return []

    def get_all_entities(self, limit: Optional[int] = None, offset: Optional[int] = None, exclude_embedding: bool = False) -> List[Entity]:
        """获取所有实体的最新版本。"""
        with self._session() as session:
            query = f"""
                MATCH (e:Entity)
                WITH e.family_id AS fid, COLLECT(e) AS ents
                UNWIND ents AS e
                WITH fid, e ORDER BY e.processed_time DESC
                WITH fid, HEAD(COLLECT(e)) AS e
                RETURN {(_ENTITY_RETURN_FIELDS if exclude_embedding else _ENTITY_RETURN_FIELDS_WITH_EMB)}
                ORDER BY e.processed_time DESC
            """
            if offset is not None and offset > 0:
                query += f" SKIP {int(offset)}"
            if limit is not None:
                query += f" LIMIT {int(limit)}"
            result = self._run(session, query)
            records = list(result)

        return [_neo4j_record_to_entity(r) for r in records]

    def get_all_entities_before_time(self, time_point: datetime, limit: Optional[int] = None,
                                      exclude_embedding: bool = False) -> List[Entity]:
        """获取指定时间点之前的所有实体最新版本。"""
        with self._session() as session:
            query = f"""
                MATCH (e:Entity)
                WHERE e.event_time <= datetime($tp)
                WITH e.family_id AS fid, COLLECT(e) AS ents
                UNWIND ents AS e
                WITH fid, e ORDER BY e.processed_time DESC
                WITH fid, HEAD(COLLECT(e)) AS e
                RETURN {(_ENTITY_RETURN_FIELDS if exclude_embedding else _ENTITY_RETURN_FIELDS_WITH_EMB)}
                ORDER BY e.processed_time DESC
            """
            if limit is not None:
                query += f" LIMIT {int(limit)}"
            result = self._run(session, query, tp=time_point.isoformat())
            records = list(result)

        return [_neo4j_record_to_entity(r) for r in records]

    def get_all_entity_names_map(self) -> Dict[str, str]:
        """Return {uuid: name} for current (non-invalidated) entities.

        Used by the relation SSE stream to resolve endpoint names.
        Endpoints are remapped to latest absolute_id before lookup.
        """
        with self._session() as session:
            result = self._run(session,
                "MATCH (e:Entity) WHERE e.invalid_at IS NULL "
                "RETURN e.uuid AS uuid, e.name AS name",
            )
            return {record["uuid"]: record["name"] for record in result}

    def get_family_ids_by_absolute_ids(self, absolute_ids: List[str]) -> Dict[str, str]:
        """批量根据 absolute_id 查询实体 family_id。"""
        if not absolute_ids:
            return {}
        with self._session() as session:
            result = self._run(session,
                """
                MATCH (e:Entity)
                WHERE e.uuid IN $uuids
                RETURN e.uuid AS uuid, e.family_id AS family_id
                """,
                uuids=absolute_ids,
            )
            return {record["uuid"]: record["family_id"] for record in result}

    def get_content_patches(self, family_id: str, section_key: str = None) -> list:
        """查询指定 family_id 的 ContentPatch 记录。"""
        _PATCH_FIELDS = (
            "cp.uuid AS uuid, cp.target_type AS target_type, "
            "cp.target_absolute_id AS target_absolute_id, cp.target_family_id AS target_family_id, "
            "cp.section_key AS section_key, cp.change_type AS change_type, "
            "cp.old_hash AS old_hash, cp.new_hash AS new_hash, "
            "cp.diff_summary AS diff_summary, cp.source_document AS source_document, "
            "cp.event_time AS event_time"
        )
        with self._session() as session:
            if section_key:
                result = self._run(session,
                    f"MATCH (cp:ContentPatch {{target_family_id: $fid, section_key: $sk}}) "
                    f"RETURN {_PATCH_FIELDS} ORDER BY cp.event_time DESC",
                    graph_id_safe=False,
                    fid=family_id, sk=section_key,
                )
            else:
                result = self._run(session,
                    f"MATCH (cp:ContentPatch {{target_family_id: $fid}}) "
                    f"RETURN {_PATCH_FIELDS} ORDER BY cp.event_time DESC",
                    graph_id_safe=False,
                    fid=family_id,
                )
            patches = []
            for record in result:
                patches.append(ContentPatch(
                    uuid=record["uuid"],
                    target_type=record["target_type"],
                    target_absolute_id=record["target_absolute_id"],
                    target_family_id=record["target_family_id"],
                    section_key=record["section_key"],
                    change_type=record["change_type"],
                    old_hash=record.get("old_hash", ""),
                    new_hash=record.get("new_hash", ""),
                    diff_summary=record.get("diff_summary", ""),
                    source_document=record.get("source_document", ""),
                    event_time=_parse_dt(record.get("event_time")),
                ))
            return patches

    def get_data_quality_report(self) -> Dict[str, Any]:
        """返回数据质量报告（合并查询，减少 Neo4j 往返次数）。"""
        with self._session() as session:
            # Query 1: All entity + relation counts in one aggregated query
            r = self._run(session, """
                CALL () {
                    MATCH (e:Entity) WHERE e.invalid_at IS NULL AND e.family_id IS NOT NULL
                    RETURN count(DISTINCT e.family_id) AS ent_valid_families, count(e) AS ent_valid_nodes
                }
                CALL () {
                    MATCH (e:Entity) WHERE e.invalid_at IS NOT NULL RETURN count(e) AS ent_invalidated
                }
                CALL () {
                    MATCH (e:Entity) WHERE e.family_id IS NULL RETURN count(e) AS ent_no_fid
                }
                CALL () {
                    MATCH (rel:Relation) WHERE rel.invalid_at IS NULL
                    RETURN count(DISTINCT rel.family_id) AS rel_valid_families, count(rel) AS rel_valid_nodes
                }
                CALL () {
                    MATCH (rel:Relation) WHERE rel.invalid_at IS NOT NULL RETURN count(rel) AS rel_invalidated
                }
                RETURN ent_valid_families, ent_valid_nodes, ent_invalidated, ent_no_fid,
                       rel_valid_families, rel_valid_nodes, rel_invalidated
            """)
            row = r.single()
            valid_families = row["ent_valid_families"]
            valid_nodes = row["ent_valid_nodes"]
            invalidated_entity_versions = row["ent_invalidated"]
            no_family_id = row["ent_no_fid"]
            valid_relation_families = row["rel_valid_families"]
            valid_relation_nodes = row["rel_valid_nodes"]
            invalidated_relation_versions = row["rel_invalidated"]

            # Query 2: Dangling ref detection (relation endpoints vs valid entity UUIDs)
            r = self._run(session, """
                CALL () {
                    MATCH (rel:Relation) WHERE rel.invalid_at IS NULL
                    WITH collect(DISTINCT rel.entity1_absolute_id) + collect(DISTINCT rel.entity2_absolute_id) AS rel_aids
                    UNWIND rel_aids AS aid
                    RETURN collect(DISTINCT aid) AS all_rel_aids
                }
                CALL () {
                    MATCH (e:Entity) WHERE e.invalid_at IS NULL
                    RETURN collect(DISTINCT e.uuid) AS ent_uuids
                }
                RETURN all_rel_aids, ent_uuids
            """)
            row = r.single()
            rel_aids = set(row["all_rel_aids"] or ())
            valid_uuids = set(row["ent_uuids"] or ())
            dangling_refs = len(rel_aids - valid_uuids)

        # 孤立实体
        isolated_count = self.count_isolated_entities()

        return {
            "entities": {
                "valid_unique": valid_families,
                "valid_versions": valid_nodes,
                "invalidated_versions": invalidated_entity_versions,
                "no_family_id": no_family_id,
                "isolated": isolated_count,
            },
            "relations": {
                "valid_unique": valid_relation_families,
                "valid_versions": valid_relation_nodes,
                "invalidated_versions": invalidated_relation_versions,
                "dangling_entity_refs": dangling_refs,
            },
            "total_nodes": valid_nodes + invalidated_entity_versions + valid_relation_nodes + invalidated_relation_versions + no_family_id,
        }

    def get_entity_absolute_ids_up_to_version(self, family_id: str, max_absolute_id: str) -> List[str]:
        """获取从最早版本到指定版本的所有 absolute_id。"""
        family_id = self.resolve_family_id(family_id)
        if not family_id:
            return []
        with self._session() as session:
            result = self._run(session,
                """
                MATCH (e:Entity {family_id: $fid})
                WHERE e.processed_time <= (
                    MATCH (e2:Entity {uuid: $max_abs}) RETURN e2.processed_time
                )
                RETURN e.uuid AS uuid
                ORDER BY e.processed_time ASC
                """,
                fid=family_id,
                max_abs=max_absolute_id,
            )
            return [r["uuid"] for r in result]

    def get_entity_by_absolute_id(self, absolute_id: str) -> Optional[Entity]:
        """根据 absolute_id 获取实体。"""
        cache_key = f"entity:by_abs:{absolute_id}"
        cached = self._cache.get(cache_key)
        if cached is not None:
            return cached
        with self._session() as session:
            result = self._run(session,
                f"""
                MATCH (e:Entity {{uuid: $uuid}})
                RETURN {_ENTITY_RETURN_FIELDS_WITH_EMB}
                """,
                uuid=absolute_id,
            )
            record = result.single()
            if not record:
                return None
            entity = _neo4j_record_to_entity(record)
            self._cache.set(cache_key, entity, ttl=60)
            return entity

    def get_entity_by_family_id(self, family_id: str) -> Optional[Entity]:
        """根据 family_id 获取最新版本的实体。"""
        from ...perf import _perf_timer
        # Fast path: check cache with raw family_id before resolve
        cache_key = f"entity:by_fid:{family_id}"
        cached = self._cache.get(cache_key)
        if cached is not None:
            return cached
        # Also check with canonical family_id in case it was previously resolved
        resolved_fid = self.resolve_family_id(family_id)
        if resolved_fid != family_id:
            cache_key2 = f"entity:by_fid:{resolved_fid}"
            cached = self._cache.get(cache_key2)
            if cached is not None:
                self._cache.set(cache_key, cached, ttl=60)  # warm raw key
                return cached
        family_id = resolved_fid
        if not family_id:
            return None
        with _perf_timer("get_entity_by_family_id"):
            with self._session() as session:
                result = self._run(session,
                    f"""
                    MATCH (e:Entity {{family_id: $fid}})
                    RETURN {_ENTITY_RETURN_FIELDS_WITH_EMB}
                    ORDER BY e.processed_time DESC LIMIT 1
                    """,
                    fid=family_id,
                )
                record = result.single()
                if not record:
                    return None
                entity = _neo4j_record_to_entity(record)
                self._cache.set(cache_key, entity, ttl=60)
                if resolved_fid != family_id:
                    self._cache.set(f"entity:by_fid:{family_id}", entity, ttl=60)
                return entity

    def get_entity_embedding_preview(self, absolute_id: str, num_values: int = 5) -> Optional[List[float]]:
        """获取实体 embedding 预览。"""
        with self._session() as session:
            result = self._run(session,
                "MATCH (e:Entity {uuid: $uuid}) RETURN e.embedding AS embedding",
                uuid=absolute_id,
            )
            record = result.single()
            if record and record["embedding"]:
                return record["embedding"][:num_values]
        return None

    def get_entity_names_by_family_ids(self, family_ids: List[str]) -> Dict[str, str]:
        """批量根据 family_id 查询实体最新名称。"""
        if not family_ids:
            return {}
        with self._session() as session:
            result = self._run(session,
                "MATCH (e:Entity) WHERE e.family_id IN $fids AND e.invalid_at IS NULL "
                "WITH e.family_id AS fid, e.name AS name ORDER BY e.processed_time DESC "
                "RETURN fid, name",
                fids=family_ids,
            )
            out = {}
            for r in result:
                if r["fid"] not in out:
                    out[r["fid"]] = r["name"]
            return out

    def get_entity_names_by_absolute_ids(self, absolute_ids: List[str]) -> Dict[str, str]:
        """批量根据 absolute_id 查询实体名称。"""
        if not absolute_ids:
            return {}
        with self._session() as session:
            result = self._run(session,
                """
                MATCH (e:Entity)
                WHERE e.uuid IN $uuids
                RETURN e.uuid AS uuid, e.name AS name
                """,
                uuids=absolute_ids,
            )
            return {record["uuid"]: record["name"] for record in result}

    def get_entity_provenance(self, family_id: str) -> List[dict]:
        """获取提及该实体的所有 Episode。

        查询所有版本的 MENTIONS 边。
        先查找 Episode->Entity 的直接 MENTIONS；如果无结果，
        再查找通过该实体参与的关系（Episode->Relation 的 MENTIONS）间接关联的 Episode。

        优化：合并为单次 Cypher 查询（collect absolute_ids + direct MENTIONS +
        indirect MENTIONS），避免 3 次串行 Neo4j 往返。

        注意：Episode 节点可能缺少 graph_id（旧数据），因此只对 Entity/Relation 侧
        进行 graph_id 过滤，不依赖 Episode 的 graph_id。
        """
        with self._session() as session:
            # Single query: collect absolute_ids, try direct MENTIONS first,
            # fall back to indirect MENTIONS via relations
            result = self._run(session, """
                MATCH (e:Entity {family_id: $fid})
                MATCH (ep:Episode)-[m:MENTIONS]->(e)
                WHERE e.graph_id = $graph_id
                RETURN DISTINCT ep.uuid AS episode_id, m.context AS context
            """, fid=family_id, graph_id=self._graph_id, graph_id_safe=False)
            provenance = [{"episode_id": r["episode_id"], "context": r.get("context", "")} for r in result]

            if provenance:
                return provenance

            # Fallback: indirect MENTIONS via relations (only if direct found nothing)
            # Re-collect abs_ids in subquery to avoid carrying state
            result = self._run(session, """
                MATCH (e:Entity {family_id: $fid})
                MATCH (r:Relation)
                WHERE (r.entity1_absolute_id = e.uuid OR r.entity2_absolute_id = e.uuid)
                      AND r.graph_id = $graph_id
                      AND r.invalid_at IS NULL
                MATCH (ep:Episode)-[m:MENTIONS]->(r)
                RETURN DISTINCT ep.uuid AS episode_id, m.context AS context
            """, fid=family_id, graph_id=self._graph_id, graph_id_safe=False)
            return [{"episode_id": r["episode_id"], "context": r.get("context", "")} for r in result]

    def get_entity_relations(self, entity_absolute_id: str, limit: Optional[int] = None,
                              time_point: Optional[datetime] = None,
                              include_candidates: bool = False) -> List[Relation]:
        """获取与指定实体相关的所有关系。"""
        with self._session() as session:
            if time_point:
                query = _q("""
                    MATCH (r:Relation)
                    WHERE (r.entity1_absolute_id = $abs_id OR r.entity2_absolute_id = $abs_id)
                    AND r.event_time <= datetime($tp)
                    AND r.invalid_at IS NULL
                    WITH r.family_id AS fid, COLLECT(r) AS rels
                    UNWIND rels AS r
                    WITH fid, r ORDER BY r.processed_time DESC
                    WITH fid, HEAD(COLLECT(r)) AS r
                    RETURN __REL_FIELDS__
                    ORDER BY r.processed_time DESC
                """)
                params = {"abs_id": entity_absolute_id, "tp": time_point.isoformat()}
            else:
                query = _q("""
                    MATCH (r:Relation)
                    WHERE (r.entity1_absolute_id = $abs_id OR r.entity2_absolute_id = $abs_id)
                    AND r.invalid_at IS NULL
                    WITH r.family_id AS fid, COLLECT(r) AS rels
                    UNWIND rels AS r
                    WITH fid, r ORDER BY r.processed_time DESC
                    WITH fid, HEAD(COLLECT(r)) AS r
                    RETURN __REL_FIELDS__
                    ORDER BY r.processed_time DESC
                """)
                params = {"abs_id": entity_absolute_id}

            if limit is not None:
                query += f" LIMIT {int(limit)}"
            result = self._run(session, query, **params)
            relations = [_neo4j_record_to_relation(r) for r in result]
            self._remap_relation_endpoints(relations, session)
            return self._filter_dream_candidates(relations, include_candidates)

    def get_entity_relations_by_family_id(self, family_id: str, limit: Optional[int] = None,
                                           time_point: Optional[datetime] = None,
                                           max_version_absolute_id: Optional[str] = None,
                                           include_candidates: bool = False) -> List[Relation]:
        """通过 family_id 获取实体的所有关系（包含所有版本）。"""
        from ...perf import _perf_timer
        with _perf_timer("get_entity_relations_by_family_id"):
            result = self._get_entity_relations_by_family_id_impl(family_id, limit, time_point, max_version_absolute_id)
            self._remap_relation_endpoints(result)
            return self._filter_dream_candidates(result, include_candidates)

    def get_entity_relations_timeline(self, family_id: str, version_abs_ids: List[str]) -> List[Dict]:
        """批量获取实体在各版本时间点的关系（消除 N+1 查询）。

        优化：合并为单次 Cypher 查询（collect abs_ids + version times + relations），
        从 3 次串行 Neo4j 往返减少为 1 次。
        """
        family_id = self.resolve_family_id(family_id)
        if not family_id or not version_abs_ids:
            return []

        with self._session() as session:
            # Single combined query: collect abs_ids + get version times + get relations
            # Uses CALL subqueries to keep the pipeline in one round-trip
            result = self._run(session, """
                // Step 1: Get version processed_times
                MATCH (e:Entity)
                WHERE e.uuid IN $version_abs_ids
                RETURN e.uuid AS uuid, e.processed_time AS pt
                ORDER BY e.processed_time ASC
            """, fid=family_id, version_abs_ids=version_abs_ids, graph_id_safe=False)
            version_times = [{"uuid": rec["uuid"], "pt": rec["pt"]} for rec in result]

            if not version_times:
                return []

            # Step 2: Get all absolute_ids for this family
            result2 = self._run(session, """
                MATCH (e:Entity {family_id: $fid})
                RETURN collect(e.uuid) AS abs_ids
            """, fid=family_id, graph_id_safe=False)
            rec2 = result2.single()
            if not rec2:
                return []
            abs_ids = rec2["abs_ids"]

            # Step 3: Get latest-version relations referencing any abs_id
            result3 = self._run(session, """
                UNWIND $aids AS aid
                MATCH (r:Relation)
                WHERE (r.entity1_absolute_id = aid OR r.entity2_absolute_id = aid)
                  AND r.invalid_at IS NULL
                WITH r.family_id AS fid, COLLECT(r) AS rels
                UNWIND rels AS r
                WITH fid, r ORDER BY r.processed_time DESC
                WITH fid, HEAD(COLLECT(r)) AS r
                RETURN r.uuid AS uuid, r.family_id AS family_id,
                       r.content AS content, r.event_time AS event_time,
                       r.processed_time AS processed_time
            """, aids=abs_ids, graph_id_safe=False)
            relations = [rec for rec in result3]

            if not relations:
                return []

            # Filter: only include relations that appeared before at least one version time point
            timeline = []
            seen = set()
            for rel in relations:
                rel_uuid = rel["uuid"]
                if rel_uuid in seen:
                    continue
                rel_pt = rel["processed_time"]
                for v in version_times:
                    v_pt = v["pt"]
                    if rel_pt and v_pt and rel_pt <= v_pt:
                        seen.add(rel_uuid)
                        timeline.append({
                            "family_id": rel["family_id"],
                            "content": rel["content"],
                            "event_time": _fmt_dt(rel["event_time"]) if rel["event_time"] else None,
                            "absolute_id": rel_uuid,
                        })
                        break
            return timeline

    def get_entity_version_at_time(self, family_id: str, time_point: datetime) -> Optional[Entity]:
        """获取实体在指定时间点的版本。"""
        family_id = self.resolve_family_id(family_id)
        if not family_id:
            return None
        with self._session() as session:
            result = self._run(session,
                f"""
                MATCH (e:Entity {{family_id: $fid}})
                WHERE e.event_time <= datetime($tp)
                RETURN {_ENTITY_RETURN_FIELDS_WITH_EMB}
                ORDER BY e.processed_time DESC LIMIT 1
                """,
                fid=family_id,
                tp=time_point.isoformat(),
            )
            record = result.single()
            if not record:
                return None
            entity = _neo4j_record_to_entity(record)
            return entity

    def get_entity_version_count(self, family_id: str) -> int:
        """获取指定 family_id 的版本数量。"""
        family_id = self.resolve_family_id(family_id)
        if not family_id:
            return 0
        with self._session() as session:
            result = self._run(session,
                "MATCH (e:Entity {family_id: $fid}) RETURN COUNT(e) AS cnt",
                fid=family_id,
            )
            record = result.single()
            return record["cnt"] if record else 0

    def get_entity_version_counts(self, family_ids: List[str]) -> Dict[str, int]:
        """批量获取多个 family_id 的版本数量。"""
        if not family_ids:
            return {}
        # 批量解析重定向
        resolved_map = self.resolve_family_ids(family_ids)
        canonical_ids = list({r for r in resolved_map.values() if r})
        if not canonical_ids:
            return {}
        with self._session() as session:
            result = self._run(session,
                """
                MATCH (e:Entity)
                WHERE e.family_id IN $fids
                RETURN e.family_id AS family_id, COUNT(e) AS cnt
                """,
                fids=canonical_ids,
            )
            canonical_counts = {record["family_id"]: record["cnt"] for record in result}
        return {fid: canonical_counts.get(canonical, 0) for fid, canonical in resolved_map.items() if canonical}

    def get_entity_versions(self, family_id: str) -> List[Entity]:
        """获取实体的所有版本。"""
        family_id = self.resolve_family_id(family_id)
        if not family_id:
            return []
        with self._session() as session:
            result = self._run(session,
                f"""
                MATCH (e:Entity {{family_id: $fid}})
                RETURN {_ENTITY_RETURN_FIELDS}
                ORDER BY e.processed_time ASC
                """,
                fid=family_id,
            )
            entities = []
            for record in result:
                entities.append(_neo4j_record_to_entity(record))
            return entities

    def get_entity_versions_batch(self, family_ids: List[str]) -> Dict[str, List[Entity]]:
        """批量获取多个 family_id 的所有版本（单次 Cypher 查询）。"""
        if not family_ids:
            return {}
        resolved_map = self.resolve_family_ids(family_ids)
        canonical_ids = list({r for r in resolved_map.values() if r})
        if not canonical_ids:
            return {}
        with self._session() as session:
            result = self._run(session,
                f"""
                UNWIND $fids AS fid
                MATCH (e:Entity {{family_id: fid}})
                RETURN e.family_id AS fid, {_ENTITY_RETURN_FIELDS}
                ORDER BY e.processed_time ASC
                """,
                fids=canonical_ids,
            )
            versions_map: Dict[str, List[Entity]] = {fid: [] for fid in canonical_ids}
            for record in result:
                fid = record["fid"]
                if fid in versions_map:
                    versions_map[fid].append(_neo4j_record_to_entity(record))
        return {orig: versions_map.get(resolved, []) for orig, resolved in resolved_map.items() if resolved}

    def get_entities_by_absolute_ids(self, absolute_ids: List[str], valid_only: bool = False) -> List[Entity]:
        """批量根据 absolute_id 获取实体。"""
        if not absolute_ids:
            return []
        with self._session() as session:
            extra_filter = " AND e.invalid_at IS NULL" if valid_only else ""
            result = self._run(session,
                f"""
                MATCH (e:Entity)
                WHERE e.uuid IN $uuids{extra_filter}
                RETURN {_ENTITY_RETURN_FIELDS}
                """,
                uuids=absolute_ids,
            )
            return [_neo4j_record_to_entity(r) for r in result]

    def get_entities_by_family_ids(self, family_ids: List[str]) -> Dict[str, "Entity"]:
        """批量根据 family_id 获取最新版本实体，返回 {family_id: Entity}。"""
        if not family_ids:
            return {}
        # 先 resolve，利用缓存
        resolved_map = self.resolve_family_ids(list(family_ids))
        valid_fids = set(resolved_map.keys()) | set(resolved_map.values())
        if not valid_fids:
            return {}
        # 检查缓存
        result: Dict[str, "Entity"] = {}
        uncached = set()
        for fid in valid_fids:
            cached = self._cache.get(f"entity:by_fid:{fid}")
            if cached is not None:
                result[fid] = cached
            else:
                uncached.add(fid)
        # 批量查询未命中缓存的
        if uncached:
            with self._session() as session:
                cypher = (
                    f"MATCH (e:Entity) WHERE e.family_id IN $fids "
                    f"WITH e ORDER BY e.processed_time DESC "
                    f"WITH e.family_id AS fid, collect(e)[0] AS latest "
                    f"RETURN fid, latest.name AS name, latest.content AS content, latest.uuid AS uuid, latest.family_id AS family_id, latest.summary AS summary, latest.attributes AS attributes, latest.confidence AS confidence, latest.content_format AS content_format, latest.community_id AS community_id, latest.valid_at AS valid_at, latest.invalid_at AS invalid_at, latest.event_time AS event_time, latest.processed_time AS processed_time, latest.episode_id AS episode_id, latest.source_document AS source_document, latest.embedding AS embedding"
                )
                records = self._run(session, cypher, fids=list(uncached)).data()
                entities = [_neo4j_record_to_entity(rec) for rec in records]
                for entity in entities:
                    fid = entity.family_id
                    result[fid] = entity
                    self._cache.set(f"entity:by_fid:{fid}", entity, ttl=60)
        # 映射原始 ID -> 实体
        for orig_fid, resolved_fid in resolved_map.items():
            if resolved_fid in result and orig_fid not in result:
                result[orig_fid] = result[resolved_fid]
        return result

    def get_graph_statistics(self) -> Dict[str, Any]:
        """返回图谱结构统计数据（仅统计有效版本，排除已失效的旧版本节点）

        优化：拆分为轻量查询，避免 UNWIND+OPTIONAL MATCH 的 O(n*m) 展开。
        度数统计直接用 RELATES_TO 边按 family_id 聚合。
        """
        cached = self._cache.get("graph_stats")
        if cached is not None:
            return cached
        with self._session() as session:
            # Query 1: 基础计数 (fast, no joins)
            r = self._run(session, """
                MATCH (all_e:Entity) WITH count(all_e) AS total_entity_versions
                MATCH (all_r:Relation) WITH total_entity_versions, count(all_r) AS total_relation_versions
                MATCH (valid_e:Entity) WHERE valid_e.invalid_at IS NULL
                WITH total_entity_versions, total_relation_versions,
                     count(DISTINCT valid_e.family_id) AS entity_count
                MATCH (valid_r:Relation) WHERE valid_r.invalid_at IS NULL
                RETURN total_entity_versions, total_relation_versions,
                       entity_count, count(DISTINCT valid_r.family_id) AS relation_count
            """)
            row = r.single()
            if row is None:
                return {}

            total_entity_versions = row["total_entity_versions"]
            total_relation_versions = row["total_relation_versions"]
            entity_count = row["entity_count"]
            relation_count = row["relation_count"]

            stats = {
                "entity_count": entity_count,
                "relation_count": relation_count,
                "total_entity_versions": total_entity_versions,
                "total_relation_versions": total_relation_versions,
                "episode_count": self.count_episodes(),
                "community_count": self.count_communities(),
            }

            if entity_count > 0:
                # Query 2: 度数统计 — direct RELATES_TO edge aggregation (no UNWIND)
                r2 = self._run(session, """
                    MATCH (e:Entity) WHERE e.invalid_at IS NULL AND e.family_id IS NOT NULL
                    WITH e.family_id AS fid
                    OPTIONAL MATCH (e2:Entity {family_id: fid})-[:RELATES_TO]-(other:Entity)
                    WHERE e2.invalid_at IS NULL
                    WITH fid, count(DISTINCT other.family_id) AS degree
                    RETURN avg(toFloat(degree)) AS avg_degree,
                           max(degree) AS max_degree_raw,
                           sum(CASE WHEN degree = 0 THEN 1 ELSE 0 END) AS isolated_count
                """)
                deg_row = r2.single()
                if deg_row:
                    stats["avg_relations_per_entity"] = round(deg_row["avg_degree"], 2)
                    stats["max_relations_per_entity"] = deg_row["max_degree_raw"]
                    stats["isolated_entities"] = deg_row["isolated_count"]
                else:
                    stats["avg_relations_per_entity"] = 0
                    stats["max_relations_per_entity"] = 0
                    stats["isolated_entities"] = entity_count

                if entity_count > 1:
                    max_possible = entity_count * (entity_count - 1) / 2
                    stats["graph_density"] = round(relation_count / max_possible, 4)
                else:
                    stats["graph_density"] = 0.0
            else:
                stats.update({
                    "avg_relations_per_entity": 0,
                    "max_relations_per_entity": 0,
                    "isolated_entities": 0,
                    "graph_density": 0.0,
                })

            # Query 3: 实体时间趋势
            r3 = self._run(session, """
                MATCH (e:Entity)
                WHERE e.invalid_at IS NULL AND e.event_time IS NOT NULL
                WITH date(e.event_time) AS d, e.family_id AS fid
                RETURN d AS date, count(DISTINCT fid) AS cnt
                ORDER BY d
                LIMIT 30
            """)
            stats["entity_count_over_time"] = [{"date": str(rec["date"]), "count": rec["cnt"]} for rec in list(r3)]

            # Query 4: 关系时间趋势
            r4 = self._run(session, """
                MATCH (r:Relation)
                WHERE r.invalid_at IS NULL AND r.event_time IS NOT NULL
                WITH date(r.event_time) AS d, r.family_id AS fid
                RETURN d AS date, count(DISTINCT fid) AS cnt
                ORDER BY d
                LIMIT 30
            """)
            stats["relation_count_over_time"] = [{"date": str(rec["date"]), "count": rec["cnt"]} for rec in list(r4)]

        self._cache.set("graph_stats", stats, ttl=60)
        return stats

    def get_graph_version(self) -> dict:
        """Return entity_count, relation_count, last_modified for cheap polling."""
        with self._session() as session:
            result = self._run(session, """
                MATCH (e:Entity) WHERE e.invalid_at IS NULL
                WITH COUNT(DISTINCT e.family_id) AS ec, max(e.processed_time) AS et
                OPTIONAL MATCH (r:Relation) WHERE r.invalid_at IS NULL
                WITH ec, et, COUNT(DISTINCT r.family_id) AS rc, max(r.processed_time) AS rt
                RETURN ec AS entity_count, rc AS relation_count,
                       CASE WHEN et IS NOT NULL AND rt IS NOT NULL AND rt > et THEN rt
                            WHEN et IS NOT NULL THEN et
                            ELSE rt END AS last_modified
            """)
            rec = result.single()
            if not rec:
                return {"entity_count": 0, "relation_count": 0, "last_modified": None}
            lm = rec["last_modified"]
            return {
                "entity_count": rec["entity_count"],
                "relation_count": rec["relation_count"],
                "last_modified": lm.isoformat() if hasattr(lm, "isoformat") else str(lm) if lm else None,
            }

    def get_isolated_entities(self, limit: int = 100, offset: int = 0) -> List[Entity]:
        """获取所有孤立实体（有效实体中没有 RELATES_TO 边的）。"""
        with self._session() as session:
            r = self._run(session, f"""
                MATCH (e:Entity)
                WHERE e.invalid_at IS NULL AND e.family_id IS NOT NULL
                AND NOT EXISTS {{ MATCH (e)-[:RELATES_TO]-() }}
                WITH e.family_id AS fid, COLLECT(e) AS ents
                UNWIND ents AS e
                WITH fid, e ORDER BY e.processed_time DESC
                WITH fid, HEAD(COLLECT(e)) AS e
                RETURN {_ENTITY_RETURN_FIELDS}
                ORDER BY e.processed_time DESC
                SKIP $offset LIMIT $limit
            """, offset=offset, limit=limit)
            return [_neo4j_record_to_entity(rec) for rec in r]

    def get_latest_entities_projection(self, content_snippet_length: Optional[int] = None) -> List[Dict[str, Any]]:
        """获取最新实体投影。"""
        snippet_length = content_snippet_length or self.entity_content_snippet_length
        entities_with_emb = self._get_entities_with_embeddings()
        version_counts = self.get_entity_version_counts([
            e.family_id for e, _ in entities_with_emb
        ])
        results: List[Dict[str, Any]] = []
        for entity, embedding_array in entities_with_emb:
            results.append({
                "entity": entity,
                "family_id": entity.family_id,
                "name": entity.name,
                "content": entity.content,
                "content_snippet": (entity.content or "")[:snippet_length],
                "version_count": version_counts.get(entity.family_id, 1),
                "embedding_array": embedding_array,
            })
        return results

    def get_section_history(self, family_id: str, section_key: str) -> list:
        """获取单个 section 的全版本变更历史。"""
        return self.get_content_patches(family_id, section_key=section_key)

    def get_stats(self) -> Dict[str, Any]:
        """返回当前图谱的基础统计：有效实体数和关系数。

        用于 GraphRegistry.get_graph_info() 显示图谱列表信息。
        """
        try:
            with self._session() as session:
                r = self._run(session,
                    "MATCH (e:Entity) WHERE e.invalid_at IS NULL "
                    "WITH count(DISTINCT e.family_id) AS ec "
                    "MATCH (r:Relation) WHERE r.invalid_at IS NULL "
                    "RETURN ec AS entity_count, count(DISTINCT r.family_id) AS relation_count"
                )
                row = r.single()
                if row is None:
                    return {"entities": 0, "relations": 0}
                return {"entities": row["entity_count"], "relations": row["relation_count"]}
        except Exception as e:
            logger.warning("get_stats failed: %s", e)
            return {"entities": 0, "relations": 0}

    def get_version_diff(self, family_id: str, v1: str, v2: str) -> dict:
        """获取两个版本之间的 section 级 diff。

        v1, v2 是两个 absolute_id（版本 uuid）。
        返回 {section_key: {"v1": content_or_None, "v2": content_or_None, "changed": bool}}
        """
        with self._session() as session:
            v1_content = ""
            v2_content = ""
            result = self._run(session,
                """
                MATCH (e:Entity) WHERE e.uuid = $v1 OR e.uuid = $v2
                RETURN e.uuid AS uid, e.content AS content
                """,
                v1=v1, v2=v2,
            )
            for record in result:
                if record["uid"] == v1:
                    v1_content = record["content"] or ""
                elif record["uid"] == v2:
                    v2_content = record["content"] or ""
            s1 = parse_markdown_sections(v1_content)
            s2 = parse_markdown_sections(v2_content)
            return compute_section_diff(s1, s2)

    def stream_all_entities(self, exclude_embedding: bool = True, since: Optional[str] = None):
        """Yield latest-version entities one by one from the Neo4j cursor.

        Unlike get_all_entities(), this does not materialize the full result
        set -- records are yielded as they arrive from the Neo4j driver's
        lazy cursor, making it suitable for SSE streaming.

        If *since* (ISO timestamp) is given, only yield entities whose latest
        version has processed_time > since.
        """
        with self._streaming_session() as session:
            fields = _ENTITY_RETURN_FIELDS if exclude_embedding else _ENTITY_RETURN_FIELDS_WITH_EMB
            params = {}
            if since:
                query = f"""
                    MATCH (e:Entity)
                    WITH e.family_id AS fid, COLLECT(e) AS ents
                    WITH fid, ents, SIZE(ents) AS vc
                    UNWIND ents AS e
                    WITH fid, vc, e ORDER BY e.processed_time DESC
                    WITH fid, vc, HEAD(COLLECT(e)) AS e
                    WHERE e.processed_time > datetime($since)
                    RETURN {fields}, vc AS version_count
                    ORDER BY e.processed_time ASC
                """
                params["since"] = since
            else:
                query = f"""
                    MATCH (e:Entity)
                    WITH e.family_id AS fid, COLLECT(e) AS ents
                    WITH fid, ents, SIZE(ents) AS vc
                    UNWIND ents AS e
                    WITH fid, vc, e ORDER BY e.processed_time DESC
                    WITH fid, vc, HEAD(COLLECT(e)) AS e
                    RETURN {fields}, vc AS version_count
                    ORDER BY e.processed_time ASC
                """
            result = self._run(session, query, **params)
            for record in result:
                entity = _neo4j_record_to_entity(record)
                yield entity, record.get("version_count", 1)
