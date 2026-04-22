"""Neo4j Neo4jBaseMixin — extracted from neo4j_store."""
import json
import logging
from datetime import datetime, timezone
from typing import Dict, List, Optional, Set

from ...models import Entity
from ...perf import _perf_timer

logger = logging.getLogger(__name__)


class Neo4jBaseMixin:
    """Neo4jBase operations for Neo4j backend.
    Shared state contract (set by Neo4jStorageManager.__init__):
        self._session()              → Neo4j session factory
        self._run(session, cypher, **kw) → execute Cypher with graph_id injection
        self._graph_id: str          → active graph ID
    """


    def _filter_dream_candidates(self, relations: list, include_candidates: bool = False) -> list:
        """Filter out dream candidate relations unless explicitly requested.

        Removes hypothesized and rejected candidates from normal search results.
        Verified/promoted candidates are always shown.
        """
        if include_candidates or not relations:
            return relations
        return [r for r in relations if not self._is_dream_candidate(r)]


    # ------------------------------------------------------------------

    def _is_dream_candidate(self, relation) -> bool:
        """Check if a relation is a dream candidate that should be hidden from normal search.

        Filters out both hypothesized and rejected candidates.
        Verified/promoted candidates are NOT filtered (they appear in normal search).
        """
        if not relation.attributes:
            return False
        try:
            attrs = json.loads(relation.attributes) if isinstance(relation.attributes, str) else relation.attributes
            tier = attrs.get("tier")
            status = attrs.get("status")
            # Filter: tier is candidate AND status is not verified
            return tier == "candidate" and status != "verified"
        except (json.JSONDecodeError, TypeError, AttributeError):
            return False


    # ------------------------------------------------------------------

    def _resolve_family_id_in_session(self, session, family_id: str) -> str:
        """沿 EntityRedirect 链解析到 canonical family_id。"""
        current_id = (family_id or "").strip()
        if not current_id:
            return ""
        seen: Set[str] = set()
        while current_id and current_id not in seen:
            seen.add(current_id)
            result = session.run(
                "MATCH (red:EntityRedirect {source_id: $sid}) RETURN red.target_id AS target",
                sid=current_id,
            )
            record = result.single()
            if not record or not record["target"] or record["target"] == current_id:
                break
            current_id = record["target"]
        return current_id


    def _tp_to_datetime(tp):
        """将 time_point 字符串转换为 Python datetime 对象，供 Neo4j DateTime 字段比较。
        Neo4j 存储的 valid_at/invalid_at 是 DateTime 类型，与字符串比较返回 null。
        必须传入 Python datetime 对象才能正确比较。
        """
        if tp is None:
            return None
        if isinstance(tp, datetime):
            return tp
        try:
            dt = datetime.fromisoformat(str(tp).replace("Z", "+00:00"))
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return dt
        except (ValueError, TypeError):
            return None



    def redirect_entity_relations(self, old_family_id: str, new_family_id: str) -> int:
        """Re-point all Relation edges referencing old_family_id's entities to new_family_id's latest entity.

        This is used during cross-window dedup to move relations from a duplicate entity
        to the canonical one before deleting the duplicate.

        Returns the number of relations updated.
        """
        old_family_id = (old_family_id or "").strip()
        new_family_id = (new_family_id or "").strip()
        if not old_family_id or not new_family_id:
            return 0

        with self._write_lock:
            with self._session() as session:
                # Get the latest entity absolute_id for the new family_id
                target_result = self._run(session,
                    "MATCH (e:Entity {family_id: $fid}) "
                    "RETURN e.uuid AS uuid ORDER BY e.processed_time DESC LIMIT 1",
                    fid=new_family_id,
                )
                target_record = target_result.single()
                if not target_record:
                    return 0
                new_abs_id = target_record["uuid"]

                # Get all absolute_ids for the old family_id
                old_abs_result = self._run(session,
                    "MATCH (e:Entity {family_id: $fid}) RETURN e.uuid AS uuid",
                    fid=old_family_id,
                )
                old_abs_ids = [r["uuid"] for r in old_abs_result]
                if not old_abs_ids:
                    return 0

                # Update entity1_absolute_id in matching relations
                upd1 = self._run(session,
                    "MATCH (r:Relation) WHERE r.entity1_absolute_id IN $old_ids "
                    "SET r.entity1_absolute_id = $new_id "
                    "RETURN count(r) AS cnt",
                    old_ids=old_abs_ids,
                    new_id=new_abs_id,
                )
                cnt1 = upd1.single()
                cnt1 = cnt1["cnt"] if cnt1 else 0

                # Update entity2_absolute_id in matching relations
                upd2 = self._run(session,
                    "MATCH (r:Relation) WHERE r.entity2_absolute_id IN $old_ids "
                    "SET r.entity2_absolute_id = $new_id "
                    "RETURN count(r) AS cnt",
                    old_ids=old_abs_ids,
                    new_id=new_abs_id,
                )
                cnt2 = upd2.single()
                cnt2 = cnt2["cnt"] if cnt2 else 0

                # Also fix RELATES_TO edges (graph traversal edges)
                # Delete old RELATES_TO edges and create new ones
                for old_abs in old_abs_ids:
                    # Outgoing: old → target  →  new → target
                    self._run(session,
                        "MATCH (old:Entity {uuid: $old_uuid})-[r:RELATES_TO]->(target) "
                        "WITH target, r "
                        "MATCH (new:Entity {uuid: $new_uuid}) "
                        "MERGE (new)-[:RELATES_TO]->(target) "
                        "DELETE r",
                        old_uuid=old_abs,
                        new_uuid=new_abs_id,
                    )
                    # Incoming: source → old  →  source → new
                    self._run(session,
                        "MATCH (source)-[r:RELATES_TO]->(old:Entity {uuid: $old_uuid}) "
                        "WITH source, r "
                        "MATCH (new:Entity {uuid: $new_uuid}) "
                        "MERGE (source)-[:RELATES_TO]->(new) "
                        "DELETE r",
                        old_uuid=old_abs,
                        new_uuid=new_abs_id,
                    )

                self._cache.invalidate("relation:")
                self._cache.invalidate("graph_stats")
                return cnt1 + cnt2

    # ------------------------------------------------------------------
    # Episode 操作（文件存储，与 StorageManager 相同逻辑）
    # ------------------------------------------------------------------



    def register_entity_redirect(self, source_family_id: str, target_family_id: str) -> str:
        """登记旧 family_id → canonical family_id 映射。"""
        source_id = (source_family_id or "").strip()
        target_id = (target_family_id or "").strip()
        if not source_id or not target_id:
            return target_id
        with self._write_lock:
            with self._session() as session:
                canonical_target = self._resolve_family_id_in_session(session, target_id)
                if not canonical_target:
                    canonical_target = target_id
                canonical_source = self._resolve_family_id_in_session(session, source_id)
                if canonical_source == canonical_target:
                    return canonical_target
                now_iso = datetime.now().isoformat()
                session.run(
                    """
                    MERGE (red:EntityRedirect {source_id: $sid})
                    SET red.target_id = $tid, red.updated_at = $now
                    """,
                    sid=source_id,
                    tid=canonical_target,
                    now=now_iso,
                )
            return canonical_target



    def resolve_family_id(self, family_id: str) -> str:
        """解析 family_id 到 canonical id。"""
        cache_key = f"resolve:{family_id}"
        cached = self._cache.get(cache_key)
        if cached is not None:
            return cached
        with _perf_timer("resolve_family_id"):
            with self._session() as session:
                resolved = self._resolve_family_id_in_session(session, family_id)
        self._cache.set(cache_key, resolved, ttl=120)
        return resolved



    def resolve_family_ids(self, family_ids: List[str]) -> Dict[str, str]:
        """批量解析 family_id 到 canonical id。利用缓存 + 一次 Cypher 查询未缓存项。

        Returns:
            {原始 family_id: canonical family_id} 映射
        """
        if not family_ids:
            return {}
        unique_ids = list(set(fid.strip() for fid in family_ids if fid and fid.strip()))
        if not unique_ids:
            return {}

        # 第一步：从缓存获取
        result: Dict[str, str] = {}
        uncached: List[str] = []
        for fid in unique_ids:
            cache_key = f"resolve:{fid}"
            cached = self._cache.get(cache_key)
            if cached is not None:
                result[fid] = cached
            else:
                uncached.append(fid)

        # 第二步：批量查询未缓存项
        if uncached:
            with _perf_timer("resolve_family_ids_batch"):
                with self._session() as session:
                    cypher = """
                    UNWIND $ids AS sid
                    MATCH (red:EntityRedirect {source_id: sid})
                    RETURN red.source_id AS source, red.target_id AS target
                    """
                    records = session.run(cypher, ids=uncached)
                    redirect_map = {r["source"]: r["target"] for r in records}

            # 解析第二跳（链式重定向）
            second_hop_sources = [t for t in redirect_map.values() if t not in result and t in uncached]
            if second_hop_sources:
                with self._session() as session:
                    cypher2 = """
                    UNWIND $ids AS sid
                    MATCH (red:EntityRedirect {source_id: sid})
                    RETURN red.source_id AS source, red.target_id AS target
                    """
                    records2 = session.run(cypher2, ids=second_hop_sources)
                    redirect_map.update({r["source"]: r["target"] for r in records2})

            for fid in uncached:
                target = redirect_map.get(fid, fid)
                # 沿链解析（通常只需 1 跳）
                final = redirect_map.get(target, target) if target != fid else fid
                result[fid] = final
                cache_key = f"resolve:{fid}"
                self._cache.set(cache_key, final, ttl=120)

        # 构建输出映射（处理可能有重复的 family_ids）
        output: Dict[str, str] = {}
        for fid in family_ids:
            key = fid.strip() if fid else ""
            output[fid] = result.get(key, key)
        return output

