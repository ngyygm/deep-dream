"""Neo4j DreamMixin — extracted from neo4j_store."""
import json as _json
import logging
import random as _random
import uuid as _uuid
from datetime import datetime
from types import SimpleNamespace
from typing import Any, Dict, List, Optional

from ...models import Entity, Episode, Relation
from ._helpers import _neo4j_record_to_relation, _q

logger = logging.getLogger(__name__)

_DREAM_TIERS = frozenset(("candidate", "verified", "rejected"))


def _new_record_id():
    """Generate a unique relation record ID."""
    return f"relation_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{_uuid.uuid4().hex[:8]}"


def _dream_source(cycle_id):
    """Build dream source_document tag."""
    return f"dream:{cycle_id}" if cycle_id else "dream"


class DreamMixin:
    """Dream operations for Neo4j backend.
    Shared state contract (set by Neo4jStorageManager.__init__):
        self._session()              → Neo4j session factory
        self._run(session, cypher, **kw) → execute Cypher with graph_id injection
        self._graph_id: str          → active graph ID
    """

    @staticmethod
    def _parse_dream_log_record(r) -> dict:
        """Convert a Neo4j DreamLog record to a dict (shared by get/list)."""
        # Short-circuit empty JSON fields ("[]" / None / "") — avoids 4x json.loads()
        _loads = _json.loads
        _raw_ins = r.get("insights")
        _raw_con = r.get("connections")
        _raw_cns = r.get("consolidations")
        _raw_epi = r.get("episode_ids")
        return {
            "cycle_id": r["cycle_id"],
            "graph_id": r["graph_id"],
            "start_time": str(r.get("start_time", "")),
            "end_time": str(r.get("end_time", "")),
            "status": r.get("status", ""),
            "narrative": r.get("narrative", ""),
            "insights": () if not _raw_ins or _raw_ins == "[]" else _loads(_raw_ins),
            "connections": () if not _raw_con or _raw_con == "[]" else _loads(_raw_con),
            "consolidations": () if not _raw_cns or _raw_cns == "[]" else _loads(_raw_cns),
            "strategy": r.get("strategy", ""),
            "entities_examined": r.get("entities_examined", 0),
            "relations_created": r.get("relations_created", 0),
            "episode_ids": () if not _raw_epi or _raw_epi == "[]" else _loads(_raw_epi),
        }


    def _dream_seeds_cross_community(self, count, exclude_uuids, community_id):
        """从不同社区各取 1 个随机实体，组合返回跨社区实体对。

        注意：cross_community 策略忽略 community_id 参数，因为它需要跨社区采样。
        """
        communities, _ = self.get_communities(limit=10, min_size=2)
        if len(communities) < 2:
            return self._dream_seeds_random(count, exclude_uuids, community_id)

        pairs = []
        for i in range(len(communities)):
            for j in range(i + 1, len(communities)):
                if len(pairs) >= count:
                    break
                c1_members = communities[i]["members"]
                c2_members = communities[j]["members"]
                # 从每个社区取 1 个非排除的随机成员
                c1_valid = [m for m in c1_members if m["uuid"] not in exclude_uuids]
                c2_valid = [m for m in c2_members if m["uuid"] not in exclude_uuids]
                if c1_valid and c2_valid:
                    e1 = _random.choice(c1_valid)
                    e2 = _random.choice(c2_valid)
                    pairs.extend([e1, e2])
            if len(pairs) >= count * 2:
                break

        return pairs[:count * 2]

    # ------------------------------------------------------------------
    # Dream candidate filtering (mirrors SQLite RelationStoreMixin)
    # ------------------------------------------------------------------



    def _dream_seeds_hub(self, count, exclude_uuids, community_id):
        with self._session() as session:
            # Use RELATES_TO edges only (not MENTIONS) to compute meaningful graph degree
            # Pre-limit candidates to avoid full-graph sort
            result = self._run(session, """
                MATCH (e:Entity)-[:RELATES_TO]-(other:Entity)
                WHERE e.invalid_at IS NULL
                  AND NOT e.uuid IN $exclude_uuids
                  AND ($cid IS NULL OR e.community_id = $cid)
                WITH e, count(DISTINCT other) AS degree
                ORDER BY degree DESC LIMIT $count
                RETURN e.uuid AS uuid, e.family_id AS family_id,
                       e.name AS name, e.content AS content,
                       e.confidence AS confidence, e.event_time AS event_time,
                       e.community_id AS community_id, degree
            """, exclude_uuids=list(exclude_uuids), cid=community_id, count=count)
            return [dict(r) for r in result]



    def _dream_seeds_low_confidence(self, count, exclude_uuids, community_id):
        with self._session() as session:
            result = self._run(session, """
                MATCH (e:Entity)
                WHERE e.invalid_at IS NULL
                  AND e.confidence IS NOT NULL AND e.confidence < 0.5
                  AND NOT e.uuid IN $exclude_uuids
                  AND ($cid IS NULL OR e.community_id = $cid)
                ORDER BY e.confidence ASC LIMIT $count
                RETURN e.uuid AS uuid, e.family_id AS family_id,
                       e.name AS name, e.content AS content,
                       e.confidence AS confidence, e.event_time AS event_time,
                       e.community_id AS community_id,
                       size([(e)-[:RELATES_TO]-() | 1]) AS degree
            """, exclude_uuids=list(exclude_uuids), cid=community_id, count=count)
            return [dict(r) for r in result]



    def _dream_seeds_orphan(self, count, exclude_uuids, community_id):
        with self._session() as session:
            result = self._run(session, """
                MATCH (e:Entity)
                WHERE e.invalid_at IS NULL
                  AND NOT (e)-[:RELATES_TO]-()
                  AND NOT e.uuid IN $exclude_uuids
                  AND ($cid IS NULL OR e.community_id = $cid)
                RETURN e.uuid AS uuid, e.family_id AS family_id,
                       e.name AS name, e.content AS content,
                       e.confidence AS confidence, e.event_time AS event_time,
                       e.community_id AS community_id, 0 AS degree
                LIMIT $count
            """, exclude_uuids=list(exclude_uuids), cid=community_id, count=count)
            return [dict(r) for r in result]



    def _dream_seeds_random(self, count, exclude_uuids, community_id):
        """Random seed selection — single query with ORDER BY rand()."""
        with self._session() as session:
            result = self._run(session, """
                MATCH (e:Entity)
                WHERE e.invalid_at IS NULL
                  AND NOT e.uuid IN $exclude_uuids
                  AND ($cid IS NULL OR e.community_id = $cid)
                WITH e, rand() AS r ORDER BY r
                LIMIT $count
                RETURN e.uuid AS uuid, e.family_id AS family_id,
                       e.name AS name, e.content AS content,
                       e.confidence AS confidence, e.event_time AS event_time,
                       e.community_id AS community_id,
                       size([(e)-[:RELATES_TO]-() | 1]) AS degree
            """, exclude_uuids=list(exclude_uuids), cid=community_id, count=count)
            return [dict(r) for r in result]



    def _dream_seeds_time_gap(self, count, exclude_uuids, community_id):
        with self._session() as session:
            result = self._run(session, """
                MATCH (e:Entity)
                WHERE e.invalid_at IS NULL
                  AND e.processed_time IS NOT NULL
                  AND NOT e.uuid IN $exclude_uuids
                  AND duration.between(e.processed_time, datetime()).days > 30
                  AND ($cid IS NULL OR e.community_id = $cid)
                ORDER BY e.processed_time ASC LIMIT $count
                RETURN e.uuid AS uuid, e.family_id AS family_id,
                       e.name AS name, e.content AS content,
                       e.confidence AS confidence, e.event_time AS event_time,
                       e.community_id AS community_id,
                       size([(e)-[:RELATES_TO]-() | 1]) AS degree
            """, exclude_uuids=list(exclude_uuids), cid=community_id, count=count)
            return [dict(r) for r in result]



    def corroborate_dream_relation(self, entity1_family_id: str, entity2_family_id: str,
                                    corroboration_source: str = "remember") -> Optional[Dict[str, Any]]:
        """当 remember 提取的关系与 dream 候选关系匹配时，自动增加佐证并可能提升。"""

        rels = self.get_relations_by_entities(entity1_family_id, entity2_family_id,
                                               include_candidates=True)
        if not rels:
            return None

        for rel in rels:
            try:
                attrs = _json.loads(rel.attributes) if rel.attributes else {}
            except (ValueError, TypeError):
                attrs = {}

            if (attrs.get("tier") == "candidate" and
                attrs.get("status") == "hypothesized" and
                rel.source_document and rel.source_document.startswith("dream")):

                count = attrs.get("corroboration_count", 0) + 1
                attrs["corroboration_count"] = count
                attrs.setdefault("corroboration_sources", []).append(corroboration_source)

                now = datetime.now()
                record_id = f"relation_{now.strftime('%Y%m%d_%H%M%S')}_{_uuid.uuid4().hex[:8]}"
                new_conf = min((rel.confidence or 0.5) + 0.1, 0.69)

                updated = Relation(
                    absolute_id=record_id,
                    family_id=rel.family_id,
                    entity1_absolute_id=rel.entity1_absolute_id,
                    entity2_absolute_id=rel.entity2_absolute_id,
                    content=rel.content,
                    event_time=now,
                    processed_time=now,
                    episode_id=rel.episode_id,
                    source_document=rel.source_document,
                    confidence=new_conf,
                    attributes=_json.dumps(attrs),
                )
                self.save_relation(updated)

                if count >= 2:
                    return self.promote_candidate_relation(
                        rel.family_id,
                        evidence_source=f"auto:{corroboration_source}",
                    )

                    return {
                        "family_id": rel.family_id,
                        "corroboration_count": count,
                        "status": "hypothesized",
                        "confidence": new_conf,
                        "message": f"佐证计数: {count}/2，需要2次佐证才能提升",
                    }

        return None

    def corroborate_dream_relations_batch(self, entity_pairs: List[tuple],
                                           corroboration_source: str = "remember") -> List[Dict[str, Any]]:
        """Batch version of corroborate_dream_relation — processes all pairs in fewer DB round-trips.

        Args:
            entity_pairs: List of (entity1_family_id, entity2_family_id) tuples.
        """
        if not entity_pairs:
            return []

        # Collect all candidate relations across all pairs in a single query
        pair_params = [{"e1": p[0], "e2": p[1]} for p in entity_pairs]
        with self._session() as session:
            rows = self._run(session, """
                UNWIND $pairs AS p
                MATCH (r:Relation)
                WHERE r.invalid_at IS NULL
                  AND r.source_document STARTS WITH 'dream'
                  AND r.attributes CONTAINS '"tier":"candidate"'
                  AND r.attributes CONTAINS '"status":"hypothesized"'
                WITH r, p
                MATCH (e1:Entity {uuid: r.entity1_absolute_id})
                MATCH (e2:Entity {uuid: r.entity2_absolute_id})
                WHERE (e1.family_id = p.e1 AND e2.family_id = p.e2)
                   OR (e1.family_id = p.e2 AND e2.family_id = p.e1)
                RETURN r.family_id AS fid, r.uuid AS uuid, r.entity1_absolute_id AS e1_abs,
                       r.entity2_absolute_id AS e2_abs, r.content AS content,
                       r.episode_id AS ep_id, r.source_document AS source,
                       r.confidence AS conf, r.attributes AS attrs
            """, pairs=pair_params)
            records = [dict(record) for record in rows]

        if not records:
            return []

        # Build all Relation objects first, then bulk save
        now = datetime.now()
        _ts_prefix = now.strftime('%Y%m%d_%H%M%S')
        pending_saves = []
        pending_promotes = []

        for rec in records:
            try:
                attrs = _json.loads(rec["attrs"]) if rec["attrs"] else {}
            except (ValueError, TypeError):
                attrs = {}

            count = attrs.get("corroboration_count", 0) + 1
            attrs["corroboration_count"] = count
            attrs.setdefault("corroboration_sources", []).append(corroboration_source)

            record_id = f"relation_{_ts_prefix}_{_uuid.uuid4().hex[:8]}"
            new_conf = min((rec["conf"] or 0.5) + 0.1, 0.69)

            updated = Relation(
                absolute_id=record_id,
                family_id=rec["fid"],
                entity1_absolute_id=rec["e1_abs"],
                entity2_absolute_id=rec["e2_abs"],
                content=rec["content"],
                event_time=now,
                processed_time=now,
                episode_id=rec["ep_id"],
                source_document=rec["source"],
                confidence=new_conf,
                attributes=_json.dumps(attrs),
            )
            pending_saves.append((updated, rec, count))

        # Bulk save all relations at once
        self.bulk_save_relations([r for r, _, _ in pending_saves])

        # Then handle promotions — batch promote all qualifying relations at once
        _to_promote = [(rec["fid"], updated.confidence) for updated, rec, count in pending_saves if count >= 2]
        if _to_promote:
            self.promote_candidate_relations_batch(
                [fid for fid, _ in _to_promote],
                evidence_source=f"auto:{corroboration_source}",
            )

        results = []
        for updated, rec, count in pending_saves:
            if count >= 2:
                results.append({"family_id": rec["fid"], "promoted": True})
            else:
                results.append({
                    "family_id": rec["fid"],
                    "corroboration_count": count,
                    "confidence": updated.confidence,
                })
        return results

    def count_candidate_relations(self, status: str = None) -> int:
        """统计候选层关系数量（Cypher COUNT 查询）。"""
        status_filter = " AND r.status = $status" if status else ""
        params = {}
        if status:
            params["status"] = status
        with self._session() as session:
            result = self._run(session, f"""
                MATCH (r:Relation) WHERE r.tier = 'candidate'{status_filter}
                RETURN count(DISTINCT r.family_id) AS cnt
            """, **params)
            return result.single()["cnt"]



    def demote_candidate_relation(self, family_id: str,
                                   reason: str = "") -> Dict[str, Any]:
        """将候选关系降级为已拒绝状态。"""

        resolved = self.resolve_family_id(family_id)
        if not resolved:
            raise ValueError(f"关系不存在: {family_id}")

        rel = self.get_relation_by_family_id(resolved)
        if not rel:
            raise ValueError(f"关系不存在: {family_id}")

        try:
            attrs = _json.loads(rel.attributes) if rel.attributes else {}
        except (ValueError, TypeError):
            attrs = {}

        old_status = attrs.get("status", "unknown")

        attrs["status"] = "rejected"
        attrs["rejected_reason"] = reason
        now = datetime.now()
        attrs["rejected_at"] = now.isoformat()

        record_id = f"relation_{now.strftime('%Y%m%d_%H%M%S')}_{_uuid.uuid4().hex[:8]}"

        relation = Relation(
            absolute_id=record_id,
            family_id=rel.family_id,
            entity1_absolute_id=rel.entity1_absolute_id,
            entity2_absolute_id=rel.entity2_absolute_id,
            content=rel.content,
            event_time=now,
            processed_time=now,
            episode_id=rel.episode_id,
            source_document=rel.source_document,
            confidence=min(rel.confidence or 0.3, 0.2),
            attributes=_json.dumps(attrs),
        )
        self.save_relation(relation)

        return {
            "family_id": resolved,
            "old_status": old_status,
            "new_status": "rejected",
            "confidence": relation.confidence,
        }


    # ------------------------------------------------------------------

    def get_candidate_relations(self, limit: int = 50, offset: int = 0,
                                 status: str = None) -> list:
        """获取候选层关系（包括已提升/已拒绝的 dream 关系）。"""

        with self._session() as session:
            # Query by source_document LIKE 'dream%' and attributes containing tier
            query_parts = ["""
                MATCH (r:Relation)
                WHERE r.source_document STARTS WITH 'dream'
                  AND r.invalid_at IS NULL
                  AND (r.attributes CONTAINS '"tier":"candidate"'
                    OR r.attributes CONTAINS '"tier":"verified"'
                    OR r.attributes CONTAINS '"tier":"rejected"'
                    OR r.attributes CONTAINS '"tier": "candidate"'
                    OR r.attributes CONTAINS '"tier": "verified"'
                    OR r.attributes CONTAINS '"tier": "rejected"')
            """]
            params = {}

            if status:
                query_parts.append(" AND (r.attributes CONTAINS $status_str OR r.attributes CONTAINS $status_str_spaced)")
                params["status_str"] = f'"status":"{status}"'
                params["status_str_spaced"] = f'"status": "{status}"'

            query_parts.append("""
                WITH r.family_id AS fid, COLLECT(r) AS rels
                UNWIND rels AS r
                WITH fid, r ORDER BY r.processed_time DESC
                WITH fid, HEAD(COLLECT(r)) AS r
                RETURN __REL_FIELDS__
                ORDER BY r.processed_time DESC
            """)

            query = _q("".join(query_parts))
            if offset > 0:
                query += f" SKIP {int(offset)}"
            query += f" LIMIT {int(limit)}"

            result = self._run(session, query, **params)
            return [_neo4j_record_to_relation(r) for r in result]



    def get_dream_log(self, cycle_id: str) -> Optional[dict]:
        """获取单条梦境日志。"""
        with self._session() as session:
            result = self._run(session, """
                MATCH (d:DreamLog {cycle_id: $cycle_id})
                RETURN d.cycle_id AS cycle_id, d.graph_id AS graph_id,
                       d.start_time AS start_time, d.end_time AS end_time,
                       d.status AS status, d.narrative AS narrative,
                       d.insights AS insights, d.connections AS connections,
                       d.consolidations AS consolidations,
                       d.strategy AS strategy, d.entities_examined AS entities_examined,
                       d.relations_created AS relations_created,
                       d.episode_ids AS episode_ids
            """, cycle_id=cycle_id)
            r = result.single()
            if not r:
                return None
            return self._parse_dream_log_record(r)

    # =========================================================
    # DeepDream Agent API — 存储层
    # =========================================================



    def get_dream_seeds(self, strategy: str = "random", count: int = 10,
                        exclude_ids: Optional[List[str]] = None,
                        community_id: Optional[int] = None) -> List[Dict[str, Any]]:
        """按策略选取梦境种子实体。

        strategy: random | orphan | hub | time_gap | cross_community | low_confidence
        """

        exclude_uuids = set()
        if exclude_ids:
            resolved_map = self.resolve_family_ids(exclude_ids)
            canonical_fids = list({v for v in resolved_map.values() if v})
            if canonical_fids:
                aids_map = self.get_latest_absolute_ids_by_family_ids(canonical_fids)
                exclude_uuids = set(aids_map.values())

        strategies = {
            "random": self._dream_seeds_random,
            "orphan": self._dream_seeds_orphan,
            "hub": self._dream_seeds_hub,
            "time_gap": self._dream_seeds_time_gap,
            "low_confidence": self._dream_seeds_low_confidence,
            "cross_community": self._dream_seeds_cross_community,
        }

        handler = strategies.get(strategy)
        if not handler:
            raise ValueError(f"未知的种子策略: {strategy}，可选: {', '.join(strategies.keys())}")

        seeds = handler(count, exclude_uuids, community_id)

        # 添加 reason 字段
        reason_map = {
            "random": "随机选取",
            "orphan": "孤立实体：无任何关系连接",
            "hub": "高连接度实体",
            "time_gap": "长时间未更新的实体",
            "low_confidence": "低置信度实体",
            "cross_community": "跨社区桥接候选",
        }
        for s in seeds:
            s["reason"] = reason_map.get(strategy, "")

        return seeds



    def list_dream_logs(self, graph_id: str = "default", limit: int = 20) -> List[dict]:
        """列出梦境日志。"""
        with self._session() as session:
            result = self._run(session, """
                MATCH (d:DreamLog)
                WHERE d.graph_id = $graph_id
                RETURN d.cycle_id AS cycle_id, d.graph_id AS graph_id,
                       d.start_time AS start_time, d.end_time AS end_time,
                       d.status AS status, d.narrative AS narrative,
                       d.insights AS insights, d.connections AS connections,
                       d.consolidations AS consolidations,
                       d.strategy AS strategy, d.entities_examined AS entities_examined,
                       d.relations_created AS relations_created,
                       d.episode_ids AS episode_ids
                ORDER BY d.start_time DESC
                LIMIT $limit
            """, graph_id=graph_id, limit=limit)
            logs = []
            for r in result:
                logs.append(self._parse_dream_log_record(r))
            return logs



    def promote_candidate_relation(self, family_id: str,
                                    evidence_source: str = "manual",
                                    new_confidence: float = None) -> Dict[str, Any]:
        """将候选关系提升为已验证状态。"""

        resolved = self.resolve_family_id(family_id)
        if not resolved:
            raise ValueError(f"关系不存在: {family_id}")

        rel = self.get_relation_by_family_id(resolved)
        if not rel:
            raise ValueError(f"关系不存在: {family_id}")

        try:
            attrs = _json.loads(rel.attributes) if rel.attributes else {}
        except (ValueError, TypeError):
            attrs = {}

        old_status = attrs.get("status", "unknown")
        old_tier = attrs.get("tier", "unknown")

        attrs["tier"] = "verified"
        attrs["status"] = "verified"
        attrs["promoted_by"] = evidence_source
        now = datetime.now()
        attrs["promoted_at"] = now.isoformat()
        attrs["corroboration_count"] = attrs.get("corroboration_count", 0) + 1

        record_id = f"relation_{now.strftime('%Y%m%d_%H%M%S')}_{_uuid.uuid4().hex[:8]}"
        new_conf = new_confidence if new_confidence is not None else max(rel.confidence or 0.5, 0.7)

        relation = Relation(
            absolute_id=record_id,
            family_id=rel.family_id,
            entity1_absolute_id=rel.entity1_absolute_id,
            entity2_absolute_id=rel.entity2_absolute_id,
            content=rel.content,
            event_time=now,
            processed_time=now,
            episode_id=rel.episode_id,
            source_document=rel.source_document,
            confidence=new_conf,
            attributes=_json.dumps(attrs),
        )
        self.save_relation(relation)

        return {
            "family_id": resolved,
            "old_status": old_status,
            "old_tier": old_tier,
            "new_status": "verified",
            "new_tier": "verified",
            "confidence": new_conf,
        }

    def promote_candidate_relations_batch(self, family_ids: List[str],
                                           evidence_source: str = "manual",
                                           new_confidence: float = None) -> List[Dict[str, Any]]:
        """Batch promote multiple candidate relations to verified status in a single session."""
        if not family_ids:
            return []
        now = datetime.now()
        now_str = now.isoformat()
        _ts_prefix = now.strftime('%Y%m%d_%H%M%S')

        resolved_map = self.resolve_family_ids(family_ids)
        canonical_fids = list({r for r in resolved_map.values() if r})
        if not canonical_fids:
            return []

        # Single session: fetch latest versions + create promoted versions
        with self._session() as session:
            # Fetch latest versions of all relations
            result = self._run(session, """
                UNWIND $fids AS fid
                MATCH (r:Relation {family_id: fid})
                WHERE r.invalid_at IS NULL
                WITH fid, r ORDER BY r.processed_time DESC
                WITH fid, collect(r)[0] AS latest
                RETURN fid, latest.uuid AS uuid, latest.family_id AS family_id,
                       latest.entity1_absolute_id AS e1, latest.entity2_absolute_id AS e2,
                       latest.content AS content, latest.episode_id AS ep_id,
                       latest.source_document AS source, latest.confidence AS conf,
                       latest.attributes AS attrs
            """, fids=canonical_fids)

            records = list(result)
            if not records:
                return []

        # Build and bulk save promoted relations
        promoted = []
        for record in records:
            try:
                attrs = _json.loads(record["attrs"]) if record["attrs"] else {}
            except (ValueError, TypeError):
                attrs = {}

            attrs["tier"] = "verified"
            attrs["status"] = "verified"
            attrs["promoted_by"] = evidence_source
            attrs["promoted_at"] = now_str
            attrs["corroboration_count"] = attrs.get("corroboration_count", 0) + 1

            record_id = f"relation_{_ts_prefix}_{_uuid.uuid4().hex[:8]}"
            new_conf = new_confidence if new_confidence is not None else max(record["conf"] or 0.5, 0.7)

            relation = Relation(
                absolute_id=record_id,
                family_id=record["family_id"],
                entity1_absolute_id=record["e1"],
                entity2_absolute_id=record["e2"],
                content=record["content"],
                event_time=now,
                processed_time=now,
                episode_id=record["ep_id"],
                source_document=record["source"],
                confidence=new_conf,
                attributes=_json.dumps(attrs),
            )
            promoted.append(relation)

        if promoted:
            self.bulk_save_relations(promoted)

        return [{"family_id": r.family_id, "promoted": True} for r in promoted]



    def reject_dream_cycle_relations(self, dream_cycle_id: str) -> Dict[str, Any]:
        """批量拒绝指定 Dream 周期产生的所有未验证候选关系（单次 Cypher 批量更新）。"""
        source_doc = _dream_source(dream_cycle_id)
        reason = f"批量拒绝: dream cycle {dream_cycle_id}"
        now_str = datetime.now().isoformat()

        with self._session() as session:
            # First fetch matching relations to update their attributes JSON in Python
            result = self._run(session, """
                MATCH (r:Relation {source_document: $src})
                WHERE r.invalid_at IS NULL
                  AND r.attributes CONTAINS '"tier":"candidate"'
                  AND r.attributes CONTAINS '"status":"hypothesized"'
                WITH r.family_id AS fid, COLLECT(r) AS rels
                WITH fid, rels[0] AS latest
                RETURN latest.uuid AS uuid, latest.family_id AS fid,
                       latest.attributes AS attrs, latest.confidence AS conf
            """, src=source_doc)
            records = [dict(r) for r in result]

        # Update each relation's attributes and confidence
        rejected = 0
        for rec in records:
            try:
                attrs = _json.loads(rec["attrs"]) if rec["attrs"] else {}
            except (ValueError, TypeError):
                attrs = {}
            attrs["status"] = "rejected"
            attrs["rejected_reason"] = reason
            attrs["rejected_at"] = now_str
            new_conf = min(rec.get("conf") or 0.5, 0.2)
            with self._session() as session:
                self._run(session, """
                    MATCH (r:Relation {uuid: $uuid})
                    SET r.attributes = $attrs, r.confidence = $conf
                """, uuid=rec["uuid"], attrs=_json.dumps(attrs, ensure_ascii=False), conf=new_conf)
            rejected += 1
            rejected = result.single()["rejected"]

        return {
            "rejected_count": rejected,
            "cycle_id": dream_cycle_id,
        }



    def save_dream_episode(self, content: str,
                           entities_examined: Optional[List[str]] = None,
                           relations_created: Optional[List[Dict]] = None,
                           strategy_used: str = "",
                           dream_cycle_id: Optional[str] = None,
                           **kwargs) -> Dict[str, Any]:
        """保存梦境 episode，同时创建 DreamLog 节点以便 dream_status/dream_logs 查询。

        Returns: {"episode_id": "...", "episode_type": "dream", "cycle_id": "..."}
        """
        now = datetime.now()
        episode_id = f"episode_dream_{now.strftime('%Y%m%d_%H%M%S')}_{_uuid.uuid4().hex[:8]}"

        # 自动生成 cycle_id（如果未提供）
        if not dream_cycle_id:
            dream_cycle_id = f"dream_{now.strftime('%Y%m%d_%H%M%S')}_{_uuid.uuid4().hex[:6]}"

        # 构建结构化内容
        _explicit_rel_count = kwargs.get("relations_created_count")
        _explicit_ent_count = kwargs.get("entities_examined_count")
        ent_count = _explicit_ent_count if _explicit_ent_count is not None else (len(entities_examined) if entities_examined else 0)
        rel_count = _explicit_rel_count if _explicit_rel_count is not None else (len(relations_created) if relations_created else 0)
        structured = {
            "narrative": content,
            "strategy": strategy_used,
            "entities_examined_count": ent_count,
            "relations_created_count": rel_count,
        }
        if relations_created:
            structured["relations_created"] = relations_created

        full_content = content
        if rel_count > 0 or ent_count > 0:
            full_content += "\n\n---\n" + _json.dumps(structured, ensure_ascii=False, indent=2)

        cache = Episode(
            absolute_id=episode_id,
            content=full_content,
            event_time=now,
            source_document=_dream_source(dream_cycle_id),
            episode_type="dream",
        )

        self.save_episode(cache)

        # 记录提及的实体（批量 resolve + 批量获取）
        if entities_examined:
            abs_ids = []
            try:
                resolved_map = self.resolve_family_ids(entities_examined)
                canonical_fids = list({r for r in resolved_map.values() if r})
                if canonical_fids:
                    entities_map = self.get_entities_by_family_ids(canonical_fids)
                    for eid in entities_examined:
                        resolved = resolved_map.get(eid)
                        if resolved:
                            entity = entities_map.get(resolved)
                            if entity and entity.absolute_id:
                                abs_ids.append(entity.absolute_id)
            except Exception:
                # Fallback to individual resolution
                for eid in entities_examined:
                    resolved = self.resolve_family_id(eid)
                    if resolved:
                        entity = self.get_entity_by_family_id(resolved)
                        if entity:
                            abs_ids.append(entity.absolute_id)
            if abs_ids:
                self.save_episode_mentions(episode_id, abs_ids, context=_dream_source(strategy_used))

        # 同时创建 DreamLog 节点，以便 dream_status/dream_logs 可以查询
        report = SimpleNamespace(
            cycle_id=dream_cycle_id,
            graph_id=self._graph_id,
            start_time=now,
            end_time=now,
            status="completed",
            narrative=content[:2000],
            insights=[],
            new_connections=relations_created or [],
            consolidations=[],
            strategy=strategy_used,
            entities_examined=ent_count,
            relations_created=rel_count,
            episode_ids=[episode_id],
        )
        self.save_dream_log(report)

        return {
            "episode_id": episode_id,
            "episode_type": "dream",
            "cycle_id": dream_cycle_id,
        }

    # ========== Version-Level CRUD (Phase 2) ==========



    def save_dream_log(self, report):
        """保存梦境日志。"""
        with self._session() as session:
            self._run(session, """
                MERGE (d:DreamLog {cycle_id: $cycle_id})
                SET d.graph_id = $graph_id,
                    d.start_time = datetime($start_time),
                    d.end_time = datetime($end_time),
                    d.status = $status,
                    d.narrative = $narrative,
                    d.insights = $insights,
                    d.connections = $connections,
                    d.consolidations = $consolidations,
                    d.strategy = $strategy,
                    d.entities_examined = $entities_examined,
                    d.relations_created = $relations_created,
                    d.episode_ids = $episode_ids
            """,
                cycle_id=report.cycle_id,
                graph_id=report.graph_id,
                start_time=report.start_time.isoformat(),
                end_time=(report.end_time or datetime.now()).isoformat(),
                status=report.status,
                narrative=report.narrative,
                insights=_json.dumps(report.insights, ensure_ascii=False),
                connections=_json.dumps(report.new_connections, ensure_ascii=False),
                consolidations=_json.dumps(report.consolidations, ensure_ascii=False),
                strategy=getattr(report, 'strategy', ''),
                entities_examined=getattr(report, 'entities_examined', 0),
                relations_created=getattr(report, 'relations_created', 0),
                episode_ids=_json.dumps(getattr(report, 'episode_ids', []), ensure_ascii=False),
            )

