"""
DreamStoreMixin — Dream and Butler methods for SQLite backend.

Provides:
  - get_isolated_entities / count_isolated_entities
  - cleanup_invalidated_versions
  - get_dream_seeds (random, orphan, hub, time_gap, low_confidence)
  - save_dream_relation
  - save_dream_episode

Relies on host-class state:
    self._get_conn()
    self._write_lock
    self._ENTITY_SELECT
    self._row_to_entity()
    self.resolve_family_id() / resolve_family_ids()
    self.get_entity_by_family_id()
    self.get_entities_by_family_ids()
    self.get_relations_by_entities()
    self.save_relation()
    self.save_episode()
    self.save_episode_mentions()
    self.save_dream_log()
"""
import json
import logging
import uuid
from datetime import datetime
from types import SimpleNamespace
from typing import Any, Dict, List, Optional

from ...models import Relation, Episode

logger = logging.getLogger(__name__)


class DreamStoreMixin:
    """Mixin providing Dream/Butler operations for SQLite StorageManager."""

    # ------------------------------------------------------------------
    # Isolated entities
    # ------------------------------------------------------------------

    def get_isolated_entities(self, limit: int = 100, offset: int = 0) -> List:
        """获取所有孤立实体（有效实体中不被任何有效 Relation 引用的）。"""
        conn = self._get_conn()
        cursor = conn.cursor()

        # Get all absolute_ids referenced by valid relations
        cursor.execute("""
            SELECT DISTINCT entity1_absolute_id FROM relations WHERE invalid_at IS NULL
            UNION
            SELECT DISTINCT entity2_absolute_id FROM relations WHERE invalid_at IS NULL
        """)
        connected = {row[0] for row in cursor.fetchall()}

        # Get latest version of each family_id
        cursor.execute(f"""
            SELECT {self._ENTITY_SELECT}
            FROM (
                SELECT *, ROW_NUMBER() OVER (PARTITION BY family_id ORDER BY processed_time DESC) AS rn
                FROM entities
                WHERE invalid_at IS NULL
            )
            WHERE rn = 1
            ORDER BY processed_time DESC
        """)

        isolated = []
        for row in cursor.fetchall():
            entity = self._row_to_entity(row)
            # An entity is isolated if none of its versions are connected
            cursor.execute(
                "SELECT id FROM entities WHERE family_id = ?",
                (entity.family_id,),
            )
            family_abs_ids = {r[0] for r in cursor.fetchall()}
            if not family_abs_ids & connected:
                isolated.append(entity)

        return isolated[offset:offset + limit]

    def count_isolated_entities(self) -> int:
        """统计孤立实体数量。"""
        conn = self._get_conn()
        cursor = conn.cursor()

        # Get all connected absolute_ids
        cursor.execute("""
            SELECT DISTINCT entity1_absolute_id FROM relations WHERE invalid_at IS NULL
            UNION
            SELECT DISTINCT entity2_absolute_id FROM relations WHERE invalid_at IS NULL
        """)
        connected = {row[0] for row in cursor.fetchall()}

        # Get distinct family_ids with their absolute_ids
        cursor.execute("""
            SELECT family_id, GROUP_CONCAT(id) AS abs_ids
            FROM entities
            WHERE invalid_at IS NULL
            GROUP BY family_id
        """)

        count = 0
        for row in cursor.fetchall():
            family_id = row[0]
            abs_ids_str = row[1] or ""
            abs_ids = set(abs_ids_str.split(","))
            if not abs_ids & connected:
                count += 1

        return count

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------

    def cleanup_invalidated_versions(self, before_date: str = None,
                                      dry_run: bool = False) -> Dict[str, Any]:
        """清理已失效的旧版本节点。"""
        date_filter = ""
        params: list = []
        if before_date:
            date_filter = " AND invalid_at < ?"
            params.append(before_date)

        conn = self._get_conn()
        cursor = conn.cursor()

        # Count
        cursor.execute(
            f"SELECT COUNT(*) FROM entities WHERE invalid_at IS NOT NULL{date_filter}",
            params,
        )
        entity_count = cursor.fetchone()[0]

        cursor.execute(
            f"SELECT COUNT(*) FROM relations WHERE invalid_at IS NOT NULL{date_filter}",
            params,
        )
        relation_count = cursor.fetchone()[0]

        if dry_run:
            return {
                "dry_run": True,
                "entities_to_remove": entity_count,
                "relations_to_remove": relation_count,
                "message": f"预览：将删除 {entity_count} 个已失效实体版本和 {relation_count} 个已失效关系版本",
            }

        # Actually delete
        with self._write_lock:
            cursor.execute(
                f"DELETE FROM entities WHERE invalid_at IS NOT NULL{date_filter}",
                params,
            )
            deleted_entities = cursor.rowcount

            cursor.execute(
                f"DELETE FROM relations WHERE invalid_at IS NOT NULL{date_filter}",
                params,
            )
            deleted_relations = cursor.rowcount

            # Also cleanup FTS
            try:
                cursor.execute(
                    "DELETE FROM entity_fts WHERE rowid NOT IN (SELECT rowid FROM entities)"
                )
            except Exception:
                pass
            try:
                cursor.execute(
                    "DELETE FROM relation_fts WHERE rowid NOT IN (SELECT rowid FROM relations)"
                )
            except Exception:
                pass

            conn.commit()

        return {
            "dry_run": False,
            "deleted_entity_versions": deleted_entities,
            "deleted_relation_versions": deleted_relations,
            "message": f"已删除 {deleted_entities} 个已失效实体版本和 {deleted_relations} 个已失效关系版本",
        }

    # ------------------------------------------------------------------
    # Dream seeds
    # ------------------------------------------------------------------

    def get_dream_seeds(self, strategy: str = "random", count: int = 10,
                        exclude_ids: Optional[List[str]] = None,
                        community_id: Optional[int] = None) -> List[Dict[str, Any]]:
        """按策略选取梦境种子实体。

        strategy: random | orphan | hub | time_gap | low_confidence
        """
        strategies = {
            "random": self._dream_seeds_random,
            "orphan": self._dream_seeds_orphan,
            "hub": self._dream_seeds_hub,
            "time_gap": self._dream_seeds_time_gap,
            "low_confidence": self._dream_seeds_low_confidence,
        }

        handler = strategies.get(strategy)
        if not handler:
            raise ValueError(f"未知的种子策略: {strategy}，可选: {', '.join(strategies.keys())}")

        seeds = handler(count, exclude_ids or [])

        reason_map = {
            "random": "随机选取",
            "orphan": "孤立实体：无任何关系连接",
            "hub": "高连接度实体",
            "time_gap": "长时间未更新的实体",
            "low_confidence": "低置信度实体",
        }
        for s in seeds:
            s["reason"] = reason_map.get(strategy, "")

        return seeds

    def _dream_seed_base_query(self, cursor, where_extra: str = "",
                                params: list = None, order_by: str = "",
                                limit: int = 10):
        """Common query: get latest version of each family_id with optional filters."""
        params = params or []
        order_clause = order_by if order_by else "e.processed_time DESC"
        where_clause = ""
        if where_extra:
            where_clause = f"AND {where_extra}"

        ent_cols = ", e.".join(self._ENTITY_SELECT.split(", "))
        cursor.execute(f"""
            SELECT e.{ent_cols}, sub.degree
            FROM (
                SELECT family_id, MAX(processed_time) as max_pt
                FROM entities
                WHERE invalid_at IS NULL
                {where_clause}
                GROUP BY family_id
            ) latest
            JOIN entities e ON e.family_id = latest.family_id AND e.processed_time = latest.max_pt
            LEFT JOIN (
                SELECT family_id, COUNT(*) as degree
                FROM (
                    SELECT family_id FROM entities
                ) GROUP BY family_id
            ) sub ON sub.family_id = e.family_id
            ORDER BY {order_clause}
            LIMIT ?
        """, params + [limit])
        return cursor.fetchall()

    def _dream_seeds_random(self, count: int, exclude_ids: list) -> List[Dict[str, Any]]:
        conn = self._get_conn()
        cursor = conn.cursor()

        # Get latest version per family_id, excluding specified ids
        exclude_set = set(exclude_ids)
        ent_cols = ", e.".join(self._ENTITY_SELECT.split(", "))
        cursor.execute(f"""
            SELECT e.{ent_cols}
            FROM (
                SELECT family_id, MAX(processed_time) as max_pt
                FROM entities
                WHERE invalid_at IS NULL
                GROUP BY family_id
            ) latest
            JOIN entities e ON e.family_id = latest.family_id AND e.processed_time = latest.max_pt
            ORDER BY RANDOM()
            LIMIT ?
        """, [count])

        seeds = []
        for row in cursor.fetchall():
            entity = self._row_to_entity(row)
            if entity.family_id in exclude_set:
                continue
            seeds.append({
                "family_id": entity.family_id,
                "name": entity.name,
                "content": entity.content[:200],
                "confidence": entity.confidence,
                "event_time": entity.event_time.isoformat() if entity.event_time else None,
                "degree": 0,
            })
        return seeds

    def _dream_seeds_orphan(self, count: int, exclude_ids: list) -> List[Dict[str, Any]]:
        """Get entities with no relations."""
        isolated = self.get_isolated_entities(limit=count)
        return [
            {
                "family_id": e.family_id,
                "name": e.name,
                "content": e.content[:200],
                "confidence": e.confidence,
                "event_time": e.event_time.isoformat() if e.event_time else None,
                "degree": 0,
            }
            for e in isolated
            if e.family_id not in set(exclude_ids)
        ]

    def _dream_seeds_hub(self, count: int, exclude_ids: list) -> List[Dict[str, Any]]:
        """Get entities with highest relation degree."""
        conn = self._get_conn()
        cursor = conn.cursor()

        # Count relations per family_id using absolute_id joins
        cursor.execute("""
            SELECT e.family_id, COUNT(DISTINCT r.id) as degree
            FROM entities e
            JOIN relations r ON (
                (r.entity1_absolute_id = e.id OR r.entity2_absolute_id = e.id)
                AND r.invalid_at IS NULL
            )
            WHERE e.invalid_at IS NULL
            GROUP BY e.family_id
            ORDER BY degree DESC
            LIMIT ?
        """, [count])

        seeds = []
        for row in cursor.fetchall():
            fid = row[0]
            degree = row[1]
            if fid in set(exclude_ids):
                continue
            entity = self.get_entity_by_family_id(fid)
            if entity:
                seeds.append({
                    "family_id": entity.family_id,
                    "name": entity.name,
                    "content": entity.content[:200],
                    "confidence": entity.confidence,
                    "event_time": entity.event_time.isoformat() if entity.event_time else None,
                    "degree": degree,
                })
        return seeds

    def _dream_seeds_time_gap(self, count: int, exclude_ids: list) -> List[Dict[str, Any]]:
        """Get entities not updated in 30+ days."""
        conn = self._get_conn()
        cursor = conn.cursor()

        ent_cols = ", e.".join(self._ENTITY_SELECT.split(", "))
        cursor.execute(f"""
            SELECT e.{ent_cols}
            FROM (
                SELECT family_id, MAX(processed_time) as max_pt
                FROM entities
                WHERE invalid_at IS NULL
                GROUP BY family_id
            ) latest
            JOIN entities e ON e.family_id = latest.family_id AND e.processed_time = latest.max_pt
            WHERE julianday('now') - julianday(e.processed_time) > 30
            ORDER BY e.processed_time ASC
            LIMIT ?
        """, [count])

        seeds = []
        for row in cursor.fetchall():
            entity = self._row_to_entity(row)
            if entity.family_id in set(exclude_ids):
                continue
            seeds.append({
                "family_id": entity.family_id,
                "name": entity.name,
                "content": entity.content[:200],
                "confidence": entity.confidence,
                "event_time": entity.event_time.isoformat() if entity.event_time else None,
                "degree": 0,
            })
        return seeds

    def _dream_seeds_low_confidence(self, count: int, exclude_ids: list) -> List[Dict[str, Any]]:
        """Get entities with lowest confidence."""
        conn = self._get_conn()
        cursor = conn.cursor()

        ent_cols = ", e.".join(self._ENTITY_SELECT.split(", "))
        cursor.execute(f"""
            SELECT e.{ent_cols}
            FROM (
                SELECT family_id, MAX(processed_time) as max_pt
                FROM entities
                WHERE invalid_at IS NULL
                GROUP BY family_id
            ) latest
            JOIN entities e ON e.family_id = latest.family_id AND e.processed_time = latest.max_pt
            WHERE e.confidence IS NOT NULL AND e.confidence < 0.5
            ORDER BY e.confidence ASC
            LIMIT ?
        """, [count])

        seeds = []
        for row in cursor.fetchall():
            entity = self._row_to_entity(row)
            if entity.family_id in set(exclude_ids):
                continue
            seeds.append({
                "family_id": entity.family_id,
                "name": entity.name,
                "content": entity.content[:200],
                "confidence": entity.confidence,
                "event_time": entity.event_time.isoformat() if entity.event_time else None,
                "degree": 0,
            })
        return seeds

    # ------------------------------------------------------------------
    # Dream relation
    # ------------------------------------------------------------------

    def save_dream_relation(self, entity1_id: str, entity2_id: str,
                            content: str, confidence: float, reasoning: str,
                            dream_cycle_id: Optional[str] = None,
                            episode_id: Optional[str] = None) -> Dict[str, Any]:
        """创建或合并梦境发现的关系。

        Returns: {"family_id": "...", "entity1_family_id": "...", "entity2_family_id": "...", "action": "created"|"merged"}
        Raises: ValueError if entities don't exist
        """
        # Resolve entities
        resolved1 = self.resolve_family_id(entity1_id)
        resolved2 = self.resolve_family_id(entity2_id)
        if not resolved1:
            raise ValueError(f"实体不存在: {entity1_id}")
        if not resolved2:
            raise ValueError(f"实体不存在: {entity2_id}")

        entity1 = self.get_entity_by_family_id(resolved1)
        entity2 = self.get_entity_by_family_id(resolved2)
        if not entity1:
            raise ValueError(f"实体不存在: {entity1_id}")
        if not entity2:
            raise ValueError(f"实体不存在: {entity2_id}")

        # Check existing relation
        existing = self.get_relations_by_entities(resolved1, resolved2)
        if existing:
            latest = existing[0]
            new_confidence = max(latest.confidence or 0, confidence)
            new_prov_entry = {
                "source": "dream",
                "dream_cycle_id": dream_cycle_id,
                "confidence": confidence,
                "reasoning": reasoning,
            }
            try:
                old_prov = json.loads(latest.provenance) if hasattr(latest, 'provenance') and latest.provenance else []
            except Exception:
                old_prov = []
            old_prov.append(new_prov_entry)

            now = datetime.now()
            record_id = f"relation_{now.strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
            source_doc = f"dream:{dream_cycle_id}" if dream_cycle_id else "dream"
            merged_content = f"{latest.content}\n[Dream update] {content}" if content != latest.content else latest.content

            # Preserve existing attributes (tier, status, corroboration state)
            try:
                merged_attrs = json.loads(latest.attributes) if latest.attributes else {}
            except (json.JSONDecodeError, TypeError):
                merged_attrs = {}
            # Track additional dream cycle
            if dream_cycle_id:
                merged_attrs.setdefault("additional_dream_cycles", [])
                merged_attrs["additional_dream_cycles"].append(dream_cycle_id)

            relation = Relation(
                absolute_id=record_id,
                family_id=latest.family_id,
                entity1_absolute_id=latest.entity1_absolute_id,
                entity2_absolute_id=latest.entity2_absolute_id,
                content=merged_content,
                event_time=now,
                processed_time=now,
                episode_id=episode_id or latest.episode_id or "",
                source_document=source_doc,
                confidence=new_confidence,
                attributes=json.dumps(merged_attrs) if merged_attrs else None,
            )
            self.save_relation(relation)
            return {
                "family_id": latest.family_id,
                "entity1_family_id": resolved1,
                "entity2_family_id": resolved2,
                "entity1_name": entity1.name,
                "entity2_name": entity2.name,
                "action": "merged",
            }

        # Create new relation
        if entity1.name <= entity2.name:
            e1_abs, e2_abs = entity1.absolute_id, entity2.absolute_id
        else:
            e1_abs, e2_abs = entity2.absolute_id, entity1.absolute_id

        now = datetime.now()
        family_id = f"rel_{uuid.uuid4().hex[:12]}"
        record_id = f"relation_{now.strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
        source_doc = f"dream:{dream_cycle_id}" if dream_cycle_id else "dream"

        relation = Relation(
            absolute_id=record_id,
            family_id=family_id,
            entity1_absolute_id=e1_abs,
            entity2_absolute_id=e2_abs,
            content=content,
            event_time=now,
            processed_time=now,
            episode_id=episode_id or "",
            source_document=source_doc,
            confidence=min(confidence, 0.5),
            attributes=json.dumps({
                "tier": "candidate",
                "status": "hypothesized",
                "corroboration_count": 0,
                "created_by_dream": dream_cycle_id or "unknown",
                "created_at": now.isoformat(),
            }),
        )

        self.save_relation(relation)

        return {
            "family_id": family_id,
            "entity1_family_id": resolved1,
            "entity2_family_id": resolved2,
            "entity1_name": entity1.name,
            "entity2_name": entity2.name,
            "action": "created",
        }

    # ------------------------------------------------------------------
    # Dream Candidate Layer — promotion / demotion / listing
    # ------------------------------------------------------------------

    def get_candidate_relations(self, limit: int = 50, offset: int = 0,
                                 status: str = None) -> List[Relation]:
        """获取候选层关系（包括已提升/已拒绝的 dream 关系）。

        Args:
            limit: 最大返回数
            offset: 分页偏移
            status: 可选过滤 "hypothesized" | "verified" | "rejected"
        """
        conn = self._get_conn()
        cursor = conn.cursor()

        status_filter = ""
        params: list = []
        if status:
            status_filter = " AND json_extract(attributes, '$.status') = ?"
            params.append(status)

        # Match all dream candidate layer relations (tier may have changed via promote/demote)
        tier_clause = "AND json_extract(attributes, '$.tier') IN ('candidate', 'verified', 'rejected')"

        cursor.execute(f"""
            SELECT {self._RELATION_SELECT}
            FROM (
                SELECT *, ROW_NUMBER() OVER (PARTITION BY family_id ORDER BY processed_time DESC) AS rn
                FROM relations
                WHERE invalid_at IS NULL
                  AND source_document LIKE 'dream%%'
                  {tier_clause}
                  {status_filter}
            )
            WHERE rn = 1
            ORDER BY processed_time DESC
            LIMIT ? OFFSET ?
        """, params + [limit, offset])

        return [self._row_to_relation(row) for row in cursor.fetchall()]

    def count_candidate_relations(self, status: str = None) -> int:
        """统计候选层关系数量（包括已提升/已拒绝的 dream 关系）。"""
        conn = self._get_conn()
        cursor = conn.cursor()

        status_filter = ""
        params: list = []
        if status:
            status_filter = " AND json_extract(attributes, '$.status') = ?"
            params.append(status)

        # Match all dream candidate layer relations (tier may have changed via promote/demote)
        tier_clause = "AND json_extract(attributes, '$.tier') IN ('candidate', 'verified', 'rejected')"

        cursor.execute(f"""
            SELECT COUNT(DISTINCT family_id)
            FROM relations
            WHERE invalid_at IS NULL
              AND source_document LIKE 'dream%%'
              {tier_clause}
              {status_filter}
        """, params)
        return cursor.fetchone()[0]

    def promote_candidate_relation(self, family_id: str,
                                    evidence_source: str = "manual",
                                    new_confidence: float = None) -> Dict[str, Any]:
        """将候选关系提升为已验证状态。

        Args:
            family_id: 关系 family_id
            evidence_source: 提升来源 ("manual" | "remember" | "dream_corroboration")
            new_confidence: 可选的新置信度（不传则自动提升到 0.7）

        Returns:
            {"family_id": "...", "old_status": "...", "new_status": "verified", "confidence": ...}
        """
        resolved = self.resolve_family_id(family_id)
        if not resolved:
            raise ValueError(f"关系不存在: {family_id}")

        rel = self.get_relation_by_family_id(resolved)
        if not rel:
            raise ValueError(f"关系不存在: {family_id}")

        # Parse current attributes
        try:
            attrs = json.loads(rel.attributes) if rel.attributes else {}
        except (json.JSONDecodeError, TypeError):
            attrs = {}

        old_status = attrs.get("status", "unknown")
        old_tier = attrs.get("tier", "unknown")

        # Update attributes
        attrs["tier"] = "verified"
        attrs["status"] = "verified"
        attrs["promoted_by"] = evidence_source
        attrs["promoted_at"] = datetime.now().isoformat()
        attrs["corroboration_count"] = attrs.get("corroboration_count", 0) + 1

        # Create new version with updated attributes and confidence
        now = datetime.now()
        record_id = f"relation_{now.strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
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
            attributes=json.dumps(attrs),
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

    def demote_candidate_relation(self, family_id: str,
                                   reason: str = "") -> Dict[str, Any]:
        """将候选关系降级为已拒绝状态。

        Args:
            family_id: 关系 family_id
            reason: 拒绝原因

        Returns:
            {"family_id": "...", "old_status": "...", "new_status": "rejected"}
        """
        resolved = self.resolve_family_id(family_id)
        if not resolved:
            raise ValueError(f"关系不存在: {family_id}")

        rel = self.get_relation_by_family_id(resolved)
        if not rel:
            raise ValueError(f"关系不存在: {family_id}")

        # Parse current attributes
        try:
            attrs = json.loads(rel.attributes) if rel.attributes else {}
        except (json.JSONDecodeError, TypeError):
            attrs = {}

        old_status = attrs.get("status", "unknown")

        # Update attributes
        attrs["status"] = "rejected"
        attrs["rejected_reason"] = reason
        attrs["rejected_at"] = datetime.now().isoformat()

        # Create new version with lower confidence
        now = datetime.now()
        record_id = f"relation_{now.strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"

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
            attributes=json.dumps(attrs),
        )
        self.save_relation(relation)

        return {
            "family_id": resolved,
            "old_status": old_status,
            "new_status": "rejected",
            "confidence": relation.confidence,
        }

    def corroborate_dream_relation(self, entity1_family_id: str, entity2_family_id: str,
                                    corroboration_source: str = "remember") -> Optional[Dict[str, Any]]:
        """当 remember 提取的关系与 dream 候选关系匹配时，自动增加佐证并可能提升。

        Args:
            entity1_family_id: 实体1 family_id
            entity2_family_id: 实体2 family_id
            corroboration_source: 佐证来源 ("remember" | "dream")

        Returns:
            提升结果 dict，或 None（未找到匹配的候选关系）
        """
        # Find candidate relation between these entities
        rels = self.get_relations_by_entities(entity1_family_id, entity2_family_id)
        if not rels:
            return None

        for rel in rels:
            try:
                attrs = json.loads(rel.attributes) if rel.attributes else {}
            except (json.JSONDecodeError, TypeError):
                attrs = {}

            # Only corroborate candidate-tier dream relations
            if (attrs.get("tier") == "candidate" and
                attrs.get("status") == "hypothesized" and
                rel.source_document and rel.source_document.startswith("dream")):

                # Increment corroboration count
                count = attrs.get("corroboration_count", 0) + 1
                attrs["corroboration_count"] = count
                attrs.setdefault("corroboration_sources", []).append(corroboration_source)

                # Auto-promote after 2 corroborations
                if count >= 2:
                    # First save updated attributes so promote reads them
                    now = datetime.now()
                    record_id = f"relation_{now.strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
                    new_conf = min((rel.confidence or 0.5) + 0.1, 0.69)

                    pre_promote = Relation(
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
                        attributes=json.dumps(attrs),
                    )
                    self.save_relation(pre_promote)

                    return self.promote_candidate_relation(
                        rel.family_id,
                        evidence_source=f"auto:{corroboration_source}",
                    )
                else:
                    # Update attributes but don't promote yet
                    now = datetime.now()
                    record_id = f"relation_{now.strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
                    new_conf = min((rel.confidence or 0.5) + 0.1, 0.69)  # cap below 0.7 until promoted

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
                        attributes=json.dumps(attrs),
                    )
                    self.save_relation(updated)

                    return {
                        "family_id": rel.family_id,
                        "corroboration_count": count,
                        "status": "hypothesized",
                        "confidence": new_conf,
                        "message": f"佐证计数: {count}/2，需要2次佐证才能提升",
                    }

        return None

    def reject_dream_cycle_relations(self, dream_cycle_id: str) -> Dict[str, Any]:
        """批量拒绝指定 Dream 周期产生的所有未验证候选关系。

        Returns:
            {"rejected_count": N, "cycle_id": "..."}
        """
        conn = self._get_conn()
        cursor = conn.cursor()

        # Find all candidate relations from this cycle
        cursor.execute(f"""
            SELECT DISTINCT family_id
            FROM relations
            WHERE source_document = ?
              AND json_extract(attributes, '$.tier') = 'candidate'
              AND json_extract(attributes, '$.status') = 'hypothesized'
              AND invalid_at IS NULL
        """, [f"dream:{dream_cycle_id}"])

        family_ids = [row[0] for row in cursor.fetchall()]

        rejected = 0
        for fid in family_ids:
            try:
                self.demote_candidate_relation(fid, reason=f"批量拒绝: dream cycle {dream_cycle_id}")
                rejected += 1
            except Exception:
                pass

        return {
            "rejected_count": rejected,
            "cycle_id": dream_cycle_id,
        }

    # ------------------------------------------------------------------
    # Dream episode
    # ------------------------------------------------------------------

    def save_dream_episode(self, content: str,
                           entities_examined: Optional[List[str]] = None,
                           relations_created: Optional[list] = None,
                           strategy_used: str = "",
                           dream_cycle_id: Optional[str] = None,
                           **kwargs) -> Dict[str, Any]:
        """保存梦境 episode，同时创建 DreamLog 以便 dream_status/dream_logs 查询。

        Returns: {"episode_id": "...", "episode_type": "dream", "cycle_id": "..."}
        """
        now = datetime.now()
        episode_id = f"episode_dream_{now.strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"

        if not dream_cycle_id:
            dream_cycle_id = f"dream_{now.strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}"

        # Build structured content
        _explicit_rel_count = kwargs.get("relations_created_count")
        _explicit_ent_count = kwargs.get("entities_examined_count")
        ent_count = _explicit_ent_count if _explicit_ent_count is not None else (len(entities_examined) if entities_examined else 0)
        rel_count = _explicit_rel_count if _explicit_rel_count is not None else (len(relations_created) if isinstance(relations_created, list) and relations_created else 0)

        structured = {
            "narrative": content,
            "strategy": strategy_used,
            "entities_examined_count": ent_count,
            "relations_created_count": rel_count,
        }
        if isinstance(relations_created, list) and relations_created:
            structured["relations_created"] = relations_created

        full_content = content
        if rel_count > 0 or ent_count > 0:
            full_content += "\n\n---\n" + json.dumps(structured, ensure_ascii=False, indent=2)

        cache = Episode(
            absolute_id=episode_id,
            content=full_content,
            event_time=now,
            source_document=f"dream:{dream_cycle_id}" if dream_cycle_id else "dream",
            episode_type="dream",
        )

        self.save_episode(cache)

        # Record entity mentions
        if entities_examined:
            abs_ids = []
            for eid in entities_examined:
                resolved = self.resolve_family_id(eid)
                if resolved:
                    entity = self.get_entity_by_family_id(resolved)
                    if entity:
                        abs_ids.append(entity.absolute_id)
            if abs_ids:
                self.save_episode_mentions(episode_id, abs_ids, context=f"dream:{strategy_used}")

        # Create DreamLog
        report = SimpleNamespace(
            cycle_id=dream_cycle_id,
            graph_id=getattr(self, '_graph_id', 'default'),
            start_time=now,
            end_time=now,
            status="completed",
            narrative=content[:2000],
            insights=[],
            new_connections=relations_created if isinstance(relations_created, list) else [],
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
