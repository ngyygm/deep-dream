"""Dream mixin — seeds, corroboration, candidate relations, logs, episodes."""

import json
import logging
import random
import uuid
from datetime import datetime
from types import SimpleNamespace
from typing import Any, Dict, List, Optional

from ...models import Relation, Episode
from .helpers import _fmt_dt, _row_to_relation

logger = logging.getLogger(__name__)


class _DreamMixin:

    def _dream_seeds_random(self, count, exclude_uuids, community_id):
        conn = self._connect()
        try:
            query = (
                "SELECT uuid, family_id, name, content, confidence, event_time, community_id FROM entity "
                "WHERE graph_id = ? "
            )
            params: list = [self._graph_id]
            if exclude_uuids:
                ph = ",".join("?" * len(exclude_uuids))
                query += f"AND uuid NOT IN ({ph}) "
                params.extend(exclude_uuids)
            if community_id is not None:
                query += "AND community_id = ? "
                params.append(str(community_id))
            query += "ORDER BY RANDOM() LIMIT ?"
            params.append(count)
            rows = conn.execute(query, params).fetchall()
        finally:
            conn.rollback()
        return [dict(r) for r in rows]

    def _dream_seeds_orphan(self, count, exclude_uuids, community_id):
        conn = self._connect()
        try:
            query = (
                "SELECT e.uuid, e.family_id, e.name, e.content, e.confidence, e.event_time, e.community_id "
                "FROM entity e WHERE e.graph_id = ? "
                "AND e.uuid NOT IN (SELECT entity1_uuid FROM relates_to WHERE graph_id = ? UNION SELECT entity2_uuid FROM relates_to WHERE graph_id = ?) "
            )
            params: list = [self._graph_id, self._graph_id, self._graph_id]
            if exclude_uuids:
                ph = ",".join("?" * len(exclude_uuids))
                query += f"AND e.uuid NOT IN ({ph}) "
                params.extend(exclude_uuids)
            if community_id is not None:
                query += "AND e.community_id = ? "
                params.append(str(community_id))
            query += "LIMIT ?"
            params.append(count)
            rows = conn.execute(query, params).fetchall()
        finally:
            conn.rollback()
        return [dict(r) for r in rows]

    def _dream_seeds_hub(self, count, exclude_uuids, community_id):
        conn = self._connect()
        try:
            query = (
                "SELECT e.uuid, e.family_id, e.name, e.content, e.confidence, e.event_time, e.community_id, "
                "COUNT(DISTINCT rt.entity2_uuid) AS degree "
                "FROM entity e "
                "INNER JOIN relates_to rt ON rt.entity1_uuid = e.uuid AND rt.graph_id = ? "
                "WHERE e.graph_id = ? "
            )
            params: list = [self._graph_id, self._graph_id]
            if exclude_uuids:
                ph = ",".join("?" * len(exclude_uuids))
                query += f"AND e.uuid NOT IN ({ph}) "
                params.extend(exclude_uuids)
            if community_id is not None:
                query += "AND e.community_id = ? "
                params.append(str(community_id))
            query += "GROUP BY e.uuid ORDER BY degree DESC LIMIT ?"
            params.append(count)
            rows = conn.execute(query, params).fetchall()
        finally:
            conn.rollback()
        return [dict(r) for r in rows]

    def _dream_seeds_time_gap(self, count, exclude_uuids, community_id):
        conn = self._connect()
        try:
            query = (
                "SELECT uuid, family_id, name, content, confidence, event_time, community_id "
                "FROM entity WHERE graph_id = ? "
                "AND processed_time IS NOT NULL AND julianday('now') - julianday(processed_time) > 30 "
            )
            params: list = [self._graph_id]
            if exclude_uuids:
                ph = ",".join("?" * len(exclude_uuids))
                query += f"AND uuid NOT IN ({ph}) "
                params.extend(exclude_uuids)
            if community_id is not None:
                query += "AND community_id = ? "
                params.append(str(community_id))
            query += "ORDER BY processed_time ASC LIMIT ?"
            params.append(count)
            rows = conn.execute(query, params).fetchall()
        finally:
            conn.rollback()
        return [dict(r) for r in rows]

    def _dream_seeds_low_confidence(self, count, exclude_uuids, community_id):
        conn = self._connect()
        try:
            query = (
                "SELECT uuid, family_id, name, content, confidence, event_time, community_id "
                "FROM entity WHERE graph_id = ? "
                "AND confidence IS NOT NULL AND confidence < 0.5 "
            )
            params: list = [self._graph_id]
            if exclude_uuids:
                ph = ",".join("?" * len(exclude_uuids))
                query += f"AND uuid NOT IN ({ph}) "
                params.extend(exclude_uuids)
            if community_id is not None:
                query += "AND community_id = ? "
                params.append(str(community_id))
            query += "ORDER BY confidence ASC LIMIT ?"
            params.append(count)
            rows = conn.execute(query, params).fetchall()
        finally:
            conn.rollback()
        return [dict(r) for r in rows]

    def _dream_seeds_cross_community(self, count, exclude_uuids, community_id):
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
                c1_valid = [m for m in c1_members if m["uuid"] not in exclude_uuids]
                c2_valid = [m for m in c2_members if m["uuid"] not in exclude_uuids]
                if c1_valid and c2_valid:
                    e1 = random.choice(c1_valid)
                    e2 = random.choice(c2_valid)
                    pairs.extend([e1, e2])
            if len(pairs) >= count * 2:
                break
        return pairs[:count * 2]

    def get_dream_seeds(self, strategy: str = "random", count: int = 10,
                        exclude_ids: Optional[List[str]] = None,
                        community_id: Optional[int] = None) -> List[Dict[str, Any]]:
        exclude_uuids = set()
        if exclude_ids:
            resolved_map = self.resolve_family_ids(exclude_ids)
            canonical_fids = list({v for v in resolved_map.values() if v})
            if canonical_fids:
                aids_map = self.get_latest_absolute_ids_by_family_ids(canonical_fids)
                exclude_uuids = set(aids_map.values())
        strategies = {
            "random": self._dream_seeds_random, "orphan": self._dream_seeds_orphan,
            "hub": self._dream_seeds_hub, "time_gap": self._dream_seeds_time_gap,
            "low_confidence": self._dream_seeds_low_confidence, "cross_community": self._dream_seeds_cross_community,
        }
        handler = strategies.get(strategy)
        if not handler:
            raise ValueError(f"Unknown seed strategy: {strategy}")
        seeds = handler(count, exclude_uuids, community_id)
        reason_map = {"random": "Random selection", "orphan": "Orphan entity: no connections",
                      "hub": "High connectivity hub", "time_gap": "Long time without updates",
                      "low_confidence": "Low confidence entity", "cross_community": "Cross-community bridge candidate"}
        for s in seeds:
            s["reason"] = reason_map.get(strategy, "")
        return seeds

    def corroborate_dream_relation(self, entity1_family_id: str, entity2_family_id: str,
                                    corroboration_source: str = "remember") -> Optional[Dict[str, Any]]:
        rels = self.get_relations_by_entities(entity1_family_id, entity2_family_id, include_candidates=True)
        if not rels:
            return None
        for rel in rels:
            try:
                attrs = json.loads(rel.attributes) if rel.attributes else {}
            except (ValueError, TypeError):
                attrs = {}
            if (attrs.get("tier") == "candidate" and attrs.get("status") == "hypothesized"
                    and rel.source_document and rel.source_document.startswith("dream")):
                count = attrs.get("corroboration_count", 0) + 1
                attrs["corroboration_count"] = count
                attrs.setdefault("corroboration_sources", []).append(corroboration_source)
                now = datetime.now()
                record_id = f"relation_{now.strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
                new_conf = min((rel.confidence or 0.5) + 0.1, 0.69)
                updated = Relation(
                    absolute_id=record_id, family_id=rel.family_id,
                    entity1_absolute_id=rel.entity1_absolute_id, entity2_absolute_id=rel.entity2_absolute_id,
                    content=rel.content, event_time=now, processed_time=now,
                    episode_id=rel.episode_id, source_document=rel.source_document,
                    confidence=new_conf, attributes=json.dumps(attrs),
                )
                self.save_relation(updated)
                if count >= 2:
                    return self.promote_candidate_relation(rel.family_id, evidence_source=f"auto:{corroboration_source}")
                return {"family_id": rel.family_id, "corroboration_count": count, "status": "hypothesized",
                        "confidence": new_conf, "message": f"Corroboration count: {count}/2"}
        return None

    def corroborate_dream_relations_batch(self, entity_pairs: List[tuple], corroboration_source: str = "remember") -> List[Dict[str, Any]]:
        if not entity_pairs:
            return []
        results = []
        for e1_fid, e2_fid in entity_pairs:
            result = self.corroborate_dream_relation(e1_fid, e2_fid, corroboration_source)
            if result:
                results.append(result)
        return results

    def count_candidate_relations(self, status: str = None) -> int:
        conn = self._connect()
        try:
            query = "SELECT COUNT(DISTINCT family_id) AS cnt FROM relation WHERE source_document LIKE 'dream%' AND graph_id = ?"
            params: list = [self._graph_id]
            if status:
                query += " AND attributes LIKE ?"
                params.append(f'%"status":"{status}"%')
            row = conn.execute(query, params).fetchone()
        finally:
            conn.rollback()
        return row["cnt"] if row else 0

    def get_candidate_relations(self, limit: int = 50, offset: int = 0, status: str = None) -> list:
        conn = self._connect()
        try:
            query = (
                f"SELECT r.* FROM relation r "
                f"INNER JOIN ("
                f"  SELECT family_id, MAX(processed_time) AS max_pt FROM relation "
                f"  WHERE graph_id = ? AND source_document LIKE 'dream%' "
                f"  GROUP BY family_id"
                f") latest ON r.family_id = latest.family_id AND r.processed_time = latest.max_pt "
                f"WHERE r.graph_id = ? "
            )
            params: list = [self._graph_id, self._graph_id]
            if status:
                query += "AND r.attributes LIKE ? "
                params.append(f'%"status":"{status}"%')
            query += f"ORDER BY r.processed_time DESC LIMIT ? OFFSET ?"
            params.extend([limit, offset])
            rows = conn.execute(query, params).fetchall()
        finally:
            conn.rollback()
        return [_row_to_relation(dict(r)) for r in rows]

    def promote_candidate_relation(self, family_id: str, evidence_source: str = "manual", new_confidence: float = None) -> Dict[str, Any]:
        resolved = self.resolve_family_id(family_id)
        if not resolved:
            raise ValueError(f"Relation not found: {family_id}")
        rel = self.get_relation_by_family_id(resolved)
        if not rel:
            raise ValueError(f"Relation not found: {family_id}")
        try:
            attrs = json.loads(rel.attributes) if rel.attributes else {}
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
        record_id = f"relation_{now.strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
        new_conf = new_confidence if new_confidence is not None else max(rel.confidence or 0.5, 0.7)
        relation = Relation(
            absolute_id=record_id, family_id=rel.family_id,
            entity1_absolute_id=rel.entity1_absolute_id, entity2_absolute_id=rel.entity2_absolute_id,
            content=rel.content, event_time=now, processed_time=now,
            episode_id=rel.episode_id, source_document=rel.source_document,
            confidence=new_conf, attributes=json.dumps(attrs),
        )
        self.save_relation(relation)
        return {"family_id": resolved, "old_status": old_status, "old_tier": old_tier,
                "new_status": "verified", "new_tier": "verified", "confidence": new_conf}

    def promote_candidate_relations_batch(self, family_ids: List[str], evidence_source: str = "manual", new_confidence: float = None) -> List[Dict[str, Any]]:
        if not family_ids:
            return []
        results = []
        for fid in family_ids:
            try:
                result = self.promote_candidate_relation(fid, evidence_source, new_confidence)
                results.append(result)
            except Exception:
                pass
        return results

    def demote_candidate_relation(self, family_id: str, reason: str = "") -> Dict[str, Any]:
        resolved = self.resolve_family_id(family_id)
        if not resolved:
            raise ValueError(f"Relation not found: {family_id}")
        rel = self.get_relation_by_family_id(resolved)
        if not rel:
            raise ValueError(f"Relation not found: {family_id}")
        try:
            attrs = json.loads(rel.attributes) if rel.attributes else {}
        except (ValueError, TypeError):
            attrs = {}
        old_status = attrs.get("status", "unknown")
        attrs["status"] = "rejected"
        attrs["rejected_reason"] = reason
        now = datetime.now()
        attrs["rejected_at"] = now.isoformat()
        record_id = f"relation_{now.strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
        relation = Relation(
            absolute_id=record_id, family_id=rel.family_id,
            entity1_absolute_id=rel.entity1_absolute_id, entity2_absolute_id=rel.entity2_absolute_id,
            content=rel.content, event_time=now, processed_time=now,
            episode_id=rel.episode_id, source_document=rel.source_document,
            confidence=min(rel.confidence or 0.3, 0.2), attributes=json.dumps(attrs),
        )
        self.save_relation(relation)
        return {"family_id": resolved, "old_status": old_status, "new_status": "rejected", "confidence": relation.confidence}

    def save_dream_log(self, report):
        conn = self._connect()
        try:
            conn.execute(
                "INSERT OR REPLACE INTO dream_log (cycle_id, graph_id, start_time, end_time, status, narrative, "
                "insights, connections, consolidations, strategy, entities_examined, relations_created, episode_ids) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    report.cycle_id, report.graph_id,
                    _fmt_dt(report.start_time), _fmt_dt(report.end_time or datetime.now()),
                    report.status, report.narrative,
                    json.dumps(report.insights, ensure_ascii=False),
                    json.dumps(getattr(report, 'new_connections', []), ensure_ascii=False),
                    json.dumps(report.consolidations, ensure_ascii=False),
                    getattr(report, 'strategy', ''),
                    getattr(report, 'entities_examined', 0),
                    getattr(report, 'relations_created', 0),
                    json.dumps(getattr(report, 'episode_ids', []), ensure_ascii=False),
                ),
            )
            conn.commit()
        finally:
            conn.rollback()

    def get_dream_log(self, cycle_id: str) -> Optional[dict]:
        conn = self._connect()
        try:
            row = conn.execute(
                "SELECT * FROM dream_log WHERE cycle_id = ? AND graph_id = ?",
                (cycle_id, self._graph_id),
            ).fetchone()
        finally:
            conn.rollback()
        if not row:
            return None
        return self._parse_dream_log_record(dict(row))

    @staticmethod
    def _parse_dream_log_record(r) -> dict:
        d = dict(r) if not isinstance(r, dict) else r
        _loads = json.loads
        _raw_ins = d.get("insights")
        _raw_con = d.get("connections")
        _raw_cns = d.get("consolidations")
        _raw_epi = d.get("episode_ids")
        return {
            "cycle_id": d["cycle_id"], "graph_id": d["graph_id"],
            "start_time": str(d.get("start_time", "")), "end_time": str(d.get("end_time", "")),
            "status": d.get("status", ""), "narrative": d.get("narrative", ""),
            "insights": () if not _raw_ins or _raw_ins == "[]" else _loads(_raw_ins),
            "connections": () if not _raw_con or _raw_con == "[]" else _loads(_raw_con),
            "consolidations": () if not _raw_cns or _raw_cns == "[]" else _loads(_raw_cns),
            "strategy": d.get("strategy", ""),
            "entities_examined": d.get("entities_examined", 0),
            "relations_created": d.get("relations_created", 0),
            "episode_ids": () if not _raw_epi or _raw_epi == "[]" else _loads(_raw_epi),
        }

    def list_dream_logs(self, graph_id: str = None, limit: int = 20) -> List[dict]:
        gid = graph_id or self._graph_id
        conn = self._connect()
        try:
            rows = conn.execute(
                "SELECT * FROM dream_log WHERE graph_id = ? ORDER BY start_time DESC LIMIT ?",
                (gid, limit),
            ).fetchall()
        finally:
            conn.rollback()
        return [self._parse_dream_log_record(dict(r)) for r in rows]

    def save_dream_episode(self, content: str, entities_examined: Optional[List[str]] = None,
                           relations_created: Optional[List[Dict]] = None, strategy_used: str = "",
                           dream_cycle_id: Optional[str] = None, **kwargs) -> Dict[str, Any]:
        now = datetime.now()
        episode_id = f"episode_dream_{now.strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
        if not dream_cycle_id:
            dream_cycle_id = f"dream_{now.strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}"
        _explicit_rel_count = kwargs.get("relations_created_count")
        _explicit_ent_count = kwargs.get("entities_examined_count")
        ent_count = _explicit_ent_count if _explicit_ent_count is not None else (len(entities_examined) if entities_examined else 0)
        rel_count = _explicit_rel_count if _explicit_rel_count is not None else (len(relations_created) if relations_created else 0)
        structured = {"narrative": content, "strategy": strategy_used,
                      "entities_examined_count": ent_count, "relations_created_count": rel_count}
        if relations_created:
            structured["relations_created"] = relations_created
        full_content = content
        if rel_count > 0 or ent_count > 0:
            full_content += "\n\n---\n" + json.dumps(structured, ensure_ascii=False, indent=2)
        source_doc = f"dream:{dream_cycle_id}" if dream_cycle_id else "dream"
        cache = Episode(
            absolute_id=episode_id, content=full_content, event_time=now,
            source_document=source_doc, episode_type="dream",
        )
        self.save_episode(cache)
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
                for eid in entities_examined:
                    resolved = self.resolve_family_id(eid)
                    if resolved:
                        entity = self.get_entity_by_family_id(resolved)
                        if entity:
                            abs_ids.append(entity.absolute_id)
            if abs_ids:
                self.save_episode_mentions(episode_id, abs_ids, context=f"dream:{strategy_used}")
        report = SimpleNamespace(
            cycle_id=dream_cycle_id, graph_id=self._graph_id,
            start_time=now, end_time=now, status="completed",
            narrative=content[:2000], insights=[], new_connections=relations_created or [],
            consolidations=[], strategy=strategy_used,
            entities_examined=ent_count, relations_created=rel_count,
            episode_ids=[episode_id],
        )
        self.save_dream_log(report)
        return {"episode_id": episode_id, "episode_type": "dream", "cycle_id": dream_cycle_id}
