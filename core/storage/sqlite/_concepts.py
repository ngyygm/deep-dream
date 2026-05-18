"""Concept mixin — unified concept model (entity, relation, observation)."""

from typing import Any, Dict, List, Optional

from .helpers import ENTITY_COLUMNS, RELATION_COLUMNS, _fmt_dt, _row_to_entity, _row_to_relation


class _ConceptMixin:

    def count_concepts(self, role: str = None, time_point: str = None) -> int:
        tp = self._tp_to_datetime(time_point)
        tp_iso = tp.isoformat() if tp else None
        conn = self._connect()
        try:
            total = 0
            if role is None or role == "entity":
                query = "SELECT COUNT(DISTINCT family_id) AS cnt FROM entity WHERE graph_id = ?"
                params = [self._graph_id]
                if tp_iso:
                    query += " AND (valid_at IS NULL OR valid_at <= ?)"
                    params.append(tp_iso)
                total += conn.execute(query, params).fetchone()["cnt"]
            if role is None or role == "relation":
                query = "SELECT COUNT(DISTINCT family_id) AS cnt FROM relation WHERE graph_id = ?"
                params = [self._graph_id]
                if tp_iso:
                    query += " AND (valid_at IS NULL OR valid_at <= ?)"
                    params.append(tp_iso)
                total += conn.execute(query, params).fetchone()["cnt"]
            if role is None or role == "observation":
                query = "SELECT COUNT(*) AS cnt FROM episode WHERE graph_id = ?"
                params = [self._graph_id]
                total += conn.execute(query, params).fetchone()["cnt"]
        finally:
            conn.rollback()
        return total

    def get_concept_by_family_id(self, family_id: str, time_point: str = None) -> Optional[dict]:
        tp = self._tp_to_datetime(time_point)
        tp_iso = tp.isoformat() if tp else None
        # Try entity first
        conn = self._connect()
        try:
            query = f"SELECT {', '.join(ENTITY_COLUMNS)} FROM entity WHERE family_id = ? AND graph_id = ?"
            params = [family_id, self._graph_id]
            if tp_iso:
                query += " AND (valid_at IS NULL OR valid_at <= ?)"
                params.append(tp_iso)
            query += " ORDER BY version_seq DESC LIMIT 1"
            row = conn.execute(query, params).fetchone()
            if row:
                entity = _row_to_entity(dict(row))
                return {"id": entity.absolute_id, "family_id": entity.family_id, "role": "entity",
                        "name": entity.name, "content": entity.content,
                        "event_time": _fmt_dt(entity.event_time), "processed_time": _fmt_dt(entity.processed_time),
                        "source_document": entity.source_document, "summary": entity.summary, "confidence": entity.confidence}
            # Try relation
            query = f"SELECT {', '.join(RELATION_COLUMNS)} FROM relation WHERE family_id = ? AND graph_id = ?"
            params = [family_id, self._graph_id]
            if tp_iso:
                query += " AND (valid_at IS NULL OR valid_at <= ?)"
                params.append(tp_iso)
            query += " ORDER BY version_seq DESC LIMIT 1"
            row = conn.execute(query, params).fetchone()
            if row:
                rel = _row_to_relation(dict(row))
                return {"id": rel.absolute_id, "family_id": rel.family_id, "role": "relation",
                        "name": "", "content": rel.content,
                        "event_time": _fmt_dt(rel.event_time), "processed_time": _fmt_dt(rel.processed_time),
                        "source_document": rel.source_document, "summary": rel.summary, "confidence": rel.confidence}
            # Try episode (by uuid)
            row = conn.execute("SELECT * FROM episode WHERE uuid = ? AND graph_id = ?", (family_id, self._graph_id)).fetchone()
            if row:
                rd = dict(row)
                return {"id": rd["uuid"], "family_id": rd["uuid"], "role": "observation",
                        "name": "", "content": rd["content"] or "",
                        "event_time": _fmt_dt(rd.get("event_time")), "processed_time": _fmt_dt(rd.get("processed_time")),
                        "source_document": rd.get("source_document", ""), "summary": None, "confidence": None}
        finally:
            conn.rollback()
        return None

    def get_concept_neighbors(self, family_id: str, max_depth: int = 1, time_point: str = None) -> List[dict]:
        concept = self.get_concept_by_family_id(family_id, time_point=time_point)
        if not concept:
            return []
        abs_id = concept.get("id")
        role = concept.get("role")
        if not abs_id or not role:
            return []
        neighbors = []
        conn = self._connect()
        try:
            if role == 'entity':
                # RELATES_TO neighbors
                edge_rows = conn.execute(
                    "SELECT entity1_uuid, entity2_uuid FROM relates_to WHERE (entity1_uuid = ? OR entity2_uuid = ?) AND graph_id = ?",
                    (abs_id, abs_id, self._graph_id),
                ).fetchall()
                neighbor_uuids = set()
                for r in edge_rows:
                    n = r["entity2_uuid"] if r["entity1_uuid"] == abs_id else r["entity1_uuid"]
                    neighbor_uuids.add(n)
                if neighbor_uuids:
                    ph = ",".join("?" * len(neighbor_uuids))
                    ent_rows = conn.execute(
                        f"SELECT DISTINCT family_id, uuid AS id, name, 'entity' AS role, content FROM entity WHERE uuid IN ({ph}) AND graph_id = ?",
                        list(neighbor_uuids) + [self._graph_id],
                    ).fetchall()
                    neighbors.extend([dict(r) for r in ent_rows])
                # Relations referencing this entity
                rel_rows = conn.execute(
                    "SELECT DISTINCT family_id, uuid AS id, '' AS name, 'relation' AS role, content FROM relation WHERE (entity1_absolute_id = ? OR entity2_absolute_id = ?) AND graph_id = ?",
                    (abs_id, abs_id, self._graph_id),
                ).fetchall()
                neighbors.extend([dict(r) for r in rel_rows])
            elif role == 'relation':
                # Endpoint entities
                rel = self.get_relation_by_absolute_id(abs_id)
                if rel:
                    eids = [rel.entity1_absolute_id, rel.entity2_absolute_id]
                    ph = ",".join("?" * len(eids))
                    ent_rows = conn.execute(
                        f"SELECT DISTINCT family_id, uuid AS id, name, 'entity' AS role, content FROM entity WHERE uuid IN ({ph}) AND graph_id = ?",
                        eids + [self._graph_id],
                    ).fetchall()
                    neighbors.extend([dict(r) for r in ent_rows])
                # Episodes mentioning this relation
                mention_rows = conn.execute(
                    "SELECT DISTINCT e.uuid AS id, e.content AS name, 'observation' AS role, e.content FROM episode e "
                    "INNER JOIN mentions m ON m.episode_uuid = e.uuid WHERE m.target_uuid = ? AND e.graph_id = ?",
                    (abs_id, self._graph_id),
                ).fetchall()
                neighbors.extend([dict(r) for r in mention_rows])
            elif role == 'observation':
                mention_rows = conn.execute(
                    "SELECT m.target_uuid, m.target_type, m.target_family_id FROM mentions m "
                    "WHERE m.episode_uuid = ? AND m.graph_id = ?",
                    (abs_id, self._graph_id),
                ).fetchall()
                for r in mention_rows:
                    rd = dict(r)
                    neighbors.append({"id": r["target_uuid"], "family_id": rd.get("target_family_id", ""),
                                      "role": r["target_type"] if r["target_type"] != "entity" else "entity"})
        finally:
            conn.rollback()
        seen = set()
        deduped = []
        for n in neighbors:
            fid = n.get('family_id', '')
            if fid and fid not in seen:
                seen.add(fid)
                deduped.append(n)
        return deduped

    def get_concept_provenance(self, family_id: str, time_point: str = None) -> List[dict]:
        concept = self.get_concept_by_family_id(family_id, time_point=time_point)
        if not concept:
            return []
        role = concept.get("role")
        abs_id = concept.get("id")
        if not abs_id:
            return []
        if role == 'entity':
            return self.get_entity_provenance(family_id)
        elif role == 'relation':
            conn = self._connect()
            try:
                abs_rows = conn.execute("SELECT uuid FROM relation WHERE family_id = ? AND graph_id = ?", (family_id, self._graph_id)).fetchall()
                abs_ids = [r["uuid"] for r in abs_rows]
                if not abs_ids:
                    return []
                ph = ",".join("?" * len(abs_ids))
                rows = conn.execute(
                    f"SELECT DISTINCT m.episode_uuid AS episode_id, m.context, e.content, e.source_document "
                    f"FROM mentions m INNER JOIN episode e ON e.uuid = m.episode_uuid "
                    f"WHERE m.target_uuid IN ({ph}) AND m.graph_id = ?",
                    abs_ids + [self._graph_id],
                ).fetchall()
            finally:
                conn.rollback()
            return [{"episode_id": r["episode_id"], "context": dict(r).get("context", ""),
                      "content": dict(r).get("content", ""), "source_document": dict(r).get("source_document", "")} for r in rows]
        elif role == 'observation':
            return [{"episode_id": abs_id, "role": "observation", "content": concept.get("content", "")}]
        return []

    def get_concept_mentions(self, family_id: str, time_point: str = None) -> List[dict]:
        return self.get_concept_provenance(family_id, time_point=time_point)

    def get_episode_concepts(self, episode_id: str) -> List[dict]:
        conn = self._connect()
        try:
            rows = conn.execute(
                "SELECT target_uuid, target_type FROM mentions WHERE episode_uuid = ? AND graph_id = ?",
                (episode_id, self._graph_id),
            ).fetchall()
            results = []
            for r in rows:
                target_uuid = r["target_uuid"]
                target_type = r["target_type"]
                if target_type == "entity":
                    ent = conn.execute(
                        "SELECT family_id, uuid AS id, 'entity' AS role, name, content FROM entity WHERE uuid = ? AND graph_id = ?",
                        (target_uuid, self._graph_id),
                    ).fetchone()
                    if ent:
                        results.append(dict(ent))
                elif target_type == "relation":
                    rel = conn.execute(
                        "SELECT family_id, uuid AS id, 'relation' AS role, '' AS name, content FROM relation WHERE uuid = ? AND graph_id = ?",
                        (target_uuid, self._graph_id),
                    ).fetchone()
                    if rel:
                        results.append(dict(rel))
        finally:
            conn.rollback()
        return results

    def list_concepts(self, role: str = None, limit: int = 50, offset: int = 0, time_point: str = None) -> List[dict]:
        results = []
        tp = self._tp_to_datetime(time_point)
        tp_iso = tp.isoformat() if tp else None
        conn = self._connect()
        try:
            if role is None or role == "entity":
                query = f"SELECT {', '.join(ENTITY_COLUMNS)} FROM entity WHERE graph_id = ?"
                params = [self._graph_id]
                if tp_iso:
                    query += " AND (valid_at IS NULL OR valid_at <= ?)"
                    params.append(tp_iso)
                query += " ORDER BY processed_time DESC LIMIT ? OFFSET ?"
                params.extend([limit, offset])
                rows = conn.execute(query, params).fetchall()
                for r in rows:
                    ent = _row_to_entity(dict(r))
                    results.append({"id": ent.absolute_id, "family_id": ent.family_id, "role": "entity",
                                    "name": ent.name, "content": ent.content,
                                    "event_time": _fmt_dt(ent.event_time), "processed_time": _fmt_dt(ent.processed_time)})
            if role is None or role == "relation":
                query = f"SELECT {', '.join(RELATION_COLUMNS)} FROM relation WHERE graph_id = ?"
                params = [self._graph_id]
                if tp_iso:
                    query += " AND (valid_at IS NULL OR valid_at <= ?)"
                    params.append(tp_iso)
                query += " ORDER BY processed_time DESC LIMIT ? OFFSET ?"
                params.extend([limit, offset])
                rows = conn.execute(query, params).fetchall()
                for r in rows:
                    rel = _row_to_relation(dict(r))
                    results.append({"id": rel.absolute_id, "family_id": rel.family_id, "role": "relation",
                                    "name": "", "content": rel.content,
                                    "event_time": _fmt_dt(rel.event_time), "processed_time": _fmt_dt(rel.processed_time)})
            if role is None or role == "observation":
                rows = conn.execute(
                    "SELECT uuid, content, event_time, processed_time FROM episode WHERE graph_id = ? ORDER BY created_at DESC LIMIT ? OFFSET ?",
                    (self._graph_id, limit, offset),
                ).fetchall()
                for r in rows:
                    results.append({"id": r["uuid"], "family_id": r["uuid"], "role": "observation",
                                    "name": "", "content": r["content"] or "",
                                    "event_time": _fmt_dt(r["event_time"]), "processed_time": _fmt_dt(r["processed_time"])})
        finally:
            conn.rollback()
        return results[:limit]

    def traverse_concepts(self, start_family_ids: List[str], max_depth: int = 2, time_point: str = None) -> dict:
        if not start_family_ids:
            return {"concepts": {}, "edges": [], "relations": [], "visited_count": 0}
        visited = set()
        queue = list(start_family_ids)
        all_concepts = {}
        all_edges = []
        for _ in range(max_depth):
            frontier = [fid for fid in queue if fid not in visited]
            if not frontier:
                break
            visited.update(frontier)
            for fid in frontier:
                concept = self.get_concept_by_family_id(fid, time_point=time_point)
                if concept:
                    all_concepts[fid] = concept
                    neighbors = self.get_concept_neighbors(fid, max_depth=1, time_point=time_point)
                    for n in neighbors:
                        nfid = n.get("family_id", "")
                        if nfid:
                            all_edges.append({"from": fid, "to": nfid, "to_role": n.get("role", ""), "to_name": n.get("name", "")})
                            if nfid not in visited:
                                queue.append(nfid)
            queue = list(set(queue))
        relation_concepts = [c for c in all_concepts.values() if c.get('role') == 'relation']
        return {"concepts": all_concepts, "edges": all_edges, "relations": relation_concepts, "visited_count": len(visited)}

    def search_concepts_by_bm25(self, query: str, role: str = None, limit: int = 20, time_point: str = None) -> List[dict]:
        if not query:
            return []
        results = []
        if role is None or role == "entity":
            entities = self.search_entities_by_bm25(query, limit=limit)
            for e in entities:
                results.append({"id": e.absolute_id, "family_id": e.family_id, "role": "entity",
                                "name": e.name, "content": e.content,
                                "event_time": _fmt_dt(e.event_time), "processed_time": _fmt_dt(e.processed_time)})
        if role is None or role == "relation":
            relations = self.search_relations_by_bm25(query, limit=limit)
            for r in relations:
                results.append({"id": r.absolute_id, "family_id": r.family_id, "role": "relation",
                                "name": "", "content": r.content,
                                "event_time": _fmt_dt(r.event_time), "processed_time": _fmt_dt(r.processed_time)})
        if role is None or role == "observation":
            episodes = self.search_episodes_by_bm25(query, limit=limit)
            for ep in episodes:
                results.append({"id": ep.absolute_id, "family_id": ep.absolute_id, "role": "observation",
                                "name": "", "content": ep.content or "",
                                "event_time": _fmt_dt(ep.event_time), "processed_time": _fmt_dt(ep.processed_time)})
        return results[:limit]

    def search_concepts_by_similarity(self, query_text: str, role: str = None,
                                       threshold: float = 0.5, max_results: int = 20,
                                       time_point: str = None) -> List[dict]:
        if not query_text or not self.embedding_client or not self.embedding_client.is_available():
            return self.search_concepts_by_bm25(query_text, role=role, limit=max_results, time_point=time_point)
        results = []
        if role is None or role == "entity":
            entities = self.search_entities_by_similarity(query_text, threshold=threshold, max_results=max_results)
            for e in entities:
                results.append({"id": e.absolute_id, "family_id": e.family_id, "role": "entity",
                                "name": e.name, "content": e.content, "score": 0.0})
        if role is None or role == "relation":
            relations = self.search_relations_by_similarity(query_text, threshold=threshold, max_results=max_results)
            for r in relations:
                results.append({"id": r.absolute_id, "family_id": r.family_id, "role": "relation",
                                "name": "", "content": r.content, "score": 0.0})
        if role == "observation" or (role is None and len(results) < max_results):
            bm25_results = self.search_concepts_by_bm25(query_text, role="observation", limit=max_results)
            for c in bm25_results:
                results.append(c)
        return results[:max_results]
