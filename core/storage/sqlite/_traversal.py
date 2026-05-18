"""Graph traversal mixin — BFS, shortest paths, neighbors, family merges."""

import logging
import time
from collections import defaultdict
from datetime import datetime
from typing import Any, Dict, List, Optional, Set, Tuple

from ...models import Entity, Relation
from .helpers import ENTITY_COLUMNS, _row_to_entity

logger = logging.getLogger(__name__)


class _TraversalMixin:

    def _build_entity_abs_id_remap(self) -> dict:
        now = time.time()
        if self._entity_remap_cache is not None and (now - self._entity_remap_cache_ts) < self._entity_remap_cache_ttl:
            return self._entity_remap_cache
        conn = self._connect()
        try:
            rows = conn.execute(
                "SELECT family_id, uuid, processed_time FROM entity WHERE graph_id = ? ORDER BY processed_time DESC",
                (self._graph_id,),
            ).fetchall()
        finally:
            conn.rollback()
        fid_to_uuids: Dict[str, List[str]] = defaultdict(list)
        for r in rows:
            fid_to_uuids[r["family_id"]].append(r["uuid"])
        remap = {}
        for fid, uuids in fid_to_uuids.items():
            latest = uuids[0]
            for u in uuids[1:]:
                if u != latest:
                    remap[u] = latest
        self._entity_remap_cache = remap
        self._entity_remap_cache_ts = now
        return remap

    def batch_bfs_traverse(self, seed_family_ids: List[str], max_depth: int = 2, max_nodes: int = 50,
                           time_point: Optional[str] = None) -> Tuple[List[Entity], List[Relation], Dict[str, int]]:
        if not seed_family_ids:
            return [], [], {}
        conn = self._connect()
        try:
            # Get seed absolute_ids
            placeholders = ",".join("?" * len(seed_family_ids))
            seed_rows = conn.execute(
                "SELECT family_id, uuid FROM entity WHERE family_id IN ({}) AND graph_id = ?".format(placeholders),
                seed_family_ids + [self._graph_id],
            ).fetchall()
            seed_abs_to_fid = {r["uuid"]: r["family_id"] for r in seed_rows}
            seed_fids = [r["family_id"] for r in seed_rows]
            if not seed_fids:
                return [], [], {}
            # BFS via relates_to table
            visited_uuids = set()
            visited_fids = set()
            hop_map: Dict[str, int] = {}
            entities = []
            # Add seeds at hop 0
            current_frontier = list(seed_abs_to_fid.keys())
            for uuid_val in current_frontier:
                visited_uuids.add(uuid_val)
            for fid in seed_fids:
                if fid not in hop_map:
                    hop_map[fid] = 0
                    visited_fids.add(fid)
            for depth in range(1, max_depth + 1):
                if not current_frontier:
                    break
                ph = ",".join("?" * len(current_frontier))
                edge_rows = conn.execute(
                    f"SELECT entity1_uuid, entity2_uuid FROM relates_to WHERE (entity1_uuid IN ({ph}) OR entity2_uuid IN ({ph})) AND graph_id = ?",
                    current_frontier + current_frontier + [self._graph_id],
                ).fetchall()
                next_frontier = []
                for r in edge_rows:
                    e1, e2 = r["entity1_uuid"], r["entity2_uuid"]
                    neighbor = e2 if e1 in visited_uuids else e1
                    if neighbor not in visited_uuids:
                        visited_uuids.add(neighbor)
                        next_frontier.append(neighbor)
                current_frontier = next_frontier
                # Get entities for this frontier
                if current_frontier:
                    ph = ",".join("?" * len(current_frontier))
                    frontier_rows = conn.execute(
                        f"SELECT {', '.join(ENTITY_COLUMNS)} FROM entity WHERE uuid IN ({ph}) AND graph_id = ?",
                        current_frontier + [self._graph_id],
                    ).fetchall()
                    for r in frontier_rows:
                        entity = _row_to_entity(dict(r))
                        if entity.family_id not in visited_fids:
                            visited_fids.add(entity.family_id)
                            hop_map[entity.family_id] = depth
                            entities.append(entity)
                if len(visited_fids) >= max_nodes:
                    break
            # Fetch seed entities
            missing_seed_fids = [fid for fid in seed_fids if fid not in hop_map]
            if missing_seed_fids:
                seed_entities = self.get_entities_by_family_ids(missing_seed_fids)
                for fid, entity in seed_entities.items():
                    if fid not in visited_fids:
                        visited_fids.add(fid)
                        hop_map[fid] = 0
                        entities.insert(0, entity)
            # Get seed entities that were already in hop_map
            seed_entity_map = self.get_entities_by_family_ids(seed_fids)
            seed_entities_list = []
            for fid in seed_fids:
                ent = seed_entity_map.get(fid)
                if ent and ent not in entities:
                    seed_entities_list.append(ent)
            all_entities = seed_entities_list + entities
            # Get relations
            discovered_fids = list(hop_map.keys())
            relations = self.get_relations_by_family_ids(discovered_fids, limit=max_nodes * 3, time_point=time_point) if discovered_fids else []
            return all_entities, relations, hop_map
        finally:
            conn.rollback()

    def batch_get_entity_degrees(self, family_ids: List[str]) -> Dict[str, int]:
        if not family_ids:
            return {}
        conn = self._connect()
        try:
            degree_map = {}
            for fid in family_ids:
                rows = conn.execute(
                    "SELECT uuid FROM entity WHERE family_id = ? AND graph_id = ?",
                    (fid, self._graph_id),
                ).fetchall()
                abs_ids = [r["uuid"] for r in rows]
                if abs_ids:
                    ph = ",".join("?" * len(abs_ids))
                    cnt = conn.execute(
                        f"SELECT COUNT(DISTINCT uuid) AS cnt FROM relation WHERE (entity1_absolute_id IN ({ph}) OR entity2_absolute_id IN ({ph})) AND graph_id = ?",
                        abs_ids + abs_ids + [self._graph_id],
                    ).fetchone()["cnt"]
                    degree_map[fid] = cnt
                else:
                    degree_map[fid] = 0
        finally:
            conn.rollback()
        for fid in family_ids:
            degree_map.setdefault(fid, 0)
        return degree_map

    def find_shortest_path_cypher(self, source_family_id: str, target_family_id: str, max_depth: int = 6) -> List[List[str]]:
        result = self.find_shortest_paths(source_family_id, target_family_id, max_depth=max_depth, max_paths=1)
        if result.get("paths"):
            return [[n.name for n in p["entities"]] for p in result["paths"]]
        return []

    def find_shortest_paths(self, source_family_id: str, target_family_id: str,
                             max_depth: int = 6, max_paths: int = 10) -> Dict[str, Any]:
        result_empty = {"source_entity": None, "target_entity": None, "path_length": -1, "total_shortest_paths": 0, "paths": []}
        _ents = self.get_entities_by_family_ids([source_family_id, target_family_id])
        source_entity = _ents.get(source_family_id)
        target_entity = _ents.get(target_family_id)
        if not source_entity or not target_entity:
            result_empty["source_entity"] = source_entity
            result_empty["target_entity"] = target_entity
            return result_empty
        if source_family_id == target_family_id:
            return {"source_entity": source_entity, "target_entity": target_entity, "path_length": 0,
                    "total_shortest_paths": 1, "paths": [{"entities": [source_entity], "relations": [], "length": 0}]}
        # BFS using relates_to table
        conn = self._connect()
        try:
            source_uuid = source_entity.absolute_id
            target_uuid = target_entity.absolute_id
            # BFS with path tracking
            queue = [(source_uuid, [source_uuid])]
            visited = {source_uuid}
            found_paths = []
            while queue and len(found_paths) < max_paths:
                next_queue = []
                for current_uuid, path in queue:
                    ph_placeholders = "?"  # single node lookup
                    edges = conn.execute(
                        "SELECT entity1_uuid, entity2_uuid FROM relates_to WHERE (entity1_uuid = ? OR entity2_uuid = ?) AND graph_id = ?",
                        (current_uuid, current_uuid, self._graph_id),
                    ).fetchall()
                    for r in edges:
                        neighbor = r["entity2_uuid"] if r["entity1_uuid"] == current_uuid else r["entity1_uuid"]
                        if neighbor == target_uuid:
                            found_paths.append(path + [neighbor])
                        elif neighbor not in visited and len(path) < max_depth:
                            visited.add(neighbor)
                            next_queue.append((neighbor, path + [neighbor]))
                if found_paths:
                    break
                queue = next_queue
        finally:
            conn.rollback()
        if not found_paths:
            return {"source_entity": source_entity, "target_entity": target_entity, "path_length": -1,
                    "total_shortest_paths": 0, "paths": []}
        # Build entity path objects
        all_uuids = set()
        for p in found_paths:
            all_uuids.update(p)
        uuid_to_entity = {}
        if all_uuids:
            ent_map = self.get_entities_by_absolute_ids(list(all_uuids))
            for e in ent_map:
                uuid_to_entity[e.absolute_id] = e
        paths_result = []
        for p in found_paths:
            path_entities = [uuid_to_entity[uid] for uid in p if uid in uuid_to_entity]
            paths_result.append({"entities": path_entities, "relations": [], "length": len(path_entities) - 1})
        path_length = paths_result[0]["length"] if paths_result else -1
        return {"source_entity": source_entity, "target_entity": target_entity, "path_length": path_length,
                "total_shortest_paths": len(paths_result), "paths": paths_result}

    def get_entity_neighbors(self, entity_uuid: str, depth: int = 1) -> Dict:
        conn = self._connect()
        try:
            center_row = conn.execute(
                "SELECT uuid, name, family_id FROM entity WHERE uuid = ? AND graph_id = ?",
                (entity_uuid, self._graph_id),
            ).fetchone()
            center_node = None
            if center_row:
                center_node = {"uuid": center_row["uuid"], "name": center_row["name"], "family_id": center_row["family_id"]}
            neighbors = {"entity": center_node, "nodes": [], "edges": []}
            # BFS via relates_to
            visited = {entity_uuid}
            current_frontier = [entity_uuid]
            for _ in range(depth):
                if not current_frontier:
                    break
                ph = ",".join("?" * len(current_frontier))
                edge_rows = conn.execute(
                    f"SELECT entity1_uuid, entity2_uuid, relation_uuid, fact FROM relates_to WHERE (entity1_uuid IN ({ph}) OR entity2_uuid IN ({ph})) AND graph_id = ? LIMIT 500",
                    current_frontier + current_frontier + [self._graph_id],
                ).fetchall()
                next_frontier = []
                seen_nodes = set()
                seen_edges = set()
                for r in edge_rows:
                    e1, e2 = r["entity1_uuid"], r["entity2_uuid"]
                    if (e1, e2) not in seen_edges:
                        seen_edges.add((e1, e2))
                        neighbors["edges"].append({"source_uuid": e1, "target_uuid": e2, "content": r["fact"] or "", "relation_uuid": r["relation_uuid"]})
                    for uid in (e1, e2):
                        if uid not in visited and uid not in seen_nodes:
                            seen_nodes.add(uid)
                            next_frontier.append(uid)
                if next_frontier:
                    ph2 = ",".join("?" * len(next_frontier))
                    node_rows = conn.execute(
                        f"SELECT uuid, name, family_id FROM entity WHERE uuid IN ({ph2}) AND graph_id = ?",
                        next_frontier + [self._graph_id],
                    ).fetchall()
                    for r in node_rows:
                        neighbors["nodes"].append({"uuid": r["uuid"], "name": r["name"], "family_id": r["family_id"]})
                        visited.add(r["uuid"])
                current_frontier = next_frontier
        finally:
            conn.rollback()
        return neighbors

    def merge_entity_families(self, target_family_id: str, source_family_ids: List[str], skip_name_check: bool = False) -> Dict[str, Any]:
        all_ids_to_resolve = [target_family_id] + [s for s in source_family_ids if s]
        resolved_map = self.resolve_family_ids(all_ids_to_resolve)
        target_family_id = resolved_map.get(target_family_id, target_family_id)
        if not target_family_id or not source_family_ids:
            return {"entities_updated": 0, "relations_updated": 0}
        if not skip_name_check:
            resolved_sources = {s: resolved_map.get(s, s) for s in source_family_ids if s}
            unique_fids = list(set([target_family_id] + list(resolved_sources.values())))
            fid_to_entity = {}
            try:
                fid_to_entity = self.get_entities_by_family_ids(unique_fids) or {}
            except Exception:
                pass
            target_entity = fid_to_entity.get(target_family_id) or self.get_entity_by_family_id(target_family_id)
            target_name = target_entity.name if target_entity else ""
            _target_chars = set(target_name) if target_name else set()
            rejected_ids = set()
            for source_id in source_family_ids:
                resolved_source = resolved_sources.get(source_id, source_id)
                if not resolved_source:
                    continue
                source_entity = fid_to_entity.get(resolved_source) or self.get_entity_by_family_id(resolved_source)
                if not source_entity:
                    continue
                source_name = source_entity.name
                if target_name and source_name:
                    _source_chars = set(source_name)
                    shared = len(_source_chars & _target_chars)
                    total = len(_source_chars | _target_chars)
                    overlap = shared / total if total > 0 else 0
                    if overlap < 0.2:
                        logger.warning("Rejecting merge: name difference too large — target=%s(%s) source=%s(%s) overlap=%.2f",
                                       target_name, target_family_id, source_name, resolved_source, overlap)
                        rejected_ids.add(resolved_source)
            if rejected_ids:
                source_family_ids = [s for s in source_family_ids if resolved_sources.get(s, s) not in rejected_ids]
        if not source_family_ids:
            return {"entities_updated": 0, "relations_updated": 0, "rejected": True}
        with self._write_lock:
            conn = self._connect()
            try:
                entities_updated = 0
                canonical_source_ids: List[str] = []
                now_iso = datetime.now().isoformat()
                resolved_sources_in_session = {s: resolved_map.get(s, s) for s in source_family_ids if s}
                for source_id in source_family_ids:
                    source_id = resolved_sources_in_session.get(source_id, source_id)
                    if not source_id or source_id == target_family_id or source_id in canonical_source_ids:
                        continue
                    canonical_source_ids.append(source_id)
                if canonical_source_ids:
                    for sid in canonical_source_ids:
                        cursor = conn.execute(
                            "UPDATE entity SET family_id = ? WHERE family_id = ? AND graph_id = ?",
                            (target_family_id, sid, self._graph_id),
                        )
                        entities_updated += cursor.rowcount
                    for sid in canonical_source_ids:
                        conn.execute(
                            "INSERT OR REPLACE INTO entity_redirect (source_id, target_id, updated_at) VALUES (?, ?, ?)",
                            (sid, target_family_id, now_iso),
                        )
                    conn.commit()
            finally:
                conn.rollback()
        self.invalidate_entity_remap_cache()
        self._invalidate_entity_cache_bulk()
        return {"entities_updated": entities_updated, "relations_updated": 0,
                "target_family_id": target_family_id, "merged_source_ids": canonical_source_ids}
