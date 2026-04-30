"""Neo4j Neo4jBaseMixin — extracted from neo4j_store."""
import json
import logging
import time
from datetime import datetime, timezone
from functools import lru_cache
from typing import Any, Dict, List, Optional, Set, Tuple

from ...models import Entity
from ...perf import _perf_timer

logger = logging.getLogger(__name__)

# Transient Neo4j error keywords that justify retry
_NEO4J_TRANSIENT_ERRORS = frozenset((
    "connection refused", "connectionerror",
    "failed to establish a new connection", "newconnectionerror",
    "temporarily unreachable", "temporary failure in name resolution",
    "name or service not known", "connection aborted",
    "connection reset", "errno 111",
    "server unavailable", "database unavailable",
    "no write leader", "leader not available",
    "transient error", "try again later",
))

_MAX_RETRIES = 3
_RETRY_BASE_DELAY = 1.0  # seconds
_MAX_RETRY_DELAY = 30.0


def _is_transient_neo4j_error(exc: BaseException) -> bool:
    """Check if exception indicates a transient Neo4j/connection error."""
    error_str = str(exc).lower()
    return any(keyword in error_str for keyword in _NEO4J_TRANSIENT_ERRORS)


class Neo4jBaseMixin:
    """Neo4jBase operations for Neo4j backend.
    Shared state contract (set by Neo4jStorageManager.__init__):
        self._session()              → Neo4j session factory
        self._run(session, cypher, **kw) → execute Cypher with graph_id injection
        self._graph_id: str          → active graph ID
    """

    def _run_with_retry(self, session, cypher: str, operation_name: str = "neo4j_query", **kwargs) -> Any:
        """Execute Cypher with retry logic for transient connection errors.

        Args:
            session: Neo4j session
            cypher: Cypher query string
            operation_name: Name of the operation for retry tracking (e.g., "bulk_save_entities")
            **kwargs: Query parameters

        Returns:
            Query result

        Raises:
            Original exception after max retries
        """
        last_error = None
        for attempt in range(_MAX_RETRIES):
            try:
                return self._run(session, cypher, **kwargs)
            except Exception as e:
                last_error = e
                if not _is_transient_neo4j_error(e):
                    # Non-transient error, raise immediately
                    raise
                # Track retry for monitoring
                if hasattr(self, '_system_monitor') and self._system_monitor:
                    self._system_monitor.retry_counter.increment("neo4j", operation_name)
                if attempt < _MAX_RETRIES - 1:
                    delay = min(_RETRY_BASE_DELAY * (2 ** attempt), _MAX_RETRY_DELAY)
                    logger.warning(
                        "Neo4j transient error (attempt %d/%d): %s. "
                        "Retrying in %.1fs...",
                        attempt + 1, _MAX_RETRIES, e, delay
                    )
                    time.sleep(delay)
                else:
                    logger.error(
                        "Neo4j transient error: max retries (%d) reached: %s",
                        _MAX_RETRIES, e
                    )
        raise last_error

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
        # Fast path: string check avoids json.loads for the vast majority of relations
        if isinstance(relation.attributes, str) and ('"candidate"' not in relation.attributes or '"tier"' not in relation.attributes):
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


    @staticmethod
    def _tp_to_datetime(tp):
        """将 time_point 字符串转换为 Python datetime 对象，供 Neo4j DateTime 字段比较。"""
        return _cached_tp_to_datetime(tp)

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
                target_result = self._run_with_retry(session,
                    "MATCH (e:Entity {family_id: $fid}) "
                    "RETURN e.uuid AS uuid ORDER BY e.processed_time DESC LIMIT 1",
                    fid=new_family_id,
                )
                target_record = target_result.single()
                if not target_record:
                    return 0
                new_abs_id = target_record["uuid"]

                # Get all absolute_ids for the old family_id
                old_abs_result = self._run_with_retry(session,
                    "MATCH (e:Entity {family_id: $fid}) RETURN e.uuid AS uuid",
                    fid=old_family_id,
                )
                old_abs_ids = [r["uuid"] for r in old_abs_result]
                if not old_abs_ids:
                    return 0

                # Update entity1_absolute_id in matching relations
                upd1 = self._run_with_retry(session,
                    "MATCH (r:Relation) WHERE r.entity1_absolute_id IN $old_ids "
                    "SET r.entity1_absolute_id = $new_id "
                    "RETURN count(r) AS cnt",
                    old_ids=old_abs_ids,
                    new_id=new_abs_id,
                )
                cnt1 = upd1.single()
                cnt1 = cnt1["cnt"] if cnt1 else 0

                # Update entity2_absolute_id in matching relations
                upd2 = self._run_with_retry(session,
                    "MATCH (r:Relation) WHERE r.entity2_absolute_id IN $old_ids "
                    "SET r.entity2_absolute_id = $new_id "
                    "RETURN count(r) AS cnt",
                    old_ids=old_abs_ids,
                    new_id=new_abs_id,
                )
                cnt2 = upd2.single()
                cnt2 = cnt2["cnt"] if cnt2 else 0

                # Also fix RELATES_TO edges (graph traversal edges)
                # Batch: redirect all old → new in two queries (not 2*N)
                if old_abs_ids:
                    # Outgoing: old → target  →  new → target
                    self._run_with_retry(session,
                        "UNWIND $old_ids AS old_uuid "
                        "MATCH (old:Entity {uuid: old_uuid})-[r:RELATES_TO]->(target) "
                        "WITH DISTINCT target, collect(r) AS rels "
                        "MATCH (new:Entity {uuid: $new_uuid}) "
                        "MERGE (new)-[:RELATES_TO]->(target) "
                        "WITH rels "
                        "UNWIND rels AS r DELETE r",
                        old_ids=old_abs_ids,
                        new_uuid=new_abs_id,
                    )
                    # Incoming: source → old  →  source → new
                    self._run_with_retry(session,
                        "UNWIND $old_ids AS old_uuid "
                        "MATCH (source)-[r:RELATES_TO]->(old:Entity {uuid: old_uuid}) "
                        "WITH DISTINCT source, collect(r) AS rels "
                        "MATCH (new:Entity {uuid: $new_uuid}) "
                        "MERGE (source)-[:RELATES_TO]->(new) "
                        "WITH rels "
                        "UNWIND rels AS r DELETE r",
                        old_ids=old_abs_ids,
                        new_uuid=new_abs_id,
                    )

                self._cache.invalidate_keys(["graph_stats"])
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

    def register_entity_redirects_batch(self, pairs: List[Tuple[str, str]]) -> None:
        """批量登记重定向映射，单次 session + UNWIND。"""
        if not pairs:
            return
        now_iso = datetime.now().isoformat()
        with self._write_lock:
            with self._session() as session:
                # Resolve all targets first via 2-hop path
                all_ids = list({
                    stripped
                    for pair in pairs
                    for val in pair
                    if val and (stripped := val.strip())
                })
                if all_ids:
                    resolve_cypher = """
                    UNWIND $ids AS sid
                    OPTIONAL MATCH (r1:EntityRedirect {source_id: sid})
                    WITH sid, r1,
                         CASE WHEN r1 IS NOT NULL THEN r1.target_id ELSE sid END AS hop1
                    OPTIONAL MATCH (r2:EntityRedirect {source_id: hop1})
                    RETURN sid AS source,
                           CASE WHEN r2 IS NOT NULL THEN r2.target_id
                                WHEN r1 IS NOT NULL THEN r1.target_id
                                ELSE sid END AS resolved
                    """
                    records = session.run(resolve_cypher, ids=all_ids)
                    resolved_map = {r["source"]: r["resolved"] for r in records}

                # Build filtered redirect rows
                rows = []
                for source_id, target_id in pairs:
                    source_id = (source_id or "").strip()
                    target_id = (target_id or "").strip()
                    if not source_id or not target_id or source_id == target_id:
                        continue
                    canonical_target = resolved_map.get(target_id, target_id)
                    canonical_source = resolved_map.get(source_id, source_id)
                    if canonical_source == canonical_target:
                        continue
                    rows.append({"sid": source_id, "tid": canonical_target})

                if rows:
                    session.run(
                        """
                        UNWIND $rows AS row
                        MERGE (red:EntityRedirect {source_id: row.sid})
                        SET red.target_id = row.tid, red.updated_at = $now
                        """,
                        rows=rows,
                        now=now_iso,
                    )



    def dedup_merge_batch(self, pairs: List[Tuple[str, str]]) -> int:
        """Batch cross-window dedup: redirect + delete + register for multiple merge pairs.

        Each pair is (old_family_id, new_family_id). All operations execute in a single
        Neo4j session, reducing N*3 sessions to 1 session for N pairs.

        Returns total number of entity versions deleted.
        """
        if not pairs:
            return 0

        # Resolve all redirects first (batch)
        all_ids = list(set(
            fid for pair in pairs for fid in pair if fid and fid.strip()
        ))
        resolved = self.resolve_family_ids(all_ids) if all_ids else {}

        # Build resolved pairs, deduplicating and skipping invalid
        resolved_pairs: List[Tuple[str, str]] = []
        for old_fid, new_fid in pairs:
            old_fid = (old_fid or "").strip()
            new_fid = (new_fid or "").strip()
            if not old_fid or not new_fid:
                continue
            old_r = resolved.get(old_fid, old_fid)
            new_r = resolved.get(new_fid, new_fid)
            if old_r == new_r:
                continue
            resolved_pairs.append((old_r, new_r))

        if not resolved_pairs:
            return 0

        total_deleted = 0
        all_abs_ids_to_clean: List[str] = []
        now_iso = datetime.now().isoformat()

        with self._write_lock:
            with self._session() as session:
                # Step 1: Batch-fetch latest absolute_ids for all target (new) fids
                new_fids_unique = list({nf for _, nf in resolved_pairs})
                if new_fids_unique:
                    target_map_result = self._run(session,
                        "UNWIND $fids AS fid "
                        "MATCH (e:Entity {family_id: fid}) "
                        "WITH fid, e.uuid AS uuid ORDER BY e.processed_time DESC "
                        "WITH fid, collect(uuid)[0] AS latest_uuid "
                        "RETURN fid, latest_uuid",
                        fids=new_fids_unique,
                    )
                    target_map = {r["fid"]: r["latest_uuid"] for r in target_map_result}
                else:
                    target_map = {}

                # Step 2: Batch-fetch old absolute_ids for all old fids
                old_fids_unique = list({of for of, _ in resolved_pairs})
                if old_fids_unique:
                    old_abs_result = self._run(session,
                        "UNWIND $fids AS fid "
                        "MATCH (e:Entity {family_id: fid}) "
                        "RETURN fid, collect(e.uuid) AS uuids",
                        fids=old_fids_unique,
                    )
                    old_abs_map = {r["fid"]: r["uuids"] for r in old_abs_result}
                else:
                    old_abs_map = {}

                # Step 3: Batch redirect Relation references + RELATES_TO edges
                redirect_pairs = []
                for old_fid, new_fid in resolved_pairs:
                    new_abs_id = target_map.get(new_fid)
                    old_abs_ids = old_abs_map.get(old_fid, [])
                    if not new_abs_id or not old_abs_ids:
                        continue
                    for oid in old_abs_ids:
                        redirect_pairs.append({"old": oid, "new": new_abs_id})
                    all_abs_ids_to_clean.extend(old_abs_ids)

                if redirect_pairs:
                    # Batch Relation entity1_absolute_id redirects
                    self._run(session,
                        "UNWIND $pairs AS p "
                        "MATCH (r:Relation) WHERE r.entity1_absolute_id = p.old "
                        "SET r.entity1_absolute_id = p.new",
                        pairs=redirect_pairs,
                    )
                    # Batch Relation entity2_absolute_id redirects
                    self._run(session,
                        "UNWIND $pairs AS p "
                        "MATCH (r:Relation) WHERE r.entity2_absolute_id = p.old "
                        "SET r.entity2_absolute_id = p.new",
                        pairs=redirect_pairs,
                    )
                    # Batch outgoing RELATES_TO redirects
                    self._run(session,
                        "UNWIND $pairs AS p "
                        "MATCH (old:Entity {uuid: p.old})-[r:RELATES_TO]->(target) "
                        "WITH DISTINCT target, p.new AS new_uuid, collect(r) AS rels "
                        "MATCH (new:Entity {uuid: new_uuid}) "
                        "MERGE (new)-[:RELATES_TO]->(target) "
                        "WITH rels UNWIND rels AS r DELETE r",
                        pairs=redirect_pairs,
                    )
                    # Batch incoming RELATES_TO redirects
                    self._run(session,
                        "UNWIND $pairs AS p "
                        "MATCH (source)-[r:RELATES_TO]->(old:Entity {uuid: p.old}) "
                        "WITH DISTINCT source, p.new AS new_uuid, collect(r) AS rels "
                        "MATCH (new:Entity {uuid: new_uuid}) "
                        "MERGE (source)-[:RELATES_TO]->(new) "
                        "WITH rels UNWIND rels AS r DELETE r",
                        pairs=redirect_pairs,
                    )

                # Step 4: Batch delete old entity nodes
                if old_fids_unique:
                    self._run(session,
                        "UNWIND $fids AS fid "
                        "MATCH (e:Entity {family_id: fid})-[r:RELATES_TO]-() DETACH DELETE r",
                        fids=old_fids_unique,
                    )
                    del_result = self._run(session,
                        "UNWIND $fids AS fid "
                        "MATCH (e:Entity {family_id: fid}) DETACH DELETE e RETURN count(e) AS cnt",
                        fids=old_fids_unique,
                    )
                    for r in del_result:
                        total_deleted += r["cnt"]

                # Step 5: Batch register redirects
                redirect_rows = [{"sid": of, "tid": nf} for of, nf in resolved_pairs]
                if redirect_rows:
                    self._run(session,
                        "UNWIND $rows AS row "
                        "MERGE (red:EntityRedirect {source_id: row.sid}) "
                        "SET red.target_id = row.tid, red.updated_at = $now",
                        rows=redirect_rows, now=now_iso,
                        graph_id_safe=False,
                    )

                # Invalidate caches
                cache_keys = [f"resolve:{of}" for of, _ in resolved_pairs]
                self._cache.invalidate_keys(cache_keys)
                for old_fid, _ in resolved_pairs:
                    self._invalidate_entity_cache(old_fid)

        self._cache.invalidate("sim_search:")
        self._cache.invalidate_keys(["graph_stats"])
        return total_deleted


    def resolve_family_id(self, family_id: str) -> str:
        """解析 family_id 到 canonical id。"""
        cache_key = f"resolve:{family_id}"
        cached = self._cache.get(cache_key)
        if cached is not None:
            return cached
        with _perf_timer("resolve_family_id"):
            with self._session() as session:
                resolved = self._resolve_family_id_in_session(session, family_id)
        # 2026-04-26: Increased TTL from 120s to 600s for long-running pipeline operations
        # Family ID redirects are stable within a single remember() call, so longer cache reduces DB round-trips
        self._cache.set(cache_key, resolved, ttl=600)
        return resolved



    def resolve_family_ids(self, family_ids: List[str]) -> Dict[str, str]:
        """批量解析 family_id 到 canonical id。利用缓存 + 一次 Cypher 查询未缓存项。

        Returns:
            {原始 family_id: canonical family_id} 映射
        """
        if not family_ids:
            return {}
        unique_ids = list({_f for fid in family_ids if fid and (_f := fid.strip())})
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

        # 第二步：单次 Cypher 查询解析最多 2 跳重定向链
        if uncached:
            with _perf_timer("resolve_family_ids_batch"):
                with self._session() as session:
                    # 使用 optional 2-hop path 一次查出完整链
                    cypher = """
                    UNWIND $ids AS sid
                    OPTIONAL MATCH (r1:EntityRedirect {source_id: sid})
                    WITH sid, r1,
                         CASE WHEN r1 IS NOT NULL THEN r1.target_id ELSE sid END AS hop1
                    OPTIONAL MATCH (r2:EntityRedirect {source_id: hop1})
                    RETURN sid AS source,
                           CASE WHEN r2 IS NOT NULL THEN r2.target_id
                                WHEN r1 IS NOT NULL THEN r1.target_id
                                ELSE sid END AS target
                    """
                    records = session.run(cypher, ids=uncached)
                    for r in records:
                        resolved = r["target"]
                        result[r["source"]] = resolved
                        self._cache.set(f"resolve:{r['source']}", resolved, ttl=600)

        # 构建输出映射（处理可能有重复的 family_ids）
        output: Dict[str, str] = {}
        for fid in family_ids:
            key = fid.strip() if fid else ""
            output[fid] = result.get(key, key)
        return output


@lru_cache(maxsize=256)
def _cached_tp_to_datetime(tp):
    """Cached version — tp is always a str, None, or datetime (all hashable)."""
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

