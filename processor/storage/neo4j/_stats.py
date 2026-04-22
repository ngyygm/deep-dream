"""Neo4j StatsMixin — extracted from neo4j_store."""
import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from ...models import Entity, Relation
from ._helpers import _ENTITY_RETURN_FIELDS, _neo4j_record_to_entity, _neo4j_record_to_relation, _q

logger = logging.getLogger(__name__)


class StatsMixin:
    """Stats operations for Neo4j backend.
    Shared state contract (set by Neo4jStorageManager.__init__):
        self._session()              → Neo4j session factory
        self._run(session, cypher, **kw) → execute Cypher with graph_id injection
        self._graph_id: str          → active graph ID
    """


    def get_changes(self, since: datetime, until: Optional[datetime] = None) -> Dict[str, Any]:
        """获取时间范围内的变更"""
        if until is None:
            until = datetime.now(timezone.utc)
        since_iso = since.isoformat()
        until_iso = until.isoformat()
        with self._session() as session:
            result = self._run(session, f"""
                MATCH (e:Entity)
                WHERE e.event_time >= datetime($since) AND e.event_time <= datetime($until)
                RETURN {_ENTITY_RETURN_FIELDS}
                ORDER BY e.event_time DESC
            """, since=since_iso, until=until_iso)
            entities = [_neo4j_record_to_entity(r) for r in result]

            result = self._run(session, _q("""
                MATCH (r:Relation)
                WHERE r.event_time >= datetime($since) AND r.event_time <= datetime($until)
                RETURN __REL_FIELDS__
                ORDER BY r.event_time DESC
            """), since=since_iso, until=until_iso)
            relations = [_neo4j_record_to_relation(r) for r in result]

        return {"entities": entities, "relations": relations}


    # ------------------------------------------------------------------

    def get_snapshot(self, time_point: datetime, limit: Optional[int] = None) -> Dict[str, Any]:
        """获取指定时间点的实体/关系快照"""
        time_iso = time_point.isoformat()
        with self._session() as session:
            result = self._run(session, f"""
                MATCH (e:Entity)
                WHERE (e.valid_at IS NULL OR e.valid_at <= datetime($time))
                  AND (e.invalid_at IS NULL OR e.invalid_at > datetime($time))
                RETURN {_ENTITY_RETURN_FIELDS}
                ORDER BY e.event_time DESC
                LIMIT $limit
            """, time=time_iso, limit=limit or 10000)
            entities = [_neo4j_record_to_entity(r) for r in result]

            result = self._run(session, _q("""
                MATCH (r:Relation)
                WHERE (r.valid_at IS NULL OR r.valid_at <= datetime($time))
                  AND (r.invalid_at IS NULL OR r.invalid_at > datetime($time))
                RETURN __REL_FIELDS__
                ORDER BY r.event_time DESC
                LIMIT $limit
            """), time=time_iso, limit=limit or 10000)
            relations = [_neo4j_record_to_relation(r) for r in result]

        return {"entities": entities, "relations": relations}

