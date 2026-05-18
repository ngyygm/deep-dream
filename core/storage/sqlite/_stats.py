import logging
from typing import Any, Dict

logger = logging.getLogger(__name__)


class _StatsMixin:

    def get_stats(self) -> Dict[str, Any]:
        try:
            conn = self._connect()
            try:
                ec = conn.execute("SELECT COUNT(DISTINCT family_id) AS cnt FROM entity WHERE graph_id = ?", (self._graph_id,)).fetchone()
                rc = conn.execute("SELECT COUNT(DISTINCT family_id) AS cnt FROM relation WHERE graph_id = ?", (self._graph_id,)).fetchone()
            finally:
                conn.rollback()
            return {"entities": ec["cnt"] if ec else 0, "relations": rc["cnt"] if rc else 0}
        except Exception as e:
            logger.warning("get_stats failed: %s", e)
            return {"entities": 0, "relations": 0}

    def get_graph_version(self) -> dict:
        conn = self._connect()
        try:
            ec = conn.execute("SELECT COUNT(DISTINCT family_id) AS cnt FROM entity WHERE graph_id = ?", (self._graph_id,)).fetchone()
            rc = conn.execute("SELECT COUNT(DISTINCT family_id) AS cnt FROM relation WHERE graph_id = ?", (self._graph_id,)).fetchone()
            et = conn.execute("SELECT MAX(processed_time) AS pt FROM entity WHERE graph_id = ?", (self._graph_id,)).fetchone()
            rt = conn.execute("SELECT MAX(processed_time) AS pt FROM relation WHERE graph_id = ?", (self._graph_id,)).fetchone()
            e_pt = et["pt"] if et and et["pt"] else None
            r_pt = rt["pt"] if rt and rt["pt"] else None
            lm = r_pt if r_pt and (not e_pt or r_pt > e_pt) else e_pt
        finally:
            conn.rollback()
        return {"entity_count": ec["cnt"] if ec else 0, "relation_count": rc["cnt"] if rc else 0, "last_modified": lm}

    def get_graph_statistics(self) -> Dict[str, Any]:
        cached = self._cache.get("graph_stats")
        if cached is not None:
            return cached
        conn = self._connect()
        try:
            total_e = conn.execute("SELECT COUNT(*) AS cnt FROM entity WHERE graph_id = ?", (self._graph_id,)).fetchone()["cnt"]
            total_r = conn.execute("SELECT COUNT(*) AS cnt FROM relation WHERE graph_id = ?", (self._graph_id,)).fetchone()["cnt"]
            valid_e = conn.execute("SELECT COUNT(DISTINCT family_id) AS cnt FROM entity WHERE graph_id = ?", (self._graph_id,)).fetchone()["cnt"]
            valid_r = conn.execute("SELECT COUNT(DISTINCT family_id) AS cnt FROM relation WHERE graph_id = ?", (self._graph_id,)).fetchone()["cnt"]

            stats = {
                "entity_count": valid_e,
                "relation_count": valid_r,
                "total_entity_versions": total_e,
                "total_relation_versions": total_r,
                "avg_relations_per_entity": 0,
                "max_relations_per_entity": 0,
                "isolated_entities": 0,
                "graph_density": 0.0,
            }

            if valid_e > 0:
                # Degree calculation
                degree_rows = conn.execute(
                    "SELECT e.family_id, COUNT(DISTINCT r.uuid) AS degree "
                    "FROM entity e "
                    "LEFT JOIN relation r ON (r.entity1_absolute_id = e.uuid OR r.entity2_absolute_id = e.uuid) AND r.graph_id = ? "
                    "WHERE e.family_id IS NOT NULL AND e.graph_id = ? "
                    "GROUP BY e.family_id",
                    (self._graph_id, self._graph_id),
                ).fetchall()
                degrees = [r["degree"] for r in degree_rows]
                isolated = sum(1 for d in degrees if d == 0)
                stats["avg_relations_per_entity"] = round(sum(degrees) / len(degrees), 2) if degrees else 0
                stats["max_relations_per_entity"] = max(degrees) if degrees else 0
                stats["isolated_entities"] = isolated
                if valid_e > 1:
                    stats["graph_density"] = round(valid_r / (valid_e * (valid_e - 1) / 2), 4)
                else:
                    stats["graph_density"] = 0.0

            # Time trends
            e_trend = conn.execute(
                "SELECT DATE(event_time) AS d, COUNT(DISTINCT family_id) AS cnt FROM entity WHERE event_time IS NOT NULL AND graph_id = ? GROUP BY d ORDER BY d LIMIT 30",
                (self._graph_id,),
            ).fetchall()
            stats["entity_count_over_time"] = [{"date": r["d"], "count": r["cnt"]} for r in e_trend]
            r_trend = conn.execute(
                "SELECT DATE(event_time) AS d, COUNT(DISTINCT family_id) AS cnt FROM relation WHERE event_time IS NOT NULL AND graph_id = ? GROUP BY d ORDER BY d LIMIT 30",
                (self._graph_id,),
            ).fetchall()
            stats["relation_count_over_time"] = [{"date": r["d"], "count": r["cnt"]} for r in r_trend]
        finally:
            conn.rollback()
        self._cache.set("graph_stats", stats, ttl=60)
        return stats

    def get_data_quality_report(self) -> Dict[str, Any]:
        conn = self._connect()
        try:
            valid_families = conn.execute("SELECT COUNT(DISTINCT family_id) AS cnt FROM entity WHERE family_id IS NOT NULL AND graph_id = ?", (self._graph_id,)).fetchone()["cnt"]
            valid_nodes = conn.execute("SELECT COUNT(*) AS cnt FROM entity WHERE graph_id = ?", (self._graph_id,)).fetchone()["cnt"]
            no_fid = conn.execute("SELECT COUNT(*) AS cnt FROM entity WHERE family_id IS NULL AND graph_id = ?", (self._graph_id,)).fetchone()["cnt"]
            valid_r_families = conn.execute("SELECT COUNT(DISTINCT family_id) AS cnt FROM relation WHERE graph_id = ?", (self._graph_id,)).fetchone()["cnt"]
            valid_r_nodes = conn.execute("SELECT COUNT(*) AS cnt FROM relation WHERE graph_id = ?", (self._graph_id,)).fetchone()["cnt"]
            # Compute old versions count (all versions minus latest per family)
            inv_e = conn.execute(
                "SELECT COUNT(*) AS cnt FROM entity WHERE graph_id = ? "
                "AND uuid NOT IN (SELECT e2.uuid FROM entity e2 WHERE e2.family_id = entity.family_id AND e2.graph_id = entity.graph_id ORDER BY e2.version_seq DESC LIMIT 1)",
                (self._graph_id,),
            ).fetchone()["cnt"]
            inv_r = conn.execute(
                "SELECT COUNT(*) AS cnt FROM relation WHERE graph_id = ? "
                "AND uuid NOT IN (SELECT r2.uuid FROM relation r2 WHERE r2.family_id = relation.family_id AND r2.graph_id = relation.graph_id ORDER BY r2.version_seq DESC LIMIT 1)",
                (self._graph_id,),
            ).fetchone()["cnt"]
        finally:
            conn.rollback()
        isolated = self.count_isolated_entities()
        return {
            "entities": {"valid_unique": valid_families, "valid_versions": valid_nodes, "old_versions": inv_e, "no_family_id": no_fid, "isolated": isolated},
            "relations": {"valid_unique": valid_r_families, "valid_versions": valid_r_nodes, "old_versions": inv_r},
            "total_nodes": valid_nodes + inv_e + valid_r_nodes + inv_r + no_fid,
        }
