"""V1.5 data integrity validation.

Each validate_* function returns a list of violations:
    [{"table": ..., "id": ..., "issue": ..., "detail": ...}, ...]
"""

import hashlib
import logging
import os
from typing import Callable

logger = logging.getLogger(__name__)

Violation = dict


def _violations() -> list:
    return []


def _add(violations: list, table: str, pk: str, issue: str,
         detail: str = "") -> None:
    violations.append({
        "table": table, "id": pk, "issue": issue, "detail": detail,
    })


def validate_document_current_version(conn) -> list:
    """current_version_id must point to this doc's active version."""
    v = _violations()
    rows = conn.execute("""
        SELECT d.document_id, d.current_version_id
        FROM documents d
        WHERE d.current_version_id IS NOT NULL
          AND d.status = 'active'
    """).fetchall()
    for doc_id, ver_id in rows:
        row = conn.execute(
            "SELECT document_id, status FROM document_versions "
            "WHERE document_version_id = ?",
            (ver_id,),
        ).fetchone()
        if not row:
            _add(v, "documents", doc_id, "current_version_id dangling",
                 f"version {ver_id} not found")
        elif row[0] != doc_id:
            _add(v, "documents", doc_id, "current_version_id wrong document",
                 f"version {ver_id} belongs to doc {row[0]}")
        elif row[1] != "active":
            _add(v, "documents", doc_id, "current_version_id not active",
                 f"version {ver_id} has status {row[1]}")
    return v


def validate_episode_document_version(conn) -> list:
    """episode.document_id must match version.document_id."""
    v = _violations()
    rows = conn.execute("""
        SELECT e.episode_id, e.document_id, e.document_version_id,
               dv.document_id AS ver_doc_id
        FROM episodes e
        JOIN document_versions dv
          ON dv.document_version_id = e.document_version_id
        WHERE e.document_id != dv.document_id
    """).fetchall()
    for ep_id, ep_doc, ver_id, ver_doc in rows:
        _add(v, "episodes", ep_id, "document_id mismatch",
             f"episode.doc={ep_doc} version.doc={ver_doc}")
    return v


def validate_entity_mentions_cache(conn) -> list:
    """Mention cache fields (entity_family_id, episode_id) must match observation."""
    v = _violations()
    rows = conn.execute("""
        SELECT em.mention_id, em.entity_family_id, em.episode_id,
               eo.entity_family_id AS obs_family, eo.episode_id AS obs_ep
        FROM entity_mentions em
        JOIN entity_observations eo ON eo.entity_id = em.entity_id
        WHERE em.entity_family_id != eo.entity_family_id
           OR em.episode_id != eo.episode_id
    """).fetchall()
    for mid, fam, ep, obs_fam, obs_ep in rows:
        _add(v, "entity_mentions", mid, "cache mismatch",
             f"fam={fam} vs obs_fam={obs_fam}, ep={ep} vs obs_ep={obs_ep}")
    return v


def validate_relation_assertions_cache(conn) -> list:
    """Assertion cache family_ids must match observations."""
    v = _violations()
    rows = conn.execute("""
        SELECT ra.relation_id,
               ra.subject_entity_family_id,
               ra.object_entity_family_id,
               s.entity_family_id AS sub_fam,
               o.entity_family_id AS obj_fam
        FROM relation_assertions ra
        JOIN entity_observations s ON s.entity_id = ra.subject_entity_id
        JOIN entity_observations o ON o.entity_id = ra.object_entity_id
        WHERE ra.subject_entity_family_id != s.entity_family_id
           OR ra.object_entity_family_id != o.entity_family_id
    """).fetchall()
    for rid, sub_fam, obj_fam, obs_sub, obs_obj in rows:
        _add(v, "relation_assertions", rid, "cache mismatch",
             f"sub={sub_fam} vs {obs_sub}, obj={obj_fam} vs {obs_obj}")
    return v


def validate_relation_same_episode(conn) -> list:
    """Subject and object observations must belong to the same episode as the assertion."""
    v = _violations()
    rows = conn.execute("""
        SELECT ra.relation_id, ra.episode_id,
               s.episode_id AS sub_ep, o.episode_id AS obj_ep
        FROM relation_assertions ra
        JOIN entity_observations s ON s.entity_id = ra.subject_entity_id
        JOIN entity_observations o ON o.entity_id = ra.object_entity_id
        WHERE s.episode_id != ra.episode_id
           OR o.episode_id != ra.episode_id
    """).fetchall()
    for rid, ep, sub_ep, obj_ep in rows:
        _add(v, "relation_assertions", rid, "same-episode violation",
             f"assertion_ep={ep} sub_ep={sub_ep} obj_ep={obj_ep}")
    return v


def validate_embeddings_owners(conn) -> list:
    """Embeddings must have existing owners. Non-null run_id must exist in pipeline_runs."""
    v = _violations()
    owner_tables = {
        "episode": "episodes",
        "entity_obs": "entity_observations",
        "relation_assert": "relation_assertions",
        "entity_family": "entity_families",
        "document_version": "document_versions",
    }
    pk_cols = {
        "episode": "episode_id",
        "entity_obs": "entity_id",
        "relation_assert": "relation_id",
        "entity_family": "entity_family_id",
        "document_version": "document_version_id",
    }
    for otype, table in owner_tables.items():
        pk_col = pk_cols[otype]
        orphans = conn.execute(f"""
            SELECT e.embedding_id, e.owner_id
            FROM embeddings e
            LEFT JOIN {table} t ON t.{pk_col} = e.owner_id
            WHERE e.owner_type = ? AND t.{pk_col} IS NULL
        """, (otype,)).fetchall()
        for eid, oid in orphans:
            _add(v, "embeddings", eid, "orphan owner",
                 f"{otype}/{oid} not in {table}")

    bad_runs = conn.execute("""
        SELECT e.embedding_id, e.run_id
        FROM embeddings e
        WHERE e.run_id IS NOT NULL
          AND e.run_id NOT IN (SELECT run_id FROM pipeline_runs)
    """).fetchall()
    for eid, rid in bad_runs:
        _add(v, "embeddings", eid, "run_id not in pipeline_runs",
             f"run_id={rid}")
    return v


def validate_document_links_episode(conn) -> list:
    """from_episode must belong to from_document/version."""
    v = _violations()
    rows = conn.execute("""
        SELECT dl.link_id, dl.from_episode_id
        FROM document_links dl
        WHERE dl.from_episode_id IS NOT NULL
          AND dl.from_episode_id NOT IN (
              SELECT episode_id FROM episodes
              WHERE document_id = dl.from_document_id
                AND document_version_id = dl.from_document_version_id
          )
    """).fetchall()
    for lid, epid in rows:
        _add(v, "document_links", lid, "from_episode not in document/version",
             f"episode={epid}")
    return v


def validate_mention_offsets(conn) -> list:
    """0 <= start <= end <= source_text length."""
    v = _violations()
    rows = conn.execute("""
        SELECT em.mention_id, em.start_offset, em.end_offset,
               length(e.source_text) AS src_len
        FROM entity_mentions em
        JOIN episodes e ON e.episode_id = em.episode_id
        WHERE em.start_offset < 0
           OR em.end_offset < em.start_offset
           OR em.end_offset > length(e.source_text)
    """).fetchall()
    for mid, start, end, src_len in rows:
        _add(v, "entity_mentions", mid, "offset out of range",
             f"start={start} end={end} src_len={src_len}")
    return v


def validate_relation_evidence_offsets(conn) -> list:
    v = _violations()
    rows = conn.execute("""
        SELECT ra.relation_id, ra.evidence_start_offset,
               ra.evidence_end_offset, length(e.source_text) AS src_len
        FROM relation_assertions ra
        JOIN episodes e ON e.episode_id = ra.episode_id
        WHERE ra.evidence_start_offset < 0
           OR ra.evidence_end_offset < ra.evidence_start_offset
           OR ra.evidence_end_offset > length(e.source_text)
    """).fetchall()
    for rid, start, end, src_len in rows:
        _add(v, "relation_assertions", rid, "evidence offset out of range",
             f"start={start} end={end} src_len={src_len}")
    return v


def validate_fts_consistency(conn) -> list:
    """FTS rows should match active episodes."""
    v = _violations()
    missing = conn.execute("""
        SELECT e.episode_id
        FROM episodes e
        WHERE e.status = 'active'
          AND e.episode_id NOT IN (SELECT episode_id FROM episodes_fts)
    """).fetchall()
    for (epid,) in missing:
        _add(v, "episodes_fts", epid, "active episode missing from FTS")

    stale = conn.execute("""
        SELECT fts.episode_id
        FROM episodes_fts fts
        LEFT JOIN episodes e ON e.episode_id = fts.episode_id
        WHERE e.status != 'active' OR e.episode_id IS NULL
    """).fetchall()
    for (epid,) in stale:
        _add(v, "episodes_fts", epid, "FTS row for non-active episode")
    return v


def validate_content_files_exist(conn, library_path: str = "") -> list:
    """version_content_path and managed_path must exist on disk."""
    v = _violations()
    if not library_path:
        return v
    rows = conn.execute("""
        SELECT document_version_id, version_content_path
        FROM document_versions
        WHERE version_content_path IS NOT NULL
          AND version_content_path != ''
    """).fetchall()
    for vid, path in rows:
        full = os.path.join(library_path, path) if library_path else path
        if not os.path.exists(full):
            _add(v, "document_versions", vid, "version_content_path missing",
                 path)
    rows = conn.execute("""
        SELECT document_id, managed_path
        FROM documents
        WHERE managed_path IS NOT NULL AND managed_path != ''
    """).fetchall()
    for did, path in rows:
        full = os.path.join(library_path, path) if library_path else path
        if not os.path.exists(full):
            _add(v, "documents", did, "managed_path missing", path)
    return v


def _compute_normalized_hash(file_path: str) -> str:
    with open(file_path, "r", encoding="utf-8") as f:
        text = f.read()
    normalized = text.replace("\r\n", "\n").replace("\r", "\n")
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def validate_content_hash_matches_file(conn, library_path: str = "") -> list:
    """File content hash must match content_hash in DB."""
    v = _violations()
    if not library_path:
        return v
    rows = conn.execute("""
        SELECT document_version_id, version_content_path, content_hash
        FROM document_versions
        WHERE version_content_path IS NOT NULL
          AND version_content_path != ''
    """).fetchall()
    for vid, path, expected_hash in rows:
        full = os.path.join(library_path, path)
        if not os.path.exists(full):
            continue
        try:
            actual_hash = _compute_normalized_hash(full)
            if actual_hash != expected_hash:
                _add(v, "document_versions", vid, "content_hash mismatch",
                     f"db={expected_hash[:16]}.. file={actual_hash[:16]}..")
        except Exception as exc:
            _add(v, "document_versions", vid, "hash computation failed",
                 str(exc))
    return v


def validate_pipeline_runs_version(conn) -> list:
    """Version-specific run types must have document_version_id."""
    v = _violations()
    version_required = {"remember", "chunk", "embedding"}
    rows = conn.execute("""
        SELECT run_id, run_type, document_id, document_version_id
        FROM pipeline_runs
        WHERE run_type IN ('remember', 'chunk', 'embedding')
          AND (document_id IS NOT NULL AND document_version_id IS NULL)
    """).fetchall()
    for rid, rtype, did, dvid in rows:
        _add(v, "pipeline_runs", rid,
             "version-specific run missing document_version_id",
             f"type={rtype} doc={did}")
    return v


def validate_managed_only_source_mode(conn) -> list:
    """V1.5 library should only have managed source_mode documents."""
    v = _violations()
    rows = conn.execute("""
        SELECT document_id, source_mode
        FROM documents
        WHERE source_mode != 'managed'
    """).fetchall()
    for did, mode in rows:
        _add(v, "documents", did, "non-managed source_mode", mode)
    return v


# ── Aggregate ──────────────────────────────────────────────

ALL_VALIDATORS: list[Callable] = [
    validate_document_current_version,
    validate_episode_document_version,
    validate_entity_mentions_cache,
    validate_relation_assertions_cache,
    validate_relation_same_episode,
    validate_embeddings_owners,
    validate_document_links_episode,
    validate_mention_offsets,
    validate_relation_evidence_offsets,
    validate_fts_consistency,
    validate_pipeline_runs_version,
    validate_managed_only_source_mode,
    # validate_content_files_exist and validate_content_hash_matches_file
    # require library_path, called separately
]


def validate_all(conn, library_path: str = "",
                 include_file_checks: bool = True) -> list:
    """Run all validators. Returns flat list of violations."""
    violations = []
    for validator in ALL_VALIDATORS:
        try:
            violations.extend(validator(conn))
        except Exception as exc:
            logger.error("validator %s failed: %s", validator.__name__, exc)
            violations.append({
                "table": "", "id": "", "issue": f"validator error: {validator.__name__}",
                "detail": str(exc),
            })

    if include_file_checks and library_path:
        violations.extend(validate_content_files_exist(conn, library_path))
        violations.extend(validate_content_hash_matches_file(conn, library_path))

    return violations
