"""Migrate old v1 schema (concept_family/version/edge) to V1.5 schema.

Usage:
    python -m core.storage.sqlite.migrate_v1_to_v15 --source <old.db> --target <library_path>
"""
from __future__ import annotations

import json
import logging
import shutil
import sqlite3
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from . import content_fs
from .repositories import (
    documents as doc_repo,
    episodes as ep_repo,
    entities as ent_repo,
    relations as rel_repo,
    embeddings as emb_repo,
    pipeline as pipe_repo,
)
from .schema_v15 import init_schema_v15

logger = logging.getLogger(__name__)


def _now_str() -> str:
    return datetime.now(timezone.utc).isoformat()


def migrate(old_db_path: str, library_path: str, *,
            copy_files: bool = True, dry_run: bool = False) -> dict:
    """Migrate old v1 schema database to V1.5 schema.

    Returns a dict with migration statistics.
    """
    old_path = Path(old_db_path)
    lib_path = Path(library_path)
    lib_path.mkdir(parents=True, exist_ok=True)

    stats = {
        "source": str(old_path),
        "target": str(lib_path),
        "documents": 0,
        "document_versions": 0,
        "episodes": 0,
        "entities": 0,
        "entity_observations": 0,
        "relations": 0,
        "relation_assertions": 0,
        "mentions": 0,
        "document_links": 0,
        "redirects": 0,
        "embeddings": 0,
        "errors": [],
    }

    # Open old DB read-only
    old_conn = sqlite3.connect(f"file:{old_path}?mode=ro", uri=True)
    old_conn.row_factory = sqlite3.Row

    # Create new V1.5 DB
    new_db_path = lib_path / "library.db"
    if new_db_path.exists() and not dry_run:
        # Backup existing
        backup = lib_path / f"library.db.pre_migration_{_now_str().replace(':', '').replace('-', '').replace('.', '')}"
        shutil.copy2(new_db_path, backup)
        new_db_path.unlink()

    new_conn = sqlite3.connect(str(new_db_path))
    new_conn.row_factory = sqlite3.Row
    init_schema_v15(new_conn)
    # Disable FK for bulk insert, re-enable after
    new_conn.execute("PRAGMA foreign_keys = OFF")

    try:
        _migrate_documents(old_conn, new_conn, lib_path, stats, copy_files, dry_run)
        _migrate_episodes(old_conn, new_conn, stats, dry_run)
        _migrate_entities(old_conn, new_conn, stats, dry_run)
        _migrate_relations(old_conn, new_conn, stats, dry_run)
        _migrate_mentions(old_conn, new_conn, stats, dry_run)
        _migrate_document_links(old_conn, new_conn, stats, dry_run)
        _migrate_redirects(old_conn, new_conn, stats, dry_run)
        _migrate_embeddings(old_conn, new_conn, stats, dry_run)

        if not dry_run:
            # Rebuild FTS
            ep_count = ep_repo.rebuild_fts_all(new_conn)
            logger.info("FTS rebuilt: %d episodes", ep_count)
            new_conn.commit()
            # Re-enable FK and verify
            new_conn.execute("PRAGMA foreign_keys = ON")
            new_conn.execute("PRAGMA foreign_key_check")
            new_conn.commit()
    except Exception as e:
        stats["errors"].append(str(e))
        logger.error("Migration failed: %s", e)
        raise
    finally:
        old_conn.close()
        new_conn.close()

    logger.info("Migration complete: %s", json.dumps(stats, indent=2))
    return stats


def _migrate_documents(old_conn, new_conn, lib_path, stats, copy_files, dry_run):
    """Migrate document_family + document_source → documents, document_versions."""
    # Map old family_id → new document_id
    doc_families = old_conn.execute(
        "SELECT cf.family_id, cf.canonical_name, cf.status, cf.created_at, cf.updated_at "
        "FROM concept_family cf WHERE cf.role = 'document'"
    ).fetchall()

    for fam in doc_families:
        fam = dict(fam)
        doc_id = fam["family_id"]

        # Get document_source
        ds = old_conn.execute(
            "SELECT * FROM document_source WHERE document_family_id = ?",
            (doc_id,),
        ).fetchone()
        ds = dict(ds) if ds else {}

        if dry_run:
            stats["documents"] += 1
            continue

        doc_repo.insert_document(
            new_conn, doc_id,
            title=ds.get("title", "") or fam.get("canonical_name", ""),
            managed_path=ds.get("managed_path", ""),
            source_mode=ds.get("source_mode", "managed") or "managed",
            created_at=fam.get("created_at", _now_str()),
            updated_at=fam.get("updated_at", _now_str()),
        )
        stats["documents"] += 1

        # Migrate document_versions
        versions = old_conn.execute(
            "SELECT * FROM document_version WHERE document_family_id = ? ORDER BY processed_time",
            (doc_id,),
        ).fetchall()
        for ver in versions:
            ver = dict(ver)
            ver_id = ver["document_version_id"]
            content_hash = ver.get("content_hash", "")

            # Copy blob to content/versions/
            if copy_files and content_hash:
                blob_path = ver.get("blob_path", "")
                if blob_path:
                    src = Path(blob_path)
                    if src.exists():
                        dest_dir = lib_path / "content" / "versions" / doc_id
                        dest_dir.mkdir(parents=True, exist_ok=True)
                        dest = dest_dir / f"{content_hash}.md"
                        if not dest.exists():
                            shutil.copy2(src, dest)

            # Read title from metadata if available
            metadata = json.loads(ver.get("metadata", "{}") or "{}")
            doc_ver_title = ver.get("title", "") or fam.get("canonical_name", "")

            doc_repo.insert_document_version(
                new_conn, ver_id, doc_id, content_hash,
                version_content_path=f"content/versions/{doc_id}/{content_hash}.md" if content_hash else "",
                title=doc_ver_title,
                frontmatter_json=ver.get("frontmatter_json", "{}"),
                tags_json=ver.get("tags_json", "[]"),
                aliases_json=ver.get("aliases_json", "[]"),
                char_count=metadata.get("char_count", 0),
                line_count=metadata.get("line_count", 0),
                byte_size=ver.get("size", 0) or metadata.get("byte_size", 0),
                mtime=ver.get("mtime"),
                processed_at=ver.get("processed_time", _now_str()),
            )
            stats["document_versions"] += 1

        # Set current version to latest
        if versions:
            latest_ver_id = dict(versions[-1])["document_version_id"]
            doc_repo.update_current_version(new_conn, doc_id, latest_ver_id,
                                              updated_at=_now_str())


def _migrate_episodes(old_conn, new_conn, stats, dry_run):
    """Migrate concept_version(role=episode) → episodes."""
    # Build doc_family → doc_id map (same in v1)
    # Build old doc_version_id → new doc_id map
    doc_ver_map = {}
    for row in old_conn.execute(
        "SELECT document_version_id, document_family_id FROM document_version"
    ).fetchall():
        doc_ver_map[row[0]] = row[1]

    episodes = old_conn.execute(
        "SELECT cv.* FROM concept_version cv "
        "JOIN concept_family cf ON cf.family_id = cv.family_id "
        "WHERE cv.role = 'episode' "
        "ORDER BY cv.document_version_id, cv.version_seq"
    ).fetchall()

    # Track chunk_index per document_version_id to ensure uniqueness
    chunk_counter: Dict[str, int] = {}

    for ep in episodes:
        ep = dict(ep)
        metadata = json.loads(ep.get("metadata", "{}") or "{}")
        ep_id = ep["version_id"]
        family_id = ep["family_id"]

        # Derive doc_id and ver_id
        doc_ver_id = ep.get("document_version_id", "") or metadata.get("document_version_id", "")
        doc_id = doc_ver_map.get(doc_ver_id) or metadata.get("document_family_id", "")
        if not doc_id:
            continue

        if dry_run:
            stats["episodes"] += 1
            continue

        ep_fam = family_id
        # Assign unique chunk_index per document_version_id
        orig_chunk = metadata.get("chunk_index", 0)
        key = doc_ver_id
        chunk_index = chunk_counter.get(key, 0)
        chunk_counter[key] = chunk_index + 1
        chunk_hash = metadata.get("chunk_hash", "")
        source_text = ep.get("source_text", "") or metadata.get("source_text", "")
        heading_path = metadata.get("heading_path", "")
        start_offset = metadata.get("start_offset", 0)
        end_offset = metadata.get("end_offset", 0)
        line_start = metadata.get("line_start", 0)
        line_end = metadata.get("line_end", 0)
        activity_type = metadata.get("activity_type", "")
        episode_type = metadata.get("episode_type", "")

        ep_repo.insert_episode(
            new_conn, ep_id, ep_fam, doc_id, doc_ver_id,
            source_text=source_text,
            memory_text=ep.get("content", ""),
            heading_path=heading_path,
            start_offset=start_offset or 0,
            end_offset=end_offset or 0,
            line_start=line_start or 0,
            line_end=line_end or 0,
            chunk_index=chunk_index or 0,
            chunk_hash=chunk_hash or "",
            name=ep.get("name", ""),
            activity_type=activity_type,
            episode_type=episode_type,
            event_time=ep.get("event_time"),
            processed_at=ep.get("processed_time", _now_str()),
        )
        stats["episodes"] += 1


def _migrate_entities(old_conn, new_conn, stats, dry_run):
    """Migrate concept_family(role=entity) + concept_version(role=entity) → entity_families + entity_observations."""
    families = old_conn.execute(
        "SELECT cf.family_id, cf.canonical_name, cf.status, cf.created_at, cf.updated_at "
        "FROM concept_family cf WHERE cf.role = 'entity'"
    ).fetchall()

    fam_count = 0
    obs_count = 0
    emb_count = 0

    for fam in families:
        fam = dict(fam)
        fid = fam["family_id"]
        if dry_run:
            fam_count += 1
            continue

        ent_repo.upsert_entity_family(
            new_conn, fid,
            fam["canonical_name"],
            canonical_content="",
            created_at=fam.get("created_at", _now_str()),
            updated_at=fam.get("updated_at", _now_str()),
        )
        fam_count += 1

        # Migrate versions as observations
        versions = old_conn.execute(
            "SELECT * FROM concept_version "
            "WHERE family_id = ? AND role = 'entity' "
            "ORDER BY version_seq",
            (fid,),
        ).fetchall()
        for ver in versions:
            ver = dict(ver)
            obs_id = ver["version_id"]
            episode_id = ver.get("episode_version_id", "")

            ent_repo.insert_entity_observation(
                new_conn, obs_id, fid, episode_id,
                name=ver.get("name", ""),
                content=ver.get("content", ""),
                processed_at=ver.get("processed_time", _now_str()),
            )
            obs_count += 1

            # Migrate embedding
            emb = ver.get("embedding")
            if emb and isinstance(emb, bytes) and len(emb) > 0:
                import hashlib
                text = ver.get("name", "")
                text_hash = hashlib.sha256(text.encode()).hexdigest()
                dim = len(emb) // 4
                emb_repo.insert_embedding(
                    new_conn, f"emb_{obs_id[:16]}", "entity_obs", obs_id,
                    "content", text_hash, "legacy", dim, emb,
                    created_at=_now_str(),
                )
                emb_count += 1

    stats["entities"] = fam_count
    stats["entity_observations"] = obs_count
    stats["embeddings"] = emb_count


def _migrate_relations(old_conn, new_conn, stats, dry_run):
    """Migrate concept_family(role=relation) + concept_version(role=relation) → relation_families + relation_assertions.

    Relation endpoints are stored in concept_edge(CONNECTS) and metadata JSON.
    """
    # Build CONNECTS edge map: relation_version_id → [(target_family_id, target_version_id)]
    connects_map: Dict[str, List[dict]] = {}
    for edge in old_conn.execute(
        "SELECT relation_version_id, target_family_id, target_version_id "
        "FROM concept_edge WHERE edge_type = 'CONNECTS' AND relation_version_id != ''"
    ).fetchall():
        edge = dict(edge)
        rv = edge["relation_version_id"]
        connects_map.setdefault(rv, []).append(edge)

    families = old_conn.execute(
        "SELECT cf.family_id, cf.canonical_name, cf.status, cf.created_at, cf.updated_at "
        "FROM concept_family cf WHERE cf.role = 'relation'"
    ).fetchall()

    fam_count = 0
    assert_count = 0

    for fam in families:
        fam = dict(fam)
        fid = fam["family_id"]

        # Get latest version to find entity endpoints from metadata
        latest = old_conn.execute(
            "SELECT * FROM concept_version "
            "WHERE family_id = ? AND role = 'relation' "
            "ORDER BY version_seq DESC LIMIT 1",
            (fid,),
        ).fetchone()
        if not latest:
            continue
        latest = dict(latest)
        metadata = json.loads(latest.get("metadata", "{}") or "{}")

        sub_fid = metadata.get("entity1_family_id", "")
        obj_fid = metadata.get("entity2_family_id", "")
        if not sub_fid or not obj_fid:
            # Try from CONNECTS edges
            connects = connects_map.get(latest["version_id"], [])
            if len(connects) >= 2:
                sub_fid = connects[0]["target_family_id"]
                obj_fid = connects[1]["target_family_id"]

        if not sub_fid or not obj_fid:
            continue

        # Check for existing relation family with same entity pair
        existing_rf = new_conn.execute(
            "SELECT relation_family_id FROM relation_families "
            "WHERE subject_entity_family_id = ? AND object_entity_family_id = ?",
            (sub_fid, obj_fid),
        ).fetchone()

        if dry_run:
            fam_count += 1
            continue

        if existing_rf:
            # Map old family_id to existing one for assertion migration
            fid = existing_rf[0]
        else:
            rel_repo.upsert_relation_family(
                new_conn, fid, sub_fid, obj_fid,
                canonical_content=latest.get("content", ""),
                created_at=fam.get("created_at", _now_str()),
                updated_at=fam.get("updated_at", _now_str()),
            )
        fam_count += 1

        # Migrate versions as assertions
        versions = old_conn.execute(
            "SELECT * FROM concept_version "
            "WHERE family_id = ? AND role = 'relation' "
            "ORDER BY version_seq",
            (fid,),
        ).fetchall()
        for ver in versions:
            ver = dict(ver)
            ver_id = ver["version_id"]
            metadata_v = json.loads(ver.get("metadata", "{}") or "{}")
            episode_id = ver.get("episode_version_id", "")

            sub_abs = metadata_v.get("entity1_absolute_id", "")
            obj_abs = metadata_v.get("entity2_absolute_id", "")
            sub_f = metadata_v.get("entity1_family_id", sub_fid)
            obj_f = metadata_v.get("entity2_family_id", obj_fid)

            # Skip duplicate assertions (same episode + family + entity pair)
            dup = new_conn.execute(
                "SELECT 1 FROM relation_assertions "
                "WHERE episode_id = ? AND relation_family_id = ? "
                "AND subject_entity_family_id = ? AND object_entity_family_id = ? "
                "AND status = 'active'",
                (episode_id, fid, sub_f, obj_f),
            ).fetchone()
            if dup:
                continue

            rel_repo.insert_relation_assertion(
                new_conn, ver_id, fid, episode_id,
                sub_abs, obj_abs, sub_f, obj_f,
                content=ver.get("content", ""),
                processed_at=ver.get("processed_time", _now_str()),
            )
            assert_count += 1

    stats["relations"] = fam_count
    stats["relation_assertions"] = assert_count


def _migrate_mentions(old_conn, new_conn, stats, dry_run):
    """Migrate concept_edge(MENTIONS) → entity_mentions."""
    edges = old_conn.execute(
        "SELECT * FROM concept_edge WHERE edge_type = 'MENTIONS'"
    ).fetchall()

    count = 0
    for edge in edges:
        edge = dict(edge)
        episode_id = edge.get("source_version_id", "") or edge.get("episode_version_id", "")
        entity_id = edge.get("target_version_id", "")
        entity_fid = edge.get("target_family_id", "")
        provenance = json.loads(edge.get("provenance", "{}") or "{}")

        if not episode_id or not entity_id:
            continue

        if dry_run:
            count += 1
            continue

        evidence = provenance.get("evidence", [{}])
        ev = evidence[0] if evidence else {}
        mention_id = edge.get("edge_id", f"ment_{count}")

        ent_repo.insert_entity_mention(
            new_conn, mention_id, entity_id, entity_fid, episode_id,
            surface_text=ev.get("quote", "") or ev.get("name", ""),
            start_offset=ev.get("start_offset", 0),
            end_offset=ev.get("end_offset", 0),
            line_start=ev.get("line_start", 0),
            line_end=ev.get("line_end", 0),
            created_at=edge.get("created_at", _now_str()),
        )
        count += 1

    stats["mentions"] = count


def _migrate_document_links(old_conn, new_conn, stats, dry_run):
    """Migrate concept_edge(DOCUMENT_LINK) → document_links."""
    edges = old_conn.execute(
        "SELECT * FROM concept_edge WHERE edge_type = 'DOCUMENT_LINK'"
    ).fetchall()

    count = 0
    for edge in edges:
        edge = dict(edge)
        from_doc = edge.get("source_family_id", "")
        to_doc = edge.get("target_family_id", "")
        provenance = json.loads(edge.get("provenance", "{}") or "{}")

        if not from_doc:
            continue

        if dry_run:
            count += 1
            continue

        # Find a valid version for from_doc
        ver = new_conn.execute(
            "SELECT document_version_id FROM document_versions "
            "WHERE document_id = ? AND status = 'active' LIMIT 1",
            (from_doc,),
        ).fetchone()

        doc_repo.insert_document_link(
            new_conn, edge.get("edge_id", f"dl_{count}"),
            from_doc, to_doc,
            from_document_version_id=ver[0] if ver else "",
            link_text=provenance.get("link_text", ""),
            link_target=provenance.get("link_target", ""),
            created_at=edge.get("created_at", _now_str()),
        )
        count += 1

    stats["document_links"] = count


def _migrate_redirects(old_conn, new_conn, stats, dry_run):
    """Migrate concept_redirect → entity_redirects."""
    redirects = old_conn.execute("SELECT * FROM concept_redirect").fetchall()

    count = 0
    for row in redirects:
        row = dict(row)
        if dry_run:
            count += 1
            continue

        new_conn.execute(
            "INSERT OR IGNORE INTO entity_redirects (source_family_id, target_family_id, created_at) "
            "VALUES (?, ?, ?)",
            (row["source_family_id"], row["target_family_id"],
             row.get("updated_at", _now_str())),
        )
        count += 1

    stats["redirects"] = count


def _migrate_embeddings(old_conn, new_conn, stats, dry_run):
    """Embeddings already migrated per entity/relation. This is a no-op unless
    the old DB has a standalone embeddings table."""
    pass


def main():
    import argparse
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    parser = argparse.ArgumentParser(description="Migrate v1 schema to V1.5")
    parser.add_argument("--source", required=True, help="Path to old graph.db")
    parser.add_argument("--target", required=True, help="Path to library directory")
    parser.add_argument("--dry-run", action="store_true", help="Count only, no writes")
    parser.add_argument("--no-copy-files", action="store_true", help="Skip file copying")
    args = parser.parse_args()

    result = migrate(args.source, args.target,
                     copy_files=not args.no_copy_files,
                     dry_run=args.dry_run)
    print(json.dumps(result, indent=2, ensure_ascii=False))

    if result["errors"]:
        sys.exit(1)


if __name__ == "__main__":
    main()
