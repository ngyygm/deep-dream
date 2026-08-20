"""``db`` command group -- database maintenance and V1.5 schema management.

Subcommands
-----------
init-v15          Initialize V1.5 schema on the current graph.db.
reset-v15         Backup old graph.db and create a fresh V1.5 database.
rebuild-fts       Full rebuild of the episodes_fts full-text search index.
validate          Run integrity validation; optionally auto-repair.
rebuild-current   Rebuild content/current/ files from database.
vacuum-embeddings Clean orphaned and inactive embeddings.
compact           VACUUM the graph.db to reclaim disk space.
quality           Data quality report with coverage and health metrics.
integrity         Per-document integrity check (and optional repair).

All commands respect the global ``--json`` / ``--quiet`` / ``--no-color``
flags handled by :class:`OutputManager`.
"""
from __future__ import annotations

import json
import os
import sqlite3
import urllib.error
import urllib.request
from datetime import datetime, timezone
from typing import Any, Dict, Optional

import click

from ._ctx import CliContext
from ._exit_codes import ARGS, ERROR, NOT_FOUND
from ._output import OutputManager


# ------------------------------------------------------------------
# Shared helpers
# ------------------------------------------------------------------

def _get_storage_path(config: Dict[str, Any]) -> str:
    """Return the storage root from config (default ``./library``)."""
    return config.get("storage_path", "./library")


def _resolve_db_path(storage_path: str) -> str:
    """Resolve the database file path, preferring library.db over graph.db."""
    for name in ("library.db", "graph.db"):
        candidate = os.path.join(storage_path, name)
        if os.path.exists(candidate):
            return candidate
    return os.path.join(storage_path, "library.db")


def _open_db_conn(config: Dict[str, Any]) -> sqlite3.Connection:
    """Open a read-write connection to the graph database.

    Looks for ``library.db`` first (V1.5 layout), then ``graph.db`` (legacy).
    Raises :class:`FileNotFoundError` when neither file exists.
    """
    storage_path = _get_storage_path(config)
    db_path = _resolve_db_path(storage_path)
    if not os.path.exists(db_path):
        raise FileNotFoundError(f"Database not found in {storage_path}")
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    return conn


def _check_icon(ok: bool) -> str:
    """Return a Rich-markup check/cross icon."""
    return "[bold green]✓[/bold green]" if ok else "[bold red]⚠[/bold red]"


def _plain_icon(ok: bool) -> str:
    """Return a plain-text check/cross icon (no Rich markup)."""
    return "✓" if ok else "⚠"


# ------------------------------------------------------------------
# Click group
# ------------------------------------------------------------------

@click.group()
def db() -> None:
    """Database maintenance and V1.5 schema management."""
    pass


# ------------------------------------------------------------------
# db init-v15
# ------------------------------------------------------------------

@db.command("init-v15")
@click.option(
    "--smoke-test",
    is_flag=True,
    default=False,
    help="Run smoke tests after initialization (default in CI).",
)
@click.pass_context
def init_v15(ctx: click.Context, smoke_test: bool) -> None:
    """Initialize V1.5 schema on the current graph.db.

    Creates all tables, indexes, the FTS virtual table, and the
    ``graph_edges`` view.  Safe to run on an already-initialized
    database (uses ``IF NOT EXISTS`` throughout).
    """
    obj: CliContext = ctx.obj
    out = OutputManager(ctx)
    config = obj.config
    storage_path = _get_storage_path(config)
    db_path = _resolve_db_path(storage_path)

    from core.storage.sqlite.schema_v15 import init_schema_v15  # deferred
    conn = sqlite3.connect(db_path)
    try:
        result = init_schema_v15(conn)

        if out.is_json:
            payload = {
                "success": True,
                "command": "db init-v15",
                "data": result,
            }
            if smoke_test:
                from core.storage.sqlite.integrity import validate_all  # deferred
                violations = validate_all(
                    conn, library_path=storage_path, include_file_checks=False,
                )
                payload["smoke_test"] = len(violations) == 0
                payload["violations"] = len(violations)
            click.echo(json.dumps(payload, ensure_ascii=False, indent=2))
            return

        out.success(f"V1.5 schema initialized at {db_path}")
        if smoke_test:
            from core.storage.sqlite.integrity import validate_all  # deferred
            violations = validate_all(
                conn, library_path=storage_path, include_file_checks=False,
            )
            if violations:
                out.console.print(
                    f"  {_check_icon(False)} Smoke test: {len(violations)} violation(s)"
                )
            else:
                out.console.print(
                    f"  {_check_icon(True)} Smoke test: all checks passed"
                )
    except Exception as exc:
        if out.is_json:
            click.echo(json.dumps({
                "success": False,
                "error": str(exc),
            }, ensure_ascii=False, indent=2))
        else:
            out.error(str(exc), code=ERROR)
    finally:
        conn.close()


# ------------------------------------------------------------------
# db reset-v15
# ------------------------------------------------------------------

@db.command("reset-v15")
@click.option(
    "--yes",
    is_flag=True,
    default=False,
    help="Required: confirm destructive reset of the database.",
)
@click.pass_context
def reset_v15(ctx: click.Context, yes: bool) -> None:
    """Backup the existing graph.db and create a fresh V1.5 database.

    \b
    DANGER: This destroys all data in the current graph.db.
    A timestamped backup is created automatically before the reset.
    Requires ``--yes`` to proceed.
    """
    obj: CliContext = ctx.obj
    out = OutputManager(ctx)
    config = obj.config
    storage_path = _get_storage_path(config)
    db_path = _resolve_db_path(storage_path)

    if not yes:
        if out.is_json:
            click.echo(json.dumps({
                "success": False,
                "error": "Destructive operation requires --yes flag.",
            }, ensure_ascii=False, indent=2))
        else:
            out.error(
                "This is a destructive operation. Pass --yes to confirm.",
                hint="A timestamped backup will be created automatically.",
                code=ARGS,
            )
        return

    if not os.path.exists(db_path):
        if out.is_json:
            click.echo(json.dumps({
                "success": False,
                "error": "No existing graph.db found.",
            }, ensure_ascii=False, indent=2))
        else:
            out.error("No existing graph.db found.", code=NOT_FOUND)
        return

    # -- create timestamped backup ------------------------------------------
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    backup_path = os.path.join(storage_path, f"graph.legacy.{ts}.db")

    src_conn = sqlite3.connect(db_path)
    try:
        src_conn.execute("PRAGMA wal_checkpoint(FULL)")
        backup_conn = sqlite3.connect(backup_path)
        src_conn.backup(backup_conn)
        backup_conn.close()
    finally:
        src_conn.close()

    if not os.path.exists(backup_path):
        if out.is_json:
            click.echo(json.dumps({
                "success": False,
                "error": "Backup failed.",
            }, ensure_ascii=False, indent=2))
        else:
            out.error("Backup failed.", code=ERROR)
        return

    # -- remove old DB and create fresh one ---------------------------------
    os.remove(db_path)
    for suffix in ("-wal", "-shm"):
        p = db_path + suffix
        if os.path.exists(p):
            os.remove(p)

    from core.storage.sqlite.schema_v15 import init_schema_v15  # deferred
    from core.storage.sqlite.integrity import validate_all  # deferred

    conn = sqlite3.connect(db_path)
    try:
        result = init_schema_v15(conn)
        violations = validate_all(
            conn, library_path=storage_path, include_file_checks=False,
        )

        if out.is_json:
            click.echo(json.dumps({
                "success": True,
                "command": "db reset-v15",
                "data": {
                    "backup": backup_path,
                    "violations": len(violations),
                    **result,
                },
            }, ensure_ascii=False, indent=2))
        else:
            out.success(f"Fresh V1.5 database created at {db_path}")
            out.console.print(f"  Backup: {backup_path}")
            out.console.print(
                f"  {_check_icon(len(violations) == 0)} "
                f"Validation: {len(violations)} violation(s)"
            )
    except Exception as exc:
        if out.is_json:
            click.echo(json.dumps({
                "success": False,
                "error": str(exc),
                "backup": backup_path,
            }, ensure_ascii=False, indent=2))
        else:
            out.error(f"Reset failed: {exc}", hint=f"Backup preserved at {backup_path}")
    finally:
        conn.close()


# ------------------------------------------------------------------
# db rebuild-fts
# ------------------------------------------------------------------

@db.command("rebuild-fts")
@click.pass_context
def rebuild_fts(ctx: click.Context) -> None:
    """Rebuild the episodes full-text search index from scratch."""
    obj: CliContext = ctx.obj
    out = OutputManager(ctx)
    config = obj.config

    try:
        conn = _open_db_conn(config)
    except FileNotFoundError as exc:
        out.error(str(exc), code=NOT_FOUND)
        return

    try:
        try:
            from core.storage.sqlite.repositories.episodes import rebuild_fts_all  # deferred
            count = rebuild_fts_all(conn)
        except NotImplementedError:
            # Fallback: direct SQL rebuild
            conn.execute("DELETE FROM episodes_fts")
            conn.execute(
                """
                INSERT INTO episodes_fts (
                    episode_id, document_id, document_version_id,
                    name, heading_path, source_text, memory_text
                )
                SELECT episode_id, document_id, document_version_id,
                       name, heading_path, source_text, memory_text
                FROM episodes
                WHERE status = 'active'
                """
            )
            count = conn.execute("SELECT changes()").fetchone()[0]
            conn.commit()

        if out.is_json:
            click.echo(json.dumps({
                "success": True,
                "command": "db rebuild-fts",
                "data": {"episodes_indexed": count},
            }, ensure_ascii=False, indent=2))
        else:
            out.success(f"FTS index rebuilt: {count} episode(s) indexed")
    except Exception as exc:
        if out.is_json:
            click.echo(json.dumps({
                "success": False,
                "error": str(exc),
            }, ensure_ascii=False, indent=2))
        else:
            out.error(str(exc), code=ERROR)
    finally:
        conn.close()


# ------------------------------------------------------------------
# db validate
# ------------------------------------------------------------------

@db.command("validate")
@click.option(
    "--repair",
    is_flag=True,
    default=False,
    help="Auto-repair fixable issues (rebuilds content/current and FTS).",
)
@click.pass_context
def validate(ctx: click.Context, repair: bool) -> None:
    """Run integrity validation on the database.

    Checks table structure, referential integrity, and file-system
    consistency.  Pass ``--repair`` to automatically rebuild the
    content/current files and the FTS index.
    """
    obj: CliContext = ctx.obj
    out = OutputManager(ctx)
    config = obj.config
    storage_path = _get_storage_path(config)

    try:
        conn = _open_db_conn(config)
    except FileNotFoundError as exc:
        out.error(str(exc), code=NOT_FOUND)
        return

    try:
        from core.storage.sqlite.integrity import validate_all  # deferred

        violations = validate_all(
            conn, library_path=storage_path, include_file_checks=True,
        )
        ok = len(violations) == 0

        files_rebuilt: Optional[int] = None
        if repair:
            from core.storage.sqlite.content_fs import rebuild_current_files  # deferred
            files_rebuilt = rebuild_current_files(conn, storage_path)

        if out.is_json:
            data: Dict[str, Any] = {
                "violations": len(violations),
                "details": violations[:50],
            }
            if repair:
                data["current_files_rebuilt"] = files_rebuilt
            click.echo(json.dumps({
                "success": ok,
                "command": f"db validate{' --repair' if repair else ''}",
                "data": data,
            }, ensure_ascii=False, indent=2))
        else:
            icon = _check_icon(ok)
            label = "validate --repair" if repair else "validate"
            out.console.print(f"  {icon} {label}: {len(violations)} violation(s)")
            if repair and files_rebuilt is not None:
                out.console.print(
                    f"  {_check_icon(True)} Rebuilt {files_rebuilt} current file(s)"
                )
            for v in violations[:10]:
                tbl = v.get("table", "")
                vid = v.get("id", "")
                issue = v.get("issue", "")
                detail = v.get("detail", "")
                parts = [p for p in [tbl, vid, issue, detail] if p]
                from rich.markup import escape as _rich_esc
                out.console.print(f"    - {' | '.join(_rich_esc(p) for p in parts)}")
            if len(violations) > 10:
                out.console.print(f"    ... and {len(violations) - 10} more")
    except Exception as exc:
        if out.is_json:
            click.echo(json.dumps({
                "success": False,
                "error": str(exc),
            }, ensure_ascii=False, indent=2))
        else:
            out.error(str(exc), code=ERROR)
    finally:
        conn.close()


# ------------------------------------------------------------------
# db rebuild-current
# ------------------------------------------------------------------

@db.command("rebuild-current")
@click.pass_context
def rebuild_current(ctx: click.Context) -> None:
    """Rebuild content/current/ files from the database.

    Regenerates the ``content/current/`` directory tree so that each
    active document version has an up-to-date file on disk.
    """
    obj: CliContext = ctx.obj
    out = OutputManager(ctx)
    config = obj.config
    storage_path = _get_storage_path(config)

    try:
        conn = _open_db_conn(config)
    except FileNotFoundError as exc:
        out.error(str(exc), code=NOT_FOUND)
        return

    try:
        from core.storage.sqlite.content_fs import rebuild_current_files  # deferred
        count = rebuild_current_files(conn, storage_path)

        if out.is_json:
            click.echo(json.dumps({
                "success": True,
                "command": "db rebuild-current",
                "data": {"files_written": count},
            }, ensure_ascii=False, indent=2))
        else:
            out.success(f"Rebuilt {count} current file(s)")
    except Exception as exc:
        if out.is_json:
            click.echo(json.dumps({
                "success": False,
                "error": str(exc),
            }, ensure_ascii=False, indent=2))
        else:
            out.error(str(exc), code=ERROR)
    finally:
        conn.close()


# ------------------------------------------------------------------
# db vacuum-embeddings
# ------------------------------------------------------------------

@db.command("vacuum-embeddings")
@click.option(
    "--inactive",
    is_flag=True,
    default=False,
    help="Also clean superseded/stale owner embeddings.",
)
@click.option(
    "--dry-run",
    is_flag=True,
    default=False,
    help="Only report counts, do not delete.",
)
@click.pass_context
def vacuum_embeddings(ctx: click.Context, inactive: bool, dry_run: bool) -> None:
    """Clean up orphaned embeddings.

    Removes embeddings whose owner rows (episodes, entity observations,
    relation assertions) no longer exist.  With ``--inactive`` also
    removes superseded/stale embeddings.  ``--dry-run`` reports counts
    without deleting.
    """
    obj: CliContext = ctx.obj
    out = OutputManager(ctx)
    config = obj.config

    try:
        conn = _open_db_conn(config)
    except FileNotFoundError as exc:
        out.error(str(exc), code=NOT_FOUND)
        return

    try:
        orphaned: int = 0
        deleted_doc: int = 0
        inactive_count: int = 0

        try:
            from core.storage.sqlite.repositories.embeddings import (  # deferred
                vacuum_orphaned,
                vacuum_deleted_documents,
                vacuum_inactive,
            )
            orphaned = vacuum_orphaned(conn, dry_run=dry_run)
            deleted_doc = vacuum_deleted_documents(conn, dry_run=dry_run)
            if inactive:
                inactive_count = vacuum_inactive(conn, dry_run=dry_run)
            if not dry_run:
                conn.commit()
        except NotImplementedError:
            # Fallback: basic orphan cleanup
            cursor = conn.execute(
                """
                DELETE FROM embeddings
                WHERE owner_type = 'episode'
                  AND owner_id NOT IN (SELECT episode_id FROM episodes)
                """
            )
            conn.commit()
            orphaned = cursor.rowcount

        if out.is_json:
            click.echo(json.dumps({
                "success": True,
                "command": "db vacuum-embeddings",
                "data": {
                    "orphaned_removed": orphaned,
                    "deleted_doc_removed": deleted_doc,
                    "inactive_removed": inactive_count,
                    "dry_run": dry_run,
                },
            }, ensure_ascii=False, indent=2))
        else:
            total = orphaned + deleted_doc + inactive_count
            label_suffix = " (dry-run)" if dry_run else ""
            out.success(f"{'Would clean' if dry_run else 'Cleaned'} {total} embedding(s){label_suffix}")
            out.console.print(f"  Orphaned          : {orphaned}")
            out.console.print(f"  Deleted documents  : {deleted_doc}")
            if inactive:
                label = "reported (dry-run)" if dry_run else "removed"
                out.console.print(f"  Inactive ({label}): {inactive_count}")
    except Exception as exc:
        if out.is_json:
            click.echo(json.dumps({
                "success": False,
                "error": str(exc),
            }, ensure_ascii=False, indent=2))
        else:
            out.error(str(exc), code=ERROR)
    finally:
        conn.close()


# ------------------------------------------------------------------
# db compact
# ------------------------------------------------------------------

@db.command("compact")
@click.option(
    "--yes",
    is_flag=True,
    default=False,
    help="Required: confirm VACUUM operation.",
)
@click.pass_context
def compact(ctx: click.Context, yes: bool) -> None:
    """VACUUM the graph.db to reclaim disk space.

    Runs SQLite ``VACUUM`` which rebuilds the database file, removing
    free pages and defragmenting the data.  Requires ``--yes``.
    """
    obj: CliContext = ctx.obj
    out = OutputManager(ctx)
    config = obj.config

    if not yes:
        if out.is_json:
            click.echo(json.dumps({
                "success": False,
                "error": "VACUUM requires --yes flag to confirm.",
            }, ensure_ascii=False, indent=2))
        else:
            out.error(
                "Pass --yes to confirm the VACUUM operation.",
                hint="VACUUM rewrites the database file and may take a moment.",
                code=ARGS,
            )
        return

    try:
        conn = _open_db_conn(config)
    except FileNotFoundError as exc:
        out.error(str(exc), code=NOT_FOUND)
        return

    try:
        conn.execute("VACUUM")

        if out.is_json:
            click.echo(json.dumps({
                "success": True,
                "command": "db compact",
                "data": {},
            }, ensure_ascii=False, indent=2))
        else:
            out.success("Database compacted (VACUUM complete)")
    except Exception as exc:
        if out.is_json:
            click.echo(json.dumps({
                "success": False,
                "error": str(exc),
            }, ensure_ascii=False, indent=2))
        else:
            out.error(str(exc), code=ERROR)
    finally:
        conn.close()


# ------------------------------------------------------------------
# db quality  (NEW)
# ------------------------------------------------------------------

@db.command("quality")
@click.pass_context
def quality(ctx: click.Context) -> None:
    """Data quality report with coverage and health metrics.

    Queries aggregate statistics from the database and presents a
    summary table with check/warning icons indicating health of
    entities, embeddings, relations, and content coverage.
    """
    obj: CliContext = ctx.obj
    out = OutputManager(ctx)
    config = obj.config

    try:
        conn = _open_db_conn(config)
    except FileNotFoundError as exc:
        out.error(str(exc), code=NOT_FOUND)
        return

    try:
        cur = conn.cursor()

        # -- Entities --------------------------------------------------------
        row = cur.execute(
            "SELECT COUNT(*) AS cnt FROM entity_families"
        ).fetchone()
        total_entities: int = row["cnt"] if row else 0

        row = cur.execute(
            """
            SELECT COUNT(*) AS cnt
            FROM entity_families ef
            WHERE EXISTS (
                SELECT 1 FROM embeddings e
                WHERE e.owner_type = 'entity_family'
                  AND e.owner_id = ef.entity_family_id
            )
            """
        ).fetchone()
        entities_with_emb: int = row["cnt"] if row else 0

        # Orphan entity families: no active observations
        row = cur.execute(
            """
            SELECT COUNT(*) AS cnt
            FROM entity_families ef
            WHERE NOT EXISTS (
                SELECT 1 FROM entity_observations eo
                WHERE eo.entity_family_id = ef.entity_family_id
                  AND eo.status = 'active'
            )
            """
        ).fetchone()
        orphan_entities: int = row["cnt"] if row else 0

        # Duplicate suspects: same canonical_name, different family_id
        row = cur.execute(
            """
            SELECT COUNT(*) AS cnt FROM (
                SELECT canonical_name
                FROM entity_families
                GROUP BY canonical_name
                HAVING COUNT(*) > 1
            )
            """
        ).fetchone()
        duplicate_suspects: int = row["cnt"] if row else 0

        # -- Confidence ------------------------------------------------------
        # entity_families does not have a confidence column; check
        # entity_observations.extra_json for confidence if available,
        # but also check if entity_observations has a confidence-like field.
        # We try to compute from observations' extra_json as a best effort.
        try:
            row = cur.execute(
                """
                SELECT
                    AVG(CAST(json_extract(extra_json, '$.confidence') AS REAL)) AS avg_conf,
                    SUM(CASE
                        WHEN CAST(json_extract(extra_json, '$.confidence') AS REAL) < 0.5
                        THEN 1 ELSE 0
                    END) AS low_count
                FROM entity_observations
                WHERE status = 'active'
                  AND json_extract(extra_json, '$.confidence') IS NOT NULL
                """
            ).fetchone()
            avg_confidence: Optional[float] = (
                round(row["avg_conf"], 3) if row and row["avg_conf"] is not None else None
            )
            low_confidence_count: int = row["low_count"] if row and row["low_count"] else 0
        except Exception:
            avg_confidence = None
            low_confidence_count = 0

        # -- Relations -------------------------------------------------------
        row = cur.execute(
            "SELECT COUNT(*) AS cnt FROM relation_families"
        ).fetchone()
        total_relations: int = row["cnt"] if row else 0

        # Relations with evidence (at least one active assertion linked to an episode)
        row = cur.execute(
            """
            SELECT COUNT(DISTINCT rf.relation_family_id) AS cnt
            FROM relation_families rf
            JOIN relation_assertions ra
              ON ra.relation_family_id = rf.relation_family_id
             AND ra.status = 'active'
            WHERE ra.evidence_text IS NOT NULL
              AND ra.evidence_text != ''
            """
        ).fetchone()
        relations_with_evidence: int = row["cnt"] if row else 0

        # -- Content coverage ------------------------------------------------
        row = cur.execute(
            "SELECT COUNT(*) AS cnt FROM documents WHERE status = 'active'"
        ).fetchone()
        total_docs: int = row["cnt"] if row else 0

        row = cur.execute(
            """
            SELECT COUNT(*) AS cnt
            FROM documents d
            WHERE d.status = 'active'
              AND d.current_version_id IS NOT NULL
              AND EXISTS (
                  SELECT 1 FROM document_versions dv
                  WHERE dv.document_version_id = d.current_version_id
                    AND dv.status = 'active'
                    AND dv.char_count > 0
              )
            """
        ).fetchone()
        docs_with_content: int = row["cnt"] if row else 0

        content_pct: float = (
            round(docs_with_content / total_docs * 100, 1) if total_docs else 0.0
        )

        # -- Embedding coverage ----------------------------------------------
        row = cur.execute(
            "SELECT COUNT(*) AS cnt FROM embeddings"
        ).fetchone()
        total_embeddings: int = row["cnt"] if row else 0

        emb_coverage_pct: float = (
            round(entities_with_emb / total_entities * 100, 1) if total_entities else 0.0
        )

        # -- Episodes --------------------------------------------------------
        row = cur.execute(
            "SELECT COUNT(*) AS cnt FROM episodes WHERE status = 'active'"
        ).fetchone()
        total_episodes: int = row["cnt"] if row else 0

        # -- Build output ----------------------------------------------------
        data = {
            "total_entities": total_entities,
            "entities_with_embeddings": entities_with_emb,
            "embedding_coverage_pct": emb_coverage_pct,
            "orphan_entities": orphan_entities,
            "duplicate_suspect_names": duplicate_suspects,
            "avg_confidence": avg_confidence,
            "low_confidence_count": low_confidence_count,
            "total_relations": total_relations,
            "relations_with_evidence": relations_with_evidence,
            "total_documents": total_docs,
            "documents_with_content": docs_with_content,
            "content_coverage_pct": content_pct,
            "total_embeddings": total_embeddings,
            "total_active_episodes": total_episodes,
        }

        if out.is_json:
            click.echo(json.dumps({
                "success": True,
                "command": "db quality",
                "data": data,
            }, ensure_ascii=False, indent=2))
            return

        # Rich table output
        rows = [
            (
                "Entities",
                str(total_entities),
                f"{emb_coverage_pct}% embedded",
                _plain_icon(orphan_entities == 0),
            ),
            (
                "Orphans",
                str(orphan_entities),
                "families w/o active observations",
                _plain_icon(orphan_entities == 0),
            ),
            (
                "Duplicates",
                str(duplicate_suspects),
                "name collisions",
                _plain_icon(duplicate_suspects == 0),
            ),
            (
                "Confidence",
                (
                    f"{avg_confidence:.1%}" if avg_confidence is not None else "N/A"
                ),
                f"{low_confidence_count} low (<50%)",
                _plain_icon(low_confidence_count == 0),
            ),
            (
                "Relations",
                str(total_relations),
                f"{relations_with_evidence} w/ evidence",
                _plain_icon(total_relations == 0 or relations_with_evidence > 0),
            ),
            (
                "Documents",
                str(total_docs),
                f"{content_pct}% content coverage",
                _plain_icon(content_pct >= 90.0 or total_docs == 0),
            ),
            (
                "Episodes",
                str(total_episodes),
                "active",
                _plain_icon(total_episodes > 0),
            ),
            (
                "Embeddings",
                str(total_embeddings),
                "total rows",
                _plain_icon(total_embeddings > 0 or total_entities == 0),
            ),
        ]

        out.table(
            title="Data Quality Report",
            columns=("Metric", "Value", "Detail", "Status"),
            rows=rows,
        )
    except Exception as exc:
        if out.is_json:
            click.echo(json.dumps({
                "success": False,
                "error": str(exc),
            }, ensure_ascii=False, indent=2))
        else:
            out.error(str(exc), code=ERROR)
    finally:
        conn.close()


# ------------------------------------------------------------------
# db integrity <doc_id>  (NEW)
# ------------------------------------------------------------------

@db.command("integrity")
@click.argument("doc_id")
@click.option(
    "--repair",
    is_flag=True,
    default=False,
    help="Submit a repair task after the integrity check.",
)
@click.pass_context
def integrity(ctx: click.Context, doc_id: str, repair: bool) -> None:
    """Per-document integrity check and optional repair.

    Makes an HTTP request to the running Deep-Dream server API to
    assess whether the specified document has missing or incomplete
    processing windows.

    \b
    DOC_ID is the document_version_id to check.
    The server must be running on port 16200 (or DEEPDREAM_PORT).
    """
    obj: CliContext = ctx.obj
    out = OutputManager(ctx)
    config = obj.config

    port = config.get("port", 16200)
    base_url = f"http://127.0.0.1:{port}"

    # -- Integrity check ----------------------------------------------------
    check_url = f"{base_url}/api/v1/documents/{doc_id}/integrity"
    try:
        req = urllib.request.Request(check_url, method="GET")
        with urllib.request.urlopen(req, timeout=30) as resp:
            body = json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = ""
        try:
            detail = exc.read().decode("utf-8", errors="replace")
        except Exception:
            pass
        if out.is_json:
            click.echo(json.dumps({
                "success": False,
                "error": f"HTTP {exc.code}: {detail}",
                "command": "db integrity",
            }, ensure_ascii=False, indent=2))
        else:
            out.error(
                f"Server returned HTTP {exc.code}",
                hint="Ensure the Deep-Dream server is running.",
                code=ERROR,
            )
        return
    except urllib.error.URLError as exc:
        if out.is_json:
            click.echo(json.dumps({
                "success": False,
                "error": f"Connection failed: {exc.reason}",
                "command": "db integrity",
            }, ensure_ascii=False, indent=2))
        else:
            out.error(
                f"Cannot reach server at {base_url}",
                hint="Start the server with `deep-dream server` first.",
                code=ERROR,
            )
        return

    # Unwrap the API envelope
    result = body.get("data", body) if isinstance(body, dict) else body

    # -- Optional repair ----------------------------------------------------
    repair_result = None
    if repair:
        repair_url = f"{base_url}/api/v1/documents/{doc_id}/repair"
        try:
            payload = json.dumps({}).encode("utf-8")
            req = urllib.request.Request(
                repair_url,
                data=payload,
                method="POST",
                headers={"Content-Type": "application/json"},
            )
            with urllib.request.urlopen(req, timeout=60) as resp:
                repair_body = json.loads(resp.read().decode("utf-8"))
                repair_result = repair_body.get("data", repair_body)
        except urllib.error.HTTPError as exc:
            detail = ""
            try:
                detail = exc.read().decode("utf-8", errors="replace")
            except Exception:
                pass
            repair_result = {"error": f"HTTP {exc.code}: {detail}"}
        except urllib.error.URLError as exc:
            repair_result = {"error": f"Connection failed: {exc.reason}"}

    # -- Output -------------------------------------------------------------
    if out.is_json:
        payload_data: Dict[str, Any] = {"integrity": result}
        if repair and repair_result is not None:
            payload_data["repair"] = repair_result
        click.echo(json.dumps({
            "success": True,
            "command": f"db integrity{' --repair' if repair else ''}",
            "data": payload_data,
        }, ensure_ascii=False, indent=2))
        return

    # Rich output
    complete = result.get("complete", None)
    total_wins = result.get("total_windows", 0)
    complete_wins = result.get("complete_windows", 0)
    missing_wins = result.get("missing_windows", 0)
    missing_indices = result.get("missing_window_indices", [])

    icon = _check_icon(complete is True)
    out.console.print(f"  {icon} Document {doc_id}")
    out.console.print(f"     Complete : {complete}")
    out.console.print(f"     Windows  : {complete_wins}/{total_wins}")
    out.console.print(f"     Missing  : {missing_wins}")
    if missing_indices:
        display = missing_indices[:20]
        suffix = f" ... +{len(missing_indices) - 20}" if len(missing_indices) > 20 else ""
        out.console.print(f"     Indices  : {display}{suffix}")

    if repair and repair_result is not None:
        if "error" in repair_result:
            out.console.print(
                f"  {_check_icon(False)} Repair failed: {repair_result['error']}"
            )
        else:
            out.console.print(
                f"  {_check_icon(True)} Repair task submitted"
            )
