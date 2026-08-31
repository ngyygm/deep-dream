"""Vault and Markdown file indexing for V1.5 schema."""
from __future__ import annotations

import logging
import hashlib
import re
import sqlite3
import uuid
from pathlib import Path
from typing import Optional

from . import content_fs
from .helpers import now_utc_str
from .repositories import documents as doc_repo, episodes as ep_repo

logger = logging.getLogger(__name__)

# ── Wikilink / Markdown-link extraction with line positions ─────────
# Matches [[target#heading|display]] and [[target|display]]
_WIKILINK_RE = re.compile(r'\[\[([^\]#|]+)(?:#[^\]|]*)?(?:\|([^\]]*))?\]\]')
# Matches [display](href) where href is not http.
# Negative lookbehind for '!' excludes image syntax ![alt](url).
_MD_LINK_RE = re.compile(r'(?<!!)\[([^\]]*)\]\(([^)]+)\)')
_SUPPORTED_SUFFIXES = frozenset({".md", ".markdown", ".txt", ".text"})
_MAX_INDEX_BYTES = 32 * 1024 * 1024


def _extract_links_with_positions(body: str) -> list[dict]:
    """Return a list of link dicts with link_type, link_target, link_text, line_start, line_end."""
    links: list[dict] = []
    for m in _WIKILINK_RE.finditer(body):
        target = m.group(1).strip()
        display = (m.group(2) or "").strip() or target
        pos = m.start()
        line_no = body.count("\n", 0, pos) + 1
        links.append({
            "link_type": "wikilink",
            "link_target": target,
            "link_text": display,
            "line_start": line_no,
            "line_end": line_no,
        })
    for m in _MD_LINK_RE.finditer(body):
        display = m.group(1)
        href = m.group(2)
        if href.startswith("http"):
            continue
        pos = m.start()
        line_no = body.count("\n", 0, pos) + 1
        links.append({
            "link_type": "markdown",
            "link_target": href,
            "link_text": display,
            "line_start": line_no,
            "line_end": line_no,
        })
    return links


def _resolve_document_id(conn: sqlite3.Connection, target: str) -> Optional[str]:
    """Try to resolve a link target to an existing document_id.

    Searches by: exact relative_path match, exact title match, then
    relative_path ending with target.md / target.markdown.
    Returns None if no document is found.
    """
    t = target.strip()
    # 1. Exact relative_path
    row = conn.execute(
        "SELECT document_id FROM documents WHERE relative_path = ? AND status = 'active' LIMIT 1",
        (t,),
    ).fetchone()
    if row:
        return row[0]

    # 2. Exact title
    row = conn.execute(
        "SELECT document_id FROM documents WHERE title = ? AND status = 'active' LIMIT 1",
        (t,),
    ).fetchone()
    if row:
        return row[0]

    # 3. relative_path ending with target.md / target.markdown
    for suffix in (".md", ".markdown"):
        like = f"%/{t}{suffix}" if "/" not in t else f"%{t}{suffix}"
        row = conn.execute(
            "SELECT document_id FROM documents WHERE relative_path LIKE ? AND status = 'active' LIMIT 1",
            (like,),
        ).fetchone()
        if row:
            return row[0]

    # 4. relative_path ending with just the target (no extension added)
    row = conn.execute(
        "SELECT document_id FROM documents WHERE relative_path LIKE ? AND status = 'active' LIMIT 1",
        (f"%{t}",),
    ).fetchone()
    if row:
        return row[0]

    return None


def parse_markdown(text: str) -> dict:
    """Extract frontmatter, title, tags, aliases, and links from Markdown text."""
    frontmatter = {}
    body = text or ""
    if body.startswith("---"):
        end = body.find("---", 3)
        if end >= 0:
            yaml_text = body[3:end].strip()
            body = body[end + 3:].lstrip("\n")
            for line in yaml_text.splitlines():
                line = line.strip()
                if ":" in line:
                    key, _, val = line.partition(":")
                    key = key.strip()
                    val = val.strip().strip('"').strip("'")
                    if key == "tags":
                        frontmatter["tags"] = [t.strip() for t in val.strip("[]").split(",") if t.strip()]
                    elif key == "aliases":
                        frontmatter["aliases"] = [a.strip().strip('"').strip("'") for a in val.strip("[]").split(",") if a.strip()]
                    else:
                        frontmatter[key] = val

    title = ""
    for line in body.splitlines():
        stripped = line.strip()
        if stripped.startswith("# "):
            title = stripped[2:].strip()
            break

    # Use the compiled regexes that handle [[target#heading|display]]
    wikilinks = [m.group(1).strip() for m in _WIKILINK_RE.finditer(body)]
    md_links = [(m.group(1), m.group(2)) for m in _MD_LINK_RE.finditer(body)
                 if not m.group(2).startswith("http")]
    tags = set(frontmatter.get("tags", []))
    tags.update(re.findall(r'(?:^|\s)#([a-zA-Z][\w-]*)', body))

    return {
        "title": title or frontmatter.get("title", ""),
        "frontmatter": frontmatter,
        "tags": sorted(tags),
        "aliases": frontmatter.get("aliases", []),
        "wikilinks": wikilinks,
        "md_links": md_links,
    }


def index_markdown_file(conn: sqlite3.Connection, library_path: Path,
                        path: str, vault_root: str = "",
                        force: bool = False) -> dict:
    """Index a single Markdown file into the V1.5 schema."""
    file_path = Path(path).expanduser()
    try:
        resolved_path = file_path.resolve(strict=True)
    except OSError:
        return {"error": f"File not found: {path}"}
    if not resolved_path.is_file():
        return {"error": f"Not a file: {path}"}
    if resolved_path.suffix.lower() not in _SUPPORTED_SUFFIXES:
        return {"error": f"Unsupported file type: {resolved_path.suffix or '(none)'}"}
    try:
        if resolved_path.stat().st_size > _MAX_INDEX_BYTES:
            return {"error": f"File exceeds {_MAX_INDEX_BYTES} byte limit: {path}"}
    except OSError:
        return {"error": f"Unable to stat file: {path}"}

    root_path: Path | None = None
    if vault_root:
        try:
            root_path = Path(vault_root).expanduser().resolve(strict=True)
            if not root_path.is_dir() or not resolved_path.is_relative_to(root_path):
                return {"error": f"File is outside vault root: {path}"}
        except (OSError, ValueError):
            return {"error": f"Invalid vault root: {vault_root}"}

    file_path = resolved_path
    text = file_path.read_text(encoding="utf-8")
    content_hash = content_fs.compute_content_hash(text)
    parsed = parse_markdown(text)
    title = parsed["title"] or file_path.stem
    rel_path = str(file_path.relative_to(root_path)) if root_path else file_path.name
    # Document identity is the canonical path, not the mutable content hash.
    # This prevents two same-content files from collapsing into one document
    # and lets edits form a real version chain.
    # For single-file indexing there is no vault root, so a basename is not
    # sufficient: ``/a/readme.md`` and ``/b/readme.md`` must remain distinct.
    # Keep the friendly basename in the stored relative_path field, but use
    # the canonical absolute path for the identity hash.
    identity_path = str(file_path) if root_path is None else rel_path
    identity = f"{root_path or ''}\0{identity_path}"
    stable_doc_id = "doc_" + hashlib.sha256(identity.encode("utf-8")).hexdigest()[:32]
    # Reuse an existing path identity where possible so upgrading a database
    # created by the old hash-based indexer does not create a second row.
    path_row = conn.execute(
        "SELECT document_id FROM documents WHERE absolute_path = ? LIMIT 1",
        (str(file_path),),
    ).fetchone()
    doc_id = path_row[0] if path_row else stable_doc_id
    # A file on disk is always "external" — only content created by the
    # remember pipeline (no real file) is "managed".
    source_mode = "external"

    # Check for an existing version with the same hash.  A superseded/deleted
    # version is not "unchanged": when a file is reverted to an older byte
    # sequence it must become the current active version again.
    existing = doc_repo.get_version_by_hash(conn, doc_id, content_hash)
    if existing and existing.get("status") == "active" and not force:
        return {"document_id": doc_id, "status": "unchanged"}

    # When force-reindexing, supersede the old active version and its
    # downstream episodes/observations/assertions so that INSERTs below
    # don't collide with UNIQUE constraints.
    if existing and existing.get("status") == "active" and force:
        # ``document_versions`` has a unique (document_id, content_hash)
        # constraint.  Re-indexing identical bytes cannot create a new
        # version safely; leave the existing version intact instead of
        # raising an IntegrityError after partially superseding its data.
        return {"document_id": doc_id, "version_id": existing.get("document_version_id"), "status": "unchanged"}

    if doc_repo.get_active_version(conn, doc_id):
        doc_repo.supersede_active_version_cascade(conn, doc_id)

    # Re-use a historical version row when the file content is reverted.  The
    # schema intentionally enforces UNIQUE(document_id, content_hash), so
    # inserting a second row for the same bytes would fail.  Old active
    # episodes are superseded and removed from FTS; their historical
    # observations remain available for audit but cannot leak into current
    # search results.
    reuse_version_id = None
    if existing and existing.get("status") != "active":
        reuse_version_id = existing.get("document_version_id")
        old_ep_rows = conn.execute(
            "SELECT episode_id FROM episodes WHERE document_version_id = ? AND status = 'active'",
            (reuse_version_id,),
        ).fetchall()
        old_ep_ids = [row[0] for row in old_ep_rows]
        if old_ep_ids:
            placeholders = ",".join("?" for _ in old_ep_ids)
            conn.execute(
                f"DELETE FROM episodes_fts WHERE episode_id IN ({placeholders})",
                old_ep_ids,
            )
            conn.execute(
                f"UPDATE entity_observations SET status = 'superseded' WHERE episode_id IN ({placeholders}) AND status = 'active'",
                old_ep_ids,
            )
            conn.execute(
                f"UPDATE relation_assertions SET status = 'superseded' WHERE episode_id IN ({placeholders}) AND status = 'active'",
                old_ep_ids,
            )
            conn.execute(
                f"UPDATE episodes SET status = 'superseded' WHERE episode_id IN ({placeholders})",
                old_ep_ids,
            )
        doc_repo.reactivate_version(conn, reuse_version_id)

    # Extract links with line positions from the body (after frontmatter).
    # Re-derive body and count how many frontmatter lines were removed so
    # link line numbers can be corrected to match full-text offsets.
    body_for_links = text or ""
    frontmatter_line_offset = 0
    if body_for_links.startswith("---"):
        end = body_for_links.find("---", 3)
        if end >= 0:
            # Count newlines in the frontmatter block (opening ---, content, closing ---)
            frontmatter_line_offset = text.count("\n", 0, end + 3) + 1
            body_for_links = body_for_links[end + 3:].lstrip("\n")
    links_with_pos = _extract_links_with_positions(body_for_links)
    # Adjust link line numbers to be relative to the full text (including frontmatter)
    if frontmatter_line_offset:
        for link_info in links_with_pos:
            link_info["line_start"] += frontmatter_line_offset
            link_info["line_end"] += frontmatter_line_offset

    # Create document
    abs_path = str(file_path)
    doc_repo.insert_document(
        conn, doc_id, title,
        managed_path="",
        source_mode=source_mode,
        absolute_path=abs_path,
        vault_root=vault_root,
        relative_path=rel_path or file_path.name,
        created_at=now_utc_str(), updated_at=now_utc_str(),
    )

    # Create (or reactivate) version.
    ver_id = reuse_version_id or f"docver_{uuid.uuid4().hex}"
    content_fs.write_version_snapshot(str(library_path), doc_id, content_hash, text)
    if reuse_version_id is None:
        doc_repo.insert_document_version(
            conn, ver_id, doc_id, content_hash,
            version_content_path=f"content/versions/{doc_id}/{content_hash}.md",
            title=title, char_count=len(text), line_count=len(text.splitlines()),
            byte_size=len(text.encode("utf-8")),
            processed_at=now_utc_str(),
        )
    else:
        conn.execute(
            """UPDATE document_versions SET title = ?, char_count = ?, line_count = ?,
               byte_size = ?, status = 'active', processed_at = ?
               WHERE document_version_id = ?""",
            (title, len(text), len(text.splitlines()),
             len(text.encode("utf-8")), now_utc_str(), ver_id),
        )
    doc_repo.update_current_version(conn, doc_id, ver_id, updated_at=now_utc_str())

    # Split into episodes
    from ...text_chunking import split_markdown_chunks
    chunks = split_markdown_chunks(text, window_size=4000, overlap=200)
    # Collect episode records for link resolution: (ep_id, line_start, line_end)
    episode_records: list[tuple[str, int, int]] = []
    for i, chunk in enumerate(chunks):
        # Compute line_start/line_end from offsets
        start_off = chunk.get("start_offset", 0)
        end_off = chunk.get("end_offset", 0)
        line_start = text.count("\n", 0, start_off) + 1
        line_end = text.count("\n", 0, end_off) + 1
        chunk_text = chunk.get("content", "") or chunk.get("text", "")
        chunk_hash = content_fs.compute_content_hash(chunk_text)[:16]
        # 复用已有 episode 行（含 superseded）：文件 revert 回旧内容时走
        # reuse_version_id 路径，旧行仍在库里占着 (document_version_id,
        # chunk_index, chunk_hash) 唯一三元组——重新 INSERT 必撞约束，
        # 该文件从此永远无法重新索引。此处复活原行（沿用 episode_id，
        # 刷新正文/行号等派生字段），无既有行时才 INSERT 新行。
        existing_ep = conn.execute(
            "SELECT episode_id FROM episodes "
            "WHERE document_version_id = ? AND chunk_index = ? AND chunk_hash = ?",
            (ver_id, i, chunk_hash),
        ).fetchone()
        if existing_ep:
            ep_id = existing_ep[0]
            conn.execute(
                """UPDATE episodes SET episode_family_id = ?, source_text = ?,
                   heading_path = ?, start_offset = ?, end_offset = ?,
                   line_start = ?, line_end = ?, name = ?,
                   status = 'active', processed_at = ?
                   WHERE episode_id = ?""",
                (f"epfam_{doc_id}_{i}", chunk_text, chunk.get("heading_path", ""),
                 start_off, end_off, line_start, line_end, chunk.get("heading", ""),
                 now_utc_str(), ep_id),
            )
        else:
            ep_id = f"ep_{uuid.uuid4().hex[:16]}"
            ep_repo.insert_episode(
                conn, ep_id, f"epfam_{doc_id}_{i}", doc_id, ver_id,
                source_text=chunk_text,
                heading_path=chunk.get("heading_path", ""),
                start_offset=start_off,
                end_offset=end_off,
                line_start=line_start,
                line_end=line_end,
                chunk_index=i,
                chunk_hash=chunk_hash,
                name=chunk.get("heading", ""),
                processed_at=now_utc_str(),
            )
        ep_repo.fts_sync_episode(conn, ep_id, doc_id, ver_id,
                                  name=chunk.get("heading", ""),
                                  heading_path=chunk.get("heading_path", ""),
                                  source_text=chunk_text)
        episode_records.append((ep_id, line_start, line_end))

    # Write document_links
    if links_with_pos:
        # Delete any old links for this version (re-index safety)
        doc_repo.delete_document_links_by_version(conn, ver_id)
        now = now_utc_str()
        for link_info in links_with_pos:
            link_target = link_info["link_target"]
            to_doc_id = _resolve_document_id(conn, link_target)
            # Find the episode containing this link's line
            containing_ep = ""
            for ep_id, ep_ls, ep_le in episode_records:
                if ep_ls <= link_info["line_start"] <= ep_le:
                    containing_ep = ep_id
                    break
            doc_repo.insert_document_link(
                conn,
                link_id=f"dl_{uuid.uuid4().hex[:16]}",
                from_document_id=doc_id,
                to_document_id=to_doc_id,
                from_document_version_id=ver_id,
                from_episode_id=containing_ep,
                link_text=link_info["link_text"],
                link_target=link_target,
                line_start=link_info["line_start"],
                line_end=link_info["line_end"],
                created_at=now,
            )

    conn.commit()
    return {"document_id": doc_id, "version_id": ver_id, "chunks": len(chunks), "status": "indexed"}


def index_vault(conn: sqlite3.Connection, library_path: Path,
                path: str, force: bool = False) -> dict:
    """Index all Markdown/text files in a directory (or a single file)."""
    vault_path = Path(path)
    if not vault_path.exists():
        return {"error": f"Path not found: {path}"}

    supported = {".md", ".markdown", ".txt", ".text"}
    if vault_path.is_dir():
        files = sorted(
            p for p in vault_path.rglob("*")
            if p.is_file() and p.suffix.lower() in supported
        )
        vault_root = str(vault_path)
    else:
        files = [vault_path]
        vault_root = ""

    indexed = 0
    errors = 0
    for f in files:
        try:
            result = index_markdown_file(conn, library_path, str(f),
                                          vault_root=vault_root, force=force)
            if "error" not in result:
                indexed += 1
            else:
                errors += 1
        except Exception as e:
            logger.warning("Failed to index %s: %s", f, e)
            try:
                conn.rollback()
            except sqlite3.Error:
                pass
            errors += 1

    return {"files": len(files), "indexed": indexed, "errors": errors}
