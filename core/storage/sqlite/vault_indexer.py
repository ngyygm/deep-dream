"""Vault and Markdown file indexing for V1.5 schema."""
from __future__ import annotations

import logging
import re
import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from . import content_fs
from .repositories import documents as doc_repo, episodes as ep_repo

logger = logging.getLogger(__name__)


def _now_str() -> str:
    return datetime.now(timezone.utc).isoformat()


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

    wikilinks = re.findall(r'\[\[([^\]#]+)(?:#[^\]]*)?\]\]', body)
    md_links = re.findall(r'\[([^\]]*)\]\(([^)]+)\)', body)
    tags = set(frontmatter.get("tags", []))
    tags.update(re.findall(r'(?:^|\s)#([a-zA-Z][\w-]*)', body))

    return {
        "title": title or frontmatter.get("title", ""),
        "frontmatter": frontmatter,
        "tags": sorted(tags),
        "aliases": frontmatter.get("aliases", []),
        "wikilinks": wikilinks,
        "md_links": [(text, href) for text, href in md_links if not href.startswith("http")],
    }


def index_markdown_file(conn: sqlite3.Connection, library_path: Path,
                        path: str, vault_root: str = "",
                        force: bool = False) -> dict:
    """Index a single Markdown file into the V1.5 schema."""
    file_path = Path(path)
    if not file_path.exists():
        return {"error": f"File not found: {path}"}

    text = file_path.read_text(encoding="utf-8")
    content_hash = content_fs.compute_content_hash(text)
    parsed = parse_markdown(text)
    title = parsed["title"] or file_path.stem
    doc_id = f"doc_{content_hash[:16]}"
    # A file on disk is always "external" — only content created by the
    # remember pipeline (no real file) is "managed".
    source_mode = "external"

    # Check for existing version with same hash
    existing = doc_repo.get_version_by_hash(conn, doc_id, content_hash)
    if existing and not force:
        return {"document_id": doc_id, "status": "unchanged"}

    # Create document
    abs_path = str(file_path)
    rel_path = ""
    if vault_root and abs_path.startswith(vault_root):
        rel_path = abs_path[len(vault_root):].lstrip("/\\")
    elif vault_root:
        try:
            rel_path = str(file_path.relative_to(vault_root))
        except ValueError:
            pass
    doc_repo.insert_document(
        conn, doc_id, title,
        managed_path="",
        source_mode=source_mode,
        absolute_path=abs_path,
        vault_root=vault_root,
        relative_path=rel_path or file_path.name,
        created_at=_now_str(), updated_at=_now_str(),
    )

    # Create version
    ver_id = f"docver_{content_hash[:16]}"
    content_fs.write_version_snapshot(str(library_path), doc_id, content_hash, text)
    doc_repo.insert_document_version(
        conn, ver_id, doc_id, content_hash,
        version_content_path=f"content/versions/{doc_id}/{content_hash}.md",
        title=title, char_count=len(text), line_count=len(text.splitlines()),
        byte_size=len(text.encode("utf-8")),
        processed_at=_now_str(),
    )
    doc_repo.update_current_version(conn, doc_id, ver_id, updated_at=_now_str())

    # Split into episodes
    from ...text_chunking import split_markdown_chunks
    chunks = split_markdown_chunks(text, window_size=4000, overlap=200)
    for i, chunk in enumerate(chunks):
        # Compute line_start/line_end from offsets
        start_off = chunk.get("start_offset", 0)
        end_off = chunk.get("end_offset", 0)
        line_start = text.count("\n", 0, start_off) + 1
        line_end = text.count("\n", 0, end_off) + 1
        chunk_text = chunk.get("content", "") or chunk.get("text", "")
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
            chunk_hash=content_fs.compute_content_hash(chunk_text)[:16],
            name=chunk.get("heading", ""),
            processed_at=_now_str(),
        )
        ep_repo.fts_sync_episode(conn, ep_id, doc_id, ver_id,
                                  name=chunk.get("heading", ""),
                                  heading_path=chunk.get("heading_path", ""),
                                  source_text=chunk_text)

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
            errors += 1

    return {"files": len(files), "indexed": indexed, "errors": errors}
