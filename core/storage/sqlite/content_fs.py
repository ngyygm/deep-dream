"""Content filesystem utilities for V1.5.

Handles content/current/ and content/versions/ directories,
content hash computation, and atomic file writes.
"""

import hashlib
import logging
import os
import re
import tempfile

logger = logging.getLogger(__name__)


def compute_content_hash(text: str) -> str:
    """sha256 of normalized UTF-8 text (\\n newlines, no frontmatter strip, no trim)."""
    normalized = text.replace("\r\n", "\n").replace("\r", "\n")
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def compute_file_hash(file_path: str) -> str:
    """Read file, normalize, compute content hash."""
    with open(file_path, "r", encoding="utf-8") as f:
        return compute_content_hash(f.read())


def _safe_title(title: str) -> str:
    """Convert title to filesystem-safe name."""
    name = re.sub(r'[<>:"/\\|?*]', '_', title)
    name = name.strip(". \t\n")
    return name or "untitled"


def _atomic_write(file_path: str, content: str) -> None:
    """Write content atomically via temp file + os.replace."""
    dir_path = os.path.dirname(file_path)
    os.makedirs(dir_path, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(dir=dir_path, suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as f:
            f.write(content)
        os.replace(tmp_path, file_path)
    except BaseException:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise


def write_version_snapshot(library_path: str, doc_id: str,
                           content_hash: str, content: str) -> str:
    """Write content to content/versions/{doc_id}/{hash}.md.

    Returns the relative path from library root.
    """
    rel_dir = os.path.join("content", "versions", doc_id)
    abs_dir = os.path.join(library_path, rel_dir)
    filename = f"{content_hash}.md"
    rel_path = os.path.join(rel_dir, filename)
    abs_path = os.path.join(library_path, rel_path)

    if os.path.exists(abs_path):
        return rel_path

    _atomic_write(abs_path, content)
    return rel_path


def write_current_file(library_path: str, title: str, content: str,
                       doc_id: str = "", disambig_hash: str = "") -> str:
    """Write content to content/current/{safe_title}.md.

    Returns the relative path from library root.
    If disambiguation needed, appends _{hash[:8]} or _{doc_id[:8]}.
    """
    safe = _safe_title(title)
    if disambig_hash:
        safe = f"{safe}_{disambig_hash[:8]}"
    elif doc_id:
        existing = os.path.join(library_path, "content", "current", f"{safe}.md")
        if os.path.exists(existing):
            existing_hash = compute_file_hash(existing)
            new_hash = compute_content_hash(content)
            if existing_hash != new_hash:
                safe = f"{safe}_{doc_id[:8]}"

    rel_path = os.path.join("content", "current", f"{safe}.md")
    abs_path = os.path.join(library_path, rel_path)
    _atomic_write(abs_path, content)
    return rel_path


def rebuild_current_files(conn, library_path: str) -> int:
    """Rebuild all content/ files from DB current_version_id.

    Returns count of files written.
    """
    content_dir = os.path.join(library_path, "content")
    os.makedirs(content_dir, exist_ok=True)

    rows = conn.execute("""
        SELECT d.document_id, d.title, d.managed_path,
               dv.version_content_path
        FROM documents d
        JOIN document_versions dv
          ON dv.document_version_id = d.current_version_id
         AND dv.status = 'active'
        WHERE d.status = 'active'
          AND d.current_version_id IS NOT NULL
    """).fetchall()

    count = 0
    for doc_id, title, managed_path, ver_path in rows:
        src = None
        if ver_path:
            candidate = os.path.join(library_path, ver_path)
            if os.path.exists(candidate):
                src = candidate
        if not src and managed_path:
            candidate = os.path.join(library_path, managed_path)
            if os.path.exists(candidate):
                src = candidate
        if not src:
            logger.warning("rebuild_content: no source for doc %s", doc_id)
            continue

        with open(src, "r", encoding="utf-8") as f:
            content = f.read()
        safe = _safe_title(title or doc_id)
        dest = os.path.join(content_dir, f"{safe}.md")
        if os.path.exists(dest) and compute_file_hash(dest) == compute_content_hash(content):
            continue
        _atomic_write(dest, content)
        count += 1

    return count


def write_submitted_content(library_path: str, source_name: str, content: str) -> str:
    """Write user-submitted content to library/content/{name}.md.

    Ensures .md extension, deduplicates by content hash.
    Returns the relative path from library root.
    """
    safe = _safe_title(source_name)
    root, ext = os.path.splitext(safe)
    if ext.lower() != ".md":
        safe = f"{root}.md"

    content_dir = os.path.join(library_path, "content")
    rel_path = os.path.join("content", safe)
    abs_path = os.path.join(library_path, rel_path)

    if os.path.exists(abs_path):
        existing_hash = compute_file_hash(abs_path)
        new_hash = compute_content_hash(content)
        if existing_hash == new_hash:
            return rel_path
        safe = f"{root}_{new_hash[:8]}.md"
        rel_path = os.path.join("content", safe)
        abs_path = os.path.join(library_path, rel_path)

    _atomic_write(abs_path, content)
    return rel_path


def cleanup_temp_files(library_path: str) -> int:
    """Remove stale .tmp files in content/ directories."""
    count = 0
    for root, _dirs, files in os.walk(os.path.join(library_path, "content")):
        for fname in files:
            if fname.endswith(".tmp"):
                try:
                    os.unlink(os.path.join(root, fname))
                    count += 1
                except OSError:
                    pass
    return count


def migrate_legacy_markdown(library_path: str) -> dict:
    """Migrate markdown files from legacy directories to content/.

    Priority: content/ > documents/managed > snapshots (fallback).

    Returns {"migrated": int, "skipped": int, "conflicts": list}.
    """
    result = {"migrated": 0, "skipped": 0, "conflicts": []}
    content_dir = os.path.join(library_path, "content")
    os.makedirs(content_dir, exist_ok=True)

    sources = []
    for legacy_dir in ["documents/managed", "snapshots"]:
        abs_dir = os.path.join(library_path, legacy_dir)
        if os.path.isdir(abs_dir):
            for fname in os.listdir(abs_dir):
                if fname.endswith(".md"):
                    sources.append(os.path.join(abs_dir, fname))

    seen = {}
    for src_path in sources:
        fname = os.path.basename(src_path)
        dest = os.path.join(content_dir, fname)

        if fname in seen:
            src_hash = compute_file_hash(src_path)
            for existing_path in seen[fname]:
                if compute_file_hash(existing_path) == src_hash:
                    result["skipped"] += 1
                    break
            else:
                safe = _safe_title(os.path.splitext(fname)[0])
                disambig = src_hash[:8]
                dest = os.path.join(content_dir, f"{safe}_{disambig}.md")
                with open(src_path, "r", encoding="utf-8") as f:
                    content = f.read()
                _atomic_write(dest, content)
                result["migrated"] += 1
                result["conflicts"].append((src_path, dest))
            seen[fname].append(src_path)
            continue

        if os.path.exists(dest):
            dest_hash = compute_file_hash(dest)
            src_hash = compute_file_hash(src_path)
            if dest_hash == src_hash:
                result["skipped"] += 1
                seen[fname] = [src_path]
                continue
            safe = _safe_title(os.path.splitext(fname)[0])
            disambig = compute_file_hash(src_path)[:8]
            dest = os.path.join(current_dir, f"{safe}_{disambig}.md")
            result["conflicts"].append((src_path, dest))

        with open(src_path, "r", encoding="utf-8") as f:
            content = f.read()
        _atomic_write(dest, content)
        seen[fname] = [src_path]
        result["migrated"] += 1

    return result
