"""Shared helper functions extracted from the legacy argparse CLI.

These are pure utility functions with no argparse dependency.  They accept
a ``SQLiteGraphStorageManager`` instance as their first argument.

NOTE: The ``SQLiteGraphStorageManager`` type hint uses ``Any`` at the
module level to avoid triggering the heavy ``core/__init__.py`` import
chain.  The actual import only happens at call time.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Iterable, List, Optional


# ------------------------------------------------------------------
# Concept helpers
# ------------------------------------------------------------------

_FAMILY_ID_PREFIXES = ("ent_", "rel_", "epfam_", "doc_", "docver_")


def _looks_like_family_id(value: str) -> bool:
    """Return True if *value* is shaped like a canonical Deep-Dream family_id."""
    return bool(value) and value.startswith(_FAMILY_ID_PREFIXES)


def _entity_family_id_by_name(storage: Any, value: str) -> Optional[str]:
    """Resolve a name to an ``ent_`` family_id via exact-then-LIKE lookup.

    Entity resolution is preferred because callers (concept neighbors/versions,
    relation evidence) need an entity/relation endpoint — an episode family id
    (``epfam_…``) is never a valid graph edge endpoint. Returns ``None`` when
    no entity family matches.
    """
    try:
        conn = storage._conn()
    except Exception:
        return None
    try:
        row = conn.execute(
            "SELECT entity_family_id FROM entity_families WHERE canonical_name = ? LIMIT 1",
            (value,),
        ).fetchone()
        if row:
            return row[0]
        row = conn.execute(
            "SELECT entity_family_id FROM entity_families WHERE canonical_name LIKE ? LIMIT 1",
            (f"%{value}%",),
        ).fetchone()
        if row:
            return row[0]
    except Exception:
        return None
    return None


def _bm25_entity_or_relation_id(storage: Any, value: str) -> Optional[str]:
    """Fall back to BM25 search, preferring entity then relation rows.

    Episode rows (``epfam_…``) are deliberately rejected: callers using
    ``resolve_concept_id`` for graph traversal / relation evidence need an
    edge endpoint, and an episode family id is never valid there.
    """
    # Prefer entity matches first.
    for role in ("entity", "relation"):
        try:
            matches = storage.search_concepts_by_bm25(value, role=role, limit=5)
        except (ZeroDivisionError, ValueError):
            # search_fts can raise ZeroDivisionError on degenerate BM25
            # scoring or ValueError on a malformed FTS query (brackets/dots
            # in user content leak straight into MATCH). Treat as no match.
            continue
        except Exception:
            continue
        for m in matches:
            fid = m.get("family_id") or m.get("entity_family_id")
            if fid and (fid.startswith("ent_") or fid.startswith("rel_")):
                return fid
    return None


def resolve_concept_id(storage: Any, value: str) -> Optional[str]:
    """Resolve a concept name or family_id to a canonical family_id.

    Lookup order (entity-preferred so graph traversal / relation evidence get
    a real edge endpoint instead of an episode family id):

      1. Exact family_id (known prefix or found via ``get_concept_by_family_id``)
      2. ``entity_families.canonical_name`` exact-then-LIKE → ``ent_`` match
      3. BM25 search restricted to ``role`` in (entity, relation); episode
         rows are rejected so callers never receive an ``epfam_`` id.
    """
    if not value:
        return None

    # 1. Exact family_id (fast path for already-canonical ids, and the only
    #    branch that may legitimately return an episode/document id).
    if _looks_like_family_id(value):
        concept = storage.get_concept_by_family_id(value)
        if concept:
            return concept["family_id"]
        # Shaped like an id but not found — do not fall through to name
        # search, the caller passed something id-like that doesn't exist.
        return None

    # Also accept a non-prefixed exact id (legacy ids) by attempting lookup.
    concept = storage.get_concept_by_family_id(value)
    if concept:
        return concept["family_id"]

    # 2. Entity canonical_name exact-then-LIKE (works without FTS/embeddings).
    ent_fid = _entity_family_id_by_name(storage, value)
    if ent_fid:
        return ent_fid

    # 3. BM25 fallback, entity/relation only — never an episode id.
    return _bm25_entity_or_relation_id(storage, value)


def concept_source_evidence(
    storage: Any,
    family_ids: Iterable[str],
    limit: int = 20,
) -> list[dict]:
    """Return source-text evidence rows for the given concept family IDs."""
    ids = [fid for fid in dict.fromkeys(family_ids) if fid]
    if not ids:
        return []
    placeholders = ",".join(f":id{i}" for i in range(len(ids)))
    params = {f"id{i}": fid for i, fid in enumerate(ids)}
    return read_sql(
        storage,
        f"""
        SELECT d.title, d.read_path, d.source_mode, ep.version_id AS episode_version_id,
               ep.document_version_id, ep.heading_path, ep.line_start, ep.line_end,
               ep.source_text, m.target_family_id, m.target_name, m.target_role
        FROM v_mentions m
        JOIN v_episodes ep ON ep.version_id = m.episode_version_id
        LEFT JOIN v_document_files d ON d.document_version_id = ep.document_version_id
        WHERE m.target_family_id IN ({placeholders})
        ORDER BY d.processed_time DESC, ep.start_offset
        """,
        params,
        limit=limit,
    )


def relation_evidence(
    storage: Any,
    concept_a: str,
    concept_b: str,
    limit: int = 50,
) -> list[dict]:
    """Return evidence rows linking two concepts via relations."""
    a = resolve_concept_id(storage, concept_a)
    b = resolve_concept_id(storage, concept_b)
    if not a or not b:
        return []
    return read_sql(
        storage,
        """
        SELECT re.relation_family_id, re.relation_version_id,
               re.relation_name, re.relation_content,
               re.entity1_name, re.entity2_name,
               d.title, d.read_path, d.source_mode,
               ep.version_id AS episode_version_id,
               ep.line_start, ep.line_end, ep.source_text
        FROM v_relation_edges re
        JOIN v_episodes ep ON ep.version_id = re.episode_version_id
        LEFT JOIN v_document_files d ON d.document_version_id = re.document_version_id
        WHERE (re.entity1_family_id = :a AND re.entity2_family_id = :b)
           OR (re.entity1_family_id = :b AND re.entity2_family_id = :a)
        ORDER BY d.title, ep.start_offset
        """,
        {"a": a, "b": b},
        limit=limit,
    )


# ------------------------------------------------------------------
# Path helpers
# ------------------------------------------------------------------

def resolve_storage_path(storage: Any, path_value: str) -> Path:
    """Resolve a possibly-relative path value against the storage root."""
    path = Path(path_value)
    if path.is_absolute():
        return path
    resolver = getattr(storage, "_resolve_storage_path", None)
    if resolver is not None:
        try:
            return resolver(path_value)
        except Exception:
            pass
    return Path(storage.storage_path) / path_value


def readable_document_path(
    storage: Any,
    doc: dict,
) -> tuple[Optional[Path], str]:
    """Return ``(resolved_path, verification_label)`` for a document row."""
    candidates: list[tuple[str, str]] = []
    if doc.get("source_mode") == "external" and doc.get("absolute_path"):
        candidates.append((doc["absolute_path"], "raw_file"))
    for key, label in (
        ("read_path", "raw_file"),
        ("managed_path", "raw_file"),
        ("snapshot_path", "snapshot"),
        ("absolute_path", "raw_file"),
    ):
        value = doc.get(key) or ""
        if value:
            candidates.append((value, label))
    seen = set()
    for value, label in candidates:
        if value in seen:
            continue
        seen.add(value)
        path = resolve_storage_path(storage, value)
        if path.is_file():
            return path, label
    return None, "missing"


def document_file_payload(storage: Any, doc: dict) -> dict:
    """Augment a document row with ``resolved_path`` and ``verification``."""
    path, verification = readable_document_path(storage, doc)
    item = dict(doc)
    item["resolved_path"] = str(path) if path else ""
    item["verification"] = verification
    return item


# ------------------------------------------------------------------
# Document queries
# ------------------------------------------------------------------

def document_rows(storage: Any, limit: int = 500) -> list[dict]:
    """Return rows from ``v_document_files`` ordered by most recent first."""
    return read_sql(
        storage,
        """
        SELECT document_version_id, document_family_id, title, source_mode,
               absolute_path, managed_path, snapshot_path, relative_path,
               vault_root, read_path, content_hash, byte_size, char_count,
               line_count, processed_time, complete_windows, total_windows,
               missing_windows
        FROM v_document_files
        ORDER BY processed_time DESC
        """,
        limit=limit,
    )


def map_path_to_documents(
    storage: Any,
    file_path: str,
    limit: int = 20,
) -> list[dict]:
    """Map a file-system path or document title to matching Deep-Dream document rows.

    Matching priority:
      1. Exact path match (absolute_path, managed_path, snapshot_path, read_path, relative_path)
      2. Resolved filesystem path match
      3. Title-based match (exact, then LIKE)
    """
    raw = str(file_path)
    resolved = str(Path(file_path).expanduser().resolve())

    # 1. Exact path match against v_document_files
    rows = read_sql(
        storage,
        """
        SELECT *
        FROM v_document_files
        WHERE absolute_path IN (:raw, :resolved)
           OR managed_path IN (:raw, :resolved)
           OR snapshot_path IN (:raw, :resolved)
           OR read_path IN (:raw, :resolved)
           OR relative_path = :raw
        ORDER BY processed_time DESC
        """,
        {"raw": raw, "resolved": resolved},
        limit=limit,
    )
    if rows:
        return rows

    # 2. Resolved filesystem path match
    matches = []
    for doc in document_rows(storage, limit=5000):
        payload = document_file_payload(storage, doc)
        if payload.get("resolved_path") and str(Path(payload["resolved_path"]).resolve()) == resolved:
            matches.append(doc)
            if len(matches) >= limit:
                break
    if matches:
        return matches

    # 3. Title-based fallback — users often pass document titles, not paths.
    title_rows = read_sql(
        storage,
        """
        SELECT *
        FROM v_document_files
        WHERE title = :title
           OR title = :basename
        ORDER BY processed_time DESC
        """,
        {"title": raw, "basename": Path(raw).name},
        limit=limit,
    )
    if title_rows:
        return title_rows

    # 4. Partial title match as last resort
    partial_rows = read_sql(
        storage,
        """
        SELECT *
        FROM v_document_files
        WHERE title LIKE :pattern
        ORDER BY processed_time DESC
        """,
        {"pattern": f"%{raw}%"},
        limit=limit,
    )
    return partial_rows


def iter_searchable_documents(
    storage: Any,
    limit: int = 1000,
) -> Iterable[dict]:
    """Yield document payloads that have a resolvable file on disk."""
    for doc in document_rows(storage, limit=limit):
        payload = document_file_payload(storage, doc)
        if payload.get("resolved_path"):
            yield payload


def search_document_files(
    storage: Any,
    pattern: str,
    regex: bool = False,
    limit: int = 50,
) -> list[dict]:
    """Search readable document files for *pattern* (literal or regex)."""
    if not pattern:
        raise ValueError("pattern cannot be empty")
    try:
        matcher = re.compile(pattern, re.IGNORECASE) if regex else None
    except re.error as exc:
        raise ValueError(f"Invalid regex pattern: {exc}") from exc
    hits: list[dict] = []
    for doc in iter_searchable_documents(storage):
        path = Path(doc["resolved_path"])
        try:
            lines = path.read_text(encoding="utf-8").splitlines()
        except UnicodeDecodeError:
            try:
                lines = path.read_text(encoding="utf-8-sig").splitlines()
            except Exception:
                continue
        except OSError:
            continue
        for line_no, line in enumerate(lines, start=1):
            matched = bool(matcher.search(line)) if matcher else pattern.lower() in line.lower()
            if not matched:
                continue
            hits.append({
                "document": {
                    "document_version_id": doc.get("document_version_id", ""),
                    "title": doc.get("title", ""),
                    "read_path": doc.get("resolved_path") or doc.get("read_path", ""),
                    "source_mode": doc.get("source_mode", ""),
                    "line_start": line_no,
                    "line_end": line_no,
                },
                "episode": None,
                "concepts": [],
                "relations": [],
                "verification": doc.get("verification", "raw_file"),
                "text": line,
            })
            if len(hits) >= limit:
                return hits
    return hits


# ------------------------------------------------------------------
# Query expansion / term search
# ------------------------------------------------------------------

def expand_query_terms(query: str, explicit_terms: Optional[str] = None) -> list[dict]:
    """Return user/agent-provided query terms without domain-specific defaults."""
    raw_terms = [query.strip()] if query and query.strip() else []
    if explicit_terms:
        raw_terms.extend(t.strip() for t in explicit_terms.split(",") if t.strip())

    seen = set()
    out = []
    for idx, term in enumerate(raw_terms):
        normalized = term.strip()
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        out.append({
            "term": normalized,
            "source": "original" if idx == 0 and normalized == query.strip() else "expanded",
        })
    return out


def search_document_terms(
    storage: Any,
    terms: list[dict],
    per_term_limit: int = 5,
    total_limit: int = 20,
) -> list[dict]:
    """Search documents for each expanded query term."""
    hits: list[dict] = []
    seen = set()
    for term_info in terms:
        term = term_info["term"]
        for hit in search_document_files(storage, term, regex=False, limit=per_term_limit):
            doc = hit.get("document") or {}
            key = (doc.get("document_version_id"), doc.get("line_start"), hit.get("text"))
            if key in seen:
                continue
            seen.add(key)
            hit["matched_term"] = term
            hit["term_source"] = term_info.get("source", "expanded")
            hits.append(hit)
            if len(hits) >= total_limit:
                return hits
    return hits


# ------------------------------------------------------------------
# Text formatting
# ------------------------------------------------------------------

def compact_text(text: str, matched_terms: Iterable[str] = (), max_chars: int = 280) -> str:
    """Compact *text* to *max_chars*, centring on the first matched term."""
    clean = re.sub(r"\s+", " ", text or "").strip()
    if len(clean) <= max_chars:
        return clean
    terms = [t for t in matched_terms if t]
    positions = [clean.find(t) for t in terms if clean.find(t) >= 0]
    if positions:
        center = min(positions)
        start = max(0, center - max_chars // 3)
    else:
        start = 0
    end = min(len(clean), start + max_chars)
    prefix = "..." if start > 0 else ""
    suffix = "..." if end < len(clean) else ""
    return f"{prefix}{clean[start:end]}{suffix}"


# ------------------------------------------------------------------
# Evidence cards
# ------------------------------------------------------------------

def evidence_cards(
    file_hits: list[dict],
    source_evidence: list[dict],
    terms: list[dict],
    limit: int,
) -> list[dict]:
    """Merge file hits and source evidence into unified evidence cards."""
    term_values = [t["term"] for t in terms]
    cards: list[dict] = []
    seen = set()
    for hit in file_hits:
        doc = hit.get("document") or {}
        key = ("file", doc.get("document_version_id"), doc.get("line_start"), hit.get("text"))
        if key in seen:
            continue
        seen.add(key)
        cards.append({
            "claim_hint": "raw document match",
            "document": doc,
            "episode": None,
            "matched_terms": [hit.get("matched_term") or ""],
            "source_excerpt": compact_text(hit.get("text", ""), [hit.get("matched_term", "")]),
            "verification": hit.get("verification", "raw_file"),
        })
        if len(cards) >= limit:
            return cards
    for ev in source_evidence:
        key = ("episode", ev.get("episode_version_id"), ev.get("target_family_id"))
        if key in seen:
            continue
        seen.add(key)
        matched = [t for t in term_values if t and t in (ev.get("source_text") or "")]
        cards.append({
            "claim_hint": ev.get("target_name") or ev.get("target_role") or "graph evidence",
            "document": {
                "document_version_id": ev.get("document_version_id", ""),
                "title": ev.get("title", ""),
                "read_path": ev.get("read_path", ""),
                "source_mode": ev.get("source_mode", ""),
                "line_start": ev.get("line_start"),
                "line_end": ev.get("line_end"),
            },
            "episode": {
                "episode_version_id": ev.get("episode_version_id", ""),
                "heading_path": ev.get("heading_path", ""),
            },
            "concepts": [{
                "family_id": ev.get("target_family_id", ""),
                "name": ev.get("target_name", ""),
                "role": ev.get("target_role", ""),
            }],
            "matched_terms": matched,
            "source_excerpt": compact_text(ev.get("source_text", ""), matched),
            "verification": "source_text",
        })
        if len(cards) >= limit:
            return cards
    return cards


# ------------------------------------------------------------------
# Raw SQL helper
# ------------------------------------------------------------------

def read_sql(
    storage: Any,
    sql: str,
    params: Any = None,
    limit: int = 200,
) -> list[dict]:
    """Execute a read-only SQL query and return rows as list[dict]."""
    return storage.read_sql(sql, params=params or {}, limit=limit)["rows"]
