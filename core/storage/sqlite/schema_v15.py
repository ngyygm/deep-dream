"""V1.5 Document-first Concept Graph Schema.

Clean schema reset. 12 tables + 1 FTS + 1 view.
No backward compatibility with old concept_* tables.
"""

import logging
import sqlite3

logger = logging.getLogger(__name__)

# ── Table DDL ──────────────────────────────────────────────

TABLES_SQL = [
    """CREATE TABLE IF NOT EXISTS documents (
    document_id TEXT PRIMARY KEY,
    title TEXT DEFAULT '',
    source_mode TEXT NOT NULL DEFAULT 'managed'
        CHECK(source_mode IN ('managed', 'external', 'vault')),
    vault_id TEXT DEFAULT NULL,
    vault_root TEXT DEFAULT NULL,
    relative_path TEXT DEFAULT NULL,
    absolute_path TEXT DEFAULT NULL,
    managed_path TEXT DEFAULT NULL,
    current_version_id TEXT DEFAULT NULL,
    status TEXT NOT NULL DEFAULT 'active'
        CHECK(status IN ('active', 'deleted')),
    created_at TEXT NOT NULL,
    last_indexed_at TEXT DEFAULT NULL,
    updated_at TEXT NOT NULL
)""",

    """CREATE TABLE IF NOT EXISTS document_versions (
    document_version_id TEXT PRIMARY KEY,
    document_id TEXT NOT NULL,
    content_hash TEXT NOT NULL,
    version_content_path TEXT DEFAULT NULL,
    title TEXT DEFAULT '',
    frontmatter_json TEXT DEFAULT '{}' CHECK(json_valid(frontmatter_json)),
    tags_json TEXT DEFAULT '[]' CHECK(json_valid(tags_json)),
    aliases_json TEXT DEFAULT '[]' CHECK(json_valid(aliases_json)),
    char_count INTEGER DEFAULT 0,
    line_count INTEGER DEFAULT 0,
    byte_size INTEGER DEFAULT 0,
    mtime TEXT DEFAULT NULL,
    status TEXT NOT NULL DEFAULT 'active'
        CHECK(status IN ('active', 'superseded', 'stale', 'deleted')),
    processed_at TEXT NOT NULL,
    extra_json TEXT DEFAULT '{}' CHECK(json_valid(extra_json)),
    FOREIGN KEY(document_id) REFERENCES documents(document_id),
    UNIQUE(document_id, content_hash),
    UNIQUE(document_id, document_version_id)
)""",

    """CREATE TABLE IF NOT EXISTS episodes (
    episode_id TEXT PRIMARY KEY,
    episode_family_id TEXT NOT NULL,
    document_id TEXT NOT NULL,
    document_version_id TEXT NOT NULL,
    parent_episode_family_id TEXT DEFAULT NULL,
    name TEXT DEFAULT '',
    source_text TEXT DEFAULT '',
    memory_text TEXT DEFAULT '',
    heading_path TEXT DEFAULT '',
    start_offset INTEGER DEFAULT 0,
    end_offset INTEGER DEFAULT 0,
    line_start INTEGER DEFAULT 0,
    line_end INTEGER DEFAULT 0,
    chunk_index INTEGER DEFAULT 0,
    chunk_hash TEXT DEFAULT '',
    episode_type TEXT DEFAULT '',
    activity_type TEXT DEFAULT '',
    status TEXT NOT NULL DEFAULT 'active'
        CHECK(status IN ('active', 'superseded', 'stale', 'deleted')),
    event_time TEXT DEFAULT NULL,
    processed_at TEXT NOT NULL,
    run_id TEXT DEFAULT NULL,
    extra_json TEXT DEFAULT '{}' CHECK(json_valid(extra_json)),
    FOREIGN KEY(document_id, document_version_id)
        REFERENCES document_versions(document_id, document_version_id),
    UNIQUE(document_version_id, chunk_index, chunk_hash)
)""",

    """CREATE TABLE IF NOT EXISTS entity_families (
    entity_family_id TEXT PRIMARY KEY,
    canonical_name TEXT NOT NULL,
    canonical_content TEXT DEFAULT '',
    created_at TEXT NOT NULL,
    last_seen_at TEXT DEFAULT NULL,
    updated_at TEXT NOT NULL
)""",

    """CREATE TABLE IF NOT EXISTS entity_observations (
    entity_id TEXT PRIMARY KEY,
    entity_family_id TEXT NOT NULL,
    episode_id TEXT,
    name TEXT NOT NULL,
    content TEXT DEFAULT '',
    status TEXT NOT NULL DEFAULT 'active'
        CHECK(status IN ('active', 'superseded', 'stale', 'deleted')),
    processed_at TEXT NOT NULL,
    run_id TEXT DEFAULT NULL,
    extra_json TEXT DEFAULT '{}' CHECK(json_valid(extra_json)),
    FOREIGN KEY(entity_family_id) REFERENCES entity_families(entity_family_id),
    FOREIGN KEY(episode_id) REFERENCES episodes(episode_id)
)""",

    """CREATE TABLE IF NOT EXISTS entity_mentions (
    mention_id TEXT PRIMARY KEY,
    entity_id TEXT NOT NULL,
    entity_family_id TEXT NOT NULL,
    episode_id TEXT,
    surface_text TEXT NOT NULL,
    start_offset INTEGER DEFAULT 0,
    end_offset INTEGER DEFAULT 0,
    line_start INTEGER DEFAULT 0,
    line_end INTEGER DEFAULT 0,
    created_at TEXT NOT NULL,
    FOREIGN KEY(entity_id) REFERENCES entity_observations(entity_id),
    FOREIGN KEY(episode_id) REFERENCES episodes(episode_id)
)""",

    """CREATE TABLE IF NOT EXISTS relation_families (
    relation_family_id TEXT PRIMARY KEY,
    subject_entity_family_id TEXT NOT NULL,
    object_entity_family_id TEXT NOT NULL,
    predicate TEXT DEFAULT '',
    is_directed INTEGER NOT NULL DEFAULT 1 CHECK(is_directed = 1),
    canonical_content TEXT DEFAULT '',
    created_at TEXT NOT NULL,
    last_seen_at TEXT DEFAULT NULL,
    updated_at TEXT NOT NULL,
    FOREIGN KEY(subject_entity_family_id) REFERENCES entity_families(entity_family_id),
    FOREIGN KEY(object_entity_family_id) REFERENCES entity_families(entity_family_id),
    UNIQUE(subject_entity_family_id, object_entity_family_id, predicate, is_directed)
)""",

    """CREATE TABLE IF NOT EXISTS relation_assertions (
    relation_id TEXT PRIMARY KEY,
    relation_family_id TEXT NOT NULL,
    episode_id TEXT,
    subject_entity_id TEXT NOT NULL,
    object_entity_id TEXT NOT NULL,
    subject_entity_family_id TEXT NOT NULL,
    object_entity_family_id TEXT NOT NULL,
    content TEXT DEFAULT '',
    evidence_text TEXT DEFAULT '',
    evidence_start_offset INTEGER DEFAULT 0,
    evidence_end_offset INTEGER DEFAULT 0,
    evidence_line_start INTEGER DEFAULT 0,
    evidence_line_end INTEGER DEFAULT 0,
    status TEXT NOT NULL DEFAULT 'active'
        CHECK(status IN ('active', 'superseded', 'stale', 'deleted')),
    processed_at TEXT NOT NULL,
    run_id TEXT DEFAULT NULL,
    extra_json TEXT DEFAULT '{}' CHECK(json_valid(extra_json)),
    FOREIGN KEY(relation_family_id) REFERENCES relation_families(relation_family_id),
    FOREIGN KEY(episode_id) REFERENCES episodes(episode_id),
    FOREIGN KEY(subject_entity_id) REFERENCES entity_observations(entity_id),
    FOREIGN KEY(object_entity_id) REFERENCES entity_observations(entity_id)
)""",

    """CREATE TABLE IF NOT EXISTS embeddings (
    embedding_id TEXT PRIMARY KEY,
    owner_type TEXT NOT NULL
        CHECK(owner_type IN ('episode', 'entity_obs', 'relation_assert',
                             'entity_family', 'document_version')),
    owner_id TEXT NOT NULL,
    text_kind TEXT NOT NULL
        CHECK(text_kind IN ('source_text', 'memory_text', 'canonical_text', 'content')),
    text_hash TEXT NOT NULL,
    embedding_model TEXT NOT NULL,
    dimensions INTEGER NOT NULL,
    vector BLOB NOT NULL,
    run_id TEXT DEFAULT NULL,
    created_at TEXT NOT NULL,
    UNIQUE(owner_type, owner_id, text_kind, embedding_model, text_hash)
)""",

    """CREATE TABLE IF NOT EXISTS pipeline_runs (
    run_id TEXT PRIMARY KEY,
    run_type TEXT NOT NULL
        CHECK(run_type IN ('remember', 'reindex', 'migration',
                           'chunk', 'embedding', 'fts_rebuild')),
    status TEXT NOT NULL
        CHECK(status IN ('running', 'succeeded', 'failed')),
    document_id TEXT DEFAULT NULL,
    document_version_id TEXT DEFAULT NULL,
    episode_count INTEGER DEFAULT 0,
    entity_count INTEGER DEFAULT 0,
    relation_count INTEGER DEFAULT 0,
    started_at TEXT NOT NULL,
    finished_at TEXT DEFAULT NULL,
    error TEXT DEFAULT NULL,
    extra_json TEXT DEFAULT '{}' CHECK(json_valid(extra_json)),
    FOREIGN KEY(document_id) REFERENCES documents(document_id),
    FOREIGN KEY(document_id, document_version_id)
        REFERENCES document_versions(document_id, document_version_id)
)""",

    """CREATE TABLE IF NOT EXISTS document_links (
    link_id TEXT PRIMARY KEY,
    from_document_id TEXT NOT NULL,
    to_document_id TEXT DEFAULT NULL,
    from_document_version_id TEXT NOT NULL,
    from_episode_id TEXT DEFAULT NULL,
    link_text TEXT DEFAULT '',
    link_target TEXT DEFAULT '',
    line_start INTEGER DEFAULT 0,
    line_end INTEGER DEFAULT 0,
    created_at TEXT NOT NULL,
    FOREIGN KEY(from_document_id, from_document_version_id)
        REFERENCES document_versions(document_id, document_version_id),
    FOREIGN KEY(to_document_id) REFERENCES documents(document_id),
    FOREIGN KEY(from_episode_id) REFERENCES episodes(episode_id)
)""",

    """CREATE TABLE IF NOT EXISTS entity_redirects (
    source_family_id TEXT PRIMARY KEY,
    target_family_id TEXT NOT NULL,
    created_at TEXT NOT NULL
)""",
]

# ── Indexes ────────────────────────────────────────────────

INDEXES_SQL = [
    "CREATE UNIQUE INDEX IF NOT EXISTS idx_docver_one_active "
    "ON document_versions(document_id) WHERE status = 'active'",

    "CREATE UNIQUE INDEX IF NOT EXISTS idx_episodes_one_active_chunk "
    "ON episodes(document_version_id, chunk_index) WHERE status = 'active'",

    "CREATE UNIQUE INDEX IF NOT EXISTS idx_entityobs_unique_active "
    "ON entity_observations(episode_id, entity_family_id) WHERE status = 'active'",

    "CREATE UNIQUE INDEX IF NOT EXISTS idx_relassert_unique_active "
    "ON relation_assertions(episode_id, relation_family_id, "
    "subject_entity_family_id, object_entity_family_id) WHERE status = 'active'",

    "CREATE INDEX IF NOT EXISTS idx_docver_document "
    "ON document_versions(document_id, processed_at DESC)",
    "CREATE INDEX IF NOT EXISTS idx_docver_hash "
    "ON document_versions(content_hash)",

    "CREATE INDEX IF NOT EXISTS idx_episodes_family "
    "ON episodes(episode_family_id, processed_at DESC)",
    "CREATE INDEX IF NOT EXISTS idx_episodes_document "
    "ON episodes(document_id, document_version_id)",

    "CREATE INDEX IF NOT EXISTS idx_entityfam_name "
    "ON entity_families(canonical_name)",

    "CREATE INDEX IF NOT EXISTS idx_entityobs_family "
    "ON entity_observations(entity_family_id, processed_at DESC)",
    "CREATE INDEX IF NOT EXISTS idx_entityobs_episode "
    "ON entity_observations(episode_id)",

    "CREATE INDEX IF NOT EXISTS idx_entitymentions_episode "
    "ON entity_mentions(episode_id)",
    "CREATE INDEX IF NOT EXISTS idx_entitymentions_family "
    "ON entity_mentions(entity_family_id)",

    "CREATE INDEX IF NOT EXISTS idx_relfam_subject "
    "ON relation_families(subject_entity_family_id)",
    "CREATE INDEX IF NOT EXISTS idx_relfam_object "
    "ON relation_families(object_entity_family_id)",

    "CREATE INDEX IF NOT EXISTS idx_relassert_family "
    "ON relation_assertions(relation_family_id, processed_at DESC)",
    "CREATE INDEX IF NOT EXISTS idx_relassert_episode "
    "ON relation_assertions(episode_id)",
    "CREATE INDEX IF NOT EXISTS idx_relassert_subject "
    "ON relation_assertions(subject_entity_family_id)",
    "CREATE INDEX IF NOT EXISTS idx_relassert_object "
    "ON relation_assertions(object_entity_family_id)",

    "CREATE INDEX IF NOT EXISTS idx_embeddings_owner "
    "ON embeddings(owner_type, owner_id)",
    "CREATE INDEX IF NOT EXISTS idx_embeddings_model "
    "ON embeddings(embedding_model)",

    "CREATE INDEX IF NOT EXISTS idx_runs_status "
    "ON pipeline_runs(status)",
    "CREATE INDEX IF NOT EXISTS idx_runs_document "
    "ON pipeline_runs(document_id)",
    "CREATE INDEX IF NOT EXISTS idx_runs_version "
    "ON pipeline_runs(document_id, document_version_id)",

    "CREATE INDEX IF NOT EXISTS idx_redirects_target "
    "ON entity_redirects(target_family_id)",
]

# ── FTS ────────────────────────────────────────────────────

_FTS_TRIGRAM_SQL = """CREATE VIRTUAL TABLE IF NOT EXISTS episodes_fts USING fts5(
    episode_id UNINDEXED,
    document_id UNINDEXED,
    document_version_id UNINDEXED,
    name,
    heading_path,
    source_text,
    memory_text,
    tokenize = 'trigram'
)"""

_FTS_DEFAULT_SQL = """CREATE VIRTUAL TABLE IF NOT EXISTS episodes_fts USING fts5(
    episode_id UNINDEXED,
    document_id UNINDEXED,
    document_version_id UNINDEXED,
    name,
    heading_path,
    source_text,
    memory_text
)"""

# ── Views ──────────────────────────────────────────────────

GRAPH_EDGES_SQL = """CREATE VIEW IF NOT EXISTS graph_edges AS
SELECT 'HAS_EPISODE' AS edge_type,
       e.document_id AS source_id,
       e.episode_id AS target_id,
       e.episode_family_id AS target_family_id
FROM episodes e
JOIN documents d ON d.document_id = e.document_id AND d.status = 'active'
JOIN document_versions dv
  ON dv.document_id = e.document_id
 AND dv.document_version_id = e.document_version_id
 AND dv.status = 'active'
WHERE e.status = 'active'

UNION ALL

SELECT 'MENTIONS' AS edge_type,
       em.episode_id AS source_id,
       em.entity_id AS target_id,
       em.entity_family_id AS target_family_id
FROM entity_mentions em
JOIN entity_observations eo ON eo.entity_id = em.entity_id AND eo.status = 'active'
JOIN episodes e ON e.episode_id = em.episode_id AND e.status = 'active'
JOIN documents d ON d.document_id = e.document_id AND d.status = 'active'
JOIN document_versions dv
  ON dv.document_id = e.document_id
 AND dv.document_version_id = e.document_version_id
 AND dv.status = 'active'

UNION ALL

SELECT 'ASSERTS' AS edge_type,
       ra.episode_id AS source_id,
       ra.relation_id AS target_id,
       ra.relation_family_id AS target_family_id
FROM relation_assertions ra
JOIN episodes e ON e.episode_id = ra.episode_id AND e.status = 'active'
JOIN documents d ON d.document_id = e.document_id AND d.status = 'active'
JOIN document_versions dv
  ON dv.document_id = e.document_id
 AND dv.document_version_id = e.document_version_id
 AND dv.status = 'active'
JOIN entity_observations s ON s.entity_id = ra.subject_entity_id AND s.status = 'active'
JOIN entity_observations o ON o.entity_id = ra.object_entity_id AND o.status = 'active'
WHERE ra.status = 'active'

UNION ALL

SELECT 'RELATES' AS edge_type,
       ra.subject_entity_id AS source_id,
       ra.object_entity_id AS target_id,
       ra.object_entity_family_id AS target_family_id
FROM relation_assertions ra
JOIN episodes e ON e.episode_id = ra.episode_id AND e.status = 'active'
JOIN documents d ON d.document_id = e.document_id AND d.status = 'active'
JOIN document_versions dv
  ON dv.document_id = e.document_id
 AND dv.document_version_id = e.document_version_id
 AND dv.status = 'active'
JOIN entity_observations s ON s.entity_id = ra.subject_entity_id AND s.status = 'active'
JOIN entity_observations o ON o.entity_id = ra.object_entity_id AND o.status = 'active'
WHERE ra.status = 'active'

UNION ALL

SELECT 'DOCUMENT_LINK' AS edge_type,
       dl.from_document_id AS source_id,
       dl.to_document_id AS target_id,
       dl.to_document_id AS target_family_id
FROM document_links dl
JOIN documents d ON d.document_id = dl.from_document_id AND d.status = 'active'
JOIN document_versions dv
  ON dv.document_id = dl.from_document_id
 AND dv.document_version_id = dl.from_document_version_id
 AND dv.status = 'active'
JOIN documents td ON td.document_id = dl.to_document_id AND td.status = 'active'
WHERE dl.to_document_id IS NOT NULL"""

# ── Capability checks ──────────────────────────────────────


def _check_json1(conn: sqlite3.Connection) -> None:
    """Raise if JSON1 is not available."""
    try:
        conn.execute("SELECT json_valid('{}')").fetchone()
    except sqlite3.OperationalError:
        raise RuntimeError(
            "SQLite JSON1 extension not available. "
            "Upgrade SQLite or rebuild Python with JSON1 support."
        )


def _check_fts5(conn: sqlite3.Connection) -> None:
    """Raise if FTS5 is not available."""
    try:
        conn.execute("CREATE VIRTUAL TABLE temp.__fts5_check USING fts5(x)")
        conn.execute("DROP TABLE temp.__fts5_check")
    except sqlite3.OperationalError:
        raise RuntimeError(
            "SQLite FTS5 extension not available. "
            "Upgrade SQLite or rebuild Python with FTS5 support."
        )


def _check_trigram(conn: sqlite3.Connection) -> bool:
    """Return True if FTS5 trigram tokenizer is available."""
    try:
        conn.execute(
            "CREATE VIRTUAL TABLE temp.__trigram_check "
            "USING fts5(x, tokenize='trigram')"
        )
        conn.execute("DROP TABLE temp.__trigram_check")
        return True
    except sqlite3.OperationalError:
        return False


# ── Schema init ────────────────────────────────────────────


def create_tables(conn: sqlite3.Connection) -> None:
    for ddl in TABLES_SQL:
        conn.execute(ddl)


def create_indexes(conn: sqlite3.Connection) -> None:
    for sql in INDEXES_SQL:
        conn.execute(sql)


def create_fts(conn: sqlite3.Connection, use_trigram: bool = True) -> None:
    sql = _FTS_TRIGRAM_SQL if use_trigram else _FTS_DEFAULT_SQL
    conn.execute(sql)


# ── CLI / DocumentService compatibility views ────────────────
#
# These views translate V1.5 tables into the column shapes expected by
# core.cli and core.documents.service so that those modules continue to
# work without rewriting all their SQL.

_COMPAT_VIEWS_SQL = """
CREATE VIEW IF NOT EXISTS v_document_files AS
SELECT
    d.document_id AS document_family_id,
    dv.document_version_id,
    COALESCE(NULLIF(dv.title, ''), d.title) AS title,
    COALESCE(d.source_mode, 'managed') AS source_mode,
    COALESCE(d.absolute_path, '') AS absolute_path,
    COALESCE(d.managed_path, '') AS managed_path,
    COALESCE(dv.version_content_path, '') AS snapshot_path,
    COALESCE(d.relative_path, '') AS relative_path,
    COALESCE(d.vault_root, '') AS vault_root,
    CASE
      WHEN COALESCE(d.source_mode, '') = 'external' AND COALESCE(d.absolute_path, '') != ''
        THEN d.absolute_path
      WHEN COALESCE(d.managed_path, '') != ''
        THEN d.managed_path
      ELSE COALESCE(dv.version_content_path, '')
    END AS read_path,
    dv.content_hash,
    dv.byte_size,
    dv.char_count,
    dv.line_count,
    dv.processed_at AS processed_time,
    0 AS complete_windows,
    0 AS total_windows,
    0 AS missing_windows,
    dv.extra_json AS metadata
FROM documents d
JOIN document_versions dv
  ON dv.document_id = d.document_id AND dv.status = 'active'
WHERE d.status = 'active';

CREATE VIEW IF NOT EXISTS v_episodes AS
SELECT
    ep.episode_id AS version_id,
    ep.episode_family_id AS family_id,
    ep.name,
    ep.source_text AS content,
    ep.source_text AS memory_content,
    ep.source_text,
    length(ep.source_text) AS source_text_length,
    1 AS version_seq,
    ep.event_time,
    ep.processed_at AS processed_time,
    ep.episode_id AS episode_version_id,
    ep.document_version_id,
    ep.name AS source_document,
    ep.document_id AS document_family_id,
    '' AS source_path,
    ep.heading_path,
    ep.start_offset,
    ep.end_offset,
    ep.line_start,
    ep.line_end,
    ep.chunk_index,
    ep.chunk_hash,
    ep.extra_json AS metadata
FROM episodes ep
WHERE ep.status = 'active';

CREATE VIEW IF NOT EXISTS v_latest_concept AS
SELECT
    eo.entity_id AS version_id,
    eo.entity_family_id AS family_id,
    'entity' AS role,
    eo.name,
    eo.content,
    '' AS source_text,
    NULL AS attributes,
    NULL AS confidence,
    'markdown' AS content_format,
    1 AS content_changed,
    1 AS version_seq,
    NULL AS valid_at,
    eo.processed_at AS event_time,
    eo.processed_at AS processed_time,
    eo.episode_id AS episode_version_id,
    '' AS document_version_id,
    '' AS source_document,
    eo.extra_json AS metadata
FROM entity_observations eo
WHERE eo.status = 'active';

CREATE VIEW IF NOT EXISTS v_mentions AS
SELECT
    em.mention_id AS edge_id,
    em.entity_family_id AS target_family_id,
    eo.entity_id AS target_version_id,
    'MENTIONS' AS edge_type,
    em.episode_id AS episode_version_id,
    '' AS document_version_id,
    '' AS source_family_id,
    '' AS source_version_id,
    'entity' AS target_role,
    em.surface_text AS target_name,
    1.0 AS weight,
    NULL AS confidence,
    '{}' AS provenance,
    em.created_at
FROM entity_mentions em
JOIN entity_observations eo
  ON eo.entity_id = em.entity_id AND eo.status = 'active'
WHERE em.episode_id != '';

CREATE VIEW IF NOT EXISTS v_relation_edges AS
SELECT
    ra.relation_id AS relation_edge_id,
    rf.relation_family_id,
    ra.relation_id AS relation_version_id,
    rf.predicate AS relation_name,
    ra.content AS relation_content,
    NULL AS relation_confidence,
    ra.processed_at,
    ra.episode_id AS episode_version_id,
    ep.document_version_id,
    ra.subject_entity_family_id AS entity1_family_id,
    ra.subject_entity_id AS entity1_version_id,
    '' AS entity1_name,
    ra.object_entity_family_id AS entity2_family_id,
    ra.object_entity_id AS entity2_version_id,
    '' AS entity2_name,
    '{}' AS provenance,
    ra.processed_at AS created_at
FROM relation_assertions ra
JOIN relation_families rf
  ON rf.relation_family_id = ra.relation_family_id
LEFT JOIN episodes ep
  ON ep.episode_id = ra.episode_id
WHERE ra.status = 'active';
"""


def create_views(conn: sqlite3.Connection) -> None:
    conn.execute(GRAPH_EDGES_SQL)
    for stmt in _COMPAT_VIEWS_SQL.split(";"):
        stmt = stmt.strip()
        if stmt:
            conn.execute(stmt)


def init_schema_v15(conn: sqlite3.Connection) -> dict:
    """Initialize the complete V1.5 schema.

    Returns a dict with capability info:
        {"fts_tokenizer": "trigram"|"default"}
    """
    conn.execute("PRAGMA foreign_keys = ON")
    conn.execute("PRAGMA journal_mode = WAL")

    _check_json1(conn)
    _check_fts5(conn)
    use_trigram = _check_trigram(conn)

    create_tables(conn)
    create_indexes(conn)
    create_fts(conn, use_trigram=use_trigram)
    create_views(conn)
    conn.commit()

    tokenizer = "trigram" if use_trigram else "default"
    logger.info("V1.5 schema initialized (fts_tokenizer=%s)", tokenizer)
    return {"fts_tokenizer": tokenizer}
