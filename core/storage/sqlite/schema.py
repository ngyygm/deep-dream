"""SQLite schema for the v1 Document-first Concept graph.

This schema intentionally replaces the legacy entity/relation/episode tables.
The old DTO names may still exist inside the remember pipeline, but storage is
now concept-family / concept-version / concept-edge based.
"""

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS concept_family (
    family_id TEXT PRIMARY KEY,
    graph_id TEXT NOT NULL,
    role TEXT NOT NULL,
    canonical_name TEXT DEFAULT '',
    status TEXT DEFAULT 'active',
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    metadata TEXT DEFAULT '{}'
);

CREATE TABLE IF NOT EXISTS concept_version (
    version_id TEXT PRIMARY KEY,
    family_id TEXT NOT NULL,
    graph_id TEXT NOT NULL,
    role TEXT NOT NULL,
    name TEXT DEFAULT '',
    content TEXT DEFAULT '',
    summary TEXT,
    attributes TEXT,
    confidence REAL,
    content_format TEXT DEFAULT 'markdown',
    content_changed INTEGER DEFAULT 1,
    version_seq INTEGER NOT NULL,
    valid_at TEXT,
    event_time TEXT,
    processed_time TEXT NOT NULL,
    episode_version_id TEXT DEFAULT '',
    document_version_id TEXT DEFAULT '',
    source_document TEXT DEFAULT '',
    embedding BLOB,
    metadata TEXT DEFAULT '{}',
    FOREIGN KEY(family_id) REFERENCES concept_family(family_id)
);

CREATE TABLE IF NOT EXISTS concept_edge (
    edge_id TEXT PRIMARY KEY,
    graph_id TEXT NOT NULL,
    edge_type TEXT NOT NULL,
    source_family_id TEXT DEFAULT '',
    source_version_id TEXT DEFAULT '',
    target_family_id TEXT DEFAULT '',
    target_version_id TEXT DEFAULT '',
    relation_family_id TEXT DEFAULT '',
    relation_version_id TEXT DEFAULT '',
    episode_version_id TEXT DEFAULT '',
    document_version_id TEXT DEFAULT '',
    weight REAL DEFAULT 1.0,
    confidence REAL,
    provenance TEXT DEFAULT '{}',
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS document_source (
    source_id TEXT PRIMARY KEY,
    graph_id TEXT NOT NULL,
    document_family_id TEXT NOT NULL,
    vault_id TEXT DEFAULT '',
    absolute_path TEXT DEFAULT '',
    relative_path TEXT DEFAULT '',
    uri TEXT DEFAULT '',
    title TEXT DEFAULT '',
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    metadata TEXT DEFAULT '{}'
);

CREATE TABLE IF NOT EXISTS document_version (
    document_version_id TEXT PRIMARY KEY,
    graph_id TEXT NOT NULL,
    document_family_id TEXT NOT NULL,
    source_id TEXT NOT NULL,
    content_hash TEXT NOT NULL,
    blob_path TEXT NOT NULL,
    title TEXT DEFAULT '',
    frontmatter_json TEXT DEFAULT '{}',
    tags_json TEXT DEFAULT '[]',
    aliases_json TEXT DEFAULT '[]',
    mtime TEXT,
    size INTEGER DEFAULT 0,
    processed_time TEXT NOT NULL,
    metadata TEXT DEFAULT '{}'
);

CREATE TABLE IF NOT EXISTS blob_manifest (
    content_hash TEXT NOT NULL,
    graph_id TEXT NOT NULL,
    blob_path TEXT NOT NULL,
    size INTEGER DEFAULT 0,
    created_at TEXT NOT NULL,
    PRIMARY KEY (graph_id, content_hash)
);

CREATE TABLE IF NOT EXISTS concept_redirect (
    source_family_id TEXT PRIMARY KEY,
    target_family_id TEXT NOT NULL,
    graph_id TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE VIRTUAL TABLE IF NOT EXISTS concept_version_fts USING fts5(
    name,
    content,
    role UNINDEXED,
    family_id UNINDEXED,
    version_id UNINDEXED,
    graph_id UNINDEXED
);
"""

INDEX_SQL = """
CREATE INDEX IF NOT EXISTS idx_cf_graph_role ON concept_family(graph_id, role);
CREATE INDEX IF NOT EXISTS idx_cf_graph_name ON concept_family(graph_id, canonical_name);

CREATE INDEX IF NOT EXISTS idx_cv_graph_family_seq ON concept_version(graph_id, family_id, version_seq DESC);
CREATE INDEX IF NOT EXISTS idx_cv_graph_role_time ON concept_version(graph_id, role, processed_time DESC);
CREATE INDEX IF NOT EXISTS idx_cv_episode ON concept_version(graph_id, episode_version_id);
CREATE INDEX IF NOT EXISTS idx_cv_document ON concept_version(graph_id, document_version_id);

CREATE INDEX IF NOT EXISTS idx_edge_graph_type_source ON concept_edge(graph_id, edge_type, source_family_id);
CREATE INDEX IF NOT EXISTS idx_edge_graph_type_target ON concept_edge(graph_id, edge_type, target_family_id);
CREATE INDEX IF NOT EXISTS idx_edge_episode ON concept_edge(graph_id, episode_version_id);
CREATE INDEX IF NOT EXISTS idx_edge_relation ON concept_edge(graph_id, relation_family_id);

CREATE INDEX IF NOT EXISTS idx_doc_source_graph_doc ON document_source(graph_id, document_family_id);
CREATE INDEX IF NOT EXISTS idx_doc_source_path ON document_source(graph_id, absolute_path, relative_path);
CREATE INDEX IF NOT EXISTS idx_doc_version_graph_doc ON document_version(graph_id, document_family_id, processed_time DESC);
CREATE INDEX IF NOT EXISTS idx_doc_version_hash ON document_version(graph_id, content_hash);
"""


def init_schema(conn):
    """Create the v1 concept graph schema."""
    for sql in (SCHEMA_SQL, INDEX_SQL):
        for stmt in sql.split(";"):
            stmt = stmt.strip()
            if stmt:
                conn.execute(stmt)
    conn.commit()
