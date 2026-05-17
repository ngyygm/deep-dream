"""Schema creation for SQLite graph storage."""

_SCHEMA_SQL = """
-- Entities (versioned by family_id, append-only version chain)
CREATE TABLE IF NOT EXISTS entity (
    uuid TEXT PRIMARY KEY,
    family_id TEXT NOT NULL,
    graph_id TEXT NOT NULL DEFAULT 'default',
    name TEXT NOT NULL DEFAULT '',
    content TEXT NOT NULL DEFAULT '',
    summary TEXT,
    attributes TEXT,
    confidence REAL,
    content_format TEXT DEFAULT 'plain',
    community_id TEXT,
    version_seq INTEGER DEFAULT 1,
    valid_at TEXT,
    event_time TEXT,
    processed_time TEXT,
    episode_id TEXT DEFAULT '',
    source_document TEXT DEFAULT '',
    embedding BLOB
);

-- Relations (versioned by family_id, append-only version chain)
CREATE TABLE IF NOT EXISTS relation (
    uuid TEXT PRIMARY KEY,
    family_id TEXT NOT NULL,
    graph_id TEXT NOT NULL DEFAULT 'default',
    entity1_absolute_id TEXT NOT NULL DEFAULT '',
    entity2_absolute_id TEXT NOT NULL DEFAULT '',
    entity1_family_id TEXT,
    entity2_family_id TEXT,
    content TEXT NOT NULL DEFAULT '',
    summary TEXT,
    attributes TEXT,
    confidence REAL,
    provenance TEXT,
    content_format TEXT DEFAULT 'plain',
    version_seq INTEGER DEFAULT 1,
    valid_at TEXT,
    event_time TEXT,
    processed_time TEXT,
    episode_id TEXT DEFAULT '',
    source_document TEXT DEFAULT '',
    embedding BLOB
);

-- Episodes
CREATE TABLE IF NOT EXISTS episode (
    uuid TEXT PRIMARY KEY,
    graph_id TEXT NOT NULL DEFAULT 'default',
    content TEXT DEFAULT '',
    source_text TEXT DEFAULT '',
    source_document TEXT DEFAULT '',
    event_time TEXT,
    processed_time TEXT,
    episode_type TEXT,
    activity_type TEXT,
    doc_hash TEXT,
    created_at TEXT,
    embedding BLOB
);

-- RELATES_TO edges (Entity <-> Entity graph edges)
CREATE TABLE IF NOT EXISTS relates_to (
    entity1_uuid TEXT NOT NULL,
    entity2_uuid TEXT NOT NULL,
    relation_uuid TEXT,
    fact TEXT,
    graph_id TEXT NOT NULL DEFAULT 'default',
    PRIMARY KEY (entity1_uuid, entity2_uuid, relation_uuid)
);

-- MENTIONS edges (Episode -> Entity/Relation)
CREATE TABLE IF NOT EXISTS mentions (
    episode_uuid TEXT NOT NULL,
    target_uuid TEXT NOT NULL,
    target_type TEXT NOT NULL DEFAULT 'entity',
    context TEXT DEFAULT '',
    entity_absolute_id TEXT,
    graph_id TEXT NOT NULL DEFAULT 'default',
    PRIMARY KEY (episode_uuid, target_uuid, target_type)
);

-- Entity redirects
CREATE TABLE IF NOT EXISTS entity_redirect (
    source_id TEXT PRIMARY KEY,
    target_id TEXT NOT NULL,
    updated_at TEXT
);

-- Content patches
CREATE TABLE IF NOT EXISTS content_patch (
    uuid TEXT PRIMARY KEY,
    target_type TEXT,
    target_absolute_id TEXT,
    target_family_id TEXT,
    section_key TEXT,
    change_type TEXT,
    old_hash TEXT,
    new_hash TEXT,
    diff_summary TEXT,
    source_document TEXT,
    event_time TEXT
);

-- Dream logs
CREATE TABLE IF NOT EXISTS dream_log (
    cycle_id TEXT PRIMARY KEY,
    graph_id TEXT NOT NULL DEFAULT 'default',
    start_time TEXT,
    end_time TEXT,
    status TEXT,
    narrative TEXT,
    insights TEXT,
    connections TEXT,
    consolidations TEXT,
    strategy TEXT,
    entities_examined INTEGER DEFAULT 0,
    relations_created INTEGER DEFAULT 0,
    episode_ids TEXT
);

-- FTS5 virtual tables for fulltext search
CREATE VIRTUAL TABLE IF NOT EXISTS entity_fts USING fts5(name, content, graph_id UNINDEXED);
CREATE VIRTUAL TABLE IF NOT EXISTS relation_fts USING fts5(content, graph_id UNINDEXED);
"""

_INDEXES_SQL = """
CREATE INDEX IF NOT EXISTS idx_entity_family_id ON entity(family_id);
CREATE INDEX IF NOT EXISTS idx_entity_graph_family ON entity(graph_id, family_id);
CREATE INDEX IF NOT EXISTS idx_entity_graph_uuid ON entity(graph_id, uuid);
CREATE INDEX IF NOT EXISTS idx_entity_name ON entity(name);
CREATE INDEX IF NOT EXISTS idx_entity_processed_time ON entity(processed_time);
CREATE INDEX IF NOT EXISTS idx_entity_event_time ON entity(event_time);
CREATE INDEX IF NOT EXISTS idx_entity_graph_family_vseq ON entity(graph_id, family_id, version_seq DESC);
CREATE INDEX IF NOT EXISTS idx_entity_valid_at ON entity(valid_at);
CREATE INDEX IF NOT EXISTS idx_entity_confidence ON entity(confidence);
CREATE INDEX IF NOT EXISTS idx_entity_community ON entity(community_id);
CREATE INDEX IF NOT EXISTS idx_entity_episode_id ON entity(episode_id);
CREATE INDEX IF NOT EXISTS idx_entity_source_document ON entity(source_document);

CREATE INDEX IF NOT EXISTS idx_relation_family_id ON relation(family_id);
CREATE INDEX IF NOT EXISTS idx_relation_graph_family ON relation(graph_id, family_id);
CREATE INDEX IF NOT EXISTS idx_relation_graph_uuid ON relation(graph_id, uuid);
CREATE INDEX IF NOT EXISTS idx_relation_entities ON relation(entity1_absolute_id, entity2_absolute_id);
CREATE INDEX IF NOT EXISTS idx_relation_graph_family_vseq ON relation(graph_id, family_id, version_seq DESC);
CREATE INDEX IF NOT EXISTS idx_relation_processed_time ON relation(processed_time);
CREATE INDEX IF NOT EXISTS idx_relation_source_document ON relation(source_document);
CREATE INDEX IF NOT EXISTS idx_relation_graph_e1_family ON relation(graph_id, entity1_family_id);
CREATE INDEX IF NOT EXISTS idx_relation_graph_e2_family ON relation(graph_id, entity2_family_id);

CREATE INDEX IF NOT EXISTS idx_episode_graph_uuid ON episode(graph_id, uuid);
CREATE INDEX IF NOT EXISTS idx_episode_doc_hash ON episode(doc_hash);
CREATE INDEX IF NOT EXISTS idx_episode_type ON episode(episode_type);

CREATE INDEX IF NOT EXISTS idx_dream_log_graph ON dream_log(graph_id);
CREATE INDEX IF NOT EXISTS idx_redirect_target ON entity_redirect(target_id);
CREATE INDEX IF NOT EXISTS idx_content_patch_target ON content_patch(target_absolute_id);
CREATE INDEX IF NOT EXISTS idx_content_patch_family ON content_patch(target_family_id);

CREATE INDEX IF NOT EXISTS idx_relates_to_e1 ON relates_to(entity1_uuid);
CREATE INDEX IF NOT EXISTS idx_relates_to_e2 ON relates_to(entity2_uuid);
CREATE INDEX IF NOT EXISTS idx_relates_to_graph ON relates_to(graph_id);
CREATE INDEX IF NOT EXISTS idx_mentions_episode ON mentions(episode_uuid);
CREATE INDEX IF NOT EXISTS idx_mentions_target ON mentions(target_uuid);
CREATE INDEX IF NOT EXISTS idx_mentions_entity_abs ON mentions(entity_absolute_id);
"""

_MIGRATION_SQL = """
-- Migration: add version_seq column to tables that lack it.
-- Called by init_schema() when detecting old schema.
"""

_MIGRATION_INDEXES_SQL = """
-- Drop old invalid_at indexes (safe IF EXISTS)
DROP INDEX IF EXISTS idx_entity_graph_family_invalid;
DROP INDEX IF EXISTS idx_entity_invalid_at;
DROP INDEX IF EXISTS idx_entity_graph_invalid;
DROP INDEX IF EXISTS idx_relation_graph_family_invalid;
DROP INDEX IF EXISTS idx_relation_invalid_at;
DROP INDEX IF EXISTS idx_relation_graph_e1_invalid;
DROP INDEX IF EXISTS idx_relation_graph_e2_invalid;
"""


def init_schema(conn):
    """Create all tables and indexes. Handles migration from old schema."""
    _migrate_if_needed(conn)
    for stmt in _SCHEMA_SQL.split(";"):
        stmt = stmt.strip()
        if stmt:
            conn.execute(stmt)
    for stmt in _INDEXES_SQL.split(";"):
        stmt = stmt.strip()
        if stmt:
            conn.execute(stmt)
    conn.commit()


def _migrate_if_needed(conn):
    """Detect and migrate old schema (with invalid_at, without version_seq)."""
    # Check if entity table exists and has version_seq column
    try:
        cols = [row[1] for row in conn.execute("PRAGMA table_info(entity)").fetchall()]
    except Exception:
        return  # Unexpected error, skip migration
    if not cols:
        return  # Table doesn't exist yet, fresh schema will be created
    if "version_seq" in cols:
        return  # Already migrated

    # Old schema detected — add version_seq column
    has_invalid_at = "invalid_at" in cols

    conn.execute("ALTER TABLE entity ADD COLUMN version_seq INTEGER DEFAULT 1")
    conn.execute("ALTER TABLE relation ADD COLUMN version_seq INTEGER DEFAULT 1")

    # Assign version_seq via ROW_NUMBER partitioned by family_id, ordered by processed_time
    conn.execute("""
        UPDATE entity SET version_seq = sub.rn
        FROM (
            SELECT uuid, ROW_NUMBER() OVER (PARTITION BY family_id, graph_id ORDER BY processed_time ASC) AS rn
            FROM entity
        ) sub
        WHERE entity.uuid = sub.uuid
    """)
    conn.execute("""
        UPDATE relation SET version_seq = sub.rn
        FROM (
            SELECT uuid, ROW_NUMBER() OVER (PARTITION BY family_id, graph_id ORDER BY processed_time ASC) AS rn
            FROM relation
        ) sub
        WHERE relation.uuid = sub.uuid
    """)

    # Drop old invalid_at indexes
    if has_invalid_at:
        for stmt in _MIGRATION_INDEXES_SQL.split(";"):
            stmt = stmt.strip()
            if stmt:
                try:
                    conn.execute(stmt)
                except Exception:
                    pass  # Index may not exist

    conn.commit()
