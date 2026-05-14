#!/usr/bin/env python3
"""
Tool schema definitions for Deep Dream MCP Server.

All _t(...) calls that define the MCP tool list, including parameter schemas
and descriptions. The TOOLS list is consumed by the main server to respond to
tools/list requests.
"""

TOOLS = []

# Shared graph_id parameter — injected into every tool for per-call graph routing.
# If omitted, uses the active graph (set by switch_graph, defaults to env var).
_GRAPH_ID_PARAM = {
    "graph_id": {
        "type": "string",
        "description": "Target graph ID. If omitted, operates on the active graph (set via switch_graph, defaults to env DEEP_DREAM_GRAPH_ID='default'). Use list_graphs to see available graphs.",
    },
}


def _t(name, desc, params, required=None):
    # Inject graph_id into every tool's parameter schema
    merged_params = {**params, **_GRAPH_ID_PARAM}
    TOOLS.append({
        "name": name,
        "description": desc,
        "inputSchema": {
            "type": "object",
            "properties": merged_params,
            **({"required": required} if required else {}),
        },
    })


# ── Remember (7) ──────────────────────────────────────────────────────────

_t("remember", "Submit text for async entity/relation extraction. Returns immediately with a task_id — poll remember_task_status(task_id='...') to check progress. The pipeline extracts entities and relations from the text and adds them to the knowledge graph. Typical workflow: remember(content='...') → remember_task_status(task_id='...') → repeat until completed.", {
    "content": {"type": "string", "description": "Text content to remember"},
    "source": {"type": "string", "description": "Source label (e.g. 'user', 'document:file.txt')"},
    "metadata": {"type": "object", "description": "Optional metadata dict"},
}, ["content"])

_t("remember_tasks", "List remember task queue. Use status filter to see pending/processing/completed/failed tasks. For a specific task, use remember_task_status(task_id=...).", {
    "status": {"type": "string", "description": "Filter by status (pending/processing/completed/failed)"},
})

_t("remember_task_status", "Get status of a specific remember task. Poll this after calling remember until status='completed'. The task_id comes from the remember response.", {
    "task_id": {"type": "string", "description": "Task ID (from remember() response)"},
}, ["task_id"])

_t("delete_remember_task", "Delete a remember task. Use this to clean up completed or failed tasks from the queue.", {
    "task_id": {"type": "string", "description": "Task ID (from remember() response)"},
}, ["task_id"])

_t("pause_remember_task", "Pause a running remember task. The task can be resumed later with resume_remember_task.", {
    "task_id": {"type": "string", "description": "Task ID (from remember() response)"},
}, ["task_id"])

_t("resume_remember_task", "Resume a paused remember task. Only paused tasks can be resumed.", {
    "task_id": {"type": "string", "description": "Task ID (from remember() response)"},
}, ["task_id"])

_t("remember_monitor", "Get remember pipeline monitor snapshot. Shows pending/processing counts. For task-level details, use remember_tasks.", {})


# ── Health (2) ────────────────────────────────────────────────────────────

_t("health_check", "Check if the Deep Dream API server is running and responsive", {})
_t("health_check_llm", "Check if the LLM backend (used for extraction/Q&A) is reachable. Use this when remember or ask calls fail — may indicate LLM provider issues.", {})


# ── Stats (3) ─────────────────────────────────────────────────────────────

_t("search_stats", "Get search engine usage statistics (query counts, cache hit rates). For graph content stats, use graph_stats or graph_summary.", {})
_t("graph_stats", "Get graph statistics: entity count and relation count. For a richer overview including backend type and embedding status, use graph_summary.", {})


# ── Find/Search (5) ──────────────────────────────────────────────────────

_t("semantic_search", "Semantic search across entities and relations. Use mode='entities' for entities only, mode='relations' for relations only. For most searches, quick_search is simpler and faster — only use semantic_search when you need: (1) specific mode control, (2) expanded graph context, or (3) custom top_k per category.", {
    "query": {"type": "string", "description": "Search query text"},
    "top_k": {"type": "integer", "description": "Max results per category (default 10)"},
    "mode": {"type": "string", "description": "Search mode: entities, relations, or all (default)"},
    "expand": {"type": "boolean", "description": "Whether to expand graph context (default false)"},
}, ["query"])

_t("search_candidates", "Find candidate entities matching a description using hybrid search. Use this before create_entity to avoid duplicates, or during entity resolution to check if a similar entity already exists. For simple name lookups, find_entity_by_name is faster.", {
    "description": {"type": "string", "description": "Entity description to match"},
    "top_k": {"type": "integer", "description": "Max candidates (default 10)"},
}, ["description"])

_t("search_entities", "Search entities by text query. Returns matching entities with names and content. Use this for broad text-based search; for semantic similarity use semantic_search, or for name lookup use find_entity_by_name.", {
    "query": {"type": "string", "description": "Search query"},
    "limit": {"type": "integer", "description": "Max results (default 20)"},
    "offset": {"type": "integer", "description": "Offset for pagination (0-based)"},
}, ["query"])

_t("search_relations", "Search relations by text query. Returns matching relations with content and connected entities. For comprehensive search returning both entities and relations, prefer quick_search.", {
    "query": {"type": "string", "description": "Search query"},
    "limit": {"type": "integer", "description": "Max results (default 20)"},
    "offset": {"type": "integer", "description": "Offset for pagination (0-based)"},
}, ["query"])

_t("traverse_graph", "BFS traverse from seed entity(s) to discover connected subgraph. Returns entities and relations within max_depth hops. Use depth=2 for immediate neighborhood, depth=3 for broader context. Good for understanding how entities are interconnected. Use this instead of entity_profile when you want to explore beyond a single entity's direct relations. Supports time_point for temporal traversal.", {
    "start_entity_id": {"type": "string", "description": "Starting entity family_id (or JSON array of family_ids for multiple seeds)"},
    "max_depth": {"type": "integer", "description": "Max traversal depth (default 2)"},
    "max_nodes": {"type": "integer", "description": "Max nodes to return (default 50)"},
    "time_point": {"type": "string", "description": "ISO 8601 timestamp — only return entities/relations valid at this time"},
}, ["start_entity_id"])


# ── Entity Query (10) ────────────────────────────────────────────────────

_t("list_entities", "List all entities with pagination. Use this only to browse the full entity catalog. For finding specific entities, prefer quick_search, find_entity_by_name, or search_entities instead.", {
    "limit": {"type": "integer", "description": "Max results (default 50)"},
    "offset": {"type": "integer", "description": "Offset for pagination (0-based)"},
})

_t("get_entity", "Get entity current version by family_id. Returns the entity's content, summary, attributes. NOTE: if you also need the entity's relations (most common case), use entity_profile instead — it returns entity + relations + version count in one call. Use get_entity only when you need just the raw entity data or the absolute_id for create_relation.", {
    "family_id": {"type": "string", "description": "Entity family ID (e.g. 'ent_abc123') — use find_entity_by_name if unknown"},
}, ["family_id"])

_t("get_entity_versions", "Get all versions of an entity. Each version represents a state change. Use this to audit how an entity evolved. For a chronological view with relation events, prefer get_entity_timeline. For comparing two specific versions, use get_entity_version_diff.", {
    "family_id": {"type": "string", "description": "Entity family ID (e.g. 'ent_abc123')"},
    "limit": {"type": "integer", "description": "Max versions to return"},
    "offset": {"type": "integer", "description": "Offset for pagination (0-based)"},
}, ["family_id"])

_t("get_entity_at_time", "Get entity state at an exact point in time (time travel). Returns the entity as it was at that timestamp. For approximate matches, use get_entity_nearest_to_time. For a time range, use get_entity_around_time.", {
    "family_id": {"type": "string", "description": "Entity family ID (e.g. 'ent_abc123')"},
    "timestamp": {"type": "string", "description": "ISO 8601 timestamp (e.g. '2024-06-01T12:00:00')"},
}, ["family_id", "timestamp"])

_t("get_entity_nearest_to_time", "Get entity version closest to a given time. Tolerates slight time mismatches.", {
    "family_id": {"type": "string", "description": "Entity family ID (e.g. 'ent_abc123')"},
    "timestamp": {"type": "string", "description": "ISO 8601 timestamp"},
}, ["family_id", "timestamp"])

_t("get_entity_around_time", "Get entity versions within a time window around a point. Useful for seeing what changed near a specific time.", {
    "family_id": {"type": "string", "description": "Entity family ID (e.g. 'ent_abc123')"},
    "timestamp": {"type": "string", "description": "ISO 8601 timestamp (center of window)"},
    "within_seconds": {"type": "number", "description": "Time window radius in seconds (e.g. 3600 for ±1 hour)"},
}, ["family_id", "timestamp"])

_t("get_entity_relations", "Get relations connected to an entity. Use relation_scope to control time range. For a complete view with entity details + relations + version count in one call, prefer entity_profile.", {
    "family_id": {"type": "string", "description": "Entity family ID (e.g. 'ent_abc123')"},
    "limit": {"type": "integer", "description": "Max results (default 50)"},
    "time_point": {"type": "string", "description": "Filter relations by time (ISO 8601)"},
    "relation_scope": {"type": "string", "description": "accumulated (default, all active), version_only (current version), all_versions (including future)"},
}, ["family_id"])

_t("get_entity_timeline", "Get entity timeline: all version changes and relation events in chronological order. Combines both version history and relation changes in one view. For version-only history use get_entity_versions, for relation-only use get_entity_relations.", {
    "family_id": {"type": "string", "description": "Entity family ID (e.g. 'ent_abc123')"},
    "limit": {"type": "integer", "description": "Max events"},
}, ["family_id"])

_t("get_entity_by_absolute_id", "Get a specific entity version by its absolute (version) ID. Use family_id for the current version instead.", {
    "absolute_id": {"type": "string", "description": "Entity absolute/version ID (UUID format, from get_entity_versions)"},
}, ["absolute_id"])

_t("get_entity_version_counts", "Get version counts for multiple entities in one call. Useful for identifying entities with many versions that may need cleanup or consolidation.", {
    "family_ids": {"type": "array", "items": {"type": "string"}, "description": "List of entity family IDs"},
}, ["family_ids"])


# ── Entity CRUD (8) ──────────────────────────────────────────────────────

_t("create_entity", "Create a new entity manually. For bulk creation from text, use remember instead.", {
    "name": {"type": "string", "description": "Entity name (required)"},
    "content": {"type": "string", "description": "Entity content/description"},
    "episode_id": {"type": "string", "description": "Episode ID to link (optional)"},
    "source_document": {"type": "string", "description": "Source document label (optional)"},
}, ["name"])

_t("update_entity", "Update entity metadata (name, summary, attributes) by family_id. Does NOT create a new version — modifies the current version in place. Use evolve_entity_summary for AI-driven summary regeneration.", {
    "family_id": {"type": "string", "description": "Entity family ID (e.g. 'ent_abc123')"},
    "name": {"type": "string", "description": "New name"},
    "summary": {"type": "string", "description": "New summary"},
    "attributes": {"type": "object", "description": "Updated attributes (merged with existing)"},
    "source": {"type": "string", "description": "Source label for this update"},
}, ["family_id"])

_t("update_entity_by_absolute_id", "Update a specific entity version by absolute_id. Does NOT create a new version.", {
    "absolute_id": {"type": "string", "description": "Entity absolute/version ID (UUID format, from get_entity_versions)"},
    "name": {"type": "string", "description": "New name"},
    "summary": {"type": "string", "description": "New summary"},
    "attributes": {"type": "object", "description": "Updated attributes"},
}, ["absolute_id"])

_t("delete_entity", "Delete entity and all its versions by family_id. This is permanent and cannot be undone. For a softer approach, consider whether the entity can be left as-is or merged with merge_entities.", {
    "family_id": {"type": "string", "description": "Entity family ID (e.g. 'ent_abc123')"},
}, ["family_id"])

_t("delete_entity_by_absolute_id", "Delete a specific entity version by absolute_id. Does NOT delete the entire entity — only the specified version snapshot.", {
    "absolute_id": {"type": "string", "description": "Entity absolute/version ID (UUID format, from get_entity_versions)"},
}, ["absolute_id"])

_t("batch_delete_entities", "Delete multiple entities at once by their family IDs. For single entity deletion, use delete_entity. For targeted version removal, use batch_delete_entity_versions.", {
    "family_ids": {"type": "array", "items": {"type": "string"}, "description": "List of entity family IDs to delete"},
}, ["family_ids"])

_t("merge_entities", "Merge multiple entities into one. All relations and versions are consolidated into the target entity. Workflow: (1) search_similar_entities or find_entity_by_name to identify duplicates, (2) batch_profiles to compare content, (3) merge_entities to consolidate. For cross-language duplicates (e.g. 'Microsoft' and '微软'), set skip_name_check=true. Irreversible — verify before merging.", {
    "family_ids": {"type": "array", "items": {"type": "string"}, "description": "All entity family IDs to merge (target + sources)"},
    "skip_name_check": {"type": "boolean", "description": "Skip name similarity check. Required for cross-language duplicates (e.g. English + Chinese names for same entity)."},
    "target_family_id": {"type": "string", "description": "Which entity to keep as the target (optional, defaults to first)"},
    "target_name": {"type": "string", "description": "New name for merged entity (optional)"},
    "target_summary": {"type": "string", "description": "New summary for merged entity (optional)"},
}, ["family_ids"])


_t("refresh_graph_edges", "Rebuild RELATES_TO traversal edges from Relation nodes. Call after entity merges, alignment, or dream cycles to ensure graph traversal is consistent. Idempotent - safe to call repeatedly. Returns count of deleted stale + created new edges.", {}, [])


_t("split_entity_version", "Separate a specific version into its own new entity. Useful when an entity has accumulated mixed topics across versions. Workflow: get_entity_versions → identify the version to split → split_entity_version. Get the version_id (absolute_id) from get_entity_versions.", {
    "family_id": {"type": "string", "description": "Source entity family ID"},
    "version_id": {"type": "string", "description": "Version absolute ID to split out"},
    "new_name": {"type": "string", "description": "Name for the new entity"},
}, ["family_id", "version_id"])


# ── Entity Intelligence (6) ──────────────────────────────────────────────

_t("evolve_entity_summary", "Use LLM to regenerate entity summary by analyzing all version history. Call this when an entity has accumulated significant new information across multiple versions and the summary is outdated or incomplete. Uses one LLM call per entity.", {
    "family_id": {"type": "string", "description": "Entity family ID (e.g. 'ent_abc123')"},
    "context": {"type": "string", "description": "Additional context to incorporate (optional)"},
}, ["family_id"])

_t("get_entity_contradictions", "Detect contradictions between entity versions. Returns list of conflicting data points with severity. Call this after remember adds new data to check for inconsistencies. Follow with resolve_entity_contradiction to fix.", {
    "family_id": {"type": "string", "description": "Entity family ID (e.g. 'ent_abc123')"},
}, ["family_id"])

_t("resolve_entity_contradiction", "Resolve a detected contradiction by choosing a strategy (keep_new, keep_old, merge, flag_for_review). Call get_entity_contradictions first to get the contradiction_id. Use 'flag_for_review' when unsure.", {
    "family_id": {"type": "string", "description": "Entity family ID (e.g. 'ent_abc123')"},
    "contradiction_id": {"type": "string", "description": "Contradiction ID from get_entity_contradictions"},
    "resolution": {"type": "string", "description": "Resolution strategy: keep_new, keep_old, merge, or flag_for_review"},
}, ["family_id", "contradiction_id", "resolution"])

_t("get_relation_contradictions", "Detect contradictions between relation versions. Returns list of conflicting data points with severity. Call this after remember adds new data to check for inconsistencies in relations. Follow with resolve_relation_contradiction to fix.", {
    "family_id": {"type": "string", "description": "Relation family ID (e.g. 'rel_abc123')"},
}, ["family_id"])

_t("resolve_relation_contradiction", "Resolve a detected relation contradiction by choosing a strategy (keep_new, keep_old, merge, flag_for_review). Call get_relation_contradictions first to get the contradiction_id. Use 'flag_for_review' when unsure.", {
    "family_id": {"type": "string", "description": "Relation family ID (e.g. 'rel_abc123')"},
    "contradiction_id": {"type": "string", "description": "Contradiction ID from get_relation_contradictions"},
    "resolution": {"type": "string", "description": "Resolution strategy: keep_new, keep_old, merge, or flag_for_review"},
}, ["family_id", "contradiction_id", "resolution"])

_t("get_entity_provenance", "Trace where entity data came from: source documents, extraction timestamps, confidence scores. Use this to verify data reliability or debug incorrect extractions. For a broader audit, combine with get_entity_versions.", {
    "family_id": {"type": "string", "description": "Entity family ID (e.g. 'ent_abc123')"},
}, ["family_id"])

_t("get_entity_version_diff", "Compare two entity versions to see what changed. Provide both from_version and to_version for meaningful results.", {
    "family_id": {"type": "string", "description": "Entity family ID (e.g. 'ent_abc123')"},
    "from_version": {"type": "string", "description": "Source version absolute ID (get from get_entity_versions)"},
    "to_version": {"type": "string", "description": "Target version absolute ID"},
}, ["family_id"])

_t("get_entity_patches", "Get incremental patches (diffs) applied to an entity over time. Shows what was added/removed at each version transition. For comparing two specific versions, use get_entity_version_diff instead.", {
    "family_id": {"type": "string", "description": "Entity family ID (e.g. 'ent_abc123')"},
    "limit": {"type": "integer", "description": "Max patches to return"},
}, ["family_id"])


# ── Relation Query (6) ───────────────────────────────────────────────────

_t("list_relations", "List all relations with pagination and optional type filter. Use this only to browse the full catalog. For finding specific relations, prefer search_relations, quick_search, or get_relations_between.", {
    "limit": {"type": "integer", "description": "Max results (default 50)"},
    "offset": {"type": "integer", "description": "Offset for pagination (0-based)"},
    "relation_type": {"type": "string", "description": "Filter by relation type label (e.g. 'related_to', 'part_of')"},
})

_t("get_relation_by_absolute_id", "Get a specific relation version by its absolute (version) ID. Use family_id for the current version instead.", {
    "absolute_id": {"type": "string", "description": "Relation absolute/version ID (UUID format, from get_relation_versions)"},
}, ["absolute_id"])

_t("get_relation_versions", "Get all versions of a relation. Each version represents a content or linkage change over time. Use absolute_ids from this for batch_delete_relation_versions.", {
    "family_id": {"type": "string", "description": "Relation family ID (e.g. 'rel_abc123')"},
    "limit": {"type": "integer", "description": "Max versions to return"},
}, ["family_id"])

_t("get_relations_between", "Find all relations connecting two entities. Returns both directions. Use this to verify if/how two entities are linked. For discovering indirect connections, use search_shortest_path or traverse_graph.", {
    "entity_a": {"type": "string", "description": "First entity family_id"},
    "entity_b": {"type": "string", "description": "Second entity family_id"},
}, ["entity_a", "entity_b"])

_t("search_shortest_path", "Find the shortest path between two entities in the graph. Returns intermediate entities and relations along the path. Use this to discover how two seemingly unrelated entities are connected. Requires Neo4j backend. Start with max_depth=5, increase for sparse graphs.", {
    "from_entity": {"type": "string", "description": "Start entity family_id"},
    "to_entity": {"type": "string", "description": "End entity family_id"},
    "max_depth": {"type": "integer", "description": "Max search depth (default 5). Increase for sparse graphs."},
}, ["from_entity", "to_entity"])

_t("search_shortest_path_cypher", "Find shortest path using native Cypher query. Same as search_shortest_path but uses Neo4j Cypher directly. Prefer search_shortest_path unless you need Cypher-specific behavior.", {
    "from_entity": {"type": "string", "description": "Start entity family_id"},
    "to_entity": {"type": "string", "description": "End entity family_id"},
    "max_depth": {"type": "integer", "description": "Max search depth (default 5)"},
}, ["from_entity", "to_entity"])


# ── Relation CRUD (7) ────────────────────────────────────────────────────

_t("create_relation", "Create a new relation between two entities. IMPORTANT: requires absolute_ids (version-specific IDs like UUIDs), NOT family_ids (like 'ent_abc123'). Workflow: (1) get_entity(family_id='ent_X') → note the absolute_id from response, (2) get_entity(family_id='ent_Y') → note absolute_id, (3) create_relation with those absolute_ids. For dream-discovered relations, prefer create_dream_relation which uses family_ids directly.", {
    "entity1_absolute_id": {"type": "string", "description": "Absolute ID (version ID) of the first entity — use get_entity(family_id=...) to find this"},
    "entity2_absolute_id": {"type": "string", "description": "Absolute ID (version ID) of the second entity — use get_entity(family_id=...) to find this"},
    "content": {"type": "string", "description": "Relation content/description"},
    "episode_id": {"type": "string", "description": "Episode ID to link (optional)"},
    "source_document": {"type": "string", "description": "Source document label (optional)"},
}, ["entity1_absolute_id", "entity2_absolute_id", "content"])

_t("update_relation", "Update relation metadata by family_id. Changes the current version without creating a new version. For fixing incorrect entity linkages, use redirect_relation instead.", {
    "family_id": {"type": "string", "description": "Relation family ID (e.g. 'rel_abc123')"},
    "content": {"type": "string", "description": "New content/description"},
    "summary": {"type": "string", "description": "New summary"},
    "attributes": {"type": "object", "description": "Updated attributes (merged with existing)"},
}, ["family_id"])

_t("update_relation_by_absolute_id", "Update a specific relation version by absolute_id. Use family_id for the current version instead.", {
    "absolute_id": {"type": "string", "description": "Relation absolute/version ID (UUID format, from get_relation_versions)"},
    "content": {"type": "string", "description": "New content/description"},
    "relation_type": {"type": "string", "description": "New relation type"},
    "summary": {"type": "string", "description": "New summary"},
}, ["absolute_id"])

_t("delete_relation", "Delete relation and all its versions by family_id. This is permanent. For a reversible approach, use invalidate_relation instead (soft-delete that can be cleaned up later).", {
    "family_id": {"type": "string", "description": "Relation family ID (e.g. 'rel_abc123')"},
}, ["family_id"])

_t("delete_relation_by_absolute_id", "Delete a specific relation version by absolute_id. Does NOT delete the entire relation — only the specified version snapshot.", {
    "absolute_id": {"type": "string", "description": "Relation absolute/version ID (UUID format, from get_relation_versions)"},
}, ["absolute_id"])

_t("batch_delete_relations", "Delete multiple relations at once by their family IDs. For single relation deletion, use delete_relation. For targeted version removal, use batch_delete_relation_versions.", {
    "family_ids": {"type": "array", "items": {"type": "string"}, "description": "List of relation family IDs to delete"},
}, ["family_ids"])

_t("redirect_relation", "Re-point one end of a relation to a different entity. Useful for fixing incorrect linkages. Use side='entity1' or 'entity2' to choose which end to redirect.", {
    "relation_family_id": {"type": "string", "description": "Relation family_id to redirect"},
    "new_target_id": {"type": "string", "description": "New target entity family_id"},
    "side": {"type": "string", "description": "Which end to redirect: 'entity1' or 'entity2'"},
}, ["relation_family_id", "new_target_id"])


# ── Episode (4) ──────────────────────────────────────────────────────────

_t("get_latest_episode", "Get the most recent Episode (snapshot of all entities/relations). Use this to see the current state of the graph. For just metadata without the heavy content, use get_latest_episode_metadata.", {})
_t("get_latest_episode_metadata", "Get metadata of the latest Episode without the full content. Faster than get_latest_episode when you only need timestamps and counts.", {})
_t("get_episode_by_id", "Get a specific Episode by its cache_id. Returns the full snapshot with entity and relation data. Use search_episodes to find relevant episodes first.", {
    "cache_id": {"type": "string", "description": "Episode cache ID (from get_latest_episode or search_episodes)"},
}, ["cache_id"])

_t("get_episode_doc", "Get the source document content associated with an Episode. Use get_episode_text for raw text instead.", {
    "cache_id": {"type": "string", "description": "Episode cache ID (from get_latest_episode or search_episodes)"},
}, ["cache_id"])


# ── Snapshot/Changes (2) ─────────────────────────────────────────────────

_t("get_snapshot", "Get a full graph snapshot at a point in time. Omit timestamp for the latest snapshot. Use get_changes(since=...) instead if you only need what changed since a specific time — much lighter weight.", {
    "timestamp": {"type": "string", "description": "ISO 8601 timestamp for point-in-time snapshot (omit for latest)"},
})

_t("get_changes", "Get all entity/relation changes in a time range. Useful for incremental sync or audit logging.", {
    "since": {"type": "string", "description": "ISO 8601 timestamp — return changes after this time"},
    "until": {"type": "string", "description": "ISO 8601 timestamp — return changes before this time (optional, defaults to now)"},
    "limit": {"type": "integer", "description": "Max changes to return"},
}, ["since"])


# ── Episodes (3) ─────────────────────────────────────────────────────────

_t("search_episodes", "Search episodes by content using semantic similarity. Returns matching episodes with metadata. Use this to find which remember operations produced specific information.", {
    "query": {"type": "string", "description": "Search query text"},
    "limit": {"type": "integer", "description": "Max results to return"},
}, ["query"])

_t("delete_episode", "Delete an episode by its cache_id. This removes the episode and its associated data.", {
    "cache_id": {"type": "string", "description": "Episode cache ID to delete"},
}, ["cache_id"])

_t("batch_ingest_episodes", "Bulk import multiple episodes at once. Each episode triggers async extraction like remember. For single texts, use remember instead. Each episode object needs at least 'content'.", {
    "episodes": {"type": "array", "items": {"type": "object"}, "description": "List of episode objects: [{\"content\": \"...\", \"source\": \"optional\", \"episode_type\": \"optional\"}]"},
}, ["episodes"])


# ── Dream (6) ────────────────────────────────────────────────────────────

_t("get_dream_status", "Get current dream consolidation status. Shows whether a dream cycle is running and its progress. Start with this before initiating dream exploration.", {})
_t("get_dream_logs", "Get dream cycle history logs. Each log entry summarizes a completed dream cycle. Use get_dream_log_detail(cycle_id=...) for full details of a specific cycle.", {
    "limit": {"type": "integer", "description": "Max logs to return (default 20)"},
})

_t("get_dream_log_detail", "Get detailed information about a specific dream cycle, including entities explored and relations discovered.", {
    "cycle_id": {"type": "string", "description": "Dream cycle ID (from get_dream_logs)"},
}, ["cycle_id"])

_t("get_dream_seeds", "Get seed entities for dream exploration. Seeds are starting points for discovering hidden connections. Workflow: get_dream_seeds → entity_profile for each seed → traverse_graph(depth=2) → create_dream_relation for discoveries → save_dream_episode. Strategies: 'hub' (highly connected), 'orphan' (isolated, good for connecting loose ends), 'recent' (newly added), 'random'.", {
    "strategy": {"type": "string", "description": "Seed selection strategy: hub, orphan, recent, or random (default)"},
    "count": {"type": "integer", "description": "Number of seeds to return (default 5)"},
})

_t("create_dream_relation", "Create a relation discovered during dream exploration. Unlike create_relation, this uses family_ids (not absolute_ids) and records confidence/reasoning metadata. Always verify with get_relations_between first to avoid duplicates.", {
    "entity1_id": {"type": "string", "description": "First entity family_id"},
    "entity2_id": {"type": "string", "description": "Second entity family_id"},
    "content": {"type": "string", "description": "Relation description / content"},
    "confidence": {"type": "number", "description": "Confidence score 0-1 (default 0.7)"},
    "reasoning": {"type": "string", "description": "Why this relation was discovered"},
    "dream_type": {"type": "string", "description": "Dream type: free_association, cross_domain, etc."},
}, ["entity1_id", "entity2_id"])

_t("save_dream_episode", "Save a dream exploration episode record. Call this AFTER completing a dream cycle to persist the summary and insights. Part of the dream workflow: get_dream_seeds → explore → create_dream_relation → save_dream_episode.", {
    "dream_type": {"type": "string", "description": "Type of dream: free_association, cross_domain, consolidation, etc."},
    "entities_explored": {"type": "array", "items": {"type": "string"}, "description": "Entity family_ids explored during this cycle"},
    "relations_found": {"type": "integer", "description": "Number of new relations found"},
    "summary": {"type": "string", "description": "Episode summary text"},
    "insights": {"type": "string", "description": "Key insights from this dream"},
}, ["dream_type", "summary"])


# ── Agent / Ask (3) ──────────────────────────────────────────────────────

_t("ask", "Ask a natural language question — AI reasons over entities and relations to produce a comprehensive answer. Best for synthesis, comparisons, summaries. For raw data, use quick_search instead. Supports context parameter for follow-ups.", {
    "question": {"type": "string", "description": "Question to ask (e.g. 'How are X and Y related?')"},
    "context": {"type": "string", "description": "Additional context to guide the answer (optional)"},
}, ["question"])

_t("explain_entity", "Get an AI-generated explanation of an entity. Optionally focus on a specific aspect. Uses LLM reasoning over entity data. For raw data without AI interpretation, use entity_profile.", {
    "family_id": {"type": "string", "description": "Entity family ID (e.g. 'ent_abc123')"},
    "question": {"type": "string", "description": "Specific aspect to explain (e.g. 'Why is this important?' or 'What changed recently?')"},
}, ["family_id"])

_t("get_suggestions", "Get AI-curated suggestions for entities worth exploring. Good starting point when you're not sure what to look at. Optionally seed from a specific entity to discover related topics. For specific searches, use quick_search instead.", {
    "entity_id": {"type": "string", "description": "Starting entity family_id (optional — omit for global suggestions)"},
    "limit": {"type": "integer", "description": "Max suggestions to return"},
})


# ── Communities (5) ──────────────────────────────────────────────────────

_t("detect_communities", "Run community detection on the graph. Returns community assignments for entities. Requires Neo4j backend. After detection, use list_communities to see results and get_community to inspect members. Higher resolution = more, smaller communities.", {
    "algorithm": {"type": "string", "description": "Algorithm: louvain (default) or label_propagation"},
    "resolution": {"type": "number", "description": "Resolution parameter for louvain (default 1.0). Higher values produce more, smaller communities."},
})

_t("list_communities", "List all detected communities with member counts and statistics. Run detect_communities first.", {})
_t("get_community", "Get details of a specific community: member entities, internal relations, and summary.", {
    "cid": {"type": "string", "description": "Community ID (from list_communities or detect_communities)"},
}, ["cid"])

_t("get_community_graph", "Get the subgraph for a specific community. Returns all entities and relations within the community for visualization. For just the member list and summary, use get_community (lighter weight).", {
    "cid": {"type": "string", "description": "Community ID (from list_communities or detect_communities)"},
}, ["cid"])

_t("clear_communities", "Remove all community detection labels from entities. Does not delete entities or relations.", {})


# ── Graphs (2) ───────────────────────────────────────────────────────────

# Note: list_graphs and create_graph don't use _GRAPH_ID_PARAM for graph routing
# (they operate across graphs), but it's still injected — just ignored.

_t("list_graphs", "List all knowledge graphs registered in the system. Each graph is an isolated namespace.", {})
_t("create_graph", "Create a new knowledge graph. Each graph is an isolated namespace for entities and relations.", {
    "graph_id": {"type": "string", "description": "Unique graph identifier (e.g. 'my_project', 'research_notes')"},
    "name": {"type": "string", "description": "Human-readable name (optional)"},
    "description": {"type": "string", "description": "Graph description (optional)"},
}, ["graph_id"])

_t("delete_graph", "Delete a knowledge graph and all its data permanently. This cannot be undone.", {
    "graph_id": {"type": "string", "description": "Graph ID to delete"},
}, ["graph_id"])

_t("switch_graph", "Switch the active graph for subsequent tool calls. All future operations will target this graph unless overridden with a per-call graph_id parameter. Returns the new active graph info.", {
    "graph_id": {"type": "string", "description": "Graph ID to switch to. Must be an existing graph — use list_graphs to see available IDs."},
}, ["graph_id"])

_t("get_active_graph", "Get the currently active graph ID and its summary. All tool calls without an explicit graph_id parameter will target this graph.", {})


# ── Docs (2) ─────────────────────────────────────────────────────────────

_t("list_docs", "List documentation files stored in the system. These are source documents that were processed via remember. Use get_doc_content(filename=...) to read a specific document.", {})
_t("get_doc_content", "Get the full content of a stored document by filename. Use list_docs first to see available documents.", {
    "filename": {"type": "string", "description": "Document filename (from list_docs)"},
}, ["filename"])


# ── Neo4j (3) ────────────────────────────────────────────────────────────

_t("get_entity_neighbors", "Get immediate graph neighbors of an entity from Neo4j. Requires the entity's UUID (Neo4j internal ID, not family_id). For most use cases, entity_profile or traverse_graph are easier as they accept family_id. Only use this for low-level Neo4j access.", {
    "uuid": {"type": "string", "description": "Entity UUID (Neo4j internal ID, not family_id)"},
    "direction": {"type": "string", "description": "Edge direction: 'outgoing', 'incoming', or 'both' (default)"},
    "limit": {"type": "integer", "description": "Max neighbors to return"},
}, ["uuid"])

_t("list_episodes", "List all episodes stored in Neo4j with pagination. For searching episode content, use search_episodes. Requires Neo4j backend.", {
    "limit": {"type": "integer", "description": "Max results to return"},
    "offset": {"type": "integer", "description": "Offset for pagination (0-based)"},
})

_t("get_neo4j_episode", "Get a specific episode by its Neo4j UUID. Returns the full episode record. For cache_id based access, use get_episode_by_id instead.", {
    "uuid": {"type": "string", "description": "Episode UUID (Neo4j internal ID)"},
}, ["uuid"])


# ── Data Quality & Maintenance (4) ──────────────────────────────────────────

_t("delete_isolated_entities", "Delete all isolated entities (entities with zero relations). Always use dry_run=true first to preview what will be deleted.", {
    "dry_run": {"type": "boolean", "description": "Preview only without deleting (default: false). Strongly recommended to run true first."},
})

_t("get_data_quality_report", "Get a comprehensive data quality report: counts of valid, invalidated, and isolated entities/relations. For a combined report with graph stats, use maintenance_health instead. For AI-powered recommendations, use butler_report.", {})

_t("cleanup_old_versions", "Remove invalidated (soft-deleted) entity/relation versions to reclaim storage. Safe — only removes already-invalidated data.", {
    "before_date": {"type": "string", "description": "ISO date string — only remove versions invalidated before this date (optional)"},
    "dry_run": {"type": "boolean", "description": "Preview only without deleting (default: false)"},
})

_t("search_similar_entities", "Find potentially duplicate entities by name similarity. Returns entities with similar names for merge review.", {
    "name": {"type": "string", "description": "Entity name to search for duplicates"},
    "similarity_threshold": {"type": "number", "description": "Minimum similarity score 0-1 (default: 0.7). Lower = more results."},
})


# ── System (6) ───────────────────────────────────────────────────────────

_t("system_dashboard", "Get the system dashboard: uptime, entity/relation counts, API stats, thread info. For graph-specific stats, use graph_summary instead.", {})
_t("system_overview", "Get a high-level system overview: version, backend status, configuration. Use health_check for a simple connectivity test.", {})
_t("system_graphs", "Get information about all graphs in the system. Includes storage backend and entity/relation counts per graph.", {})
_t("system_tasks", "Get running and queued system tasks. Shows active remember tasks, dream cycles, and maintenance operations.", {})
_t("system_logs", "Get system log entries. Filter by level for targeted debugging.", {
    "level": {"type": "string", "description": "Log level filter: 'info', 'warn', or 'error'"},
    "limit": {"type": "integer", "description": "Max log entries to return"},
})

_t("system_access_stats", "Get API access statistics: request counts, latencies, endpoint usage.", {})


# ── Relation Invalidation (2) ────────────────────────────────────────────

_t("invalidate_relation", "Soft-delete a relation by family_id. The relation is marked as invalidated but not permanently removed. Can be cleaned up later with cleanup_old_versions.", {
    "family_id": {"type": "string", "description": "Relation family ID to invalidate"},
    "reason": {"type": "string", "description": "Reason for invalidation (optional, stored for audit)"},
}, ["family_id"])

_t("list_invalidated_relations", "List all soft-deleted (invalidated) relations. These can be permanently removed with cleanup_old_versions.", {
    "limit": {"type": "integer", "description": "Max results to return (default 100)"},
})


# ── Version Management (2) ────────────────────────────────────────────────

_t("batch_delete_entity_versions", "Delete specific entity version snapshots by their absolute IDs. Unlike delete_entity (which removes all versions), this surgically removes individual versions. Get absolute_ids from get_entity_versions.", {
    "absolute_ids": {"type": "array", "items": {"type": "string"}, "description": "List of entity absolute (version) IDs to permanently delete"},
}, ["absolute_ids"])

_t("batch_delete_relation_versions", "Delete specific relation version snapshots by their absolute IDs. Unlike delete_relation (which removes all versions), this surgically removes individual versions. Get absolute_ids from get_relation_versions.", {
    "absolute_ids": {"type": "array", "items": {"type": "string"}, "description": "List of relation absolute (version) IDs to permanently delete"},
}, ["absolute_ids"])


# ── Section History (1) ───────────────────────────────────────────────────

_t("get_section_history", "Track how a specific Markdown section of an entity evolved over time. Section keys correspond to headings in entity content (e.g. '## Summary', '## Details'). Useful for auditing changes to a particular aspect of an entity.", {
    "family_id": {"type": "string", "description": "Entity family ID (e.g. 'ent_abc123')"},
    "section": {"type": "string", "description": "Markdown heading key (e.g. '## Summary', '## Details', '## Key Facts')"},
}, ["family_id", "section"])


# ── Episode Text (1) ──────────────────────────────────────────────────────

_t("get_episode_text", "Get the original raw text that was submitted when creating an episode. Use get_episode_doc for the processed document instead.", {
    "cache_id": {"type": "string", "description": "Episode cache ID (from list_episodes or get_latest_episode)"},
}, ["cache_id"])


# ── Isolated Entities (1) ─────────────────────────────────────────────────

_t("list_isolated_entities", "List all isolated entities (entities with zero relations). These may be extraction artifacts. Review before bulk-deleting with delete_isolated_entities.", {
    "limit": {"type": "integer", "description": "Max results (default 100)"},
    "offset": {"type": "integer", "description": "Offset for pagination (0-based)"},
})

# ── Aggregation Tools (1 call replaces 3-5) ─────────────────────────────────
_t("entity_profile", "Get a complete entity profile in one call: current entity data + all connected relations + version count. Use this INSTEAD of get_entity when you also need relations (which is almost always). Only use get_entity if you need just the raw entity data without any relation context.", {
    "family_id": {"type": "string", "description": "Entity family ID (e.g. 'ent_abc123')"},
}, ["family_id"])

_t("graph_summary", "Get graph overview in one call: total entity/relation counts, storage backend type, embedding model status. Use this as your FIRST call to understand the graph before performing operations. For just entity/relation counts, graph_stats is lighter.", {})

_t("maintenance_health", "Combined health check: graph statistics + data quality report (valid/invalidated/isolated counts) + isolated entity list. Use this before running cleanup operations.", {})

_t("maintenance_cleanup", "One-click cleanup that combines two operations: (1) remove all invalidated (soft-deleted) versions, (2) delete isolated entities with zero relations. Always use dry_run=true first to preview.", {
    "dry_run": {"type": "boolean", "description": "Preview only without deleting (default false). Strongly recommended to run true first."},
})

_t("butler_report", "Comprehensive AI-powered health report combining graph stats + data quality + dream status. Returns actionable recommendations (e.g. 'cleanup_isolated', 'evolve_summaries'). Workflow: butler_report → review recommendations → butler_execute(actions=[...]). Use dry_run=true on butler_execute to preview before applying.", {})

_t("butler_execute", "Execute butler optimization actions on the memory graph. Get action names from butler_report. Available actions: cleanup_isolated (remove isolated entities), cleanup_invalidated (remove soft-deleted versions), detect_communities (run community detection), evolve_summaries (regenerate entity summaries with LLM).", {
    "actions": {
        "type": "array",
        "items": {"type": "string"},
        "description": "Actions to execute: cleanup_isolated, cleanup_invalidated, detect_communities, evolve_summaries",
    },
    "dry_run": {"type": "boolean", "description": "Preview only without executing (default false). Strongly recommended for evolve_summaries since it uses LLM calls."},
}, ["actions"])

_t("quick_search", "All-in-one search: returns entities + relations in one call. RECOMMENDED for most searches. After results: entity_profile(family_id) for details, traverse_graph for connections, ask(question) for AI answers. Use this instead of semantic_search unless you need specific mode control or expand context. Typical use: 'What does the graph know about X?' → quick_search(query='X').", {
    "query": {"type": "string", "description": "Search query text"},
    "max_entities": {"type": "integer", "description": "Max entities to return (default 10, max 50)"},
    "max_relations": {"type": "integer", "description": "Max relations to return (default 20, max 100)"},
    "similarity_threshold": {"type": "number", "description": "Min similarity score (default 0.4). Lower = more results, higher = more precise."},
}, ["query"])

_t("find_entity_by_name", "Fast entity lookup by name using fuzzy matching. Returns the entity and its connected relations. Prefer this over search_entities when you know the entity name. Threshold guide: 1.0=exact, 0.7+=strict (recommended default), 0.5=broad, 0.3=very broad.", {
    "name": {"type": "string", "description": "Entity name to search for (supports partial/fuzzy matching)"},
    "threshold": {"type": "number", "description": "Min similarity threshold (default 0.5). Use 0.7+ for strict matching, 0.3 for broad."},
    "limit": {"type": "integer", "description": "Max candidate entities to return (default 5)"},
}, ["name"])

_t("batch_profiles", "Get profiles for up to 20 entities in one call. Each profile includes entity details + relations + version count. More efficient than calling entity_profile in a loop.", {
    "family_ids": {"type": "array", "items": {"type": "string"}, "description": "List of entity family IDs (max 20)"},
}, ["family_ids"])

_t("recent_activity", "Get a snapshot of recent graph activity: newest entities, newest relations, and current statistics. Useful as a dashboard summary or to see what changed recently. For specific entity details, follow up with entity_profile or batch_profiles.", {
    "limit": {"type": "integer", "description": "Max items per category (default 10, max 50)"},
})


# ── Concepts — unified concept query (7) ──────────────────────────────────

_t("search_concepts", "Unified concept search across all roles (entity, relation, observation). Searches the unified concept space using BM25 text matching. Optionally filter by role. Returns concepts with their role, family_id, and content.", {
    "query": {"type": "string", "description": "Search query text"},
    "role": {"type": "string", "description": "Optional role filter: 'entity', 'relation', or 'observation'"},
    "limit": {"type": "integer", "description": "Max results (default 20, max 100)"},
}, ["query"])

_t("list_concepts", "List concepts with pagination and optional role filter. Returns concepts from the unified concept table across all roles.", {
    "role": {"type": "string", "description": "Optional role filter: 'entity', 'relation', or 'observation'"},
    "limit": {"type": "integer", "description": "Max results per page (default 50, max 100)"},
    "offset": {"type": "integer", "description": "Pagination offset (default 0)"},
})

_t("get_concept", "Get a concept by family_id. Works for any role — entity, relation, or observation. Returns the latest version of the concept.", {
    "family_id": {"type": "string", "description": "Concept family ID (e.g. 'ent_abc123', 'rel_abc123', or episode ID)"},
}, ["family_id"])

_t("get_concept_neighbors", "Get neighbors of a concept, regardless of its role. For entities: returns connected relations. For relations: returns connected entities. For observations: returns mentioned concepts.", {
    "family_id": {"type": "string", "description": "Concept family ID"},
    "max_depth": {"type": "integer", "description": "Neighbor depth (default 1, max 3)"},
}, ["family_id"])

_t("get_concept_provenance", "Trace a concept back to its source observations. Returns all episodes (observations) that mention this concept, enabling full provenance tracking.", {
    "family_id": {"type": "string", "description": "Concept family ID"},
}, ["family_id"])

_t("traverse_concepts", "BFS traverse the concept graph starting from one or more seed concepts. Discovers connected concepts across all roles in a unified graph traversal.", {
    "start_family_ids": {"type": "array", "items": {"type": "string"}, "description": "List of starting concept family IDs"},
    "max_depth": {"type": "integer", "description": "Max traversal depth (default 2, max 5)"},
}, ["start_family_ids"])

_t("get_concept_mentions", "Get all episodes that mention a given concept. Alias for get_concept_provenance but with a clearer name for the 'which episodes mention this concept' use case.", {
    "family_id": {"type": "string", "description": "Concept family ID"},
}, ["family_id"])



# -- Composite workflow tools --------------------------------------------------

_t("remember_and_explore", "Remember text AND immediately show extracted results. Combines remember + quick_search in one call. RECOMMENDED over bare remember() when you want to verify what was stored. Follow up with entity_profile for specific entities.", {
    "content": {"type": "string", "description": "Text content to remember"},
    "source": {"type": "string", "description": "Source label"},
}, ["content"])

_t("explore_topic", "Deep-explore a topic: search + traverse in one call. Returns entities, relations, and connections as a knowledge map. Follow up with entity_profile for specific entities or ask(question) for AI synthesis.", {
    "topic": {"type": "string", "description": "Topic or question to explore"},
    "depth": {"type": "integer", "description": "Traversal depth from found entities (default 2)"},
}, ["topic"])

_t("graph_overview", "CALL THIS FIRST at session start. Returns graph stats, recent activity, and health. Use results to decide next: quick_search for queries, remember for new data, dream_quick_start for consolidation, butler_report for maintenance.", {})

_t("dream_quick_start", "Start a dream cycle (offline consolidation that discovers hidden connections). Combines status check + dream start. After completion, use get_dream_logs to review discoveries or quick_search to find new relations.", {
    "max_cycles": {"type": "integer", "description": "Number of dream cycles (default 5)"},
    "strategies": {"type": "array", "items": {"type": "string"}, "description": "Dream strategies (default: free_association, cross_domain, leap)"},
})
