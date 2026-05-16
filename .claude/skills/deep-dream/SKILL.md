---
name: deep-dream
description: >
  Use when interacting with a Deep-Dream knowledge graph server — remembering text,
  searching entities/relations, running dream exploration, managing concepts, or
  performing graph maintenance. Triggers: user mentions knowledge graph, entities,
  relations, episodes, dream, remember, find, concepts, or asks to store/retrieve
  structured memory.
---

# Deep-Dream Knowledge Graph

Natural-language memory graph. **Remember** (write → auto-extract entities/relations) + **Find** (semantic retrieval). All concepts are one primitive with roles: `entity`, `relation`, `observation`.

```
BASE_URL=http://localhost:16200/api/v1   |   graph_id=default (query param)   |   Response: {success, data}
```

All endpoints accept `?graph_id=<id>` to target a specific graph. **Always add `&compact=true`** to strip embeddings and truncate content — responses can be 10x+ larger without it.

## Quick Start

```bash
# Health check
curl -s $BASE_URL/health

# LLM health (check before remember/dream)
curl -s $BASE_URL/health/llm

# Graph overview
curl -s "$BASE_URL/find/graph-summary?graph_id=default&compact=true"

# List all graphs
curl -s $BASE_URL/graphs
```

## Decision Guide

| Intent | Method | Endpoint | Key Params |
|---|---|---|---|
| Store text (sync) | POST | `/remember` | `{text, wait:true, timeout:120}` |
| Store text (async) | POST | `/remember` | `{text}` → poll task_id |
| Check remember task | GET | `/remember/tasks/{id}` | — |
| Search everything | POST | `/find` | `{query, search_mode:"hybrid"}` |
| Search entities | GET | `/find/entities/search` | `query_name=X` |
| Search relations | GET | `/find/relations/search` | `query_text=X` |
| Find by name | GET | `/find/entities/by-name/{name}` | `threshold=0.7` |
| Entity profile | GET | `/find/entities/{fid}/profile` | — |
| Quick search | POST | `/find` | `{query, search_mode:"hybrid"}` (modes: `hybrid`, `semantic`, `bm25`) |
| Traverse graph | POST | `/find/traverse` | `{seed_family_ids:["ent_abc",...], max_depth:2}` |
| Shortest path | POST | `/find/paths/shortest` | `{family_id_a, family_id_b}` |
| Create entity | POST | `/find/entities/create` | `{name, content}` |
| Create relation | POST | `/find/relations/create` | `{entity1_family_id, entity2_family_id, content}` |
| Update entity | PUT | `/find/entities/{fid}` | `{name, summary, attributes}` |
| Update relation | PUT | `/find/relations/{fid}` | `{content, summary, attributes, confidence}` (content optional for metadata-only updates) |
| Delete entity | DELETE | `/find/entities/{fid}` | `?cascade=true` to remove connected relations (default: false, leaves orphans) |
| Delete relation | DELETE | `/find/relations/{fid}` | `?cascade=false` |
| Merge entities | POST | `/find/entities/merge` | `{source_family_ids:[...], target_family_id:...}` |
| Dream cycle | POST | `/find/dream/run` | `{strategy, seed_count}` |
| Dream status | GET | `/find/dream/status` | — |
| Dream logs | GET | `/find/dream/logs` | — |
| Ask NL question | POST | `/find/ask` | `{question}` |
| Butler report | GET | `/butler/report` | — |
| Butler execute | POST | `/butler/execute` | `{actions:[...], dry_run:true}` |
| Health report | GET | `/find/maintenance/health` | — |
| Fix dangling refs | POST | `/butler/execute` | `{actions:["fix_dangling_refs"]}` |
| Cleanup stale redirects | POST | `/butler/execute` | `{actions:["cleanup_stale_redirects"]}` |
| Detect communities | POST | `/communities/detect` | `{algorithm:"louvain"}` |
| List communities | GET | `/communities` | `min_size=3, limit=50` (returns `data.communities` list, requires `detect_communities` first) |
| Entity neighbors | GET | `/find/entities/{fid}/neighbors` | `depth=1` (accepts family_id) |
| Concept provenance | GET | `/concepts/{fid}/provenance` | `time_point=ISO8601` |
| Concept mentions | GET | `/concepts/{fid}/mentions` | `time_point=ISO8601` |
| Search episodes | POST | `/find/episodes/search` | `{query, limit:20}` |
| Episode text | GET | `/find/episodes/{cache_id}/text` | — |
| Recent activity | GET | `/find/recent-activity` | — |
| Refresh graph edges | POST | `/find/entities/refresh-edges` | — |

## Response Format

All responses: `{"success": bool, "data": ..., "elapsed_ms": float}`

- `data` type varies by endpoint:
  - **Single item**: `data: {family_id, name, content, ...}` (create)
  - **By-name** (nested): `data: {entity: {...}, relations: [...]}` — same structure as profile. Returns 404 `{success: false}` for non-existent names (not `{success: true, entity: null}`)
  - **Profile** (nested): `data: {entity: {...}, relations: [...], relation_count, version_count}` (profile)
  - **List**: `data: [{...}, ...]` (search, find entities, list)
  - **Aggregation**: `data: {entities: [...], relations: [...], ...}` (find)
  - **Graph summary**: `data: {embedding_available: bool, graph_id: str, statistics: {entity_count, relation_count, ...}, storage_backend: str}` (graph-summary — returns counts/flags, NOT entity/relation lists)
  - **Recent activity**: `data: {latest_entities: [...], latest_relations: [...], statistics: {...}}` (note: `latest_entities`/`latest_relations`, not `entities`/`relations`)
  - **Counts**: `data: {total, count, ...}` (routes, counts)
  - **Ask**: `data: {answer: string, query_plan: {query_text, query_type, ...}, results: {entities: [...], relations: [...]}}` (natural language Q&A)
  - **Neighbors**: `data: {entity: {uuid, name, family_id}, nodes: [{uuid, name, family_id}], edges: [{source_uuid, target_uuid, source_name, target_name, content, relation_uuid}]}` (neighbors)
  - **Traverse**: `data: {entities: [...], relations: [...], visited_count}` (traverse)
  - **Update**: `data: {family_id, name, content, summary, community_id, ...}` (update — returns full entity)
- Errors: `{"success": false, "error": "message", "hint": "actionable guidance", "elapsed_ms": float}`
- Error messages may be in Chinese. The `hint` field provides English guidance.
- Add `compact=true` query param to strip embeddings and truncate content.

## Common Workflows

### Write and Verify
```bash
# Store text (sync, wait for extraction)
curl -s -X POST "$BASE_URL/remember?graph_id=default" \
  -H 'Content-Type: application/json' \
  -d '{"text":"Alice is a software engineer at Google","wait":true,"timeout":120}'

# Verify extraction
curl -s "$BASE_URL/find/entities/search?query_name=Alice&graph_id=default"
```

### Remember Async Polling
```bash
# Async remember returns immediately with task_id
RESP=$(curl -s -X POST "$BASE_URL/remember?graph_id=default" \
  -H 'Content-Type: application/json' \
  -d '{"text":"Long document text here"}')
TASK_ID=$(echo $RESP | jq -r '.data.task_id')

# Poll every 5 seconds until complete (typical: 30s-5min)
curl -s "$BASE_URL/remember/tasks/$TASK_ID?graph_id=default"
# Response: {"success":true,"data":{"status":"completed",...}}
# or: {"success":true,"data":{"status":"running","progress":0.5},...}
```

### Create Entity + Relation
```bash
# Create two entities
E1=$(curl -s -X POST "$BASE_URL/find/entities/create?graph_id=default" \
  -H 'Content-Type: application/json' \
  -d '{"name":"Project A","content":"A research project"}' | jq -r '.data.family_id')

E2=$(curl -s -X POST "$BASE_URL/find/entities/create?graph_id=default" \
  -H 'Content-Type: application/json' \
  -d '{"name":"Project B","content":"Another project"}' | jq -r '.data.family_id')

# Create relation using family_ids (no need for absolute_ids)
curl -s -X POST "$BASE_URL/find/relations/create?graph_id=default" \
  -H 'Content-Type: application/json' \
  -d "{\"entity1_family_id\":\"$E1\",\"entity2_family_id\":\"$E2\",\"content\":\"A and B are related\"}"
```

**Note:** Relations are undirected — the API normalizes entity order by absolute_id. Your `entity1`/`entity2` may appear swapped in the response. Also accepts optional `summary`, `confidence`, and `attributes` fields.

### Merge Duplicate Entities
```bash
# Merge multiple entities into one target
curl -s -X POST "$BASE_URL/find/entities/merge?graph_id=default" \
  -H 'Content-Type: application/json' \
  -d '{"source_family_ids":["ent_dup1","ent_dup2"],"target_family_id":"ent_main"}'

# Merge auto-redirects Relation endpoints and refreshes RELATES_TO edges — no manual refresh needed
```

### Explore Entity Connections
```bash
# Get entity profile (includes relations, version count)
curl -s "$BASE_URL/find/entities/{fid}/profile?graph_id=default&compact=true"

# Get neighbors (graph traversal via RELATES_TO edges)
curl -s "$BASE_URL/find/entities/{fid}/neighbors?graph_id=default&depth=2"

# Find shortest path between two entities
curl -s -X POST "$BASE_URL/find/paths/shortest?graph_id=default" \
  -H 'Content-Type: application/json' \
  -d '{"family_id_a":"ent_abc","family_id_b":"ent_xyz"}'
# Returns path_length=-1 if entities exist but are disconnected (not 404)
```

### Dream Cycle
```bash
# Check dream status
curl -s "$BASE_URL/find/dream/status?graph_id=default"

# Run a dream cycle
curl -s -X POST "$BASE_URL/find/dream/run?graph_id=default" \
  -H 'Content-Type: application/json' \
  -d '{"strategy":"cross_community","seed_count":5,"max_depth":2}'

# Check dream logs
curl -s "$BASE_URL/find/dream/logs?graph_id=default"
```

### Graph Maintenance
```bash
# Health report (isolated entities, data quality)
curl -s "$BASE_URL/find/maintenance/health?graph_id=default"

# Butler report (detailed, 10-15s)
curl -s "$BASE_URL/butler/report?graph_id=default"

# Preview cleanup (dry_run only works for butler/execute, NOT merge)
curl -s -X POST "$BASE_URL/butler/execute?graph_id=default" \
  -H 'Content-Type: application/json' \
  -d '{"actions":["cleanup_isolated","cleanup_invalidated"],"dry_run":true}'

# Execute cleanup
curl -s -X POST "$BASE_URL/butler/execute?graph_id=default" \
  -H 'Content-Type: application/json' \
  -d '{"actions":["cleanup_isolated","cleanup_invalidated"]}'

# Regenerate RELATES_TO graph edges (fixes missing traversal paths)
curl -s -X POST "$BASE_URL/find/entities/refresh-edges?graph_id=default"
```

### Version History and Provenance
```bash
# Get entity versions (ordered by processed_time, NOT event_time)
curl -s "$BASE_URL/find/entities/{fid}/versions?graph_id=default&compact=true"

# Detect contradictions between versions
curl -s "$BASE_URL/find/entities/{fid}/contradictions?graph_id=default"

# Concept provenance (which episodes contributed)
curl -s "$BASE_URL/concepts/{fid}/provenance?graph_id=default"

# Concept mentions (episodes referencing this concept)
curl -s "$BASE_URL/concepts/{fid}/mentions?graph_id=default"

# Search episodes by content
curl -s -X POST "$BASE_URL/find/episodes/search?graph_id=default" \
  -H 'Content-Type: application/json' \
  -d '{"query":"machine learning","limit":20}'
```

## ID System

- **family_id** (`ent_*`/`rel_*`): stable ID for most operations. Example: `ent_abc123`
- **absolute_id** (UUID): version-specific snapshot. Used for: `split_entity_version`, version diff. Example: `entity_20260514_231113_6e558246`
- `create_relation` accepts **family_ids** (`entity1_family_id`/`entity2_family_id`) — no need to resolve to absolute_id
- Relation responses include both `entity1_absolute_id`/`entity2_absolute_id` AND `entity1_family_id`/`entity2_family_id`
- Relation `entity1_name`/`entity2_name` fields may be `null` for some relations

## Entity Disambiguation

When searching returns multiple entities with the same or similar names:
1. Check `content` field to distinguish — each entity's content describes what it represents
2. Use `GET /find/entities/by-name/{name}` for name lookup (returns 404 if no close match exists; uses exact → prefix → BM25 → embedding cascade)
3. Use `GET /find/entities/search?query_name=X` for scored list of candidates
4. Use entity profile `GET /find/entities/{fid}/profile` to see full context including relations

## Key Behavioral Differences

| Endpoint | Returns | Behavior |
|---|---|---|
| `by-name` | Nested `{entity, relations}` | Fuzzy match, returns single best match with relations. NOT flat. |
| `search` | Scored list | Multiple results with scores |
| `neighbors` | Graph nodes/edges | Traverses RELATES_TO graph edges |
| Relations search | List of Relation nodes | Queries Relation node properties directly |
| `profile` | Nested `{entity, relations}` | Not flat — access `data.entity`, `data.relations` |
| `shortest_path` | `{path_length, paths}` | Returns `path_length=-1` for disconnected, 404 for missing entities |

## Auto-named Entities

When the extraction pipeline cannot determine an entity name, it creates `auto_XXXXXXXX` placeholder names. These indicate:
- The entity was extracted but lacked a clear name in the source text
- You can find all auto-named entities by searching: `GET /find/entities/search?query_name=auto_`
- To clean up: rename them with `PUT /find/entities/{fid}` with `{name:"Better Name"}`, or merge into an existing entity

## Parameter Pitfalls

- **Entity prefix search**: There is no dedicated prefix/name-filter endpoint. `GET /find/entities/search?query_name=auto_` uses semantic similarity, not text prefix — results may miss some `auto_*` entities and include false positives. For reliable prefix matching, use `POST /find` with `search_mode:"bm25"` or filter client-side from a full entity list.
- **search_mode validation**: `POST /find` accepts only `hybrid`, `semantic`, `bm25`. Invalid modes return 400 with hint.
- **Content truncation**: Entity `content` is truncated to ~2000 chars with `content_truncated: true` flag. For longer content, split into multiple entities or use shorter summaries.
- **split-version on single-version entity**: Splits the only version into a new family_id, leaving the original family_id empty (returns 404). This is a move, not a copy.
- **Episode search language**: `POST /find/episodes/search` uses substring matching (`CONTAINS`). Query language must match content language — use Chinese queries for Chinese content, English for English.
- **BM25 search ranking for Chinese**: BM25 does character-level token matching. Searching "张三" may rank "桃园三结义" higher than the actual entity "张三" due to character overlap. Use `GET /find/entities/by-name/{name}` for precise name lookup.
- `remember`: use `wait:true` for sync mode; default is async (returns task_id to poll)
- `remember` sync mode: if extraction takes longer than `timeout` seconds (default 300), returns HTTP 202 with `status:"running"` — continue polling via task endpoint
- `remember` async mode: tasks queue serially — a stuck task blocks all subsequent tasks. If polling shows persistent `"queued"` status, the pipeline may be stuck. Try `POST /find/entities/create` + `POST /find/relations/create` as fallback.
- Entity search uses `query_name` param, not `q` or `query`
- `ask` endpoint uses `question` param (not `query` or `query_name`) — different from other search endpoints
- Shortest path uses `family_id_a`/`family_id_b` (or aliases `entity1_family_id`/`entity2_family_id`)
- `update_entity`: returns full updated entity. name/content changes create a new version (preserves summary, confidence, community_id); summary/attribute changes are in-place. Entity rename auto-propagates to relation records and refreshes RELATES_TO edges
- `update_relation` (PUT /find/relations/{fid}): creates a new version. `content` is optional — omit for metadata-only updates. Pass `summary`, `confidence`, `attributes` to override; omitted fields carry forward from previous version. Returns full relation dict.
- `delete_entity`: default `cascade=false` leaves orphaned relations connected to the deleted entity. Use `?cascade=true` to also delete connected relations. After deletion, absolute_id lookups may return stale data briefly (cache invalidation is immediate but in-flight requests may complete with cached data).
- `relations/between`: returns only the latest valid version per relation (excludes invalidated versions)
- `shortest_path`: depends on RELATES_TO graph edges (same as traverse), NOT on Relation records visible in profile. Returns 404 if either entity family_id doesn't exist; returns `path_length:-1` if entities exist but lack connecting RELATES_TO edges. If entities have Relations but shortest-path returns -1, run `POST /find/entities/refresh-edges` to regenerate RELATES_TO edges
- `merge`: target entity stays canonical (name/content preserved); source entities are absorbed. Auto-redirects Relation endpoints and refreshes RELATES_TO edges
- `merge`: rejects with HTTP 409 if source/target names have insufficient word overlap (uses word-level Jaccard for multi-word names, character-level for single-word); pass `skip_name_check: true` in body to override
- `merge`: returns 400 if target is in source list (self-merge) or if source entities don't exist
- `merge`: response data is nested at `data.merged_count` (not flat in `data`)
- `merge`: if a relation connects source and target entities, both endpoints resolve to target after merge (self-loop). No warning is returned
- `merge`: auto-updates source entity names to target name and refreshes RELATES_TO edges
- Auto-named entities (`auto_XXXXXXXX`) may outrank real entities in search results — filter by checking `content` field
- `dry_run:true` only works for `butler/execute`, NOT for merge or other destructive ops
- Valid `butler_execute` actions: `cleanup_isolated`, `cleanup_invalidated`, `fix_dangling_refs`, `cleanup_stale_redirects`, `detect_communities`, `evolve_summaries` (NOT `run_dream`)
- `butler_execute`: returns `success: true` even if individual actions fail — check each action's `status` field for errors
- If remember returns 0 entities, check LLM health: `GET /health/llm`
- `profile`: returns 404 `{success: false, error: "..."}` for nonexistent family_ids (not `success: true` with null). `neighbors` returns `success: true` with empty arrays for nonexistent IDs
- Chinese characters in curl URLs: use `--data-urlencode` or Python urllib to avoid encoding issues
- Entity versions are ordered by `processed_time` (ingestion time), NOT `event_time` (when the event occurred)
- `traverse`: requires RELATES_TO edges to exist. If traverse returns empty but neighbors works, run `POST /find/entities/refresh-edges` first
- `communities/list`: returns empty or 500 if `detect_communities` hasn't been called — always POST `/communities/detect` first
- `search` results may show stale family_ids for entities that were merged/redirected — profile endpoint resolves correctly

## Slow Endpoints

These endpoints may take 5-15+ seconds. Use `timeout` param or increase curl timeout:
- `POST /find/ask` — LLM-powered natural language Q&A (~5-20s depending on graph size)
- `POST /find/dream/run` — LLM-powered exploration (~30s-5min)
- `POST /remember` with `wait:true` — extraction pipeline (~30s-5min)
- `GET /find/graph-summary` — aggregation over all nodes (~3-15s depending on graph size)
- `POST /communities/detect` — graph loading + Louvain algorithm (~5-30s)
- `POST /find/entities/refresh-edges` — regenerates all RELATES_TO edges (~5-30s)
- `GET /butler/report` — scans full graph (~0.2-15s depending on graph size)
- `GET /find/maintenance/health` — quality + statistics scan (~0.4-15s)
- `GET /find/entities/{fid}/contradictions` — LLM-powered version analysis (~2-5s)

## Episode Text Availability

- `/find/episodes/{cache_id}/text` returns the original source text for an episode
- **Episode search returns `uuid` field** (e.g. `cache_abc123`) — use this value as `cache_id` in the text endpoint URL
- **Older episodes may return 404** — source text can be evicted from cache over time
- Episode search (`POST /find/episodes/search`) always works, but `source_text` field may be empty
- Recent episodes (within current session) are most likely to have text available

## Full API Reference

100+ endpoints with parameters and examples: [references/api-reference.md](references/api-reference.md)
