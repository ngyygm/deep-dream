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
| List remember tasks | GET | `/remember/tasks` | Lists all tasks (omit `{id}`) |
| Check remember task | GET | `/remember/tasks/{id}` | Check specific task |
| Search everything | POST | `/find` | `{query, search_mode:"hybrid"}` |
| Search entities | GET | `/find/entities/search` | `query_name=X`, `limit=N` (or `max_results=N`) |
| Search relations | GET | `/find/relations/search` | `query_text=X` |
| Find by name | GET | `/find/entities/by-name/{name}` | `threshold=0.7` |
| Entity profile | GET | `/find/entities/{fid}/profile` | — |
| Quick search | POST | `/find` | `{query, search_mode:"hybrid"}` (modes: `hybrid`, `semantic`, `bm25`) |
| Traverse graph | POST | `/find/traverse` | `{seed_family_ids:["ent_abc",...], max_depth:2}` (max 5, max_nodes 200) |
| Shortest path | POST | `/find/paths/shortest` | `{family_id_a, family_id_b}` |
| Create entity | POST | `/find/entities/create` | `{name, content}` |
| Create relation | POST | `/find/relations/create` | `{entity1_family_id, entity2_family_id, content}` |
| Update entity | PUT | `/find/entities/{fid}` | `{name, summary, attributes}` |
| Get relation | GET | `/find/relations/{fid}` | Returns latest valid version |
| Update relation | PUT | `/find/relations/{fid}` | `{content, summary, attributes, confidence}` (content optional for metadata-only updates) |
| Delete entity | DELETE | `/find/entities/{fid}` | `?cascade=true` to remove connected relations (default: false, leaves orphans) |
| Delete relation | DELETE | `/find/relations/{fid}` | Deletes all versions |
| Relation versions | GET | `/find/relations/{fid}/versions` | List all versions, ordered by processed_time |
| Delete relation version | DELETE | `/find/relations/absolute/{aid}` | Delete single version, others unaffected |
| Relations between | GET | `/find/relations/between` | `family_id_a=X&family_id_b=Y` |
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
| Graph version | GET | `/find/graph/version` | Returns `{entity_count, relation_count, last_modified}` |
| Refresh graph edges | POST | `/find/entities/refresh-edges` | — |
| Stream entities (SSE) | GET | `/find/graph/stream/entities` | `since=ISO8601` for incremental; returns SSE events |
| Stream relations (SSE) | GET | `/find/graph/stream/relations` | `since=ISO8601` for incremental; returns SSE events |

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
2. Use `GET /find/entities/by-name/{name}` for name lookup (returns 404 if no close match exists; uses exact → prefix → BM25 → embedding cascade). Response includes `match_method` and `match_score` — check these before trusting results. BM25 with Chinese can produce false positives (e.g., "林黛玉" matches "贾宝玉" with 0.96 score). If `match_score < 0.8`, a `hint` field warns about low confidence
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
- **Entity rename collision**: renaming an entity to include another entity's name can cause `by-name` to return the wrong entity (e.g., renaming to "Sarah Chen Metadata" may shadow "Dr. Sarah Chen"). Prefer unique, descriptive names.
- `update_relation` (PUT /find/relations/{fid}): creates a new version. `content` is optional — omit for metadata-only updates. Pass `summary`, `confidence`, `attributes` to override; omitted fields carry forward from previous version. Returns full relation dict.
- `delete_entity`: default `cascade=false` leaves orphaned relations connected to the deleted entity. Use `?cascade=true` to also delete connected relations. After deletion, absolute_id lookups may return stale data briefly (cache invalidation is immediate but in-flight requests may complete with cached data).
- `relations/between`: returns only the latest valid version per relation (excludes invalidated versions)
- `shortest_path`: depends on RELATES_TO graph edges (same as traverse), NOT on Relation records visible in profile. Returns 404 if either entity family_id doesn't exist; returns `path_length:-1` if entities exist but lack connecting RELATES_TO edges. If entities have Relations but shortest-path returns -1, run `POST /find/entities/refresh-edges` to regenerate RELATES_TO edges
- `shortest_path`: response can be very large (90KB+) as it includes full entity/relation data for the entire path. No `compact` parameter available — client-side filtering recommended
- `shortest_path`: response paths use `entities` and `relations` arrays (NOT `nodes` and `edges`). Each path has `{entities: [...], relations: [...], length: N}`
- `merge`: target entity stays canonical (name/content preserved); source entities are absorbed. Auto-redirects Relation endpoints and refreshes RELATES_TO edges
- `merge`: rejects with HTTP 409 if source/target names have insufficient word overlap (uses word-level Jaccard for multi-word names, character-level for single-word); pass `skip_name_check: true` in body to override. **Chinese courtesy names (字) always need `skip_name_check`** since they share zero characters (e.g., 孔明 vs 诸葛亮)
- `merge`: returns 400 if target is in source list (self-merge) or if source entities don't exist
- `merge`: response data is nested at `data.merged_count` (not flat in `data`). Response includes `relations_updated` count
- `merge`: after merge, butler dry-run may still show the absorbed entity family_id until caches expire (TTL ~30s)
- `merge`: if a relation connects source and target entities, both endpoints resolve to target after merge (self-loop). No warning is returned
- `merge`: auto-updates source entity names to target name and refreshes RELATES_TO edges
- Auto-named entities (`auto_XXXXXXXX`) may outrank real entities in search results — filter by checking `content` field. The auto_* document-wrapper entity often scores higher than the actual extracted entity
- **by-name vs search score scales differ**: by-name reports raw BM25 scores (can be 5-20+), search reports normalized scores (0-1). An exact by-name match shows `match_score: 1.0`. These are not comparable across endpoints
- `dry_run:true` only works for `butler/execute`, NOT for merge or other destructive ops
- Valid `butler_execute` actions: `cleanup_isolated`, `cleanup_invalidated`, `fix_dangling_refs`, `cleanup_stale_redirects`, `detect_communities`, `evolve_summaries` (NOT `run_dream`)
- `butler_execute`: returns `success: true` even if individual actions fail — check each action's `status` field for errors
- If remember returns 0 entities, check LLM health: `GET /health/llm`
- `profile`: returns 404 `{success: false, error: "..."}` for nonexistent family_ids (not `success: true` with null). `neighbors` also returns 404 for nonexistent IDs
- Chinese characters in curl URLs: use `--data-urlencode` or Python urllib to avoid encoding issues
- Entity versions are ordered by `processed_time` (ingestion time), NOT `event_time` (when the event occurred)
- `traverse`: requires RELATES_TO edges to exist. If traverse returns empty but neighbors works, run `POST /find/entities/refresh-edges` first
- `communities/list`: returns empty or 500 if `detect_communities` hasn't been called — always POST `/communities/detect` first
- `search` results may show stale family_ids for entities that were merged/redirected — profile endpoint resolves correctly
- **neighbors edge naming**: edge fields use `source_uuid`/`target_uuid` (not `source_family_id`/`target_family_id`). Inconsistent with other endpoints' `entity1_family_id`/`entity2_family_id` naming
- **Semantic false positives**: entity search may return entities whose name contains a query substring — e.g., "Google AI Quantum" appears for "AI safety" queries because "AI" matches. Use `search_mode:"bm25"` or filter by `community_id` for precision
- **LLM health response**: `GET /health/llm` may return Chinese messages in the `message` field during cooldown. Check `llm_available: true/false` as the definitive field
- **Community detect response size**: `POST /communities/detect` returns the full community assignment map (30KB+). Follow with `GET /communities` for structured/paginated view
- **Graph fragmentation**: `shortest_path` returns -1 for many cross-community pairs even when RELATES_TO edges exist. The graph naturally fragments into disconnected clusters — this is expected, not an error. Cross-domain queries typically show no path
- **by-name search strictness**: `GET /find/entities/by-name/{name}` uses BM25 with threshold 0.7. Short names or translated names may not match if the stored entity name differs. Try `POST /find` with `search_mode:"hybrid"` as fallback
- **remember pipeline reliability**: if the server is under load, the relation alignment step (step 10/10) may hit Neo4j Bolt connection timeouts. If a task stalls at 99%, extraction results are likely saved but final relation writes failed — entities should still be searchable
- **remember stuck task blocks server**: a stuck remember task (stalled at step 9-10) can make the entire server unresponsive to other requests. If endpoints return empty/error responses unexpectedly, check for running tasks via `GET /remember/tasks`
- **remember timeout for large graphs**: graphs with 500+ entities may need `timeout:600` instead of default 300. Steps 9-10 (entity/relation alignment) are the bottleneck
- **Content language mismatch**: the LLM generates entity content/summaries in its default language (Chinese in most deployments) regardless of input text language. Manually update summaries to English via `PUT /find/entities/{fid}` if needed
- **Post-ingestion merge workflow**: the remember pipeline creates new entities even when similar ones exist. After ingestion, recommended workflow: search for potential duplicates (`GET /find/entities/search` or `by-name`), then `POST /find/entities/merge` to consolidate
- **Dream cycle side effects**: dream cycles may create new entities without connecting them, increasing the isolated entity count. Check isolated count before/after dream runs. Historical dream cycles frequently create 0 new relations
- **Dream status caching**: `GET /find/dream/status` may return stale data (previous cycle) after a new cycle completes. The log endpoint (`GET /find/dream/logs`) is more reliable for recent activity
- **Count discrepancies between endpoints**: `graph-summary` and `maintenance/health` return different entity/relation counts. `graph-summary` is faster (cached) but may be slightly stale; `maintenance/health` does a fresh scan
- **Contradictions endpoint**: returns empty (no error) when an entity has versions but no contradictions detected. Slow even for empty results (~3s) due to LLM call
- **compact=true truncates relation content**: relation `content` is truncated to ~80 chars with `compact=true`. For full content, omit compact or use the absolute_id lookup. Also strips `entity1_family_id`/`entity2_family_id` from relation objects (shows `null`)
- **Community response structure**: community list uses `members` array (not `entities`), each member has `family_id`, `name`, `uuid`. Community `name` field returns `N/A` (not auto-named)
- **Dream cycle response**: `POST /dream/run` returns synchronously with `{explored, seeds, stats, cycle_summary}` — NOT async. The `GET /dream/status` endpoint returns the previous completed cycle, not the current one
- **Dream cycle effectiveness**: dream creates relations via cross-neighbor pair generation. Use `discovery_mode:true` to lower confidence threshold (0.5→0.3) and find more connections. Best with `cross_community` strategy. Without discovery_mode, dream may still create 0 relations if entity pairs lack strong connections
- **Neighbors endpoint reliability**: may return transient 500 errors for entities with many relations. Retry after 1-2 seconds
- **Merge 0-update warning**: when merge rejects all sources (name similarity check), response includes `warning` field. Check `data.merged_count.entities_updated` — if 0, pass `skip_name_check: true`

- **Malformed JSON detection**: all POST endpoints now return `"请求体不是有效的 JSON（请检查格式）"` with 400 status when body is present but not valid JSON (previously returned misleading "missing field" errors)
- **Summary update behavior**: updating only `summary` or `attributes` (without name/content) is in-place — does NOT create a new version. If name/content is also updated, a new version IS created AND the metadata updates are applied to it
- **Relation create entity names**: `POST /find/relations/create` may return `entity1_name`/`entity2_name` as `null` in the response. Entity order may be swapped (sorted by absolute_id). Verify via the returned `family_id` fields, not names
- **GET search for Chinese**: the server auto-corrects URL encoding issues for Chinese query parameters. Both raw UTF-8 and percent-encoded work correctly
- **Concurrent load**: when 3+ clients hit the API simultaneously (especially during remember pipeline), Neo4j connection pool may exhaust causing transient 500 errors on traverse, neighbors, and search endpoints. All such errors resolve on retry after 2-3 seconds. Sequential usage is reliable
- **Concept data model differences**: `observation` concepts lack `family_id` (use `id`/UUID instead). `relation` concepts have `name: null` (use `content` for display). These are by design, not bugs
- **Orphaned relations after delete**: `DELETE /entities/{fid}?cascade=false` leaves relations intact but the deleted entity's name resolves to empty string in relation lists. Use `cascade=true` to clean up relations, or filter orphaned relations client-side
- **Concept traverse response size**: `POST /concepts/traverse` with depth 3 can return 500KB+ responses (502 concepts, 2281 edges). Use smaller `max_depth` or filter client-side
- **Stats count discrepancies**: `GET /find/graph-summary` uses cached counts; `GET /find/stats` uses live counts. Differences of 5-10 entities are normal due to cache TTL (~30s)
- **Remember input validation**: rejects empty text, whitespace-only text, and punctuation-only text (e.g. `"..."`, `"!!!"`). Returns 400 with Chinese error message
- **Graph counts in /graphs**: `GET /graphs` now shows accurate entity/relation counts for all graphs, including those not yet accessed in the current server session
- **DELETE /graphs/{id}**: removes the graph data and metadata completely. Graph will no longer appear in `/graphs` listing

## Slow Endpoints

These endpoints may take 5-15+ seconds. Use `timeout` param or increase curl timeout:
- `POST /find/ask` — LLM-powered natural language Q&A (~5-20s depending on graph size). Responses include full entity/relation data and can be 60KB+; `compact=true` query param does NOT reduce ask response size
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
