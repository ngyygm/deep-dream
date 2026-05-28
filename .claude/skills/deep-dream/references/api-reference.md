# Deep-Dream v1 API Reference

Base URL: `http://localhost:16200/api/v1`

All graph-scoped endpoints use `graph_id` from query string, JSON body, form data, or `X-Graph-Id`. Missing `graph_id` defaults to `default`.

Response format: `{success: bool, data: any, error: string|null, elapsed_ms: float}`

## Remember

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/remember` | Write text or file. JSON: `{text, graph_id?, source_name?, event_time?, wait?, timeout?}`. Multipart: `file` field + optional `graph_id`. Supported formats: .txt .md .json .html .pdf .docx .csv .log (max 10MB) |
| GET | `/remember/tasks` | List remember tasks. Query: `?graph_id` |
| GET | `/remember/tasks/<task_id>` | Get task status. Query: `?graph_id` |
| DELETE | `/remember/tasks/<task_id>` | Cancel/delete task |
| POST | `/remember/tasks/<task_id>/pause` | Pause running task |
| POST | `/remember/tasks/<task_id>/resume` | Resume paused task |
| POST | `/remember/tasks/resume-all` | Resume all paused tasks |
| GET | `/remember/monitor` | Real-time pipeline snapshot (windows, threads, storage counts) |

## Concepts

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/concepts` | List concepts. Query: `?role=entity\|relation\|document\|episode&limit&offset` |
| POST | `/concepts/search` | Search. Body: `{query, role?, limit?, search_mode?, fields?, group?, expand?, threshold?, source_document?, max_name_length?, reranker?, time_point?}`. `search_mode`: semantic/bm25/hybrid. `fields`: comma-separated field list for lightweight responses (always includes `family_id`). `group`: cluster results thematically. |
| GET | `/concepts/suggest` | Autocomplete/disambiguation. Query: `?query&role&limit&source_document&max_name_length`. Min 2 chars. Returns `{family_id, name, relevance, role, source_document}`. GET only. |
| GET | `/concepts/duplicates` | Find potential duplicate entities |
| GET | `/concepts/<family_id>` | Concept details. Query: `?compact` |
| PATCH | `/concepts/<family_id>` | Update concept content. Body: `{name?, content?, confidence?, metadata?}` |
| GET | `/concepts/<family_id>/versions` | Version history |
| GET | `/concepts/<family_id>/provenance` | Which episodes asserted this. Query: `?time_point` |
| GET | `/concepts/<family_id>/mentions` | Episodes mentioning this concept |
| GET | `/concepts/<family_id>/neighbors` | Connected concepts. Query: `?compact&fields&max_depth&max_results&time_point` |

## Search & Traverse

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/agent/sql` | Agent read-only SQL workbench. Body: `{sql, params?, limit?, timeout_seconds?, explain?, graph_id?}`. Allows SELECT/WITH/EXPLAIN QUERY PLAN only. |
| POST | `/agent/semantic-search` | Agent semantic candidate recall. Body: `{query, role?, top_k?, threshold?, source_document?, graph_id?}` |
| POST | `/find` | Combined entity+relation search. Body: `{query, limit?, ...}` |
| POST | `/traverse` | BFS graph traversal. Body: `{start_family_ids, max_depth?, max_results?, edge_types?, time_point?}` |
| POST | `/concepts/traverse` | Alias for `/traverse` |
| GET | `/find/stats` | Graph statistics (entity/relation/episode counts) |
| GET | `/graph/stats` | Alias for find/stats |

`edge_types` filter: `DOCUMENT_LINK HAS_EPISODE MENTIONS ASSERTS CONNECTS`

## Documents

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/documents` | List documents (paginated). Query: `?graph_id&limit&offset`. Returns `{documents, total, limit, offset}` |
| POST | `/documents/graph` | Document->Episode->Concept subgraph. Body: `{document_version_ids: [str], document_family_ids?: [str]}` |
| POST | `/documents/graph/outline` | Document graph outline |
| POST | `/documents/graph/chunk` | Progressive graph chunk |
| GET | `/documents/<id>/content` | Document Markdown content |
| DELETE | `/documents/<id>` | Delete document version |
| POST | `/vaults/index` | Index Obsidian/Markdown vault. Body: `{path, graph_id?, force?}` |

## Graphs

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/graphs` | List all graphs |
| POST | `/graphs` | Create graph. Body: `{graph_id, name?, description?}` |
| GET | `/graphs/<graph_id>` | Graph details |
| DELETE | `/graphs/<graph_id>` | Delete graph (data + metadata) |
| POST | `/graphs/<graph_id>/clear` | Clear all data, keep graph |

## System

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/health` | Service health (storage backend, graph_id) |
| GET | `/health/llm` | LLM connectivity check |
| GET | `/routes` | Dynamic route index |
| GET | `/system/overview` | Uptime, threads, graph count |
| GET | `/system/graphs` | All graphs summary |
| GET | `/system/graphs/<graph_id>` | Single graph detailed status |
| GET | `/system/tasks` | All graphs task list |
| GET | `/system/config` | Read service config |
| PATCH | `/system/config` | Update service config |
| GET | `/system/logs` | Recent logs. Query: `?level&limit&source` |
| GET | `/system/access-stats` | API access statistics |
| GET | `/system/dashboard` | Combined dashboard |

## Chat Sessions

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/chat/sessions` | List sessions |
| POST | `/chat/sessions` | Create session |
| GET | `/chat/sessions/<sid>` | Session details |
| PUT | `/chat/sessions/<sid>` | Update metadata |
| DELETE | `/chat/sessions/<sid>` | Delete session |
| POST | `/chat/sessions/<sid>/close` | Close session |
| POST | `/chat/sessions/<sid>/stream` | Send message (SSE stream) |

## Common Query Parameters

| Param | Description |
|-------|-------------|
| `graph_id` | Target graph (default: "default") |
| `limit` / `offset` | Pagination |
| `max_results` | Cap BFS traversal results (default 500, max 2000) |
| `role` | Filter: `document`, `episode`, `entity`, `relation` |
| `search_mode` | `semantic`, `bm25`, `hybrid` (default) |
| `time_point` | ISO timestamp for time-travel queries |
| `compact` | `true` strips embeddings, truncates content |
| `wait` | Block until remember pipeline finishes |
| `timeout` | Max seconds to wait (default 300) |
