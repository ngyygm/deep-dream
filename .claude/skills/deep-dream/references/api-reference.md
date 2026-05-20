# Deep-Dream v1 API Reference

Base URL: `http://localhost:16200/api/v1`

All graph-scoped endpoints use `graph_id` from query string, JSON body, form data, or `X-Graph-Id`. Missing `graph_id` defaults to `default`.

## Graphs

- `GET /graphs`
- `POST /graphs` with `{graph_id, name?, description?}`
- `GET /graphs/<graph_id>`
- `DELETE /graphs/<graph_id>`
- `POST /graphs/<graph_id>/clear`

## Remember

- `POST /remember`
  - JSON: `{text, graph_id?, source_name?, event_time?, wait?, timeout?}`
  - Multipart: `file`, optional `graph_id`
- `GET /remember/tasks`
- `GET /remember/tasks/<task_id>`
- `DELETE /remember/tasks/<task_id>`
- `POST /remember/tasks/<task_id>/pause`
- `POST /remember/tasks/<task_id>/resume`
- `GET /remember/monitor`

## Vaults And Documents

- `POST /vaults/index` with `{path, graph_id?, force?}`
- `GET /documents?graph_id=<id>&limit=50&offset=0`

## Concepts

- `GET /concepts?role=document|episode|entity|relation&limit=50&offset=0`
- `POST /concepts/search` with `{query, role?, limit?, search_mode?}`
- `GET /concepts/<family_id>`
- `GET /concepts/<family_id>/versions`
- `GET /concepts/<family_id>/provenance`
- `GET /concepts/<family_id>/mentions`
- `GET /concepts/<family_id>/neighbors`
- `POST /traverse` with `{start_family_ids, max_depth?, edge_types?, time_point?}`
- `POST /concepts/traverse` same body as `/traverse`

## System

- `GET /health`
- `GET /health/llm`
- `GET /routes`
- `GET /find/stats`
- `GET /graph/stats`
- `GET /system/overview`
- `GET /system/graphs`
- `GET /system/graphs/<graph_id>`
- `GET /system/tasks`
- `GET /system/logs`
- `GET /system/access-stats`
