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

All endpoints accept `?graph_id=<id>` to target a specific graph. Add `&compact=true` to strip embeddings and truncate content for agent-friendly responses.

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
| Quick search | POST | `/find` | `{query}` |
| Traverse graph | POST | `/find/traverse` | `{seed_family_ids:[...], max_depth}` |
| Shortest path | POST | `/find/paths/shortest` | `{family_id_a, family_id_b}` |
| Create entity | POST | `/find/entities/create` | `{name, content}` |
| Create relation | POST | `/find/relations/create` | `{entity1_family_id, entity2_family_id, content}` |
| Update entity | PUT | `/find/entities/{fid}` | `{name, summary, attributes}` |
| Merge entities | POST | `/find/entities/merge` | `{source_family_ids:[...], target_family_id:...}` |
| Dream cycle | POST | `/find/dream/run` | `{strategy, seed_count}` |
| Dream status | GET | `/find/dream/status` | — |
| Ask NL question | POST | `/find/ask` | `{question}` |
| Butler report | GET | `/butler/report` | — |
| Butler execute | POST | `/butler/execute` | `{actions:[...], dry_run:true}` |
| Health report | GET | `/find/maintenance/health` | — |
| Detect communities | POST | `/communities/detect` | `{algorithm:"louvain"}` |

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

### Butler Health Check
```bash
# Get report
curl -s "$BASE_URL/butler/report?graph_id=default"

# Preview cleanup (dry_run)
curl -s -X POST "$BASE_URL/butler/execute?graph_id=default" \
  -H 'Content-Type: application/json' \
  -d '{"actions":["cleanup_isolated","cleanup_invalidated"],"dry_run":true}'
```

## ID System

- **family_id** (`ent_*`/`rel_*`): stable ID for most operations. Example: `ent_abc123`
- **absolute_id** (UUID): version-specific snapshot. Used for: `split_entity_version`, version diff. Example: `entity_20260514_231113_6e558246`
- `create_relation` accepts **family_ids** (`entity1_family_id`/`entity2_family_id`) — no need to resolve to absolute_id

## Response Format

All responses: `{"success": bool, "data": {...}, "elapsed_ms": float}`

Errors: `{"success": false, "error": "message", "hint": "actionable guidance", "elapsed_ms": float}`

- Error messages may be in Chinese. The `hint` field provides English guidance.
- Add `compact=true` query param to strip embeddings and truncate content.

## Parameter Pitfalls

- `remember`: use `wait:true` for sync mode; default is async (returns task_id to poll)
- Entity search uses `query_name` param, not `q` or `query`
- Relations between entities uses `family_id_a`/`family_id_b` param names
- `update_entity`: name/content changes create a new version; summary/attribute changes are in-place
- Destructive ops: pass `dry_run:true` in body to preview before executing
- Valid `butler_execute` actions: `cleanup_isolated`, `cleanup_invalidated`, `detect_communities`, `evolve_summaries` (NOT `run_dream`)
- If remember returns 0 entities, check LLM health: `GET /health/llm`

## Full API Reference

100+ endpoints with parameters and examples: [references/api-reference.md](references/api-reference.md)
