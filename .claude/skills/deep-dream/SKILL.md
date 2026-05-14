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

Natural-language memory graph. **Remember** (write → auto-extract entities/relations) + **Find** (semantic retrieval). All concepts are one primitive with roles: `entity`, `relation`, `observation`. Concept tools (`search_concepts` etc.) provide cross-role access; use specialized tools (`create_entity`, `create_relation`) for type-specific ops.

```
BASE_URL=http://localhost:16200/api/v1   |   graph_id=default   |   Response: {success, data}
```

## CRITICAL: Always Use MCP Tools

**NEVER use raw `curl` or direct HTTP calls to the API.** Use the MCP tools listed below (e.g. `get_entity_versions`, `entity_profile`, `search_entities`). The MCP server handles URL construction, response parsing, error handling, and ID validation automatically.

If you catch yourself writing `curl -s "http://localhost:16200/api/v1/..."`, stop — use the corresponding MCP tool instead.

### Fallback: REST API Direct Access

If MCP tools are unavailable, use the REST API at `http://localhost:16200/api/v1/`. All endpoints require `?graph_id=default` (or your target graph).

| MCP Tool | REST Endpoint | Method |
|---|---|---|
| `graph_overview` | `/find/graph-summary` | GET |
| `remember` | `/remember` | POST body: `{text, wait, timeout}` |
| `quick_search` | `/find` | POST body: `{query}` |
| `entity_profile` | `/find/entities/{fid}/profile` | GET |
| `create_entity` | `/find/entities/create` | POST body: `{name, content}` |
| `create_relation` | `/find/relations/create` | POST body: `{entity1_family_id, entity2_family_id, content}` |
| `search_entities` | `/find/entities/search?query_name=X` | GET |
| `dream_run` | `/find/dream/run` | POST body: `{strategy, seed_count}` |
| `butler_report` | `/butler/report` | GET |
| `butler_execute` | `/butler/execute` | POST body: `{actions, dry_run}` |
| `health_check_llm` | `/health/llm` | GET |

Error format: `{"success": false, "error": "...", "elapsed_ms": N}`. Error messages may be in Chinese.

## Decision Guide

| Intent | Tool | Key Notes |
|---|---|---|
| First call / overview | `graph_overview` | Stats + activity + health |
| Check LLM health | `health_check_llm` | Verify LLM is reachable before remember/dream |
| Store text | `remember` (wait=true) | Sync; async returns task_id |
| Store + explore | `remember_and_explore` | 1 call = write + search |
| Search anything | `quick_search` | query → entities + relations |
| Deep topic dive | `explore_topic` | search + traverse |
| NL question | `ask` | AI reasoning over graph |
| Find by name | `find_entity_by_name` | Fuzzy match |
| Entity details | `entity_profile` | Details + relations + versions |
| Batch entities | `batch_profiles` | Up to 20 |
| Cross-role search | `search_concepts` | Unified concept search |
| Concept provenance | `get_concept_provenance` | Trace to source observation |
| Concept traversal | `traverse_concepts` | Cross-role BFS |
| Concept neighbors | `get_concept_neighbors` | Unified graph traversal |
| Dream (full) | `dream_run` | 1 call = full cycle |
| Dream (quick) | `dream_quick_start` | Lightweight start |
| Merge entities | `merge_entities` | target + sources |
| Split mixed entity | `split_entity_version` | Needs absolute_id |
| Update confidence | API: `PUT .../confidence` | No MCP tool yet |
| Dream candidates | API: `GET /dream/candidates` | No MCP tool yet |
| Health / cleanup | `butler_report` → `butler_execute` | 2 calls; valid actions: cleanup_isolated, cleanup_invalidated, detect_communities, evolve_summaries |
| Communities | `detect_communities` → `get_community` | Neo4j only |

### More Tools

**Search**: `semantic_search`, `search_entities`, `search_relations`, `traverse_graph`, `search_shortest_path`

**Entity**: `get_entity`, `get_entity_versions`, `get_entity_timeline`, `create_entity`, `update_entity`, `delete_entity`, `evolve_entity_summary`, `get_entity_contradictions`, `refresh_graph_edges`

**Relation**: `get_relations_between`, `create_relation`, `update_relation`, `delete_relation`, `invalidate_relation`, `redirect_relation`

**Concept**: `search_concepts`, `list_concepts`, `get_concept`, `get_concept_neighbors`, `get_concept_provenance`, `traverse_concepts`, `get_concept_mentions`

**Maintenance**: `maintenance_health`, `maintenance_cleanup`, `cleanup_old_versions`, `detect_communities`

**Graph**: `switch_graph`, `list_graphs`, `create_graph`, `delete_graph`

**Write**: `remember`, `remember_and_explore`, `batch_ingest_episodes`

## ID System

`family_id` (stable, `ent_*`/`rel_*`) — most ops | `absolute_id` (UUID version snapshot) — split_entity_version, version diff

## Dream Mode

Triggered by "dream"/"做梦". Use `dream_run` (1 call) or manual: graph_summary → dream/seeds → entity_profile → traverse → create_dream_relation → save_dream_episode. 8 types: free_association, cross_domain, leap, contrastive, temporal_bridge, orphan_adoption, hub_remix, narrative (see `references/dream-types/`). **Rules**: evidence required, honest confidence (0.3-0.5 if unsure), check existing first, always `save_dream_episode`.

Note: Dream relation discovery requires a running LLM. If `dream_run` returns 0 relations with a warning, check `health_check_llm` first.

## Butler Mode

`butler_report` → `butler_execute`. Or manual: `maintenance_health` → cleanup/communities/evolve → verify with `graph_summary`.

Valid `butler_execute` actions: `cleanup_isolated`, `cleanup_invalidated`, `detect_communities`, `evolve_summaries`. Note: `run_dream` may appear in butler_report recommendations but is NOT a valid butler_execute action — use `dream_run` or `dream_quick_start` instead.

## Parameter Pitfalls

- `remember`: set `wait=true` for sync mode (blocks until extraction done); default is async (returns task_id to poll)
- `remember`: if extraction returns 0 entities, check `health_check_llm` — LLM may be down
- `create_relation`: accepts family_id (`entity1_family_id`) or absolute_id (`entity1_absolute_id`); family_id is simpler
- `create_dream_relation`: uses family_id (not absolute_id)
- `update_entity`: name/content changes create a new version; summary/attribute changes are in-place
- `split_entity_version`: needs absolute_id — `get_entity_versions` first
- Concept tools: accept any prefix (`ent_*`, `rel_*`, `episode`)
- Destructive ops: **default dry_run=true** (must pass false to execute)
- search_mode: semantic / bm25 / hybrid
- relation_scope: accumulated / version_only / all_versions

## Common Mistakes

| Don't | Do |
|---|---|
| `get_entity` + `get_entity_relations` | `entity_profile` (1 call) |
| Manual write + search | `remember_and_explore` (1 call) |
| Jump to search on first visit | `graph_overview` first |
| Only `search_entities` for concepts | `search_concepts` (cross-role) |
| 15-25 manual dream calls | `dream_run` (1 call) |
| Forget to poll remember | Use `wait=true` |
| Use `health_check` only | `health_check_llm` to verify LLM before remember/dream |
| Ignore 0-extraction results | Check `health_check_llm`, may be LLM connectivity issue |

MCP server auto-protects: ID type detection, empty result hints, pagination warnings, response truncation, destructive op safety defaults.

## Full API Reference

100+ endpoints with parameters and examples: [references/api-reference.md](references/api-reference.md)
