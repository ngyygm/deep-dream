---
name: deep-dream
description: >
  Use when interacting with a Deep-Dream knowledge graph server for remembering
  Markdown/text, indexing an Obsidian-style vault, searching concepts, tracing
  provenance, traversing the concept graph, or managing isolated graph stores.
  Triggers: knowledge graph, memory, remember, find, markdown vault, Obsidian,
  document, episode, concept, entity, relation, provenance, graph.
---

# Deep-Dream Memory Graph

Deep-Dream v1 stores a Document-first concept graph:

`Document -> Episode -> Concept`

- Markdown documents are read-only sources of truth.
- Documents are split into heading-first episodes.
- Entity and relation mentions become concept families and concept versions.
- Every cross-episode mention creates a new concept version.
- Each graph is physically isolated under `storage_root/graphs/<graph_id>/`.

```
BASE_URL=http://localhost:16200/api/v1
graph_id=default
response={success,data,error,elapsed_ms}
```

Always pass `graph_id` by query string, JSON body, or `X-Graph-Id`.

## Core Endpoints

| Intent | Method | Endpoint | Body / Query |
|---|---|---|---|
| Health check | GET | `/health` | `?graph_id=default` |
| Create graph | POST | `/graphs` | `{graph_id,name,description}` |
| List graphs | GET | `/graphs` | none |
| Delete graph | DELETE | `/graphs/<graph_id>` | none |
| Remember text/file | POST | `/remember` | `{text, wait:true, timeout:300}` or multipart `file` |
| List remember tasks | GET | `/remember/tasks` | `?graph_id=default` |
| Index vault | POST | `/vaults/index` | `{path, graph_id, force:false}` |
| List documents | GET | `/documents` | `?graph_id=default&limit=50` |
| Search concepts | POST | `/concepts/search` | `{query, role, limit, search_mode:"bm25"}` |
| Get concept | GET | `/concepts/<family_id>` | `?graph_id=default` |
| Concept versions | GET | `/concepts/<family_id>/versions` | `?graph_id=default` |
| Concept provenance | GET | `/concepts/<family_id>/provenance` | `?graph_id=default&time_point=...` |
| Traverse graph | POST | `/traverse` | `{start_family_ids:[...], max_depth:2, edge_types:[...]}` |

## Roles

- `document`: Markdown source document.
- `episode`: a source span/chunk inside a document.
- `entity`: extracted entity concept.
- `relation`: extracted relation concept.

## Edge Types

- `DOCUMENT_LINK`: explicit Markdown/Obsidian document links.
- `HAS_EPISODE`: document version to episode version.
- `MENTIONS`: episode to entity/relation concept.
- `ASSERTS`: episode asserts a relation concept.
- `CONNECTS`: relation concept connects endpoint entity concepts.

## Typical Workflows

### Remember Text

```bash
curl -s -X POST "$BASE_URL/remember?graph_id=default" \
  -H "Content-Type: application/json" \
  -d '{"text":"Alice works with Bob on Project Orion.","wait":true,"timeout":300}'
```

### Index an Obsidian Vault

```bash
curl -s -X POST "$BASE_URL/vaults/index?graph_id=default" \
  -H "Content-Type: application/json" \
  -d '{"path":"C:/notes/my-vault","force":false}'
```

### Search and Trace

```bash
curl -s -X POST "$BASE_URL/concepts/search?graph_id=default" \
  -H "Content-Type: application/json" \
  -d '{"query":"Project Orion","role":"entity","limit":10}'

curl -s "$BASE_URL/concepts/confam_example/provenance?graph_id=default"
```

### Traverse by Explicit Edge Types

```bash
curl -s -X POST "$BASE_URL/traverse?graph_id=default" \
  -H "Content-Type: application/json" \
  -d '{"start_family_ids":["confam_example"],"max_depth":2,"edge_types":["MENTIONS","CONNECTS"]}'
```

## Operational Notes

- Use a separate `graph_id` per project or tenant; graphs do not share SQLite DBs, blobs, indexes, or logs.
- Vault indexing is read-only; Deep-Dream never writes back to the Markdown vault.
- `force:false` skips unchanged Markdown files by content hash.
- Same episode + same concept family writes one version; a later episode writes a fresh version even if content is unchanged.
