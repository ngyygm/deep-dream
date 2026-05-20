# Deep-Dream

Deep-Dream is a Document-first memory graph for AI agents.

The v1 storage model treats Markdown documents as read-only sources of truth, splits them into episodes, and stores extracted entities and relations as unified concept families, concept versions, and concept edges.

## Current Model

```text
Document -> Episode -> Concept
```

- `document`: a Markdown source document or remembered text source.
- `episode`: a heading-first source span/chunk inside a document.
- `entity`: an extracted entity concept.
- `relation`: an extracted relation concept.

Relations are concepts too. A relation version uses `CONNECTS` edges to point to endpoint entity concepts and `ASSERTS` edges to trace the episode that asserted it.

Each episode mention creates a new concept version across episodes. Within the same episode, the same concept family is written once.

## Storage Layout

Each graph is physically isolated:

```text
{storage_root}/
  graphs/
    {graph_id}/
      graph.db
      blobs/
      artifacts/
      indexes/
      logs/
  registry.json
```

Graphs do not share SQLite databases, blobs, indexes, or logs.

## API

Base URL: `http://localhost:16200/api/v1`

Core endpoints:

- `POST /remember`
- `POST /vaults/index`
- `GET /documents`
- `GET /concepts`
- `GET /concepts/<family_id>`
- `GET /concepts/<family_id>/versions`
- `GET /concepts/<family_id>/provenance`
- `POST /concepts/search`
- `POST /traverse`
- `GET /graphs`
- `POST /graphs`
- `DELETE /graphs/<graph_id>`

All graph-scoped endpoints accept `graph_id`; omitted `graph_id` defaults to `default`.

## Development

Run focused storage tests:

```bash
python -m pytest core/tests/test_concept_store_v1.py
```

Run the API server:

```bash
python -m core.server.api --config service_config.json --skip-llm-check
```
