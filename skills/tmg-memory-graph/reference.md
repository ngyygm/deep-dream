# TMG API Reference

This reference complements `SKILL.md` with concrete endpoint choices and payload examples.

## Bootstrapping The Service

Use this when the user wants to start using TMG from scratch.

### 0. Clone the repository

```bash
git clone https://github.com/ngyygm/Temporal_Memory_Graph
cd Temporal_Memory_Graph
```

### 1. Create a Python environment and install dependencies

Recommended:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Core dependencies currently include:

- `openai>=1.0.0`
- `sentence-transformers>=2.2.0`
- `python-dateutil>=2.8.0`
- `numpy>=1.21.0`
- optional visualization packages such as `pyvis`, `networkx`, `matplotlib`

### 2. Check config files

- Main config: `Temporal_Memory_Graph/service_config.json`
- Example config: `Temporal_Memory_Graph/service_config.example.json`

Important fields in the current project config:

```json
{
  "host": "0.0.0.0",
  "port": 16200,
  "storage_path": "./graph/tmg_storage",
  "llm": {
    "api_key": "ollama",
    "model": "qwen3.5:4b",
    "base_url": "http://127.0.0.1:11434/v1"
  },
  "embedding": {
    "model": "/home/linkco/exa/models/Qwen3-Embedding-0.6B",
    "device": "cpu"
  }
}
```

Check that:

- `storage_path` can be created or written
- `llm.base_url` matches the running model service
- `embedding.model_path` exists if provided (or `embedding.model` as local path)

If `service_config.json` is missing, create it from the example first:

```bash
cp service_config.example.json service_config.json
```

### 3. Start the API service

From `Temporal_Memory_Graph/`:

```bash
python service_api.py --config service_config.json
```

### 4. Verify readiness

```http
GET /health
```

Expected successful fields:

- `storage_path`
- `embedding_available`
- `elapsed_ms`

## Service Discovery

- Config file: `Temporal_Memory_Graph/service_config.json`
- Current default base URL in this project: `http://127.0.0.1:16200`
- Health check:

```http
GET /health
```

## Remember

### Text input

```json
POST /api/remember
{
  "text": "罗辑是一名社会学教授，他被选为面壁者之一。",
  "source_name": "三体测试-文本"
}
```

### Local file path

```json
POST /api/remember
{
  "file_path": "/home/linkco/exa/datas/docs/三体2黑暗森林.txt",
  "source_name": "三体2黑暗森林"
}
```

### Multipart upload

Use form-data:

- `file`: uploaded file
- `source_name`: optional

## Unified Find

Use this as the default retrieval entrypoint.

```json
POST /api/find
{
  "query": "罗辑为什么被选为面壁者",
  "similarity_threshold": 0.5,
  "max_entities": 10,
  "max_relations": 20,
  "expand": true,
  "create_subgraph": false
}
```

Useful optional fields:

- `time_before`
- `time_after`
- `create_subgraph`

## Atomic Find Endpoints

### Entity semantic search

```http
GET /api/find/entities/search?query_name=罗辑&max_results=5&threshold=0.3
```

### Relation semantic search

```http
GET /api/find/relations/search?query_text=面壁者&max_results=5&threshold=0.3
```

### Latest entity version

```http
GET /api/find/entities/<entity_id>
```

### Entity version history

```http
GET /api/find/entities/<entity_id>/versions
```

### Entity at a specific time

```http
GET /api/find/entities/<entity_id>/at-time?time_point=2026-03-10T10:00:00
```

### All relations for an entity

```http
GET /api/find/relations/by-entity/<entity_id>?limit=20
```

## Subgraph Workflow

Use only when a multi-step graph exploration is needed.

### Create

```json
POST /api/find/subgraph
{
  "query_text": "白狐",
  "similarity_threshold": 0.5,
  "max_entities": 20,
  "max_relations": 50
}
```

### Read

```http
GET /api/find/subgraph/<subgraph_id>/entities
GET /api/find/subgraph/<subgraph_id>/relations
```

### Expand

```json
POST /api/find/subgraph/<subgraph_id>/expand
{
  "query_text": "罗辑"
}
```

### Filter

```json
POST /api/find/subgraph/<subgraph_id>/filter
{
  "time_before": "2026-12-31T23:59:59"
}
```

### Release

```http
DELETE /api/find/subgraph/<subgraph_id>
```

## Response Interpretation

Typical success envelope:

```json
{
  "success": true,
  "data": {
    "entity_count": 5,
    "relation_count": 8
  },
  "elapsed_ms": 123.45
}
```

Typical error envelope:

```json
{
  "success": false,
  "error": "请提供 text、file_path 或上传文件",
  "elapsed_ms": 4.52
}
```

## Notes For Agents

- TMG is one unified graph. Do not assume per-library routing.
- `remember` accepts natural-language text or documents. Avoid tag-heavy payloads.
- `find` returns candidates and local graph context. Final selection belongs to the caller.
- If the user asks for performance or latency, report `elapsed_ms`.
- If the user has not started the service yet, help them configure and start it before using the endpoints.
