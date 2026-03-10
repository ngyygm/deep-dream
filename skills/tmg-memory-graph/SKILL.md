---
name: tmg-memory-graph
description: Deploys, configures, starts, and uses the Temporal Memory Graph API to write natural-language memories and query the unified memory graph. Use when setting up this project's memory service, checking whether it is running, remembering text or documents, running semantic find queries, inspecting entities or relations, or helping another agent integrate with TMG endpoints.
---

# TMG Memory Graph

Use this skill when the user wants to set up or interact with the `Temporal_Memory_Graph` service through its API.

## What This System Is

- TMG is a unified natural-language memory graph, not a multi-library or tag-based system.
- All memory is written into one global graph.
- The system is responsible for `remember` and `find`.
- The caller is responsible for `select`, reranking, and final answer generation.

## Default Project Context

- GitHub repository: `https://github.com/ngyygm/Temporal_Memory_Graph`
- Project root: `Temporal_Memory_Graph/`
- Dependency file: `Temporal_Memory_Graph/requirements.txt`
- Default config: `Temporal_Memory_Graph/service_config.json`
- Example config template: `Temporal_Memory_Graph/service_config.example.json`
- Default API base URL in this project: `http://127.0.0.1:16200`
- Health endpoint: `GET /health`
- Startup command: `python service_api.py --config service_config.json`

Before using the API, prefer this order:

1. Read `Temporal_Memory_Graph/service_config.json` if the host or port may have changed.
2. Check whether the service is already running with `GET /health`.
3. If it is not running, inspect configuration and start the service if the user asked for setup, deployment, startup, or usage.
4. Only after the service is healthy should you proceed to `remember` or `find`.

## Setup And Deployment Workflow

Follow this workflow when the user wants to use TMG but the service may not exist yet.

### Step 0: Clone the repository

If the project is not already present locally, start from:

```bash
git clone https://github.com/ngyygm/Temporal_Memory_Graph
cd Temporal_Memory_Graph
```

### Step 1: Set up the environment

Recommended baseline flow:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

If the user already has a preferred environment manager such as conda, uv, or poetry, follow that instead.

### Step 2: Inspect configuration

Read these files first:

- `Temporal_Memory_Graph/service_config.json`
- `Temporal_Memory_Graph/service_config.example.json` if the main config is missing or incomplete

Confirm these fields:

- `host`
- `port`
- `storage_path`
- `llm.api_key`
- `llm.model`
- `llm.base_url`
- `embedding.model`: 本地路径或 HuggingFace 模型名（路径存在则用本地，否则自动下载）
- `embedding.device`

### Step 3: Validate runtime assumptions

Check:

- The configured `storage_path` is writable
- The LLM endpoint is expected to be reachable
- The embedding model path exists if `embedding.model` is a local path
- The Python environment has the packages from `requirements.txt`

If something is missing, tell the user exactly what needs to be configured instead of guessing.

### Step 4: Start the service

Default startup command:

```bash
python service_api.py --config service_config.json
```

If the user wants background execution or service management, follow their preferred method, but keep the API contract unchanged.

### Step 5: Verify readiness

Run or call:

```http
GET /health
```

Only consider the service ready if the response succeeds.

### Step 6: Use the API

After health is confirmed:

- Use `POST /api/remember` to ingest memory
- Use `POST /api/find` for normal semantic retrieval
- Use atomic `find` endpoints only for advanced inspection

## Core Rules

1. Prefer the HTTP API over direct Python processor calls unless the user explicitly asks for in-process usage.
2. Do not invent tags, categories, memory libraries, or scopes when writing memory.
3. Prefer `POST /api/find` for retrieval unless the user explicitly needs low-level or multi-step control.
4. Use atomic `find` endpoints only when the task specifically needs versions, exact entity lookup, relation lookup, or manual subgraph workflows.
5. When summarizing API results, include useful counts, key ids, and `elapsed_ms` if present.
6. If setup is part of the task, handle configuration, startup, and health verification before any data operations.
7. If the service is already healthy, do not restart it unnecessarily.
8. If the repository is missing locally, clone it before doing any setup work.

## Remember Workflow

Use `POST /api/remember`.

Choose the input mode like this:

- If the user gives plain text: send JSON with `text`.
- If the file already exists on the same machine as the service: send JSON with `file_path`.
- If the file must be transmitted to the service: use multipart upload with `file`.

Optional fields:

- `source_name`: preferred human-readable source label
- `load_cache_memory`: whether to continue from the latest memory cache chain

Remember request defaults:

- Keep input natural-language.
- Do not add artificial structure unless the user explicitly wants it.
- Use the original filename as `source_name` when that improves traceability.

## Find Workflow

Default retrieval path:

1. Use `POST /api/find` with a natural-language `query`.
2. Start with `expand: true`.
3. Use moderate limits unless the user asks otherwise:
   - `max_entities: 10-20`
   - `max_relations: 20-50`
4. Add `time_before` or `time_after` only when the user asks for temporal filtering.
5. Use `create_subgraph: true` only when you need follow-up expand/filter/entity/relation calls.

Use atomic endpoints when needed:

- `GET /api/find/entities/search` for entity-focused semantic lookup
- `GET /api/find/relations/search` for relation-focused semantic lookup
- `GET /api/find/entities/<entity_id>/versions` for version history
- `GET /api/find/entities/<entity_id>/at-time` for temporal replay
- `POST /api/find/subgraph` and related endpoints for explicit multi-step graph exploration

## Response Handling

Successful responses use:

```json
{"success": true, "data": {...}, "elapsed_ms": 12.34}
```

Error responses use:

```json
{"success": false, "error": "...", "elapsed_ms": 12.34}
```

When reporting results to the user:

- Highlight the most relevant entities and relations, not the whole payload.
- Preserve exact ids when they may be used in the next step.
- Mention `elapsed_ms` when performance matters or when the user asks.

## Recommended Patterns

### Pattern 1: Write text memory

- Health check
- `POST /api/remember` with `{"text": "...", "source_name": "..."}`
- Report `memory_cache_id`, `chunks_processed`, and `elapsed_ms`

### Pattern 2: Write document memory

- Prefer `file_path` if the file is already local to the service host
- Fall back to multipart upload when needed
- For very large files, warn the user that ingest may take time

### Pattern 3: Semantic recall

- `POST /api/find` with the user’s natural-language question
- Summarize returned entities and relations
- If needed, follow up with entity version or subgraph endpoints

### Pattern 4: End-to-end setup and use

- Clone the repository if missing
- Create or activate a Python environment
- Install `requirements.txt`
- Read config
- Start the service if not already running
- Verify `GET /health`
- Write memory with `POST /api/remember`
- Query with `POST /api/find`
- Report concrete API base URL, key ids, and elapsed time

## Avoid

- Do not treat TMG as a multi-tenant vector store with namespaces unless the project changes.
- Do not force tags onto the input.
- Do not pretend `find` already performed `select`.
- Do not omit failures from the user; surface API errors clearly.

## Additional Resources

- For endpoint details and example payloads, read [reference.md](reference.md)
