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
    "model": "model_name",
    "base_url": "http://127.0.0.1:11434"
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

可选：检查大模型是否真正可用（会发起一次极短 LLM 调用）：

```http
GET /health/llm
```

成功时 `llm_available: true`；失败时 HTTP 503。

## Service Discovery

- Config file: `Temporal_Memory_Graph/service_config.json`
- Current default base URL in this project: `http://127.0.0.1:16200`
- Health check:

```http
GET /health
```

**完整 HTTP 路由与参数以仓库内 [`service_api.py`](../../service_api.py) 为准**（本文件只列与 `SKILL.md` 配套的常用入口与原子查询示例）。

## 推荐测试库：`tmg_storage` 与目录结构（最新格式）

本地开发与联调时，建议将 `service_config.json` 中的 **`storage_path`** 指向仓库内 **`Temporal_Memory_Graph/graph/tmg_storage`**（相对路径写作 `./graph/tmg_storage`，启动时工作目录为项目根目录）。该目录在仓库中已具备**最新布局**的示例数据（含 `graph.db`、`memory_caches/`、`originals/` 等），便于对照接口行为。

在任意 `storage_path` 根目录下，核心子路径与含义如下：

| 路径 | 含义 |
|------|------|
| `graph.db` | SQLite：概念实体、关系、embedding 元数据等，**统一记忆图**本体。 |
| `memory_caches/json/`、`memory_caches/md/` | 流水线产生的记忆缓存（滑窗摘要链）；与实体/关系抽取对齐。 |
| `originals/*.txt` | **每次** `GET /api/remember` 成功受理时写入的**原文快照**（一份请求一个文件）。文件名形如：`{source_name 净化后}_{时间戳}_{随机8位}.txt`。 |
| `remember_journal/{task_id}.json` | 异步任务状态与结果摘要；进程崩溃重启后，未完成的 `queued`/`running` 会据此**从对应 `originals` 文件重新入队**处理。 |

### `originals` 与接口的对应关系

1. **写入**：调用 `GET /api/remember` 后，HTTP 202 的 `data.original_path` 即为磁盘上该次提交的 **`originals/...txt` 绝对路径**；`remember_journal/{task_id}.json` 里也会记录同一 `original_path`。
2. **任务进度**：`GET /api/remember/status/<task_id>` 返回的 `original_path` 与上述一致；`completed` 时 `result` 内也会带上处理统计信息。
3. **检索图内容（不直接读 txt）**：图结构在 `graph.db` 中；应用侧应优先用 **`POST /api/find`**（自然语言）或 **原子接口**（实体/关系/时间）查询，而不是把 `originals` 当全文检索库。`originals` 的作用是：**审计、恢复重跑、与人类对齐「当时入库原文」**。
4. **按实体/关系深挖**：从 `find` 得到 `entity_id` / `id`（版本绝对 id）后，可继续调用 `GET /api/find/entities/<entity_id>/versions`、`GET /api/find/relations/by-entity/<entity_id>` 等（见下文 Atomic 小节）。

### 想查什么、用什么接口（速查）

| 目的 | 推荐接口 |
|------|----------|
| 自然语言回忆、唤醒一片相关记忆 | `POST /api/find` |
| 按名称/语义找实体 | `GET /api/find/entities/search?query_name=...` |
| 按语义找关系 | `GET /api/find/relations/search?query_text=...` |
| 某逻辑实体最新版本 / 版本链 / 某时间点版本 | `GET /api/find/entities/<entity_id>`、`.../versions`、`.../at-time?time_point=...` |
| 与某实体相连的关系 | `GET /api/find/relations/by-entity/<entity_id>` |
| 与「按条件抽一批」等价的批量拉取 | `POST /api/find/query-one`（JSON body） |
| 记忆缓存链（调试/对齐） | `GET /api/find/memory-cache/latest`、`/api/find/memory-cache/<cache_id>` 等 |
| 统计总量 | `GET /api/find/stats` |

### Agent 自行编程、组合查询

TMG **不强制**单一对话式调用：Agent 可以用 **任意语言**（Python `requests`、Node `fetch`、Shell `curl` 等）编排多个 HTTP 请求，例如：

- `POST /api/find` 召回候选 → 解析返回的 `entities[].entity_id` → 再请求 `GET /api/find/entities/<entity_id>/versions` 做版本对比；
- `GET /api/find/entities/search` 收窄实体 → `GET /api/find/relations/between?from_entity_id=...&to_entity_id=...`；
- 先 `GET /health` 确认 `embedding_available`，再决定走语义检索还是纯原子接口。

组合逻辑、重试与缓存策略由 **Agent 侧代码**决定；服务端只保证各 REST 接口语义稳定。完整路径仍以 [`service_api.py`](../../service_api.py) 为准。

## Remember

**仅 GET**（查询参数）。长正文请用 `text_b64`。批量写入、避免一两句一调。

**默认异步** — HTTP 202，立即返回 `task_id`；后台 worker 处理。`flask_threaded` 默认为 true 时，**remember 运行期间 find 仍可并发处理**。

崩溃恢复：任务状态落在 `storage_path/remember_journal/`；重启后 `queued` / `running` 会重新入队，从 `originals/` 保存的原文 **整段重跑**（与上次是否处理到一半无关）。

### Request (GET)

```http
GET /api/remember?text=...&source_name=三体测试&event_time=2026-03-09T14:00:00&load_cache_memory=false
```

或使用 Base64（推荐长文本）：

```http
GET /api/remember?text_b64=SGVsbG8...&source_name=三体测试
```

| Query param | Required | Description |
|-------------|----------|-------------|
| `text` | One of `text` / `text_b64` | URL-encoded UTF-8 body |
| `text_b64` | One of `text` / `text_b64` | Standard Base64 of UTF-8 text |
| `source_name` / `doc_name` | No | Source label (default `api_input`) |
| `event_time` | No | ISO 8601 — actual event time for this batch |
| `load_cache_memory` | No | `true`/`false`/`1`/`0` |

### Response (202)

```json
{ "success": true, "data": { "task_id": "abc123", "status": "queued", "original_path": "..." } }
```

### 查询任务状态

```http
GET /api/remember/status/<task_id>
```

返回 `status`（`queued` / `running` / `completed` / `failed`）、`result`（完成时）、`error`（失败时）。若任务已从内存淘汰，会尝试从 `remember_journal` 读取。

### 查看任务队列

```http
GET /api/remember/queue?limit=50
```

正文保存到 `storage_path/originals/`，`original_path` 在响应与 journal 中返回。

## Unified Find

Use this as the default retrieval entrypoint.

```json
POST /api/find
{
  "query": "罗辑为什么被选为面壁者",
  "similarity_threshold": 0.5,
  "max_entities": 10,
  "max_relations": 20,
  "expand": true
}
```

Useful optional fields:

- `time_before`
- `time_after`

### Batch fetch by same conditions (`query-one`)

与「按条件从主图取一批实体/关系」相关的辅助接口（无服务端持久化子图，一次性返回）：

```json
POST /api/find/query-one
{
  "query_text": "罗辑",
  "similarity_threshold": 0.5,
  "max_entities": 20,
  "max_relations": 50,
  "include_entities": true,
  "include_relations": true
}
```

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
  "error": "text 为必填字段",
  "elapsed_ms": 4.52
}
```

## Notes For Agents

- TMG is one unified graph. Do not assume per-library routing.
- `remember` is **GET only** — use query params `text` or `text_b64`. No POST JSON, no file_path or multipart.
- `remember` is **async** — returns `task_id` immediately (HTTP 202); poll `/api/remember/status/<task_id>`. Unfinished jobs persist under `remember_journal/` and are re-queued after restart.
- Batch content in remember calls — do not send one or two sentences at a time.
- Use `event_time` when the content describes events that happened earlier than the request time.
- `find` returns candidates and local graph context. Final selection belongs to the caller.
- If the user asks for performance or latency, report `elapsed_ms`.
- If the user has not started the service yet, help them configure and start it before using the endpoints.
