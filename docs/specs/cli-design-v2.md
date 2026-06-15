# Deep-Dream CLI 完整设计规格

**版本**: 2.0  
**日期**: 2026-06-01  
**框架**: Click 8+ / Rich 13+  
**状态**: Final Draft  

---

## 目录

1. [设计原则](#1-设计原则)
2. [架构概览](#2-架构概览)
3. [全局选项](#3-全局选项)
4. [退出码](#4-退出码)
5. [输出格式规范](#5-输出格式规范)
6. [错误信息规范](#6-错误信息规范)
7. [完整命令树](#7-完整命令树)
8. [命令详细设计](#8-命令详细设计)
9. [交互与确认规范](#9-交互与确认规范)
10. [进度指示器](#10-进度指示器)
11. [Shell 补全](#11-shell-补全)
12. [环境变量](#12-环境变量)
13. [配置管理](#13-配置管理)
14. [向后兼容策略](#14-向后兼容策略)
15. [实施路线](#15-实施路线)

---

## 1. 设计原则

基于 16 条 CLI 设计原则，Deep-Dream CLI 遵循以下核心理念：

| # | 原则 | 应用方式 |
|---|------|---------|
| 1 | **任务优先** | 高频操作 `find`、`remember`、`explore` 短路径；低频维护 `db vacuum-embeddings` 显式路径 |
| 2 | **自然语言结构** | 统一 `<资源> <动作>` 模式：`concept search`、`docs list`、`task status` |
| 3 | **安全默认值** | 所有破坏性操作需 `--yes` 确认；支持 `--dry-run` 预览 |
| 4 | **参数命名统一** | `--limit`、`--graph`、`--force`、`--yes` 全局一致 |
| 5 | **有用的 --help** | 每个命令包含示例、默认值标注、参数说明 |
| 6 | **双模式输出** | 默认 Rich 表格/面板；`--json` 机器可读 JSON |
| 7 | **标准流与退出码** | stdout=结果, stderr=提示/进度, 语义化退出码 |
| 8 | **可操作的错误** | `Error:` + `Hint:` 格式，告诉用户下一步怎么做 |
| 9 | **自动化友好** | `--yes`、`--json`、`--quiet`、`--no-color`、环境变量 |
| 10 | **配置透明** | `config show/get/set` 支持 dot-path 访问 |
| 11 | **克制交互** | TTY 检测，管道自动非交互，所有操作可用参数完成 |
| 12 | **进度反馈** | 耗时操作用 Rich spinner + 阶段步骤 |
| 13 | **可撤销/预览** | `--dry-run` 预览，`reset-v15` 自动备份 |
| 14 | **启动性能** | `--help`/`--version` 不加载 pipeline，Click lazy group |
| 15 | **Shell 补全** | `completion bash/zsh/fish`，补全命令、资源名、参数值 |
| 16 | **向后兼容** | 旧命令保留并显示 deprecation warning |

---

## 2. 架构概览

### 2.1 文件结构

```
core/cli/
├── __init__.py          # Re-exports: cli, main
├── _main.py             # Root Click group + global options
├── _ctx.py              # CliContext (config, storage, registry helpers)
├── _output.py           # Rich/JSON dual output, tables, spinner, error_out
├── _exit_codes.py       # Exit code constants
├── _helpers.py          # Shared helpers (resolve_concept, document path, etc.)
├── cmd_version.py       # version
├── cmd_doctor.py        # doctor
├── cmd_config.py        # config show/get/set
├── cmd_server.py        # server start/stop/status/logs
├── cmd_task.py          # task list/status/cancel/pause/resume/retry
├── cmd_graph.py         # graph list/create/use/stats/rebuild
├── cmd_library.py       # library migrate
├── cmd_vault.py         # vault index/tree
├── cmd_remember.py      # remember
├── cmd_find.py          # find
├── cmd_explore.py       # explore
├── cmd_docs.py          # docs list/path/search/grep/map/content/delete
├── cmd_concept.py       # concept search/get/trace/neighbors/versions/mentions/merge/suggest
├── cmd_episode.py       # episode get/concepts/content
├── cmd_relation.py      # relation evidence
├── cmd_sql.py           # sql
└── cmd_db.py            # db init-v15/reset-v15/rebuild-fts/validate/...
```

### 2.2 依赖

```toml
# pyproject.toml
dependencies = [
    # ... existing ...
    "click>=8.1",
    "rich>=13.0",
]
```

### 2.3 入口点

```toml
[project.scripts]
deep-dream = "core.cli:main"
```

---

## 3. 全局选项

所有命令继承这些选项（Click root group 定义）：

| 选项 | 短 | 环境变量 | 默认值 | 说明 |
|------|-----|---------|--------|------|
| `--config` | | `DEEPDREAM_CONFIG` | `service_config.json` | 配置文件路径 |
| `--json` | | `DEEPDREAM_JSON` | off | 机器可读 JSON 输出 |
| `--no-color` | | `NO_COLOR` / `DEEPDREAM_NO_COLOR` | off | 禁用彩色输出 |
| `--quiet` | `-q` | `DEEPDREAM_QUIET` | off | 静默模式 |
| `--version` | | | | 显示版本 |
| `--help` | `-h` | | | 显示帮助 |

Click `auto_envvar_prefix="DEEPDREAM"` 自动为所有选项创建环境变量覆盖。

> 注：`--verbose` / `--dry-run` **不是** root group 选项。需要它们的命令
> 各自声明（例如 `remember`、`server`、`graph`、`db`），避免无意义的
> 继承噪声。

---

## 4. 退出码

```python
# core/cli/_exit_codes.py
OK          = 0   # 成功
ERROR       = 1   # 通用/运行时错误
ARGS        = 2   # 参数错误
AUTH        = 3   # 认证/API key 问题
NETWORK     = 4   # 网络/连接失败
NOT_FOUND   = 5   # 资源不存在
CONFLICT    = 6   # 资源冲突（已存在等）
TIMEOUT     = 7   # 操作超时
PARTIAL     = 8   # 部分成功（如批量操作部分失败）
```

---

## 5. 输出格式规范

### 5.1 模式切换

```python
# core/cli/_output.py
class OutputMode(Enum):
    RICH = "rich"      # 默认：Rich 表格/面板
    JSON = "json"      # --json：纯 JSON 到 stdout
    QUIET = "quiet"    # --quiet：最小输出
```

### 5.2 人类可读格式（默认）

使用 Rich 库：
- **表格数据** → `rich.table.Table`（带列对齐、颜色）
- **单条记录** → `rich.panel.Panel`（带标题、键值对）
- **列表/树** → `rich.tree.Tree`
- **进度** → `rich.progress.Progress`（spinner + 步骤）
- **错误** → `rich.console.Console(stderr=True)` 红色面板

### 5.3 JSON 格式（--json）

所有命令的 JSON 输出遵循统一信封：

```json
{
  "success": true,
  "command": "find",
  "data": { ... },
  "meta": {
    "graph_id": "library",
    "count": 5,
    "elapsed_ms": 234
  }
}
```

错误时：

```json
{
  "success": false,
  "command": "find",
  "error": {
    "code": 5,
    "message": "Concept not found: xyz-123",
    "hint": "Use 'deep-dream concept search \"xyz\"' to find similar concepts."
  }
}
```

### 5.4 通道分配

| 通道 | 内容 | 目的地 |
|------|------|--------|
| stdout | 命令结果（表格/JSON） | 可管道/重定向 |
| stderr | 状态提示、进度、警告、错误 | 不污染 stdout |
| Rich Console | spinner、颜色、面板（stderr） | 仅交互终端 |

---

## 6. 错误信息规范

### 6.1 格式

```
Error: <简洁描述>
  Hint: <修复建议或下一步操作>
```

### 6.2 错误信息目录

| 场景 | Error | Hint | 退出码 |
|------|-------|------|--------|
| 缺少必填参数 | `Error: Provide --file or --text.` | `Hint: Example: deep-dream remember --file notes.md` | ARGS (2) |
| 概念不存在 | `Error: Concept not found: "xyz-123"` | `Hint: Use 'deep-dream concept search "xyz"' to find similar concepts.` | NOT_FOUND (5) |
| 图谱不存在 | `Error: Graph not found: my-proj` | `Hint: Use 'deep-dream graph list' to see available graphs.` | NOT_FOUND (5) |
| 服务器不可达 | `Error: Cannot reach server at http://127.0.0.1:16200` | `Hint: Start with 'deep-dream server start'.` | NETWORK (4) |
| SQL 只读 | `Error: Only SELECT queries are allowed.` | `Hint: Use 'deep-dream db' commands for maintenance operations.` | ARGS (2) |
| 需要认证 | `Error: API key required.` | `Hint: Set DEEPDREAM_API_KEY or use --api-key.` | AUTH (3) |
| 操作超时 | `Error: Operation timed out after 60s.` | `Hint: Use --timeout to increase the limit.` | TIMEOUT (7) |
| 资源已存在 | `Error: Graph "my-proj" already exists.` | `Hint: Use 'deep-dream graph list' to see existing graphs.` | CONFLICT (6) |
| 文件不存在 | `Error: File not found: ./notes.md` | `Hint: Check the file path. Use 'deep-dream docs roots' to see indexed roots.` | NOT_FOUND (5) |
| 编码错误 | `Error: Failed to decode file as utf-8.` | `Hint: Try --encoding latin-1 or --encoding cp1252.` | ERROR (1) |
| 认证失败 | `Error: LLM API returned 401 Unauthorized.` | `Hint: Check your LLM API key in config. Run 'deep-dream config get llm.api_key'.` | AUTH (3) |
| 无图谱数据 | `Error: No graph data. Run 'remember' first.` | `Hint: Example: deep-dream remember --file notes.md` | NOT_FOUND (5) |

---

## 7. 完整命令树

```
deep-dream
├── version                    显示版本信息
├── doctor                     健康检查与诊断
├── config                     配置管理
│   ├── show                   显示完整配置
│   ├── get <key>              获取配置值（dot-path）
│   └── set <key> <value>      设置配置值
├── server                     服务器生命周期
│   ├── start                  启动服务器
│   ├── stop                   停止服务器
│   ├── status                 检查服务器状态
│   └── logs                   查看服务器日志
├── task                       任务队列管理
│   ├── list                   列出任务
│   ├── status <task_id>       查看任务详情
│   ├── cancel <task_id>       取消任务
│   ├── pause <task_id>        暂停任务
│   ├── resume <task_id>       恢复任务
│   └── retry <task_id>        重试失败任务
├── remember                   记忆（摄入文档/文本）
├── find <query>               快速概念搜索（BM25）
├── explore <question>         深度多策略探索
├── graph                      图谱管理（兼容层）
│   ├── list
│   ├── create
│   ├── use
│   ├── stats
│   └── rebuild                ⚠️ 破坏性
├── library                    库级操作
│   └── migrate                迁移旧格式数据
├── vault                      Vault 索引
│   ├── index <path>           索引 Vault
│   └── tree                   显示 Vault 文件树
├── docs                       文档操作
│   ├── roots                  文档根目录
│   ├── list                   文档列表
│   ├── path <id>              文档路径解析
│   ├── search <pattern>       字面搜索
│   ├── grep <pattern>         正则搜索
│   ├── map <path>             路径映射
│   ├── content <id>           读取文档内容
│   └── delete <id>            ⚠️ 删除文档
├── concept                    概念操作
│   ├── search <query>         概念搜索（BM25/语义/混合）
│   ├── get <family_id>        获取概念详情
│   ├── trace <family_id>      溯源追踪
│   ├── neighbors <family_id>  邻居扩展
│   ├── versions <family_id>   版本历史
│   ├── mentions <family_id>   引用该概念的 episodes
│   ├── update <family_id>     手动更新概念
│   ├── suggest <prefix>       名称建议/补全
│   ├── duplicates             检测重复实体
│   └── merge <source> <target> ⚠️ 合并实体
├── episode                    Episode 操作
│   ├── from-file <path>       将文件路径/行映射到 episodes
│   ├── get <id>               获取 episode 详情
│   ├── concepts <id>          Episode 关联概念
│   └── content <id>           读取 episode 内容
├── relation                   关系操作
│   └── evidence <a> <b>       关系证据
├── sql                        SQL 查询（只读）
└── db                         数据库维护
    ├── init-v15               初始化 V1.5 schema
    ├── reset-v15              ⚠️ 备份并重置
    ├── rebuild-fts            重建全文索引
    ├── validate               完整性验证
    ├── rebuild-current        重建 content/current/
    ├── vacuum-embeddings      清理孤立嵌入
    ├── compact                ⚠️ VACUUM 压缩
    ├── quality                数据质量报告
    └── integrity <doc_id>     文档完整性检查
```

---

## 8. 命令详细设计

---

### 8.1 `deep-dream version`

```
Usage: deep-dream version [OPTIONS]

Show version information.

Examples:
  deep-dream version
  deep-dream version --json
```

**人类输出：**
```
Deep-Dream v0.2.0
  Python:     3.11.5
  Storage:    /home/user/deep-dream/library
  Database:   SQLite 3.42.0
  Framework:  Click 8.1 / Rich 13.7
```

**JSON 输出：**
```json
{
  "success": true,
  "command": "version",
  "data": {
    "deep-dream": "0.2.0",
    "python": "3.11.5",
    "storage_path": "/home/user/deep-dream/library"
  },
  "meta": {
    "graph_id": "library"
  }
}
```

版本号以 `pyproject.toml` 为单一来源，`deep-dream --version` 与
`deep-dream version` 在运行时通过 `importlib.metadata.version("deep-dream")`
读取同一字符串。

---

### 8.2 `deep-dream doctor`

```
Usage: deep-dream doctor [OPTIONS]

Inspect configuration, storage health, and API reachability.

Examples:
  deep-dream doctor
  deep-dream doctor --json
  deep-dream doctor --api-base http://localhost:5001/api/v1

Options:
  --api-base TEXT  API base URL  [default: http://127.0.0.1:16200/api/v1]
```

**人类输出：**
```
✓ Deep-Dream Doctor

  Storage:     /home/user/deep-dream/library (exists, 2.4 GB)
  Config:      service_config.json (loaded)
  LLM:         connected (gpt-4o-mini via openai)
  Embeddings:  available (text-embedding-3-small, dim=1024)
  API Server:  online at http://127.0.0.1:16200

           Graphs
┌──────────┬───────────┬──────────┬─────────┬───────┐
│ ID       │ Documents │ Concepts │ Episodes │ Edges │
├──────────┼───────────┼──────────┼─────────┼───────┤
│ library  │       142 │      891 │    2,341 │   567 │
└──────────┴───────────┴──────────┴─────────┴───────┘

All checks passed. ✅
```

**有错误时：**
```
✗ Deep-Dream Doctor

  Storage:    /home/user/deep-dream/library (exists)
  Config:     service_config.json (loaded)
  LLM:        ✗ connection refused (api.openai.com:443)
  Embeddings: ✗ unavailable (no embedding client)
  API Server: ✗ offline

  Error: LLM and API server are unreachable.
    Hint: Check your network and API key. Run 'deep-dream config get llm.api_key'.
```

---

### 8.3 `deep-dream config show/get/set`

```
Usage: deep-dream config [COMMAND]...

View and manage configuration.

Examples:
  deep-dream config show
  deep-dream config get llm.model
  deep-dream config get pipeline --verbose
  deep-dream config set llm.model gpt-4o

Commands:
  show            Display resolved configuration
  get <key>       Get a value by dot-path (e.g. llm.model)
  set <key> <val> Set a value (with confirmation)

Options:
  --secrets   Show/redact API keys (default: redacted)
```

**`config show` 人类输出：**
```
Configuration (service_config.json)

  storage_path:        ./library
  server.host:         0.0.0.0
  server.port:         16200
  llm.model:           gpt-4o-mini
  llm.api_key:         sk-****redacted****
  llm.provider:        openai
  embedding.model:     text-embedding-3-small
  embedding.dimension: 1024
  pipeline.chunk_size: 4000
  pipeline.overlap:    400

  Use --secrets to reveal API keys.
  Use 'deep-dream config get <key>' for a specific value.
```

**`config set` 确认流程：**
```
$ deep-dream config set llm.model gpt-4o

  This will update llm.model:
    "gpt-4o-mini" → "gpt-4o"

  Apply change? [y/N]: y
  ✓ Updated llm.model = "gpt-4o"

  Hint: Restart the server for changes to take effect: deep-dream server restart
```

**`config set --yes`（无交互）：**
```
$ deep-dream config set llm.model gpt-4o --yes
  ✓ Updated llm.model = "gpt-4o"
```

---

### 8.4 `deep-dream server start/stop/status/logs`

```
Usage: deep-dream server [COMMAND]...

Manage the Deep-Dream Flask server.

Examples:
  deep-dream server start
  deep-dream server start --port 5001 --detach
  deep-dream server start --verbose
  deep-dream server status
  deep-dream server stop
  deep-dream server logs --lines 50

Commands:
  start   Start the server
  stop    Stop a running server
  status  Check server status
  logs    View recent server logs
```

**`server start`：**
```
Usage: deep-dream server start [OPTIONS]

Start the Deep-Dream Flask server.

Options:
  --host TEXT     Bind host  [default: 0.0.0.0]
  --port INT     Bind port  [default: 16200]
  --detach       Run in background (daemon mode)
  --verbose      Show startup details
```

**人类输出：**
```
⠋ Starting Deep-Dream server...
  ✓ Storage initialized: ./library
  ✓ LLM client ready: gpt-4o-mini
  ✓ Embedding client ready: text-embedding-3-small

  Deep-Dream server running at http://0.0.0.0:16200
  Press Ctrl+C to stop.
```

**`server start --detach`：**
```
⠋ Starting Deep-Dream server in background...
  ✓ Server started (PID 12345)
  ✓ Listening on http://0.0.0.0:16200

  Use 'deep-dream server stop' to shut down.
  Use 'deep-dream server logs' to view output.
```

**`server stop`：**
```
$ deep-dream server stop

  Stop Deep-Dream server (PID 12345)? [y/N]: y
  ✓ Server stopped.

  Or use --yes to skip confirmation.
```

**`server status`：**
```
  Deep-Dream Server

  Status:    ● online
  PID:       12345
  Host:      0.0.0.0:16200
  Uptime:    2h 34m
  Requests:  1,247 today
  Tasks:     0 running, 3 queued
```

---

### 8.5 `deep-dream task list/status/cancel/pause/resume/retry`

```
Usage: deep-dream task [COMMAND]...

Manage the async remember task queue (requires running server).

Examples:
  deep-dream task list
  deep-dream task status abc-123
  deep-dream task cancel abc-123 --yes
  deep-dream task retry abc-123

Commands:
  list              List all tasks
  status <id>       Show task details
  cancel <id>       Cancel/delete a task
  pause <id>        Pause a running task
  resume <id>       Resume a paused task
  retry <id>        Retry failed/missing windows
  resume-all        Resume all paused tasks
```

**`task list` 人类输出：**
```
             Tasks (3 total)
┌──────────┬──────────────┬─────────┬──────────┬─────────────────────┐
│ ID       │ Source       │ Status  │ Progress │ Created             │
├──────────┼──────────────┼─────────┼──────────┼─────────────────────┤
│ abc-123  │ notes.md     │ done    │ 100%     │ 2026-06-01 14:30:22 │
│ def-456  │ research.pdf │ running │  67%     │ 2026-06-01 15:01:10 │
│ ghi-789  │ diary.md     │ failed  │  45%     │ 2026-06-01 15:10:03 │
└──────────┴──────────────┴─────────┴──────────┴─────────────────────┘

  Use 'deep-dream task status <id>' for details.
```

**`task status abc-123` 人类输出：**
```
  Task abc-123

  Source:    notes.md
  Status:    ● running
  Progress:  67% (8/12 windows)
  Created:   2026-06-01 15:01:10
  Started:   2026-06-01 15:01:12

  Pipeline Steps:
    ✓ Step 1: Cache update
    ✓ Step 2: Entity extraction
    ✓ Step 3: Entity dedup
    ✓ Step 4: Content writing
    ⠋ Step 5: Quality gate (3/5 entities)
    ○ Step 6: Relation discovery
    ○ Step 7: Relation content
    ○ Step 8: Relation quality gate
    ○ Step 9: Entity alignment
    ○ Step 10: Relation alignment

  Elapsed: 45s
  Use 'deep-dream task pause abc-123' to pause.
```

---

### 8.6 `deep-dream remember`

```
Usage: deep-dream remember [OPTIONS]

Ingest text or a file into the concept graph.
Provide exactly one of --file or --text.

Examples:
  deep-dream remember --file notes.md
  deep-dream remember --text "Key insight about quantum computing"
  deep-dream remember --file doc.md --source "research-paper" -v
  deep-dream remember --file large.md --start-chunk 5

Options:
  -f, --file PATH       File to remember
  -t, --text TEXT       Inline text to remember
  -s, --source TEXT     Source label  [default: filename or "cli:text"]
      --encoding TEXT   File encoding  [default: utf-8]
      --graph TEXT      Graph ID  [default: library]
      --start-chunk INT Start from this chunk number
  -v, --verbose         Show processing details
```

**人类输出（正常）：**
```
⠋ Remembering notes.md...

  ✓ Parsed 3 episodes (1,247 chars)
  ✓ Extracted 12 entities, 5 relations
  ✓ Aligned with 3 existing concepts
  ✓ Saved to graph "library"

  Summary:
    Episodes:    3
    Entities:    12 new, 3 aligned
    Relations:   5 new, 1 aligned

  Completed in 8.3s
```

**人类输出（--verbose）：**
```
⠋ Remembering notes.md...

  [1/3] Episode 1 (chars 1-412)
    ✓ Extracted 5 entities: "attention mechanism", "transformer", ...
    ✓ Extracted 2 relations: "attention mechanism" → uses → "softmax"
    ✓ Quality gate: all entities passed
  [2/3] Episode 2 (chars 413-824)
    ✓ Extracted 4 entities: "RAG", "retrieval", ...
    ✓ 1 entity aligned with existing: "RAG" ← "retrieval-augmented generation"
    ✓ Cross-window alignment: merged with window 1 entity "RAG"
  [3/3] Episode 3 (chars 825-1247)
    ✓ Extracted 3 entities: "embedding", "vector store", ...
    ✓ 2 relations aligned with window 1

  Summary:
    Episodes:    3
    Entities:    12 new, 3 aligned
    Relations:   5 new, 1 aligned

  Completed in 8.3s
```

**JSON 输出：**
```json
{
  "success": true,
  "command": "remember",
  "data": {
    "source": "notes.md",
    "graph_id": "library",
    "episodes": 3,
    "entities": { "new": 12, "aligned": 3 },
    "relations": { "new": 5, "aligned": 1 },
    "elapsed_ms": 8300
  }
}
```

---

### 8.7 `deep-dream find`

```
Usage: deep-dream find [OPTIONS] QUERY

Quick concept search using BM25 full-text search.
For semantic search, use 'deep-dream concept search --semantic'.

Examples:
  deep-dream find "machine learning"
  deep-dream find "transformer" --role entity --limit 10
  deep-dream find "attention" --time-point 2026-05-01T00:00:00

Options:
  --role TEXT         Filter: entity|relation|document|episode
  --limit INT         Max results  [default: 20]
  --time-point TEXT   Temporal snapshot (ISO timestamp)
  --graph TEXT        Graph ID  [default: library]
```

**人类输出：**
```
  Results for "machine learning" (5 found)

┌──────────────────────┬──────────┬───────────┬────────────────────────────┐
│ Concept              │ Role     │ Conf.     │ Summary                    │
├──────────────────────┼──────────┼───────────┼────────────────────────────┤
│ machine learning     │ entity   │ 0.95      │ Statistical methods for... │
│ deep learning        │ entity   │ 0.92      │ Subset of ML using neural… │
│ ML pipeline          │ entity   │ 0.88      │ End-to-end ML workflow...  │
│ learning rate        │ entity   │ 0.85      │ Hyperparameter control...  │
│ ML → statistics      │ relation │ 0.82      │ ML builds on statistics... │
└──────────────────────┴──────────┴───────────┴────────────────────────────┘

  Use 'deep-dream concept get <family_id>' for details.
  Use 'deep-dream concept search "..." --semantic' for semantic search.
```

---

### 8.8 `deep-dream explore`

```
Usage: deep-dream explore [OPTIONS] QUESTION

Multi-strategy deep exploration combining document search,
concept search, graph traversal, and relation evidence.

Examples:
  deep-dream explore "how does attention mechanism work"
  deep-dream explore "RAG pipeline design" --limit 30
  deep-dream explore "causal inference" --terms "do-calculus,counterfactual"
  deep-dream explore "graph neural network" --depth 3

Options:
  --limit INT           Max results per strategy  [default: 20]
  --depth INT           Graph traversal depth  [default: 2]
  --terms TEXT          Additional search terms (comma-separated)
  --graph TEXT          Graph ID  [default: library]
  --no-documents        Skip document file search
  --no-semantic         Skip semantic concept search
  --no-neighbors        Skip graph neighbor expansion
  --no-relations        Skip relation evidence
```

**人类输出：**
```
⠋ Exploring "how does attention mechanism work"...

  Strategy 1: Document Search (2 results)
  ────────────────────────────────────────
    📄 attention-is-all-you-need.md (score: 0.94)
       "We propose a new network architecture, the Transformer..."
    📄 transformer-deep-dive.md (score: 0.87)
       "Multi-head attention allows the model to jointly attend..."

  Strategy 2: Concept Search (6 results)
  ────────────────────────────────────────
    🔵 attention mechanism   (entity, conf: 0.95)
       Core component of Transformer architecture...
    🔵 multi-head attention  (entity, conf: 0.93)
       Parallel attention computations with different projections...
    🔵 self-attention        (entity, conf: 0.91)
       Attention within a single sequence...
    🔵 scaled dot-product    (entity, conf: 0.88)
       Attention score = softmax(QK^T / sqrt(d_k)) V...
    🔵 query/key/value       (entity, conf: 0.85)
       Three projections used in attention computation...
    🔵 softmax               (entity, conf: 0.82)
       Normalization function used in attention weights...

  Strategy 3: Graph Neighbors (depth=2)
  ──────────────────────────────────────
    attention mechanism
    ├── uses → scaled dot-product
    ├── variant → multi-head attention
    │   └── used-in → Transformer
    ├── variant → self-attention
    │   └── used-in → BERT, GPT
    └── component → query/key/value
        └── output → softmax

  Strategy 4: Relation Evidence (3 pairs)
  ────────────────────────────────────────
    attention mechanism ──uses──→ scaled dot-product
      Source: attention-is-all-you-need.md (episode 3)
      "We compute attention using scaled dot-product..."
    attention mechanism ──variant──→ multi-head attention
      Source: transformer-deep-dive.md (episode 5)

  Total: 8 concepts, 3 relation pairs, 2 documents found in 2.1s
```

---

### 8.9 `deep-dream docs`

```
Usage: deep-dream docs [COMMAND]...

Document discovery, search, and content access.

Examples:
  deep-dream docs list --limit 20
  deep-dream docs search "attention mechanism"
  deep-dream docs content doc-abc123

Commands:
  roots               List searchable document root directories
  list                List indexed documents
  path <id>           Resolve document ID to file path
  search <pattern>    Literal text search over documents
  grep <pattern>      Regex search over documents
  map <path>          Map file path to document records
  content <id>        Read document content
  delete <id>         Delete a document version  ⚠️ DESTRUCTIVE
```

**`docs list` 人类输出：**
```
           Documents (142 total, showing 20)
┌──────────┬───────────────────────────────┬─────────┬─────────────────────┐
│ ID       │ Path                          │ Windows │ Indexed             │
├──────────┼───────────────────────────────┼─────────┼─────────────────────┤
│ doc-a1b2 │ notes/ml-paper.md             │      5  │ 2026-05-30 14:22:10 │
│ doc-c3d4 │ vault/daily/2026-05-29.md     │      3  │ 2026-05-29 09:15:33 │
│ doc-e5f6 │ research/attention-is-all.md  │      8  │ 2026-05-28 11:45:02 │
│ ...      │ ...                           │    ...  │ ...                 │
└──────────┴───────────────────────────────┴─────────┴─────────────────────┘

  Use 'deep-dream docs content <id>' to read document content.
  Use 'deep-dream docs list --limit 0' for all.
```

**`docs content doc-a1b2` 人类输出：**
```
  Document: doc-a1b2 (notes/ml-paper.md)
  Indexed:  2026-05-30 14:22:10
  Windows:  5

────────────────────────────────────────────────────────

# Machine Learning Paper Notes

We propose a new network architecture, the Transformer,
based solely on attention mechanisms...

[truncated at 80 lines. Use --full or --lines 200 for more]
```

**`docs delete` 确认流程：**
```
$ deep-dream docs delete doc-a1b2

  ⚠ This will permanently delete:
    Document: doc-a1b2 (notes/ml-paper.md)
    Windows:  5
    This also removes associated episodes and concepts.

  Type the document ID to confirm: doc-a1b2
  ✓ Deleted document doc-a1b2 and all associated data.
```

---

### 8.10 `deep-dream concept`

```
Usage: deep-dream concept [COMMAND]...

Concept search, inspection, and management.

Examples:
  deep-dream concept search "attention"
  deep-dream concept search "transformer" --semantic --limit 5
  deep-dream concept get entity-abc-123
  deep-dream concept versions entity-abc-123
  deep-dream concept merge entity-old entity-new

Commands:
  search <query>         Search concepts (BM25, semantic, or hybrid)
  get <family_id>        Get concept details
  trace <family_id>      Trace concept provenance across versions
  neighbors <family_id>  Expand graph neighbors
  versions <family_id>   List all versions of a concept
  mentions <family_id>   Episodes mentioning this concept
  update <family_id>     Manually update a concept  ⚠️ modifies data
  suggest <prefix>       Name suggestions (autocomplete)
  duplicates             Detect potential duplicate entities
  merge <src> <target>   Merge source into target  ⚠️ DESTRUCTIVE
```

**`concept search`：**
```
Usage: deep-dream concept search [OPTIONS] QUERY

Search concepts by keyword (BM25), semantic similarity, or hybrid.

Examples:
  deep-dream concept search "attention"
  deep-dream concept search "transformer" --semantic
  deep-dream concept search "RAG pipeline" --mode hybrid --limit 5
  deep-dream concept search "learning" --role entity

Options:
  --mode TEXT         Search mode: bm25|semantic|hybrid  [default: bm25]
  --role TEXT         Filter: entity|relation
  --limit INT         Max results  [default: 20]
  --threshold FLOAT   Similarity threshold for semantic  [default: 0.6]
  --graph TEXT        Graph ID  [default: library]
```

**`concept search` 人类输出：**
```
  Concept Search: "attention" (bm25, 8 results)

┌──────────────────┬────────┬───────────┬────────────────────────────────────┐
│ Concept          │ Role   │ Conf.     │ Content Summary                    │
├──────────────────┼────────┼───────────┼────────────────────────────────────┤
│ attention mec…   │ entity │     0.95  │ Core mechanism for weighing input… │
│ self-attention   │ entity │     0.93  │ Attention within a single sequen…  │
│ multi-head att…  │ entity │     0.91  │ Parallel attention heads with di…  │
│ attention → QKV  │ relation│    0.88  │ Attention operates on query/key/…  │
│ ...              │        │           │                                    │
└──────────────────┴────────┴───────────┴────────────────────────────────────┘

  Use 'deep-dream concept get <family_id>' for details.
  Use --semantic for similarity search, --mode hybrid for combined.
```

**`concept get entity-abc-123` 人类输出：**
```
┌─────────────────────────────────────────────────────────┐
│ Concept: attention mechanism                             │
├─────────────────────────────────────────────────────────┤
│ Family ID:   entity-abc-123                              │
│ Absolute ID: entity-abc-123-v3                           │
│ Role:        entity                                      │
│ Confidence:  0.95                                        │
│ Versions:    3 (latest: v3, 2026-05-30 14:22:10)        │
│                                                          │
│ Content:                                                 │
│   Core mechanism for weighting input signals in neural   │
│   networks. Computes relevance scores between query and  │
│   key vectors, producing weighted value outputs.         │
│                                                          │
│ Relations (4):                                           │
│   ├── uses → scaled dot-product (conf: 0.92)            │
│   ├── variant → multi-head attention (conf: 0.90)       │
│   ├── variant → self-attention (conf: 0.88)             │
│   └── component → query/key/value (conf: 0.85)          │
│                                                          │
│ Sources:                                                 │
│   • attention-is-all-you-need.md (episode 3)            │
│   • transformer-deep-dive.md (episode 5)                │
└─────────────────────────────────────────────────────────┘

  Use 'deep-dream concept versions entity-abc-123' for history.
  Use 'deep-dream concept neighbors entity-abc-123' for graph expansion.
```

**`concept versions` 人类输出：**
```
  Concept: attention mechanism (entity-abc-123) — 3 versions

┌─────┬─────────────────────┬───────────┬──────────────────────────────────┐
│ Ver │ Timestamp           │ Conf.     │ Change Summary                   │
├─────┼─────────────────────┼───────────┼──────────────────────────────────┤
│ v1  │ 2026-05-28 10:00:00 │     0.80  │ Initial extraction               │
│ v2  │ 2026-05-29 14:30:00 │     0.88  │ Content enriched; merged with... │
│ v3  │ 2026-05-30 14:22:10 │     0.95  │ Corroborated by 2nd source       │
└─────┴─────────────────────┴───────────┴──────────────────────────────────┘

  Use 'deep-dream concept trace entity-abc-123' for full provenance.
```

**`concept merge` 确认流程：**
```
$ deep-dream concept merge entity-old entity-new

  ⚠ This will merge two concepts:
    Source:  entity-old ("attention mecanism" — note typo)
    Target:  entity-new ("attention mechanism")

  The source will be deleted. All its relations and mentions
  will be redirected to the target. This cannot be undone.

  Proceed? [y/N]: y
  ✓ Merged "attention mecanism" → "attention mechanism"
    Redirected 4 relations and 2 episode mentions.
```

**`concept duplicates` 人类输出：**
```
  Duplicate Detection (by core-name matching)

┌───────────────────────────────┬───────────────────────────────┬────────┐
│ Concept A                     │ Concept B                     │ Action │
├───────────────────────────────┼───────────────────────────────┼────────┤
│ attention mecanism            │ attention mechanism           │ merge  │
│ RAG                           │ retrieval-augmented gen…      │ merge  │
│ neural net                    │ neural network                │ merge  │
└───────────────────────────────┴───────────────────────────────┴────────┘

  3 potential duplicates found.
  Use 'deep-dream concept merge <source> <target>' to merge.
```

---

### 8.11 `deep-dream episode`

```
Usage: deep-dream episode [COMMAND]...

Episode inspection and content access.

Examples:
  deep-dream episode from-file notes/ml-paper.md --line 12
  deep-dream episode get ep-abc-123
  deep-dream episode concepts ep-abc-123
  deep-dream episode content ep-abc-123

Commands:
  from-file <path>  Map a file path (and optional line) to episodes
  get <id>          Get episode details
  concepts <id>     List concepts mentioned in an episode
  content <id>      Read episode source content
```

**`episode from-file`：**

将一个文件路径（及可选行号）解析到它所覆盖的 episodes。
按原始文件位置（`line_start`/`line_end` 与字符 offset）匹配，因此对
外部文件（`source_mode: external`，未快照进库）同样有效——这是 skill
用于「定位到 episode」的核心入口。

```
Usage: deep-dream episode from-file [OPTIONS] PATH

Map a file path (and optional line) to episodes.

Examples:
  deep-dream episode from-file notes/ml-paper.md
  deep-dream episode from-file notes/ml-paper.md --line 12
  deep-dream episode from-file vault/daily/2026-05-30.md --limit 20

Options:
  --line INT     Filter to episodes overlapping this line number.
  --limit INT    Maximum episodes to return  [default: 50]
  --graph TEXT   Graph ID  [default: library]
```

**`episode from-file notes/ml-paper.md --line 12` 人类输出：**
```
  Episodes from notes/ml-paper.md (2)

┌──────────────┬──────────────────────┬──────────┬──────────────────┐
│ Episode ID   │ Heading              │ Lines    │ Offset           │
├──────────────┼──────────────────────┼──────────┼──────────────────┤
│ ep-abc-123   │ ## Attention          │ 8-15     │ 412-824          │
│ ep-def-456   │ ### Scaled Dot-Product│ 12-18    │ 640-1024         │
└──────────────┴──────────────────────┴──────────┴──────────────────┘
```

**`episode from-file` JSON 输出：**
```json
{
  "success": true,
  "command": "episode from-file",
  "data": {
    "documents": [
      {
        "document_version_id": "doc-a1b2",
        "source_path": "notes/ml-paper.md",
        "source_mode": "external",
        "resolved_path": "/abs/notes/ml-paper.md",
        "line_start": 1,
        "line_end": 80
      }
    ],
    "episodes": [
      {
        "episode_version_id": "ep-abc-123",
        "document_version_id": "doc-a1b2",
        "heading_path": "## Attention",
        "start_offset": 412,
        "end_offset": 824,
        "line_start": 8,
        "line_end": 15,
        "source_path": "notes/ml-paper.md",
        "source_text": "..."
      }
    ],
    "total": 1
  },
  "meta": {
    "graph_id": "library",
    "count": 1
  }
}
```

**`episode get ep-abc-123` 人类输出：**
```
  Episode ep-abc-123

  Document:  doc-a1b2 (notes/ml-paper.md)
  Window:    3 of 5
  Position:  chars 825-1247
  Created:   2026-05-30 14:22:10

  Concepts (4):
    • attention mechanism (entity, conf: 0.95)
    • multi-head attention (entity, conf: 0.91)
    • attention → QKV (relation, conf: 0.88)
    • scaled dot-product (entity, conf: 0.85)

  Use 'deep-dream episode content ep-abc-123' for source text.
```

---

### 8.12 `deep-dream relation`

```
Usage: deep-dream relation [COMMAND]...

Relation inspection and evidence.

Examples:
  deep-dream relation evidence entity-a entity-b
  deep-dream relation evidence "attention" "softmax" --limit 5

Commands:
  evidence <a> <b>   Find evidence linking two concepts

Options:
  --limit INT   Max evidence items  [default: 10]
  --graph TEXT  Graph ID  [default: library]
```

**人类输出：**
```
  Evidence: attention mechanism ↔ softmax (3 items)

┌─────────────────────┬──────────┬──────────────────────────────────────┐
│ Source              │ Conf.    │ Evidence Text                        │
├─────────────────────┼──────────┼──────────────────────────────────────┤
│ ml-paper.md (ep 3)  │     0.92 │ "Attention weights are computed by…  │
│ transformer-dive.md │     0.88 │ "The softmax function normalizes...  │
│ neural-nets.md      │     0.85 │ "Scaled dot-product attention uses…" │
└─────────────────────┴──────────┴──────────────────────────────────────┘
```

---

### 8.13 `deep-dream sql`

```
Usage: deep-dream sql [OPTIONS] QUERY

Execute a read-only SQL query against the graph database.

Examples:
  deep-dream sql "SELECT * FROM entities LIMIT 5"
  deep-dream sql "SELECT count(*) FROM episodes" --explain

Options:
  --explain     Show query execution plan
  --limit INT   Max rows  [default: 100]
  --graph TEXT  Graph ID  [default: library]
```

**人类输出：**
```
  SQL Results (3 rows)

┌──────────┬──────────────────────┬──────────┬───────────┐
│ family_id│ name                 │ role     │ confidence│
├──────────┼──────────────────────┼──────────┼───────────┤
│ ent-001  │ attention mechanism  │ entity   │      0.95 │
│ ent-002  │ transformer          │ entity   │      0.93 │
│ ent-003  │ multi-head attention │ entity   │      0.91 │
└──────────┴──────────────────────┴───────────┴──────────┘
```

---

### 8.14 `deep-dream graph`

```
Usage: deep-dream graph [COMMAND]...

Graph management commands.
Note: Deep-Dream uses single-library mode. All operations
target the default "library" graph.

Examples:
  deep-dream graph list
  deep-dream graph stats
  deep-dream graph rebuild --dry-run

Commands:
  list              List all graphs
  create            Create a new graph (single-library mode)
  use               Set active graph
  stats             Show graph statistics
  rebuild           ⚠️ Clear graph data for re-indexing
```

**`graph stats` 人类输出：**
```
  Graph: library

┌───────────────────┬──────────┐
│ Metric            │ Count    │
├───────────────────┼──────────┤
│ Documents         │      142 │
│ Episodes          │    2,341 │
│ Unique Entities   │      891 │
│ Unique Relations  │      567 │
│ Embeddings        │    1,458 │
│ Orphan Entities   │       12 │
│ Storage Size      │   2.4 GB │
└───────────────────┴──────────┘
```

**`graph rebuild` 确认流程：**
```
$ deep-dream graph rebuild

  ⚠ This will DELETE all concept and relation data from graph "library".
  Documents and raw files will be preserved.
  You will need to run 'remember' again to rebuild the graph.

  Proceed? [y/N]: n
  Aborted.

  Use --dry-run to preview what would be affected.
  Use --yes to skip this confirmation.
```

---

### 8.15 `deep-dream vault`

```
Usage: deep-dream vault [COMMAND]...

Markdown/Obsidian vault indexing.

Examples:
  deep-dream vault index ./my-vault
  deep-dream vault index ./vault --force
  deep-dream vault tree

Commands:
  index <path>    Index a vault directory
  tree            Show indexed vault file tree

Options (index):
  --force       Re-index even if already indexed
  --graph TEXT  Graph ID  [default: library]
```

**`vault index` 人类输出：**
```
⠋ Indexing vault: ./my-vault
  ✓ Scanned 47 files (42 .md, 5 attachments)
  ✓ Indexed 39 new files, 3 updated, 5 unchanged
  ✓ Created 127 episodes, 89 entities, 34 relations

  Completed in 12.4s
```

**`vault tree` 人类输出：**
```
  Vault Tree (39 indexed documents)

📁 my-vault/
├── 📁 daily-notes/
│   ├── 📄 2026-05-28.md (3 windows)
│   ├── 📄 2026-05-29.md (2 windows)
│   └── 📄 2026-05-30.md (4 windows)
├── 📁 projects/
│   ├── 📄 ml-research.md (8 windows)
│   └── 📄 cli-design.md (5 windows)
└── 📄 README.md (1 window)

  39 documents, 127 windows total
```

---

### 8.16 `deep-dream db`

```
Usage: deep-dream db [COMMAND]...

Database maintenance, schema management, and integrity tools.

Examples:
  deep-dream db validate
  deep-dream db validate --repair
  deep-dream db rebuild-fts
  deep-dream db vacuum-embeddings --dry-run

Commands:
  init-v15              Initialize V1.5 schema
  reset-v15             ⚠️ Backup old DB and create fresh V1.5
  rebuild-fts           Rebuild full-text search index
  validate              Run integrity validation
  rebuild-current       Rebuild content/current/ files
  vacuum-embeddings     Clean orphaned embeddings
  compact               ⚠️ VACUUM to reclaim disk space
  quality               Data quality report
  integrity <doc_id>    Check/repair document integrity

Shared Options:
  --dry-run             Preview without making changes
  --yes                 Skip confirmation prompts
  --graph TEXT          Graph ID  [default: library]
```

**`db validate` 人类输出：**
```
⠋ Running integrity validation...

  ✓ Entity embeddings: 891/891 present
  ✓ Relation embeddings: 567/567 present
  ✓ Episode windows: all valid
  ✓ Document references: all valid
  ✓ Content files: 142/142 current

  All checks passed. ✅
```

**`db validate --repair` 人类输出：**
```
⠋ Running integrity validation (with repair)...

  ✓ Entity embeddings: 891/891 present
  ⚠ Relation embeddings: 3 missing
    → Repairing: recomputing 3 embeddings... ✓
  ✓ Episode windows: all valid
  ⚠ Content files: 2 stale
    → Repairing: rebuilding 2 current files... ✓

  Repaired: 3 embeddings, 2 content files
  All checks now pass. ✅
```

**`db quality` 人类输出：**
```
  Data Quality Report (graph: library)

┌────────────────────────────┬──────────┬──────────┐
│ Metric                     │ Value    │ Status   │
├────────────────────────────┼──────────┼──────────┤
│ Total entities             │      891 │          │
│ Entities with embeddings   │      891 │ ✓ 100%   │
│ Orphan entities (no edges) │       12 │ ⚠  1.3%  │
│ Duplicate suspects         │        3 │ ⚠        │
│ Avg confidence             │     0.88 │          │
│ Low confidence (<0.5)      │       15 │ ⚠  1.7%  │
│ Total relations            │      567 │          │
│ Relations with evidence    │      567 │ ✓ 100%   │
│ Avg relation confidence    │     0.84 │          │
│ Content coverage           │   98.6%  │ ✓        │
└────────────────────────────┴──────────┴──────────┘

  Use 'deep-dream concept duplicates' to review suspected duplicates.
  Use 'deep-dream db vacuum-embeddings' to clean orphans.
```

**`db reset-v15` 确认流程：**
```
$ deep-dream db reset-v15

  ⚠ This will:
    1. Backup current database to graph.db.bak.<timestamp>
    2. DELETE all data in the database
    3. Create a fresh V1.5 schema

  This cannot be undone (except by manually restoring the backup).

  Type "reset" to confirm: reset
  ✓ Backed up to graph.db.bak.20260601-151022
  ✓ Created fresh V1.5 database
  ✓ Schema initialized

  Use 'deep-dream remember --file ...' to rebuild the graph.
```

---

### 8.17 `deep-dream library`

```
Usage: deep-dream library [COMMAND]...

Library-level operations.

Commands:
  migrate              Migrate legacy multi-graph data to single-library layout

Options:
  --dry-run            Preview without making changes
  --backup             Create backup before migration  [default: true]
  --yes                Skip confirmation
```

---

### 8.18 `deep-dream completion`

```
Usage: deep-dream completion [OPTIONS] SHELL

Generate shell completion scripts.

Supported shells: bash, zsh, fish

Examples:
  # Bash
  echo 'eval "$(deep-dream completion bash)"' >> ~/.bashrc

  # Zsh
  deep-dream completion zsh > ~/.zfunc/_deep_dream

  # Fish
  deep-dream completion fish > ~/.config/fish/completions/deep-dream.fish
```

---

## 9. 交互与确认规范

### 9.1 确认分级

| 级别 | 操作 | 确认方式 | 示例 |
|------|------|---------|------|
| **L0 无需确认** | 读取、搜索、查询 | 无 | `find`, `concept search`, `sql` |
| **L1 简单确认** | 修改配置、停止服务 | `[y/N]` | `config set`, `server stop` |
| **L2 输入确认** | 删除数据、重置 | 要求输入标识符 | `docs delete <id>`, `db reset-v15` |
| **L3 严格确认** | 不可逆批量操作 | 输入特定词 + `--yes` | `graph rebuild`, `concept merge` |

### 9.2 TTY 检测

```python
import sys

def is_interactive():
    return sys.stdin.isatty() and sys.stdout.isatty()

# 非交互（管道/脚本/CI）时：
# - 自动使用 --yes 行为（跳过确认）
# - 禁用 spinner/颜色
# - 输出纯文本/JSON
```

### 9.3 --yes 行为

```python
# 所有破坏性操作的通用模式
@click.command()
@click.option('--yes', is_flag=True, help='Skip confirmation prompts')
def dangerous_op(yes):
    if not yes:
        if is_interactive():
            click.confirm('Proceed?', abort=True)
        else:
            click.echo('Error: --yes required in non-interactive mode.', err=True)
            raise SystemExit(ARGS)
```

---

## 10. 进度指示器

### 10.1 使用 Rich Progress

以下命令显示进度：

| 命令 | 进度类型 | 显示内容 |
|------|---------|---------|
| `remember` | Spinner + 步骤 | `[2/3] Episode 2: extracting entities...` |
| `vault index` | 进度条 | `Scanning vault: ████████░░ 80% (32/40 files)` |
| `explore` | Spinner | `Searching concepts...` → `Expanding neighbors...` |
| `db validate` | Spinner + 计数 | `Checking embeddings: 456/891...` |
| `db rebuild-fts` | 进度条 | `Rebuilding FTS: ████████░░ 80%` |
| `db compact` | Spinner | `VACUUMing database...` |

### 10.2 管道检测

当输出被重定向时（`> file` 或 `| pipe`），自动：
- 禁用 spinner（Rich 自动处理）
- 输出纯文本进度到 stderr
- 结果仍输出到 stdout

---

## 11. Shell 补全

### 11.1 命令补全

Click 8+ 自动生成补全脚本。通过 `deep-dream completion <shell>` 安装。

### 11.2 动态补全

为以下命令提供动态补全：

| 命令 | 补全内容 |
|------|---------|
| `concept get/trace/neighbors/versions/...` | 从数据库获取 family_id 列表 |
| `docs content/path/delete` | 从数据库获取 document ID 列表 |
| `episode get/concepts/content` | 从数据库获取 episode ID 列表 |
| `task status/cancel/pause/resume/retry` | 从 API 获取 task ID 列表 |
| `concept search --mode` | `bm25` / `semantic` / `hybrid` |
| `find --role` | `entity` / `relation` / `document` / `episode` |
| `completion` | `bash` / `zsh` / `fish` |

### 11.3 补全实现

```python
# Click 8.1+ 动态补全
@click.command()
@click.argument('family_id', shell_complete=get_concept_ids)
def concept_get(family_id):
    ...

def get_concept_ids(ctx, param, incomplete):
    """从数据库获取概念 ID 补全列表"""
    storage = get_storage(ctx)
    concepts = storage.list_concepts(limit=50)
    return [c['family_id'] for c in concepts if c['family_id'].startswith(incomplete)]
```

---

## 12. 环境变量

Click `auto_envvar_prefix="DEEPDREAM"` 自动映射所有选项：

| 环境变量 | 对应选项 | 说明 |
|---------|---------|------|
| `DEEPDREAM_CONFIG` | `--config` | 配置文件路径 |
| `DEEPDREAM_JSON` | `--json` | JSON 输出 |
| `DEEPDREAM_QUIET` | `--quiet` | 静默模式 |
| `DEEPDREAM_VERBOSE` | `--verbose` | 详细输出 |
| `DEEPDREAM_NO_COLOR` | `--no-color` | 无颜色 |
| `NO_COLOR` | `--no-color` | 行业标准无颜色 |
| `DEEPDREAM_API_KEY` | N/A | API 密钥 |
| `DEEPDREAM_LLM_MODEL` | N/A | LLM 模型覆盖 |

**优先级：命令行参数 > 环境变量 > 配置文件 > 默认值**

---

## 13. 配置管理

### 13.1 配置文件位置

```
./service_config.json          # 项目级（优先）
~/.deep-dream/config.json      # 用户级（备选）
```

### 13.2 配置结构

```json
{
  "storage_path": "./library",
  "server": {
    "host": "0.0.0.0",
    "port": 16200
  },
  "llm": {
    "model": "gpt-4o-mini",
    "api_key": "sk-...",
    "provider": "openai",
    "base_url": null
  },
  "embedding": {
    "model": "text-embedding-3-small",
    "dimension": 1024
  },
  "pipeline": {
    "chunk_size": 4000,
    "overlap": 400,
    "search": {
      "similarity_threshold": 0.6
    }
  }
}
```

### 13.3 Dot-Path 访问

```bash
deep-dream config get llm.model              # → "gpt-4o-mini"
deep-dream config get server.port            # → 16200
deep-dream config get pipeline.search        # → {"similarity_threshold": 0.6}
deep-dream config set llm.model gpt-4o       # 确认后更新
deep-dream config set server.port 5001       # 确认后更新
```

### 13.4 配置验证

`config set` 自动验证值类型：
- `server.port` 必须是整数
- `llm.model` 必须是非空字符串
- `pipeline.chunk_size` 必须 > 0

无效值给出具体错误：
```
Error: Invalid value for "server.port": "abc" is not a valid integer.
  Hint: Port must be a number between 1 and 65535.
```

---

## 14. 向后兼容策略

### 14.1 命令名保留

所有现有命令名完整保留：

| 旧命令 | 新命令 | 变化 |
|--------|--------|------|
| `deep-dream doctor` | `deep-dream doctor` | 无变化，增加 Rich 输出 |
| `deep-dream remember --text ...` | `deep-dream remember --text ...` | 增加 `-t` 短参数 |
| `deep-dream find ...` | `deep-dream find ...` | 无变化 |
| `deep-dream explore ...` | `deep-dream explore ...` | 参数名可能调整 |
| `deep-dream docs list` | `deep-dream docs list` | 无变化 |
| `deep-dream concept search` | `deep-dream concept search` | 增加 `--mode` 参数 |
| `deep-dream db validate` | `deep-dream db validate` | 无变化 |

### 14.2 JSON 输出兼容

`--json` 输出保持相同的顶层结构（详见 5.3），确保脚本不受影响：
```json
{
  "success": true,
  "command": "<command>",
  "data": { ... },
  "meta": {
    "graph_id": "library",
    "count": 5
  }
}
```

`graph_id`、`count` 等元数据统一收纳在 `meta` 内，顶层仅保留
`success` / `command` / `data`（成功）或 `error`（失败）四个键。
新增字段仅添加到 `data` 或 `meta` 内部，不修改顶层键集合。

### 14.3 Deprecation 处理

如果未来需要废弃命令或参数：

```python
# 废弃参数示例
@click.option('--old-param', hidden=True, callback=deprecated_option)
def cmd(old_param):
    ...

def deprecated_option(ctx, param, value):
    if value is not None:
        click.echo(
            'Warning: --old-param is deprecated. Use --new-param instead.\n'
            'This option will be removed in v3.0.',
            err=True
        )
```

### 14.4 默认 JSON 环境变量

为了向后兼容，设置 `DEEPDREAM_JSON_OUTPUT=1` 时自动启用 `--json`（旧 CLI 的行为）。新 CLI 下 `DEEPDREAM_JSON=1` 是标准方式，但旧名继续支持。

---

## 15. 实施路线

### Phase 1: 基础架构（2-3 天）

1. 创建 `core/cli/` 包结构
2. 实现 `_main.py`（root Click group + 全局选项）
3. 实现 `_ctx.py`（CliContext）
4. 实现 `_output.py`（Rich/JSON 双输出）
5. 实现 `_exit_codes.py`
6. 更新 `pyproject.toml`（依赖 + 入口点）

### Phase 2: 新命令（1-2 天）

1. `cmd_version.py`
2. `cmd_doctor.py`
3. `cmd_config.py`
4. `cmd_completion.py`

### Phase 3: 迁移核心命令（3-4 天）

按使用频率排序迁移：

1. `cmd_find.py` — 高频
2. `cmd_remember.py` — 高频
3. `cmd_explore.py` — 高频
4. `cmd_concept.py` — 高频（含新增子命令）
5. `cmd_docs.py`（含新增 content/delete）
6. `cmd_episode.py`（含新增 get/content）
7. `cmd_relation.py`

### Phase 4: 管理命令（2-3 天）

1. `cmd_server.py`
2. `cmd_task.py`
3. `cmd_graph.py`
4. `cmd_vault.py`（含新增 tree）
5. `cmd_library.py`

### Phase 5: 维护命令（1-2 天）

1. `cmd_db.py`（含新增 quality/integrity）
2. `cmd_sql.py`

### Phase 6: 打磨与测试（2-3 天）

1. Shell 补全测试（bash/zsh/fish）
2. 集成测试（每个命令）
3. 错误场景测试
4. 性能测试（`--help`/`--version` 启动时间 < 200ms）
5. 旧 CLI 兼容测试
6. 更新文档

### 总计：约 12-17 天

---

## 设计检查清单

| 维度 | 检查项 | 状态 |
|------|--------|------|
| **可发现性** | 用户只看 `--help` 能否完成核心操作？ | ✅ 每个命令有示例、默认值 |
| **一致性** | 命令、参数、输出风格是否统一？ | ✅ 统一 `<资源> <动作>` 模式 |
| **安全性** | 危险操作默认确认？支持 `--dry-run`？ | ✅ 4 级确认分级 |
| **自动化** | 稳定退出码、JSON 输出、非交互模式？ | ✅ 9 个退出码、`--json`、`--yes` |
| **错误处理** | 错误信息告诉用户修复方法？ | ✅ Error + Hint 格式 |
| **性能** | `--help`/`--version` 是否立即返回？ | ✅ Click lazy group + 延迟导入 |
| **兼容性** | 更新后是否破坏旧脚本？ | ✅ 所有命令名保留，JSON 结构兼容 |
| **可测试性** | 核心命令、异常路径是否有测试？ | 📋 Phase 6 |
