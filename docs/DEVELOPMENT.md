# Deep Dream 开发文档

> 面向开发者与 AI Agent（如 Ralph）的项目技术文档。

---

## 目录

- [项目概览](#项目概览)
- [技术栈](#技术栈)
- [项目结构](#项目结构)
- [快速开始](#快速开始)
- [核心架构](#核心架构)
- [模块详解](#模块详解)
  - [processor/ — 数据处理管线](#processor--数据处理管线)
  - [server/ — API 服务与前端](#server--api-服务与前端)
  - [scripts/ — 迁移与查询工具](#scripts--迁移与查询工具)
  - [tests/ — 测试套件](#tests--测试套件)
- [数据模型](#数据模型)
- [配置参考](#配置参考)
- [API 端点概览](#api-端点概览)
- [开发工作流](#开发工作流)
- [常见任务](#常见任务)

---

## 项目概览

**Deep Dream** 是一个以自然语言为核心的记忆图谱服务。两个核心职责：

| 职责 | 说明 |
|------|------|
| **Remember** | 接收自然语言文本，自动抽取实体与关系，写入带版本链的知识图谱 |
| **Find** | 通过语义检索（Embedding / BM25 / 混合）唤醒局部记忆区域 |

第三大能力 **Dream**（梦境巩固）：自主探索知识图谱，发现已有实体间隐藏的关系，模拟人类睡眠中的记忆重组。

- **版本**: 0.2.0
- **许可**: MIT
- **Python**: >= 3.10

---

## 技术栈

| 层 | 技术 |
|----|------|
| 图数据库 | Neo4j 5.x Community（可选，默认 SQLite） |
| 向量搜索 | sqlite-vec (ANN KNN) |
| LLM | OpenAI 兼容协议（GLM / Ollama / LM Studio 等） |
| Embedding | sentence-transformers（默认 Qwen3-Embedding-0.6B） |
| Web 框架 | Flask + 原生 SPA Dashboard |
| 前端 | Tailwind CSS + vis-network.js + Lucide Icons |
| MCP | Model Context Protocol（Claude Code / Cursor 集成） |
| 测试 | pytest（默认 mock 模式，`USE_REAL_LLM=1` 启用真实 LLM 测试） |

---

## 项目结构

```
deep-dream/
├── processor/                  # 核心数据处理管线
│   ├── __init__.py             # 公共 API 导出
│   ├── models.py               # 数据模型（Entity, Relation, Episode, ContentPatch）
│   ├── content_schema.py       # Markdown section 解析与 diff
│   ├── utils.py                # 工具函数（日志、hash 等）
│   ├── llm/                    # LLM 客户端与提示词
│   │   ├── client.py           # LLMClient（OpenAI 兼容协议 + Ollama 原生）
│   │   ├── prompts.py          # 所有 LLM prompt 模板
│   │   ├── entity_extraction.py
│   │   ├── relation_extraction.py
│   │   ├── entity_resolution.py
│   │   ├── consolidation.py
│   │   ├── contradiction.py
│   │   ├── content_merger.py
│   │   └── memory_ops.py
│   ├── pipeline/               # 处理管线编排
│   │   ├── orchestrator.py     # TemporalMemoryGraphProcessor（主入口）
│   │   ├── extraction.py       # 抽取 mixin（多轮、去重、重排）
│   │   ├── entity.py           # EntityProcessor（实体处理 + LLM 两步分析）
│   │   ├── entity_merge.py     # 多实体合并逻辑
│   │   ├── relation.py         # RelationProcessor
│   │   ├── relation_ops.py     # 关系操作（去重、重定向、自环检测）
│   │   ├── consolidation.py    # 知识图谱巩固（自环消除、演化摘要）
│   │   ├── section_merge.py    # Section 级增量合并
│   │   └── document.py         # 文档分窗处理
│   ├── storage/                # 存储层
│   │   ├── manager.py          # StorageManager（SQLite 抽象层）
│   │   ├── neo4j_store.py      # Neo4j 存储（图遍历、社区、Episode）
│   │   └── embedding.py        # EmbeddingClient
│   ├── search/                 # 检索层
│   │   ├── hybrid.py           # HybridSearcher（语义 + BM25 + 混合）
│   │   └── graph_traversal.py  # GraphTraversalSearcher（BFS 遍历）
│   └── perf.py                 # 性能计时工具
├── server/                     # Flask API 服务
│   ├── api.py                  # 主入口：路由定义、请求处理（85+ 端点）
│   ├── config.py               # 配置加载与默认值
│   ├── registry.py             # GraphRegistry（多图谱隔离管理）
│   ├── monitor.py              # SystemMonitor（运行时监控）
│   ├── task_queue.py           # RememberTaskQueue（异步任务队列）
│   ├── sse.py                  # SSE 流式响应工具
│   ├── dashboard.py            # 仪表盘聚合端点
│   ├── mcp/                    # MCP 协议服务器
│   │   └── deep_dream_server.py
│   ├── static/                 # 前端 SPA
│   │   ├── index.html          # 单页入口
│   │   ├── css/
│   │   │   ├── app.css         # 主样式（含暗色主题）
│   │   │   └── chat.css        # Chat 终端风格样式
│   │   └── js/
│   │       ├── app.js          # 路由、初始化、通用 UI
│   │       ├── graph-explorer.js  # 图谱可视化（vis-network）
│   │       ├── graph-utils.js     # 图谱工具函数
│   │       ├── path-finder.js     # 最短路径可视化
│   │       ├── chat/              # Chat 模块
│   │       │   ├── renderer.js
│   │       │   └── stream.js
│   │       ├── components/        # UI 组件
│   │       │   ├── modal.js
│   │       │   └── toast.js
│   │       ├── i18n/              # 国际化（zh / en / ja）
│   │       │   ├── index.js
│   │       │   ├── zh.js
│   │       │   ├── en.js
│   │       │   └── ja.js
│   │       └── pages/             # 页面模块（SPA 路由）
│   │           ├── chat.js
│   │           ├── dashboard.js
│   │           ├── graph.js
│   │           ├── memory.js
│   │           ├── search.js
│   │           ├── entities.js
│   │           ├── relations.js
│   │           ├── episodes.js
│   │           ├── communities.js
│   │           └── api-test.js
│   └── templates/
│       └── graph.html           # 独立图谱可视化页面
├── scripts/                    # 迁移与查询脚本
│   ├── migrate_sqlite_to_neo4j.py
│   ├── migrate_build_episodes.py
│   ├── migrate_plain_to_markdown.py
│   ├── migrate_normalize_embeddings.py
│   ├── migrate_rename_family_id.py
│   ├── comprehensive_query.py
│   ├── query_tool.py
│   ├── query_single_entity.py
│   ├── query_entity_relation.py
│   ├── query_relations_network.py
│   ├── search_entities.py
│   ├── search_memory_cache.py
│   └── expand_context.py
├── tests/                      # 测试套件
│   ├── conftest.py             # 共享 fixtures
│   ├── test_pipeline.py
│   ├── test_pipeline_real_llm.py
│   ├── test_storage.py
│   ├── test_api.py
│   ├── test_comprehensive.py
│   ├── test_server_config.py
│   ├── test_llm_client.py
│   ├── test_cross_window_prefetch.py
│   ├── test_queue_progress.py
│   └── test_extraction_reflow.py
├── deploy/                     # 部署配置
│   └── nginx-llm-openai-lb.example.conf
├── docs/                       # 文档（不纳入 git）
├── graph/                      # 图数据库数据（不纳入 git）
├── data/                       # 本地数据（不纳入 git）
├── .claude/skills/deep-dream/  # Claude Code Skill 定义
│   └── SKILL.md
├── docker-compose.yml          # Neo4j 容器编排
├── pyproject.toml              # 包元数据
├── requirements.txt            # 依赖列表
├── service_config.example.json # 配置模板
├── .mcp.json                   # MCP 服务器配置
└── bench_perf.py               # 性能基准测试
```

---

## 快速开始

### 环境要求

- Python >= 3.10
- （可选）Neo4j 5.x — 通过 `docker-compose up -d` 启动
- （可选）本地 LLM — Ollama / LM Studio 等

### 安装

```bash
# 克隆仓库
git clone https://github.com/ngyygm/deep-dream.git
cd deep-dream

# 安装依赖（二选一）
pip install -r requirements.txt          # 方式 1
pip install -e ".[dev]"                  # 方式 2（含 dev 依赖）

# 复制配置模板并编辑
cp service_config.example.json service_config.json
# 编辑 service_config.json: 配置 LLM base_url / model / api_key / embedding 等
```

### 启动服务

```bash
# 启动 API 服务器（默认端口 16200）
python -m server.api --config service_config.json

# （可选）启动 Neo4j
docker-compose up -d
```

浏览器打开 **http://127.0.0.1:16200/** 查看管理面板。

### 验证安装

```bash
# 健康检查
curl http://localhost:16200/api/v1/health?graph_id=default

# 写入记忆
curl -s -X POST http://localhost:16200/api/v1/remember \
  -H 'Content-Type: application/json' \
  -d '{"text":"测试文本","source_name":"test"}'

# 搜索
curl -s -X POST http://localhost:16200/api/v1/find \
  -H 'Content-Type: application/json' \
  -d '{"query":"测试","search_mode":"hybrid"}'
```

---

## 核心架构

### 数据流

```
用户文本 / 文档
     │
     ▼
┌──────────────────────────────────────────────────┐
│  Remember 管线（TemporalMemoryGraphProcessor）    │
│                                                   │
│  1. 文档分窗（DocumentProcessor）                 │
│  2. 整体摘要（Phase 1）                           │
│  3. 逐窗口抽取实体 + 关系（Phase 2）              │
│     ├─ LLM 抽取 → EntityProcessor                │
│     │   ├─ 相似实体检索（Embedding + Jaccard）    │
│     │   ├─ LLM 两步分析（初筛 + 精细化）          │
│     │   └─ 合并 / 新建 / 关联                     │
│     └─ LLM 抽取 → RelationProcessor              │
│         ├─ 端点解析（嵌入名 / Jaccard 匹配）      │
│         └─ 去重 + 版本链更新                      │
│  4. 跨窗口合并（SectionMerge）                    │
│  5. 巩固（自环消除、演化摘要）                    │
│  6. 结果入库 + Embedding 计算                     │
└───────────────┬──────────────────────────────────┘
                │
                ▼
┌──────────────────────────────────────────────────┐
│           统一记忆知识图谱                         │
│                                                   │
│  Entity（版本链）  ·  Relation（版本链）           │
│  Episode  ·  ContentPatch  ·  Community           │
│                                                   │
│  存储后端: SQLite + sqlite-vec  |  Neo4j 5.x     │
└───────────────┬──────────────────────────────────┘
                │
        ┌───────┴───────┐
        ▼               ▼
   Find 检索         Dream 巩固
  （语义/BM25/      （自主探索，
   混合搜索,          发现隐藏
   BFS 遍历）         关系）
```

### 多图谱隔离

`GraphRegistry` 按 `graph_id` 隔离不同的知识图谱：

- 每个 graph_id 对应 `{storage_path}/{graph_id}/` 下的独立数据库
- 不同图谱各自拥有独立的 Processor + TaskQueue
- 所有图谱共享一个 EmbeddingClient 实例（线程安全）

---

## 模块详解

### processor/ — 数据处理管线

#### processor/models.py — 核心数据模型

四个核心 dataclass：

| 类 | 说明 |
|----|------|
| `Episode` | 知识图谱的一等节点，每次写入产生一个 Episode |
| `Entity` | 实体，带版本链（family_id 相同的多个版本） |
| `Relation` | 无向关系，带版本链，端点按字母序排列 |
| `ContentPatch` | Section 级变更记录 |

关键字段说明：
- `absolute_id`: 版本唯一标识符（主键）
- `family_id`: 同一实体/关系的不同版本共享
- `valid_at` / `invalid_at`: 时间版本链
- `content_format`: `"plain"`（旧）或 `"markdown"`（新）

#### processor/content_schema.py — Section 级版本控制

将 Entity/Relation 的 content 字段解析为结构化 section：

- Entity sections: `概述` / `类型与属性` / `详细描述` / `关键事实`
- Relation sections: `关系概述` / `关系类型` / `详细描述` / `上下文`
- 支持 section 级 diff（`compute_section_diff`）
- 生成 `ContentPatch` 记录变更历史

#### processor/llm/ — LLM 交互层

**client.py — LLMClient**

- 支持 OpenAI 兼容协议和 Ollama 原生 `/api/chat`
- `PrioritySemaphore`：7 级优先级的并发控制
  - STEP1~STEP7 对应管线不同阶段，避免低优先级任务阻塞关键路径
- 核心方法：
  - `extract_entities()` / `extract_relations()`: 信息抽取
  - `analyze_entity_candidates_preliminary()` / `analyze_entity_pair_detailed()`: 实体分析两步流程
  - `merge_multiple_entity_contents()`: 多实体内容合并
  - `judge_content_need_update()`: 判断是否需要更新
  - `merge_entity_name()`: 实体名称合并

**prompts.py** — 所有 prompt 模板集中管理

其余文件按职责拆分：
- `entity_extraction.py`: 实体抽取 prompt 构建与结果解析
- `relation_extraction.py`: 关系抽取
- `entity_resolution.py`: 实体消解（合并/关联决策）
- `consolidation.py`: 知识巩固
- `contradiction.py`: 矛盾检测与裁决
- `content_merger.py`: 内容合并
- `memory_ops.py`: 记忆缓存操作

#### processor/pipeline/ — 管线编排

**orchestrator.py — TemporalMemoryGraphProcessor**

主处理入口，核心方法：

| 方法 | 阶段 | 说明 |
|------|------|------|
| `process_documents()` | 主入口 | 文档处理管线（分窗 → Phase1 → Phase2 → 巩固） |
| `remember_text()` | 写入 | 单条文本记忆写入 |
| `remember_phase1_overall()` | Phase 1 | 整体摘要与实体初筛 |
| `remember_phase2_windows()` | Phase 2 | 逐窗口抽取实体+关系 |

**extraction.py — _ExtractionMixin**

- 多轮抽取控制
- 跨窗口去重（`dedupe_extraction_lists`）
- LLM 调用编排

**entity.py — EntityProcessor**

实体处理核心流程：

1. Embedding + Jaccard 相似实体检索
2. LLM 两步分析：
   - 初筛（content snippet 快速筛选）
   - 精细化（完整 content 精确判断）
3. 决策：merge（合并到已有实体） / create_relation（创建关系） / no_action
4. 多合并目标处理（选择版本数最多的实体为主要目标）
5. Section 级 patch 记录

**relation.py / relation_ops.py**

- 关系端点解析（嵌入名匹配 + Jaccard 匹配）
- 去重（无向对 + 内容比对）
- 自环检测、重定向

**consolidation.py — 巩固阶段**

- 自环关系消除
- 实体演化摘要（LLM 生成）
- 关系重定向链清理

**section_merge.py** — 跨窗口 section 级增量合并

**document.py** — 文档分窗（滑窗切分 + overlap）

#### processor/storage/ — 存储层

**manager.py — StorageManager**

SQLite 抽象层，核心方法：

- Entity CRUD: `save_entity()`, `get_entity_by_family_id()`, `get_entity_versions()`
- Relation CRUD: `save_relation()`, `get_relations_by_entities()`
- 搜索: `search_entities_by_similarity()`, `search_relations_by_text()`
- 合并: `merge_entity_families()`
- 版本链管理: `invalidate_entity_version()`, `get_entity_version_counts()`
- Content Patch: `save_content_patches()`
- Embedding: `get_all_entity_embeddings()`

**neo4j_store.py — Neo4jStore**

Neo4j 专属功能：
- 图遍历（BFS）
- 社区检测（Louvain / Label Propagation）
- Episode 管理
- 邻居查询
- 最短路径搜索

**embedding.py — EmbeddingClient**

- 基于 sentence-transformers 的本地 embedding
- 批量编码与 L2 归一化
- 线程安全

#### processor/search/ — 检索层

**hybrid.py — HybridSearcher**

三种搜索模式：
- `semantic`: Embedding 向量搜索（sqlite-vec KNN）
- `bm25`: 关键词搜索（Jaccard + 文本匹配）
- `hybrid`: 混合搜索（语义 + BM25 结果融合）

支持 Reranker（RRF / MMR / node_degree）。

**graph_traversal.py — GraphTraversalSearcher**

- BFS 多跳遍历
- 种子扩展
- 关系收集

---

### server/ — API 服务与前端

#### server/api.py — Flask 路由

主入口文件（约 1800 行），包含 85+ API 端点：

| 端点组 | 路径前缀 | 说明 |
|--------|---------|------|
| 记忆写入 | `/api/v1/remember` | 异步任务提交、进度查询、暂停/恢复 |
| 统一检索 | `/api/v1/find` | 语义/BM25/混合搜索 |
| 自然语言问答 | `/api/v1/find/ask` | LLM 元查询（含 SSE 流式） |
| 实体 CRUD | `/api/v1/find/entities` | 创建/读取/更新/删除/合并/拆分 |
| 关系 CRUD | `/api/v1/find/relations` | 创建/更新/失效/重定向 |
| 时间旅行 | `/api/v1/find/snapshot` | 快照/变更/时间点版本 |
| 梦境巩固 | `/api/v1/find/dream` | 种子/关系/episode/状态 |
| 社区检测 | `/api/v1/communities` | 检测/列表/子图（Neo4j 专属） |
| Episode | `/api/v1/episodes` | 列表/搜索/批量导入/删除 |
| Chat 会话 | `/api/v1/chat/sessions` | 创建/消息/SSE 流式 |
| 图谱管理 | `/api/v1/graphs` | 列出/创建图谱 |
| 系统 | `/api/v1/system` | 健康/日志/统计/仪表盘 |
| 静态文件 | `/` | SPA Dashboard |

启动命令行参数：
```bash
python -m server.api --config service_config.json [--host 0.0.0.0] [--port 16200]
```

#### server/config.py — 配置加载

- 从 JSON 文件加载配置
- 缺失项使用默认值填充
- 支持旧格式自动转换（如 `remember_workers` → `runtime.concurrency.queue_workers`）

#### server/registry.py — GraphRegistry

多图谱注册表：
- 延迟初始化：首次访问 graph_id 时才创建 Processor
- 自动注册到 SystemMonitor
- 共享 EmbeddingClient

#### server/monitor.py — SystemMonitor

运行时监控：
- 请求计数与延迟统计
- 任务队列状态
- 内存使用
- API 访问统计

#### server/task_queue.py — RememberTaskQueue

异步任务队列：
- 后台 worker 线程处理 Remember 任务
- 支持暂停 / 恢复 / 删除
- 进度追踪（6 阶段：Phase1 → Phase2 → ... → 巩固）

#### server/mcp/ — MCP 协议服务器

`deep_dream_server.py` 实现 MCP 协议（JSON-RPC over stdio），将 HTTP API 包装为 MCP tools，供 Claude Code 等 Agent 直接调用。

**注意**: 此文件在 import 时修改 sys.stdout/stdin，不能在同一进程中直接导入。

#### 前端 SPA

原生单页应用（无构建步骤）：

| 页面 | 文件 | 功能 |
|------|------|------|
| Chat | `pages/chat.js` | Claude Code 风格终端聊天（SSE 流式） |
| Dashboard | `pages/dashboard.js` | 系统概览、实时统计 |
| Graph | `pages/graph.js` | vis-network 图谱可视化 |
| Memory | `pages/memory.js` | 记忆写入与任务管理 |
| Search | `pages/search.js` | 多模式检索 |
| Entities | `pages/entities.js` | 实体 CRUD |
| Relations | `pages/relations.js` | 关系 CRUD |
| Episodes | `pages/episodes.js` | Episode 管理 |
| Communities | `pages/communities.js` | 社区检测与可视化 |
| API Test | `pages/api-test.js` | API 调试工具 |

支持三语 i18n（中文 / English / 日本語）和暗色/亮色主题切换。

---

### scripts/ — 迁移与查询工具

#### 迁移脚本

| 脚本 | 用途 |
|------|------|
| `migrate_sqlite_to_neo4j.py` | SQLite → Neo4j 全量迁移（实体、关系、Embedding） |
| `migrate_build_episodes.py` | 从 docs/ 构建 Neo4j Episode 节点和 MENTIONS 边 |
| `migrate_plain_to_markdown.py` | 旧格式 plain text → 结构化 Markdown（LLM 转换） |
| `migrate_normalize_embeddings.py` | L2 归一化所有向量 |
| `migrate_rename_family_id.py` | Neo4j 属性重命名（entity_id → family_id） |

#### 查询脚本

| 脚本 | 用途 |
|------|------|
| `comprehensive_query.py` | 自动分类问题类型并编排多步查询策略 |
| `query_tool.py` | 通用查询工具（实体/关系/历史/图谱） |
| `query_single_entity.py` | 单实体深度查询 |
| `query_entity_relation.py` | 两实体间关系查询（含别名解析） |
| `query_relations_network.py` | BFS 关系网络（导出 JSON/DOT/Markdown） |
| `search_entities.py` | 多策略实体搜索 |
| `search_memory_cache.py` | 记忆缓存搜索 |
| `expand_context.py` | 上下文扩展（时间邻居/关系图遍历） |

---

### tests/ — 测试套件

### 运行测试

```bash
# 运行所有测试（默认 mock 模式，无需 LLM API Key）
pytest

# 运行特定测试文件
pytest tests/test_storage.py

# 启用真实 LLM 测试
USE_REAL_LLM=1 pytest tests/test_pipeline_real_llm.py

# 带覆盖率
pytest --cov=processor --cov=server
```

### 测试文件说明

| 文件 | 覆盖范围 |
|------|---------|
| `test_pipeline.py` | TemporalMemoryGraphProcessor 全流程（mock LLM） |
| `test_pipeline_real_llm.py` | 同上，使用真实 LLM 后端 |
| `test_storage.py` | StorageManager CRUD、搜索、合并、并发、边界情况 |
| `test_api.py` | Flask API 85+ 端点（创建/查询/错误处理/CORS） |
| `test_comprehensive.py` | 高级 API 测试（搜索模式、时间旅行、社区、梦境） |
| `test_server_config.py` | 配置加载与格式转换 |
| `test_llm_client.py` | LLMClient、PrioritySemaphore、JSON 清理 |
| `test_cross_window_prefetch.py` | 跨窗口预取逻辑 |
| `test_queue_progress.py` | 任务队列进度追踪 |
| `test_extraction_reflow.py` | Steps 2-5 重流（实体修剪、去重、关系端点补充） |

---

## 数据模型

### 版本链机制

```
Entity / Relation 的版本链:

  valid_at                 invalid_at
     │                        │
     ▼                        ▼
  ┌──────────┐            ┌──────────┐
  │ Version 1│ ──────────▶│ Version 2│ ──────▶ null (当前有效)
  │ (absolute_id_1)        │ (absolute_id_2)
  │ family_id = "ent_xxx"  │ family_id = "ent_xxx"
  └──────────┘            └──────────┘
```

- 同一逻辑实体/关系的所有版本共享 `family_id`
- 每个版本有唯一的 `absolute_id`
- `valid_at` / `invalid_at` 构成时间版本链
- 当前有效版本的 `invalid_at` 为 null

### 存储后端

| 特性 | SQLite + sqlite-vec | Neo4j |
|------|-------------------|-------|
| 实体/关系 CRUD | ✅ | ✅ |
| 向量搜索 | ✅ (KNN) | ✅ (通过 sqlite-vec) |
| 图遍历 (BFS) | ✅ (SQL 递归) | ✅ (原生) |
| 社区检测 | ❌ | ✅ (Louvain) |
| Episode 管理 | ✅ (SQLite) | ✅ (Neo4j 节点) |
| 邻居图 | ❌ | ✅ |
| 最短路径 | ❌ | ✅ |

---

## 配置参考

配置文件为 `service_config.json`，完整模板见 `service_config.example.json`。

### 核心配置项

```jsonc
{
  // 服务器
  "host": "0.0.0.0",
  "port": 16200,
  "log_mode": "detail",  // "detail" | "monitor"

  // 存储
  "storage": {
    "backend": "sqlite",  // "sqlite" | "neo4j"
    "neo4j": {
      "uri": "bolt://localhost:7687",
      "user": "neo4j",
      "password": "password"
    },
    "vector_dim": 1024
  },

  // LLM 配置（支持三级：主 LLM / alignment LLM / dream LLM）
  "llm": {
    "api_key": "your-key",
    "model": "model_name",
    "base_url": "http://127.0.0.1:11434",
    "think": false,
    "max_tokens": 3000,
    "context_window_tokens": 8000,
    "max_concurrency": 3,
    "alignment": { /* 对齐专用 LLM */ },
    "dream": { /* 梦境专用 LLM */ }
  },

  // Embedding
  "embedding": {
    "model": "Qwen/Qwen3-Embedding-0.6B",  // HuggingFace 名称或本地路径
    "device": "cpu"  // "cpu" | "cuda"
  },

  // 文档分窗
  "chunking": {
    "window_size": 1000,
    "overlap": 200
  },

  // 运行时并发
  "runtime": {
    "concurrency": {
      "queue_workers": 1,   // Remember 任务并行数
      "window_workers": 1   // 窗口处理并行数
    }
  },

  // 管线参数
  "pipeline": {
    "search": {
      "similarity_threshold": 0.7,
      "max_similar_entities": 10
    },
    "extraction": {
      "extraction_rounds": 1,
      "entity_extraction_rounds": 1,
      "relation_extraction_rounds": 1,
      "entity_post_enhancement": false
    }
  }
}
```

---

## API 端点概览

### 常用端点

| 方法 | 路径 | 说明 |
|------|------|------|
| `POST` | `/api/v1/remember` | 写入记忆（异步，返回 task_id） |
| `POST` | `/api/v1/find` | 统一检索（semantic/bm25/hybrid） |
| `POST` | `/api/v1/find/ask` | 自然语言问答（Agent 元查询） |
| `POST` | `/api/v1/find/ask/stream` | 自然语言问答（SSE 流式） |
| `GET` | `/api/v1/health` | 健康检查 |
| `GET` | `/api/v1/find/stats` | 图谱统计 |
| `GET` | `/api/v1/system/dashboard` | 系统仪表盘 |

完整 85+ 端点列表见 `.claude/skills/deep-dream/SKILL.md` 或启动服务后访问 `/api/v1/routes`。

---

## 开发工作流

### 代码风格

- 使用 `ruff` 进行 lint（`pip install -e ".[dev]"` 后可用）
- 中文注释与 docstring
- 数据类使用 `@dataclass`
- 配置通过 `service_config.json` 管理，不使用环境变量

### Git 工作流

```bash
# 主分支
main

# 提交消息风格（参考 git log）
feat: 功能描述
fix: 修复描述
refactor: 重构描述
```

### 不纳入版本控制的内容

参见 `.gitignore`：
- `docs/` / `graph/` / `data/` — 本地数据
- `service_config.json` — 含 API Key 等敏感信息
- `distill_pipeline/` — 管线蒸馏输出
- `__pycache__/` / `.pytest_cache/` — 缓存

---

## 常见任务

### 添加新的 LLM 提示词

1. 在 `processor/llm/prompts.py` 中定义 prompt 模板
2. 在 `processor/llm/client.py` 中添加调用方法
3. 在对应模块（如 `entity_extraction.py`）中集成
4. 在 `tests/test_llm_client.py` 中添加测试

### 添加新的 API 端点

1. 在 `server/api.py` 中添加路由函数
2. 如需新的管线功能，在 `processor/pipeline/` 中实现
3. 在 `tests/test_api.py` 或 `tests/test_comprehensive.py` 中添加测试

### 切换存储后端

在 `service_config.json` 中修改：
```json
"storage": {
  "backend": "neo4j",
  "neo4j": {
    "uri": "bolt://localhost:7687",
    "user": "neo4j",
    "password": "your-password"
  }
}
```

### 使用 MCP 集成

项目自带 `.mcp.json`，Claude Code 会自动发现并连接 Deep Dream MCP 服务器：
- 确保 API 服务器在 `localhost:16200` 运行
- MCP 服务器通过 stdio 与 Claude Code 通信

### 性能基准测试

```bash
python bench_perf.py
```

### Docker 部署 Neo4j

```bash
docker-compose up -d
# 验证
docker exec tmg-neo4j cypher-shell -u neo4j -p tmg2024secure "RETURN 1"
```
