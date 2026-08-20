<div align="center">

![Deep-Dream](docs/picture/deep-dream-logo.png)

# Deep-Dream

**文档优先的概念图谱记忆服务器**

[![Python 3.10+](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![MIT License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Version](https://img.shields.io/badge/Version-0.2.0-orange.svg)](pyproject.toml)

将本地文档转化为人类可审查、AI 可召回的结构化记忆系统。原始文件始终可读可编辑，系统在其之上构建可追溯的文档、Episode、概念、关系和图谱视图。

[English](README_EN.md) · 简体中文

</div>

---

## 核心特性

- **文档优先** — Markdown / 纯文本文件始终是 Source of Truth，概念图谱是语义叠加层
- **Remember 管线** — 多步提取：分块 → 实体提取 → 关系发现 → 对齐合并 → 去重写入
- **混合搜索** — BM25 全文检索 + 向量嵌入 + 图谱 BFS 扩展，RRF 融合排序
- **CLI 控制台** — Click 8+ / Rich 13+，17 个命令，人类可读 Rich 输出 + `--json` 机器模式
- **Web UI** — Dashboard、记忆上传、交互式图谱、语义搜索、社区发现、设置管理
- **概念版本化** — 每个概念维护 `family_id`（稳定身份）和版本链（随 Episode 演化）
- **Vault 索引** — 支持 Obsidian / Markdown 库的 Wikilink 提取和标题解析
- **多语言界面** — 中文 / English / 日本語，深色 / 浅色主题
- **本地优先** — SQLite + 本地 Embedding 模型，支持 Ollama 等 OpenAI 兼容端点
- **异步任务队列** — 支持 pause / resume / retry，磁盘持久化和断电恢复

## Remember 管线

`remember` 流程将原始文本转化为结构化的、有证据支撑的记忆：分块输入 → 提取实体与关系 → 质量门控 → 与已有概念对齐 → 写入本地图谱，全程保留来源证据。

**管线步骤（strong-v1 单遍抽取）：**

1. **文档分块** — Markdown 标题感知的智能分块，支持重叠窗口（strong-v1 默认 6000/300 大窗口）
2. **Episode 生成** — 每个分块作为一个 Episode（记忆事件）
3. **单遍抽取** — 每窗口一次 LLM 调用同时产出实体、实体内容与关系，保留证据文本和行号
4. **概念对齐** — 窗口内批量对齐 + 与已有概念匹配合并（保守策略）
5. **跨窗口去重合并** — 同一文档内跨 Episode 的同名概念合并，内容走统一合并器
6. **写入存储** — FamilyWriteGate 家族级写入门控，写入 SQLite，嵌入同步更新

## 快速开始

### 安装

```bash
# 克隆项目
git clone <repo-url>
cd deep-dream

# 安装依赖
pip install -e .
```

### 配置

```bash
# 从示例配置创建你的配置文件
cp service_config.example.json service_config.json
```

编辑 `service_config.json`，配置你的 LLM 端点：

```json
{
  "llm": {
    "api_key": "your-api-key",
    "model": "your-model-name",
    "base_url": "http://127.0.0.1:11434",
    "max_tokens": 3000,
    "context_window_tokens": 8000
  },
  "embedding": {
    "model": "Qwen/Qwen3-Embedding-0.6B",
    "device": "cpu"
  }
}
```

支持任何 OpenAI 兼容端点（Ollama、LM Studio、GLM、Xinference 等）。

### 启动

```bash
# CLI 方式启动
deep-dream server start --config service_config.json

# 或直接运行
python -m core.server.api --config service_config.json

# Windows 一键启动
start.bat
```

服务器默认运行在 `http://localhost:16200`。

## CLI

CLI 是面向人类和 Agent 的控制面板：任务优先的命令结构、安全默认值、Rich 格式化输出，以及 `--json` 自动化模式。

**17 个命令：**

| 命令 | 说明 |
|------|------|
| `deep-dream version` | 显示版本信息 |
| `deep-dream doctor` | 系统健康检查 |
| `deep-dream config` | 查看 / 编辑配置 |
| `deep-dream remember` | 将文本 / 文件写入记忆图谱 |
| `deep-dream find <query>` | 语义搜索概念 |
| `deep-dream explore` | 概念语义探索 |
| `deep-dream concept` | 概念 CRUD 操作 |
| `deep-dream episode` | Episode 查看 |
| `deep-dream relation` | 关系查看 |
| `deep-dream docs` | 文档管理 |
| `deep-dream graph` | 图谱管理 |
| `deep-dream vault` | Obsidian / Markdown 库索引 |
| `deep-dream server` | 启动 / 管理 API 服务器 |
| `deep-dream task` | 任务队列管理 |
| `deep-dream db` | 数据库维护 |
| `deep-dream sql` | 直接 SQL 查询 |
| `deep-dream completion` | Shell 补全设置 |

**全局选项：** `--json` · `--no-color` · `-q` · `--config`

### 评测与论文工作

实验测评 harness（LoCoMo / LongMemEval / MEME 等）与论文工程位于 `research/`，与系统本体无关，用法见 [research/README.md](research/README.md)。

## Web UI

Deep-Dream 提供功能完整的单页应用界面：

- **Dashboard** — 系统概览、任务进度、实时日志、统计信息
- **记忆管理** — 文本 / 文件上传、任务监控、文档浏览
- **图谱可视化** — 基于 vis-network 的交互式图谱，支持生长动画、文档子图、时间线回放、角色着色（文档=紫色、Episode=蓝色、实体=青色、关系=琥珀色）
- **语义搜索** — 三种模式（普通 / 多查询 / 遍历），路径查找器，阈值和时间筛选，搜索历史
- **社区发现** — Louvain 算法社区检测，社区浏览和可视化
- **API 测试** — 原始 API 请求测试界面
- **设置** — 在线配置编辑器

## 数据模型

```text
Document → Episode → Concept (entity / relation / observation)
```

- **Document** — Markdown 源文档或记忆文本来源（managed / external / vault 三种模式）
- **Episode** — 文档内的一个标题级源文片段，是记忆提取的基本单元
- **Entity** — 提取的实体概念，维护版本链，每次 Episode 提及生成新版本
- **Relation** — 提取的关系概念，连接两个实体，保留证据文本和行号偏移

**统一概念模型：** 实体、关系、观察都是 `Concept` 的不同角色（role），共享统一的 `family_id`（稳定身份）和版本链（演化历史）。

**Schema V1.5（12 张表）：**

| 表 | 说明 |
|----|------|
| `documents` | 源文档 |
| `document_versions` | 文档版本快照 |
| `episodes` | 源文分块 |
| `entity_families` | 实体身份（跨版本） |
| `entity_observations` | 每 Episode 的实体观察 |
| `entity_mentions` | 文本提及及偏移 |
| `relation_families` | 关系身份 |
| `relation_assertions` | 每 Episode 的关系断言 |
| `embeddings` | 通用嵌入存储 |
| `pipeline_runs` | 管线执行跟踪 |
| `document_links` | Wikilink / Markdown 链接 |
| `entity_redirects` | 实体合并重定向 |

另含 `episodes_fts`（FTS5，trigram 分词支持中日韩）和 `graph_edges` 视图。

## API

Base URL: `http://localhost:16200/api/v1`

**核心端点：**

| 类别 | 端点 | 说明 |
|------|------|------|
| 记忆 | `POST /remember` | 提交文本 / 文件进行记忆写入 |
| | `GET /remember/tasks` | 任务队列列表 |
| 搜索 | `POST /concepts/search` | 语义搜索概念 |
| | `POST /traverse` | 图谱遍历 |
| 概念 | `GET /concepts` | 概念列表 |
| | `GET /concepts/<family_id>` | 概念详情 |
| | `GET /concepts/<family_id>/versions` | 版本历史 |
| | `GET /concepts/<family_id>/provenance` | 来源追溯 |
| 文档 | `GET /documents` | 文档列表 |
| | `GET /documents/<id>/content` | 文档内容 |
| | `GET /documents/search` | 搜索原始文件 |
| Vault | `POST /vaults/index` | 索引 Markdown / Obsidian 库 |
| | `GET /vaults/tree` | Vault 文件树 |
| 系统 | `GET /health` | 健康检查 |
| | `GET /stats/counts` | 概念统计 |
| | `POST /communities/detect` | 社区检测 |

**Agent 工作流：**

```text
1. 先搜索和阅读原始文件
2. 需要图谱上下文时，将文件映射为 document id
3. 通过 Episode 获取源文片段和行级证据
4. 通过 concepts/relations 进行语义扩展和对齐
5. 用原始文本或 episode source_text 验证最终结论
```

## 存储布局

```text
library/                     # 默认存储路径
  library.db                 # SQLite 主数据库
  documents/
    managed/                 # 系统管理的文档
    external/                # 外部引用的文档
  snapshots/                 # 文档快照
  artifacts/                 # 附件
  indexes/                   # 索引
  logs/                      # 日志
  tasks/                     # 任务持久化
  library.json               # 库元数据
```

## 配置说明

配置文件为 `service_config.json`，主要配置项：

```json
{
  "host": "0.0.0.0",
  "port": 16200,
  "storage_path": "./library",
  "storage": { "backend": "sqlite", "vector_dim": 1024 },
  "llm": {
    "model": "model_name",
    "base_url": "http://127.0.0.1:11434",
    "max_concurrency": 3,
    "alignment": {
      "model": "alignment_model_name",
      "base_url": "http://127.0.0.1:11435"
    }
  },
  "embedding": {
    "model": "Qwen/Qwen3-Embedding-0.6B",
    "device": "cpu"
  },
  "chunking": { "window_size": 1000, "overlap": 200 },
  "pipeline": {
    "remember": { "profile": "strong-v1", "alignment_policy": "conservative" }
  }
}
```

**LLM 双协议支持：** 同时支持 Ollama 原生（`/api/chat`）和 OpenAI 兼容（`/v1/chat/completions`）协议。

**Embedding：** 默认使用本地 sentence-transformers 模型，LRU 缓存 + SHA-256 键 + TTL 自动过期。

## 开发

```bash
# 安装开发依赖
pip install -e ".[dev]"

# 运行测试
python -m pytest core/tests/

# 启动服务器（跳过 LLM 检查）
python -m core.server.api --config service_config.json --skip-llm-check

# 代码检查
ruff check core/
```

## 技术栈

- **后端：** Python / Flask / SQLite (FTS5)
- **CLI：** Click 8+ / Rich 13+
- **LLM：** OpenAI SDK（兼容 Ollama、LM Studio、GLM 等）
- **Embedding：** sentence-transformers（本地模型）
- **前端：** 原生 SPA（vis-network、Tailwind CSS、Lucide Icons）
- **搜索：** BM25 + 向量检索 + 图遍历，RRF 融合

## License

[MIT](LICENSE)
