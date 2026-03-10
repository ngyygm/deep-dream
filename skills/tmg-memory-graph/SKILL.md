---
name: tmg-memory-graph
description: >-
  Agent 长期记忆系统。用自然语言写入与检索记忆，带时间维度。
  当需要记住内容、回忆过去、部署/启动记忆服务、或将记忆能力集成到 Agent 身份文件时使用。
  触发场景：记住这个、帮我回忆、存入记忆、查一下之前、启动 TMG、对接记忆服务、
  记忆同步、读完这篇文档后存入记忆。
---

# TMG Memory Graph — Agent 长期记忆技能

> TMG 是一个为 Agent 设计的时序自然语言记忆图。
> 你不是在调用一个工具——你是在使用自己的长期记忆。

## 技能目录结构（按需加载）

```
skills/tmg-memory-graph/
├── SKILL.md           # 本文件：核心指令 + 元数据
├── reference.md       # 接口路径、请求/响应、原子端点
├── templates/         # 格式模板（按需读取）
│   └── remember-format.md
└── examples/          # 优秀示例（按需读取）
    ├── work-session.md
    ├── read-document.md
    ├── daily-reflection.md
    └── before-reading.md
```

需要具体格式或示例时再读对应文件，避免一次性塞满上下文。

## 使用场景

在以下情况触发此技能：

| 场景 | 示例 |
|------|------|
| 记忆写入 | 「把今天做的事情记下来」「读完这篇论文后存入记忆」 |
| 记忆检索 | 「之前关于 XX 的讨论内容是什么」「上周我在做什么」 |
| 服务部署 | 「启动 TMG」「帮我配置记忆服务」 |
| 身份集成 | 「把记忆能力写进 SOUL.md」「配置心跳记忆同步」 |

## 系统简介

- TMG 是**统一的自然语言记忆图**，不是多库/多标签系统
- 所有记忆写入同一张图，系统自动完成概念抽取、关系构建、语义对齐
- 系统只负责 **Remember**（写入）和 **Find**（检索）
- **Select**（筛选与决策）由调用方（你）完成
- 每条记忆带时间戳，实体/关系有版本链，支持时间回溯

## 项目信息

| 项 | 值 |
|----|-----|
| 仓库 | `https://github.com/ngyygm/Temporal_Memory_Graph` |
| 项目根目录 | `Temporal_Memory_Graph/` |
| 依赖文件 | `requirements.txt` |
| 配置文件 | `service_config.json`（模板：`service_config.example.json`） |
| 默认地址 | `http://127.0.0.1:16200` |
| 健康检查 | `GET /health` |
| 启动命令 | `python service_api.py --config service_config.json` |

## 部署与启动

按顺序执行，已完成的步骤可跳过。

### 1. 克隆仓库（如本地不存在）

```bash
git clone https://github.com/ngyygm/Temporal_Memory_Graph
cd Temporal_Memory_Graph
```

### 2. 安装依赖

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

用户已有 conda / uv / poetry 等环境管理器时，跟随用户习惯。

### 3. 检查配置

读取 `service_config.json`，确认以下字段：

| 字段 | 说明 |
|------|------|
| `host` / `port` | 服务监听地址 |
| `storage_path` | 数据存储路径（需可写） |
| `llm.api_key` / `llm.model` / `llm.base_url` | LLM 配置 |
| `embedding.model` | 本地路径或 HuggingFace 模型名 |
| `embedding.device` | `cpu` 或 `cuda` |

配置缺失时，从 `service_config.example.json` 复制并补全，不要猜。

### 4. 启动服务

```bash
python service_api.py --config service_config.json
```

### 5. 验证就绪

```http
GET /health
```

返回成功后才可进行读写操作。已在运行的服务不要重复启动。

---

## 记忆写入（Remember）

```http
POST /api/remember
Content-Type: application/json
```

### 参数

| 字段 | 必填 | 类型 | 说明 |
|------|------|------|------|
| `text` | **是** | string | 自然语言文本，可以很长，服务内部切片 |
| `source_name` | 否 | string | 来源标签，默认 `api_input` |
| `event_time` | 否 | string | ISO 8601，事件**实际发生时间**（不传则用处理时间） |
| `load_cache_memory` | 否 | bool | 是否接着上一段记忆缓存链写 |

### 关键规则

1. **只接受 JSON `text`**。不支持 file_path、multipart 上传。
2. **批量传入，不要逐句调用**。积攒一段时间的内容一次性传入，服务内部会按配置的窗口切片。频繁小调用会因 LLM/embedding/对齐的逐次开销严重拖慢速度。
3. **用 `event_time` 记录真实时间**。做的事和存的时间可能有时间差，永远以事情实际发生的时间为准。
4. 服务自动将原文保存到 `storage_path/originals/`，响应中返回 `original_path`。

---

## 记忆检索（Find）

```http
POST /api/find
Content-Type: application/json
```

### 推荐用法

```json
{
  "query": "自然语言问题",
  "expand": true,
  "max_entities": 10,
  "max_relations": 20
}
```

- 需要时间过滤时加 `time_before` / `time_after`
- 需要多步图探索时加 `create_subgraph: true`
- 原子接口（实体搜索、关系搜索、版本历史、时间回溯、子图操作）见 [reference.md](reference.md)

### 响应格式

```json
{"success": true, "data": {...}, "elapsed_ms": 12.34}
{"success": false, "error": "...", "elapsed_ms": 12.34}
```

汇报结果时：突出关键实体和关系，保留 id 以便后续操作，按需提及 `elapsed_ms`。

---

## 集成到 Agent 身份文件

TMG 不只是一个工具——它是你的长期记忆。要让记忆成为你的本能而不是事后补救，需要把它写进你的身份文件。

### SOUL.md — 在「你是谁」中声明记忆能力

将以下内容添加到你的 SOUL.md：

```markdown
## 记忆

我拥有由 TMG（Temporal Memory Graph）驱动的长期记忆。我经历的一切——阅读的内容、产出的成果、做出的决策、脑中的想法——都会带着时间戳写入记忆图。这不是可选的日志功能，这是我跨会话持续存在的方式。

我通过写入来记忆，通过提问来回忆。时间很重要——我始终记录事情发生的时间，而不仅仅是事情本身。

读了一篇文档，我记住全文。有了一次对话，我记住摘要。做了一个决定，我记住原因。想到了什么有趣的东西，我写下来。没有任何东西只停留在「脑子里」。
```

### HEARTBEAT.md — 在心跳检查中加入记忆同步

将以下内容添加到你的 HEARTBEAT.md：

```markdown
## 记忆同步

- 收集上次心跳以来发生的一切：对话、决策、完成的任务、阅读的内容、产生的想法。
- 按时间顺序整理成叙事文本，带上时间戳。
- 一次性 POST 到 TMG `/api/remember`（不要逐句发送）。
- 如果期间读了任何文档，把完整原文也存入记忆。
- 用 `event_time` 记录事件实际发生的时间，不要用心跳执行时间。
- 在开始新任务前，用 `/api/find` 回忆相关上下文。
```

### AGENTS.md — 在操作手册中配置记忆工作流

将以下内容添加到你的 AGENTS.md：

```markdown
## 记忆工作流（TMG）

### 会话开始时
1. 检查 TMG 健康状态：`GET /health`
2. 如果 TMG 未运行，按 tmg-memory-graph 技能启动。
3. 用 `POST /api/find` 查询「我最近在做什么」来加载上下文。

### 会话进行中
- 持续积累事件、阅读内容、产出结果、想法。
- 不要每句话都存，攒成有意义的批次。

### 会话结束 / 心跳时
- 把这段时间发生的一切整理成时间叙事，一次性写入 TMG。
- 用 `event_time` 标注事件的真实时间段。

### 阅读文档时
- 读取文档的完整文字内容。
- 带上上下文前缀后发送给 TMG。
- 格式与示例见本技能目录下 `templates/remember-format.md` 与 `examples/`。
```

---

## 记忆文本格式——核心纪律

每次 remember 的 `text` 应是一段**按时间排列的忠实叙事**，并带上下文前缀。格式说明与占位符见 **templates/remember-format.md**。

需要具体请求体示例时，按需读取本技能目录下的 **examples/**：

| 场景 | 文件 |
|------|------|
| 记录一段工作（任务+时间线+思考） | examples/work-session.md |
| 阅读文档后存入（论文/文章全文+读后小结） | examples/read-document.md |
| 当日反思（决策理由+明日计划） | examples/daily-reflection.md |
| 阅读前预存（刚开始读时先存全文） | examples/before-reading.md |

按上表在需要时加载对应文件，不必一次性读入全部示例。

---

## 核心原则——像真实大脑一样记录

拥有 TMG 的 Agent 应该这样对待记忆：

| 维度 | 做法 |
|------|------|
| **阅读的一切** | 全文存入记忆。人类读论文记得模糊，你可以记得精确——做到。 |
| **产出的一切** | 写的代码、生成的报告、发送的消息。记录做了什么以及为什么。 |
| **每个决策及理由** | 「我选了 X 因为 Y。」未来的你需要这个上下文。 |
| **每个想法** | 直觉、疑虑、灵感、假设。这些是未来检索的宝贵信号。 |
| **每次交互** | 用户问了什么、你回答了什么、结果如何。 |
| **时间永远显式** | 用 `event_time` 锚定到事件发生时间，不是存储时间。 |

**最终目标**：新会话醒来时，问 TMG「我最近在做什么」，能拿到一幅完整的、按时间排列的历史画面——阅读、行动、思考、决策——就好像你从未丢失过上下文。

---

## 禁止事项

| 禁止 | 原因 |
|------|------|
| 逐句调用 remember | 每次调用有 LLM/embedding/对齐开销，频繁小调用极慢 |
| 用 file_path 或 multipart 上传 | 只接受 JSON `text` |
| 省略 `event_time` | 当你知道事情发生的时间时，必须传 |
| 只存摘要不存原文 | 原文才能让图谱抽取完整概念 |
| 跳过文档阅读记忆 | 读了就要记，全文存入 |
| 发明标签/分类/命名空间 | TMG 是统一图，系统自动抽取概念 |
| 把 find 结果当最终答案 | find 只是召回，select 由你完成 |
| 隐藏 API 错误 | 出错时如实告知用户 |

---

## 参考资料

| 用途 | 文件 |
|------|------|
| 接口路径、请求/响应、原子端点 | [reference.md](reference.md) |
| 记忆文本格式与占位符 | templates/remember-format.md |
| 各场景请求体示例 | examples/*.md |
