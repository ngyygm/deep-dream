<p align="center">
  <img src="https://img.shields.io/github/stars/ngyygm/deep-dream?style=for-the-badge&logo=github" alt="GitHub stars"/>
  <img src="https://img.shields.io/github/forks/ngyygm/deep-dream?style=for-the-badge&logo=github" alt="GitHub forks"/>
  <img src="https://img.shields.io/github/license/ngyygm/deep-dream?style=for-the-badge" alt="License"/>
  <img src="https://img.shields.io/badge/python-3.8+-blue?style=for-the-badge&logo=python" alt="Python"/>
  <img src="https://img.shields.io/badge/Neo4j-5.x-018BFF?style=for-the-badge&logo=neo4j" alt="Neo4j"/>
</p>

<p align="center">
  <img src="docs/images/logo.jpeg" alt="Deep Dream Logo" width="180"/>
</p>

<h1 align="center">🌊 Deep Dream</h1>

<p align="center">
  <em>Agent 的全生命周期记忆 — 像人一样记忆、回溯、做梦</em>
</p>

<p align="center">
  <a href="README.md">🇨🇳 中文</a> · <a href="README.en.md">🇬🇧 English</a> · <a href="README.ja.md">🇯🇵 日本語</a>
</p>

---

<p align="center">
  <img src="docs/images/hero.svg" alt="Deep Dream Hero" width="100%"/>
</p>

> 💤 **人一生三分之一的时间在睡眠中度过。**
>
> 这并非浪费。睡眠中，大脑在**重播**经历、**重组**碎片、**发现**清醒时来不及注意的隐藏联系。
> 每一次 REM 睡眠，都是一次自主的知识巩固——将散落片段编织成网络，将模糊直觉凝固为洞见。
>
> **Deep Dream 让 AI Agent 拥有了同样的能力。**

---

## ✨ 三大核心能力

<table>
<tr>
<td width="33%" align="center"><b>🧠 Remember</b><br/>清醒时写入</td>
<td width="33%" align="center"><b>🔍 Find</b><br/>需要时检索</td>
<td width="33%" align="center"><b>💭 Dream</b><br/>睡眠时巩固</td>
</tr>
<tr>
<td>

文本 → 实体<br/>
文档 → 关系<br/>
版本化写入

</td>
<td>

语义检索<br/>
图谱扩展<br/>
时间回溯

</td>
<td>

自主策略选择<br/>
工具调用循环<br/>
关系发现

</td>
</tr>
</table>

<p align="center">
  <img src="docs/images/architecture.jpeg" alt="Deep Dream Architecture" width="650"/>
</p>

---

## 🤔 为什么 Agent 需要做梦？

| 🧑 人类记忆 | 🤖 Deep Dream |
|:---:|:---:|
| 白天经历 → 写入记忆 | 文本/文档 → **Remember** 写入知识图谱 |
| 回忆往事 → 提取记忆 | 自然语言提问 → **Find** 语义检索 |
| 夜间睡眠 → 重组巩固 | Dream Agent → **DeepDream** 自主发现新关系 |

传统知识图谱是**静态**的——写入什么就是什么。但人类记忆不是这样工作的。DeepDream 赋予 Agent 同样的能力：

- 🌉 **跨越语义鸿沟** — 不仅发现相似实体，还能跨越巨大语义距离找到意想不到的连接
- 🦘 **跳跃性思维** — 像梦境中的自由联想，从概念 A 跳到看似无关的概念 B
- 🔄 **多策略做梦** — 8 种策略循环轮换，覆盖联想、对比、时间、跨域等不同维度
- ♾️ **永不停止** — 只要 Agent 在"睡眠"中，梦境就持续进行，无限迭代

> ⚠️ **关键约束：** Dream Agent 只能发现**已有实体间的新关系**，绝不编造不存在的实体。
> 就像人类在梦中重组已有记忆，而非凭空创造新人物。所有梦境发现带有 `source: dream` 标记。

---

## 🏗️ 核心架构

```
Remember（清醒时）          Find（需要时）           Dream（睡眠时）
┌──────────────┐      ┌──────────────┐      ┌────────────────────┐
│ 📝 文本→实体  │      │ 🔍 语义检索   │      │ 💭 Dream Agent     │
│ 📄 文档→关系  │      │ 🕸️ 图谱扩展   │      │   ├─ 策略选择       │
│ 📦 版本化写入 │      │ ⏳ 时间回溯   │      │   ├─ LLM 规划       │
│              │      │              │      │   ├─ 工具执行       │
└──────┬───────┘      └──────┬───────┘      │   ├─ 观察反思       │
       │                     │              │   └─ 关系保存       │
       ▼                     ▼              └────────┬───────────┘
   ┌───────────────────────────────────────────────────▼─────────┐
   │                 🧬 统一记忆知识图谱                           │
   │     Entity 版本链 · Relation 版本链 · Episode · Community    │
   └──────────────────────────────────────────────────────────────┘
```

Dream Agent 不是一个硬编码的循环——它是一个**自主智能体**，通过工具调用循环自主决定：
1. 📋 选择哪种策略获取种子实体
2. 🔭 需要遍历和观察哪些实体与关系
3. 💡 何时提出新的关系假设
4. 📝 何时记录梦境发现

---

## 🚀 快速开始

### 安装

```bash
git clone https://github.com/ngyygm/deep-dream.git
cd deep-dream
pip install -r requirements.txt
cp service_config.example.json service_config.json
# 编辑 service_config.json: 配置 LLM 和 Embedding
python -m server.api --config service_config.json
```

浏览器打开 **http://127.0.0.1:16200/** 即可看到管理面板 🎉

### 📝 写入记忆

```bash
curl -s -X POST http://localhost:16200/api/v1/remember \
  -H "Content-Type: application/json" \
  -d '{"text":"林嘿嘿是考古学博士，在山洞遇见了会说话的白狐。白狐说已守护山洞三百年。","event_time":"2026-03-09T14:00:00"}'
```

### 🔍 检索记忆

```bash
curl -s -X POST http://localhost:16200/api/v1/find \
  -H "Content-Type: application/json" \
  -d '{"query": "林嘿嘿和白狐之间发生了什么"}'
```

### 💭 启动梦境巩固

```bash
curl -s -X POST http://localhost:16200/api/v1/find/dream/agent/start \
  -H "Content-Type: application/json" \
  -d '{
    "max_cycles": 10,
    "strategies": ["free_association", "cross_domain", "leap"],
    "strategy_mode": "round_robin",
    "confidence_threshold": 0.6
  }'
```

---

## 🌈 8 种梦境策略

| 策略 | 🎭 类比 | 🎯 目标 |
|------|---------|---------|
| `free_association` | 🔗 自由联想 | 随机实体间寻找隐藏连接 |
| `contrastive` | ⚖️ 对比分析 | 相似实体间的差异与对比 |
| `temporal_bridge` | ⏳ 时间穿越 | 跨越时间长河发现演变规律 |
| `cross_domain` | 🌉 跨域灵感 | 不同领域间的意外桥梁 |
| `orphan_adoption` | 🏠 孤儿收容 | 为孤立实体寻找归属 |
| `hub_remix` | 🔀 枢纽重组 | 核心节点间的新路径发现 |
| `leap` | 🦘 思维跳跃 | 创造性的远距离联想 |
| `narrative` | 📖 故事编织 | 将零散片段串联成叙事线 |

---

## 🛠️ Dream Agent 工具箱

Dream Agent 通过 8 个工具与知识图谱交互，LLM 自主决定调用哪些工具：

| 工具 | 📌 用途 |
|------|---------|
| `get_seeds` | 按策略获取种子实体（起点） |
| `get_entity` | 查看实体详情及其直接关系 |
| `traverse` | BFS 多跳扩展，发现邻居 |
| `search_similar` | 语义相似度搜索 |
| `search_bm25` | BM25 关键词搜索 |
| `get_community` | 获取社区及其成员 |
| `create_relation` | 保存梦境发现的新关系 |
| `create_episode` | 记录梦境周期发现 |

---

## 📋 API 参考

### 梦境 Agent

```
POST /api/v1/find/dream/agent/start
```

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `max_cycles` | int | 10 | 梦境周期数（1-50） |
| `strategies` | string[] | `["free_association","cross_domain","leap"]` | 使用的策略列表 |
| `strategy_mode` | string | `"round_robin"` | 策略模式: `round_robin` / `random` / `adaptive` |
| `confidence_threshold` | float | 0.6 | 关系保存的最低置信度 |
| `max_tool_calls_per_cycle` | int | 15 | 每周期最大工具调用次数 |

### 记忆读写

| 端点 | 说明 |
|------|------|
| `POST /api/v1/remember` | 写入记忆（异步） |
| `POST /api/v1/find` | 统一语义检索 |
| `POST /api/v1/find/traverse` | BFS 图遍历 |
| `GET /api/v1/find/entities` | 实体列表/搜索 |
| `GET /api/v1/find/relations` | 关系列表/搜索 |
| `GET /api/v1/find/snapshot` | 时间旅行快照 |
| `POST /api/v1/find/ask` | Agent 元查询（自然语言） |

---

## ⚙️ 配置

参考 `service_config.example.json`，关键配置项：

| 配置 | 说明 |
|------|------|
| `host` / `port` | 服务地址，默认 `0.0.0.0:16200` |
| `storage.backend` | 存储后端: `"sqlite"` / `"neo4j"` |
| `llm` | LLM 配置（Ollama / OpenAI 兼容 / 智谱 GLM 等） |
| `embedding` | Embedding 模型（本地路径或 HuggingFace 名称） |
| `dream_llm` | 梦境专用 LLM（可单独配置轻量模型） |
| `chunking` | 滑窗大小和重叠 |
| `runtime.concurrency.*` | 三层并发控制 |

---

## 🔌 Agent 集成

Deep Dream 提供 Skill，使任何支持技能调用的 Agent（Cursor、Claude Code 等）能直接使用记忆和梦境功能：

- **Skill 名称**：`deep-dream`
- **路径**：`.claude/skills/deep-dream/`
- **触发词**：`"开始做梦"` / `"dream"` / `"深度复习"`
- **集成方式**：将 Skill 加入 Agent 的技能目录，Agent 即可自主 Remember、Find、Dream

---

## 🧪 技术栈

| 层 | 技术 |
|----|------|
| 图数据库 | Neo4j 5.x Community |
| 向量搜索 | sqlite-vec (ANN KNN) |
| LLM | OpenAI 兼容协议（GLM / Ollama / LM Studio） |
| Embedding | 本地模型 / HuggingFace |
| Web | Flask + 原生 SPA Dashboard |
| Agent 模式 | Tool-based Agent Loop（参考 claude-code-rev） |

---

## 📄 License

见仓库根目录 [LICENSE](LICENSE) 文件。
