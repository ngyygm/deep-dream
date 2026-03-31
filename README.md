<p align="center">
  <img src="https://img.shields.io/github/stars/ngyygm/deep-dream?style=for-the-badge&logo=github" alt="GitHub stars"/>
  <img src="https://img.shields.io/github/forks/ngyygm/deep-dream?style=for-the-badge&logo=github" alt="GitHub forks"/>
  <img src="https://img.shields.io/github/license/ngyygm/deep-dream?style=for-the-badge" alt="License"/>
  <img src="https://img.shields.io/badge/python-3.8+-blue?style=for-the-badge&logo=python" alt="Python"/>
</p>

<p align="center">
  <strong>Deep Dream</strong>
</p>
<p align="center">
  Agent 的全生命周期记忆 — 像人一样记忆、回溯、做梦。
</p>

<p align="center">
  <img src="docs/images/logo.jpeg" alt="Deep Dream Logo" width="200"/>
</p>

<p align="center">
  <a href="README.md">中文</a> · <a href="README.en.md">English</a> · <a href="README.ja.md">日本語</a>
</p>

---

## 人一生三分之一的时间在睡眠中度过

这并非浪费。睡眠中，大脑并没有闲着——它在**重播**白天的经历，**重组**记忆碎片，**发现**那些清醒时来不及注意的隐藏联系。每一次快速眼动（REM）睡眠，都是一次自主的知识巩固：将散落的片段编织成网络，将模糊的直觉凝固为洞见。

**Deep Dream 让 AI Agent 拥有了同样的能力。**

Deep Dream 是一个面向 Agent 的长期记忆系统。Agent 在清醒时写入记忆（Remember），在需要时检索记忆（Find）。而当 Agent 进入"睡眠"状态，**DeepDream 自主梦境巩固**启动——一个自主的 Dream Agent 无限循环地遍历知识图谱，像大脑在夜间的自由联想一样，不断发现实体间的隐藏关系、构建新的概念桥梁。

---

## 为什么 Agent 需要做梦？

| 人类记忆 | Deep Dream |
|----------|-----------|
| 白天经历 → 写入记忆 | 文本/文档 → **Remember** 写入知识图谱 |
| 回忆往事 → 提取记忆 | 自然语言提问 → **Find** 语义检索 |
| 夜间睡眠 → 重组巩固 | Dream Agent → **DeepDream** 自主发现新关系 |

传统的知识图谱是**静态**的——写入什么就是什么。但人类的记忆不是这样工作的。我们在梦中重新连接记忆碎片，发现清醒时无法注意到的模式。DeepDream 赋予 Agent 同样的能力：

- **不局限于最相近的** — 不仅发现相似实体间的关系，还能跨越巨大语义距离，找到意想不到的连接
- **跳跃性思维** — 像梦境中的自由联想，从一个概念跳到看似无关的另一个概念
- **多策略做梦** — 自由联想、对比分析、时间桥接、跨域发现、孤立收养、枢纽混搭……多种策略循环轮换
- **永不停止** — 只要 Agent 在"睡眠"中，梦境就持续进行，无限次迭代

### 关键约束

Dream Agent 在做梦时严格遵守一条规则：**只能发现已有实体间的新关系，绝不编造不存在的实体。** 这就像人类在梦中重组已有记忆，而非凭空创造全新的人物和事件。所有梦境发现都带有明确的来源标记（`source: dream`），与清醒时写入的记忆清晰区分。

---

## 核心架构

Deep Dream 提供三层能力：

<p align="center">
  <img src="docs/images/architecture.jpeg" alt="Deep Dream Architecture" width="600"/>
</p>

```
Remember（清醒时）          Find（需要时）           Dream（睡眠时）
┌──────────────┐      ┌──────────────┐      ┌────────────────────┐
│ 文本 → 实体   │      │ 语义检索      │      │ Dream Agent        │
│ 文档 → 关系   │      │ 图谱扩展      │      │  ├─ 策略选择        │
│ 版本化写入    │      │ 时间回溯      │      │  ├─ 工具调用        │
│              │      │              │      │  ├─ 关系发现        │
└──────┬───────┘      └──────┬───────┘      │  └─ 无限迭代        │
       │                     │              └────────┬───────────┘
       ▼                     ▼                       ▼
   ┌───────────────────────────────────────────────────────┐
   │                统一记忆知识图谱                         │
   │         Entity 版本链 · Relation 版本链 · Episode      │
   └───────────────────────────────────────────────────────┘
```

Dream Agent 不是一个硬编码的循环——它是一个**自主智能体**，通过 skill 接入 Deep Dream 的 API，像使用工具一样自主决定：
- 什么时候选择哪种做梦策略
- 需要遍历哪些实体和关系
- 什么时候提出新的关系假设
- 什么时候记录梦境发现

---

## 快速开始

```bash
git clone https://github.com/ngyygm/deep-dream.git
cd deep-dream
pip install -r requirements.txt
cp service_config.example.json service_config.json
# 编辑 service_config.json: 配置 LLM 和 Embedding
python -m server.api --config service_config.json
```

浏览器访问 **http://127.0.0.1:16200/** 打开管理面板。

### 写入记忆

```bash
curl -s -X POST http://localhost:16200/api/v1/remember \
  -H "Content-Type: application/json" \
  -d '{"text":"林嘿嘿是考古学博士，在山洞遇见了会说话的白狐。白狐说已守护山洞三百年。","event_time":"2026-03-09T14:00:00"}'
```

### 检索记忆

```bash
curl -s -X POST http://localhost:16200/api/v1/find \
  -H "Content-Type: application/json" \
  -d '{"query": "林嘿嘿和白狐之间发生了什么"}'
```

### 启动梦境巩固

```bash
curl -s -X POST http://localhost:16200/api/v1/find/dream/agent/start \
  -H "Content-Type: application/json" \
  -d '{"max_cycles": 10, "strategies": ["free_association", "cross_domain", "leap"]}'
```

---

## Dream Agent 的做梦策略

| 策略 | 类比 | 目标 |
|------|------|------|
| `free_association` | 自由联想 | 随机实体间寻找隐藏连接 |
| `contrastive` | 对比分析 | 相似实体间的差异与对比 |
| `temporal_bridge` | 时间穿越 | 跨越时间长河发现演变规律 |
| `cross_domain` | 跨域灵感 | 不同领域间的意外桥梁 |
| `orphan_adoption` | 孤儿收容 | 为孤立实体寻找归属 |
| `hub_remix` | 枢纽重组 | 核心节点间的新路径发现 |
| `leap` | 思维跳跃 | 创造性的远距离联想 |
| `narrative` | 故事编织 | 将零散片段串联成叙事线 |

---

## 配置参考

参考 `service_config.example.json`，关键配置项：

| 配置 | 说明 |
|------|------|
| `host` / `port` | 服务地址，默认 `0.0.0.0:16200` |
| `llm` | LLM 配置（Ollama / OpenAI 兼容 / 智谱 GLM 等） |
| `embedding` | Embedding 模型（本地路径或 HuggingFace 名称） |
| `chunking` | 滑窗大小和重叠 |
| `runtime.concurrency.*` | 三层并发控制 |

---

## Agent 集成

Deep Dream 提供 Skill，使任何支持技能调用的 Agent（Cursor、Claude Code 等）能直接使用记忆和梦境功能：

- **Skill 名称**：`deep-dream`（通过 skill-creator 创建）
- **路径**：`.claude/skills/deep-dream/`
- **集成方式**：将 Skill 加入 Agent 的技能目录，Agent 即可自主 Remember、Find、Dream

---

## License

见仓库根目录 [LICENSE](LICENSE) 文件。
