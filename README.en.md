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
  <em>Full-lifecycle memory for AI agents — remember, recall, and dream like a human.</em>
</p>

<p align="center">
  <a href="README.md">🇨🇳 中文</a> · <a href="README.en.md">🇬🇧 English</a> · <a href="README.ja.md">🇯🇵 日本語</a>
</p>

---

<p align="center">
  <img src="docs/images/hero.svg" alt="Deep Dream Hero" width="100%"/>
</p>

> 💤 **Humans spend one-third of their lives asleep.**
>
> This is not wasted time. During sleep, the brain **replays** experiences, **reorganizes** fragments, and **discovers** hidden connections that waking consciousness never noticed.
> Every REM cycle is autonomous knowledge consolidation — weaving scattered fragments into networks, crystallizing vague intuitions into insight.
>
> **Deep Dream gives AI agents the same ability.**

---

## ✨ Three Core Capabilities

<table>
<tr>
<td width="33%" align="center"><b>🧠 Remember</b><br/>Write while awake</td>
<td width="33%" align="center"><b>🔍 Find</b><br/>Retrieve when needed</td>
<td width="33%" align="center"><b>💭 Dream</b><br/>Consolidate during sleep</td>
</tr>
<tr>
<td>

Text → Entities<br/>
Docs → Relations<br/>
Versioned writes

</td>
<td>

Semantic search<br/>
Graph expansion<br/>
Time travel

</td>
<td>

Autonomous strategy<br/>
Tool-calling loop<br/>
Relation discovery

</td>
</tr>
</table>

<p align="center">
  <img src="docs/images/architecture.jpeg" alt="Deep Dream Architecture" width="650"/>
</p>

---

## 🤔 Why do agents need to dream?

| 🧑 Human Memory | 🤖 Deep Dream |
|:---:|:---:|
| Daily experience → encode memory | Text/documents → **Remember** into knowledge graph |
| Recall the past → retrieve memory | Natural-language query → **Find** via semantic search |
| Nightly sleep → consolidate & reorganize | Dream Agent → **DeepDream** discovers new relations |

Traditional knowledge graphs are **static** — what you write is what you get. Human memory doesn't work that way. DeepDream gives agents the same capability:

- 🌉 **Beyond nearest neighbors** — Discovers connections across vast semantic distances
- 🦘 **Associative leaps** — Free-association jumps between seemingly unrelated concepts
- 🔄 **Multi-strategy dreaming** — 8 strategies rotate to cover association, contrast, time, cross-domain, and more
- ♾️ **Never stops** — As long as the agent is "asleep," dreaming continues infinitely

> ⚠️ **Key constraint:** The Dream Agent can only discover new relations between **existing entities — never fabricate entities.** Like humans recombining existing memories in dreams. All dream discoveries carry `source: dream` provenance markers.

---

## 🏗️ Architecture

```
Remember (awake)            Find (when needed)        Dream (asleep)
┌──────────────┐      ┌──────────────┐      ┌────────────────────┐
│ 📝 Text→Entity│      │ 🔍 Semantic   │      │ 💭 Dream Agent     │
│ 📄 Docs→Rel.  │      │ 🕸️ Graph expand│      │   ├─ Strategy      │
│ 📦 Versioned  │      │ ⏳ Time travel │      │   ├─ LLM planning  │
│   write       │      │               │      │   ├─ Tool exec     │
└──────┬───────┘      └──────┬───────┘      │   ├─ Observe       │
       │                     │              │   └─ Save relations│
       ▼                     ▼              └────────┬───────────┘
   ┌───────────────────────────────────────────────────▼─────────┐
   │                  🧬 Unified Memory Knowledge Graph           │
   │    Entity versions · Relation versions · Episode · Community │
   └──────────────────────────────────────────────────────────────┘
```

The Dream Agent is not a hardcoded loop — it is an **autonomous agent** that decides through tool-calling loops:
1. 📋 Which strategy to use for seed entity selection
2. 🔭 Which entities and relations to traverse and observe
3. 💡 When to propose new relation hypotheses
4. 📝 When to record dream discoveries

---

## 🚀 Quick Start

### Installation

```bash
git clone --recurse-submodules https://github.com/ngyygm/deep-dream.git
cd deep-dream
pip install -r requirements.txt
cp service_config.example.json service_config.json
# Edit service_config.json: configure LLM and Embedding
```

### Install Claude Code CLI (required for chat)

Chat functionality is provided through the Claude Code CLI. Ensure Claude Code (v2.1.90+) is installed:

```bash
# Verify Claude Code is installed
claude --version
```

Claude Code automatically gains Deep Dream MCP tools via `--mcp-config`, no additional configuration needed.

### Start the server

```bash
python -m server.api --config service_config.json
```

Open **http://127.0.0.1:16200/** in your browser for the management dashboard 🎉

### 📝 Write memory

```bash
curl -s -X POST http://localhost:16200/api/v1/remember \
  -H "Content-Type: application/json" \
  -d '{"text":"Lin Heihei is an archaeology PhD who met a talking white fox in a cave. The fox said it has guarded the cave for 300 years.","event_time":"2026-03-09T14:00:00"}'
```

### 🔍 Retrieve memory

```bash
curl -s -X POST http://localhost:16200/api/v1/find \
  -H "Content-Type: application/json" \
  -d '{"query": "What happened between Lin Heihei and the white fox?"}'
```

### 💭 Start dream consolidation

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

## 🌈 8 Dream Strategies

| Strategy | 🎭 Analogy | 🎯 Goal |
|----------|-----------|---------|
| `free_association` | 🔗 Free association | Find hidden connections between random entities |
| `contrastive` | ⚖️ Contrastive analysis | Discover differences between similar entities |
| `temporal_bridge` | ⏳ Time travel | Find evolution patterns across time |
| `cross_domain` | 🌉 Cross-domain insight | Unexpected bridges between different fields |
| `orphan_adoption` | 🏠 Orphan rescue | Find connections for isolated entities |
| `hub_remix` | 🔀 Hub recombination | New paths between core nodes |
| `leap` | 🦘 Creative leap | Far-distance associative jumps |
| `narrative` | 📖 Story weaving | Weave scattered fragments into narrative threads |

---

## 🛠️ Dream Agent Toolbox

The Dream Agent interacts with the knowledge graph through 8 tools. The LLM autonomously decides which tools to call:

| Tool | 📌 Purpose |
|------|-----------|
| `get_seeds` | Get seed entities by strategy (starting points) |
| `get_entity` | View entity details and direct relations |
| `traverse` | BFS multi-hop expansion to discover neighbors |
| `search_similar` | Semantic similarity search |
| `search_bm25` | BM25 keyword search |
| `get_community` | Get community and its members |
| `create_relation` | Save a dream-discovered relation |
| `create_episode` | Record dream cycle discoveries |

---

## 📋 API Reference

### Dream Agent

```
POST /api/v1/find/dream/agent/start
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `max_cycles` | int | 10 | Number of dream cycles (1-50) |
| `strategies` | string[] | `["free_association","cross_domain","leap"]` | Strategies to use |
| `strategy_mode` | string | `"round_robin"` | Mode: `round_robin` / `random` / `adaptive` |
| `confidence_threshold` | float | 0.6 | Minimum confidence for saving relations |
| `max_tool_calls_per_cycle` | int | 15 | Max tool calls per cycle |

### Memory Operations

| Endpoint | Description |
|----------|-------------|
| `POST /api/v1/remember` | Write memory (async) |
| `POST /api/v1/find` | Unified semantic search |
| `POST /api/v1/find/traverse` | BFS graph traversal |
| `GET /api/v1/find/entities` | Entity list/search |
| `GET /api/v1/find/relations` | Relation list/search |
| `GET /api/v1/find/snapshot` | Time-travel snapshot |
| `POST /api/v1/find/ask` | Agent meta-query (natural language) |

---

## ⚙️ Configuration

See `service_config.example.json` for details:

| Config | Description |
|--------|-------------|
| `host` / `port` | Service address, default `0.0.0.0:16200` |
| `storage.backend` | Backend: `"sqlite"` / `"neo4j"` |
| `llm` | LLM config (Ollama / OpenAI-compatible / GLM etc.) |
| `embedding` | Embedding model (local path or HuggingFace name) |
| `dream_llm` | Dedicated LLM for dreaming (can use a lighter model) |
| `chunking` | Sliding window size and overlap |
| `runtime.concurrency.*` | Three-tier concurrency control |

---

## 🔌 Agent Integration

Deep Dream ships a Skill so any agent that supports skill invocation (Cursor, Claude Code, etc.) can directly use memory and dream capabilities:

- **Skill name**: `deep-dream`
- **Path**: `.claude/skills/deep-dream/`
- **Triggers**: `"开始做梦"` / `"dream"` / `"深度复习"`
- **Integration**: Add the Skill to the agent's skill directory

---

## 🧪 Tech Stack

| Layer | Technology |
|-------|-----------|
| Graph Database | Neo4j 5.x Community |
| Vector Search | sqlite-vec (ANN KNN) |
| LLM | OpenAI-compatible protocol (GLM / Ollama / LM Studio) |
| Embedding | Local model / HuggingFace |
| Web | Flask + native SPA Dashboard |
| Agent Pattern | Tool-based Agent Loop (powered by Claude Code + MCP) |

---

## 📄 License

See [LICENSE](LICENSE) in the repository root.
