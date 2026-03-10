<p align="center">
  <img src="https://img.shields.io/github/stars/ngyygm/Temporal_Memory_Graph?style=for-the-badge&logo=github" alt="GitHub stars"/>
  <img src="https://img.shields.io/github/forks/ngyygm/Temporal_Memory_Graph?style=for-the-badge&logo=github" alt="GitHub forks"/>
  <img src="https://img.shields.io/github/license/ngyygm/Temporal_Memory_Graph?style=for-the-badge" alt="License"/>
  <img src="https://img.shields.io/badge/python-3.8+-blue?style=for-the-badge&logo=python" alt="Python"/>
</p>

<p align="center">
  <strong>Temporal Memory Graph (TMG)</strong>
</p>
<p align="center">
  <b>Long-term memory for AI agents</b> — store, recall, and traverse in time, the way humans do.
</p>

<p align="center">
  <a href="README.md">中文</a> · <a href="README.en.md">English</a> · <a href="README.ja.md">日本語</a>
</p>

---

## Overview

TMG gives AI agents **temporal, natural-language memory**: long-term storage and retrieval designed for agents, with human-like semantics (natural language in and out) and **time as a first-class citizen**—every memory is traceable, and entities and relations carry version chains. Experiences are written into a single unified graph; natural-language queries wake up relevant regions and support questions like “what happened then?”

| Focus | Description |
|-------|-------------|
| **Agent-oriented** | Built for agents: long-term memory read/write, not human-facing notes or knowledge bases. |
| **Human-like** | Natural language in, natural language out; no predefined tags; the system does concept extraction and relation building. |
| **Time as first-class** | Memories are timestamped; entities and relations have version chains and support time-range or point-in-time queries. |
| **Unified graph** | All memories live in one graph; semantic retrieval plus graph expansion returns “a region of related memory.” |

System boundary: TMG provides **Remember** (write) and **Find** (retrieve) only; **Select** (what to use, how to use it) is left to the caller.

### Compared to traditional knowledge graphs

| Aspect | Traditional KG | TMG |
|--------|----------------|-----|
| Relations | Fixed types (e.g. is_a, located_in) | Natural-language descriptions (concept edges) |
| Write | Structured input and schema | Raw text/documents; system extracts and aligns |
| Time | Static or simple timestamps | Version chains + timestamps; time-travel queries |
| Updates | Often overwrite | Append-only; full history kept |
| Retrieval | Structured queries, tag filters | Semantic search + graph neighborhood expansion |

---

## Architecture

```mermaid
flowchart TB
    subgraph Input["Input"]
        T[Text / Documents]
        F[File upload]
    end

    subgraph Pipeline["Memory pipeline"]
        W[Sliding window]
        M[Memory Agent]
        M --> M1[Update memory cache]
        M --> M2[Entity extraction]
        M --> M3[Relation extraction]
        M --> M4[Graph alignment]
        M --> M5[Versioned write]
    end

    subgraph Storage["Unified memory graph"]
        E[(Entity versions)]
        R[(Relation versions)]
        C[(MemoryCache)]
    end

    subgraph Find["Retrieval"]
        Q[Natural-language query]
        S[Semantic recall]
        G[Graph expansion]
        Tf[Time filter]
        Out[Local memory region]
    end

    T --> W
    F --> W
    W --> M
    M --> E
    M --> R
    M --> C
    Q --> S
    S --> G
    G --> Tf
    Tf --> Out
    E -.-> S
    R -.-> S
```

---

## Quick start

```bash
cp service_config.example.json service_config.json
# Edit service_config.json: LLM and embedding
python service_api.py --config service_config.json
```

**Remember:**

```bash
curl -s -X POST http://localhost:16200/api/remember \
  -H "Content-Type: application/json" \
  -d '{"text": "Lin Heihei is an archaeology PhD who met a talking white fox in a cave. The fox said it had guarded the cave for three hundred years."}' | jq
```

**Find:**

```bash
curl -s -X POST http://localhost:16200/api/find \
  -H "Content-Type: application/json" \
  -d '{"query": "What happened between Lin Heihei and the white fox?"}' | jq
```

---

## Using the Skill (agent integration)

TMG ships a **Skill** so that Cursor, Claude, and similar agents can deploy, configure, start, and call the API by following the documentation—no hand-written HTTP client.

### Where the Skill lives

- **Path:** `Temporal_Memory_Graph/skills/tmg-memory-graph/`
- **Files:** `SKILL.md` (agent instructions), `reference.md` (API quick reference)
- **Purpose:** Any agent that can “read docs and act” can use TMG after reading the Skill (when to use, how to deploy, how to call the API).

### Three steps to give an agent access

1. **Expose the Skill to the agent**  
   - **Cursor:** In rules, add “When using TMG memory, read and follow `Temporal_Memory_Graph/skills/tmg-memory-graph/SKILL.md`,” or copy key points into `.cursor/rules`.  
   - **Claude / others:** Add `skills/tmg-memory-graph/` to the agent’s skill directory or knowledge base.

2. **Trigger with natural language**  
   When the user says “remember this,” “look up what we knew about X,” or “connect to TMG memory,” the agent reads the Skill and runs the flow (check service → remember/find).

3. **What the agent will do**  
   - If the service is not running: clone repo → configure `service_config.json` → run `python service_api.py` → verify with `GET /health`.  
   - Remember: `POST /api/remember` (body `text` or `file_path`, or multipart upload).  
   - Find: `POST /api/find` with natural-language `query`; use entity/relation/version/subgraph endpoints when needed.

---

## API summary

### Remember — write

| Mode | Description |
|------|-------------|
| Text | JSON body: `{"text": "..."}` |
| Local file | JSON body: `{"file_path": "/path/to/file"}` (server path) |
| Upload | Multipart: `file=@/path/to/file` (txt / md / pdf / docx) |

Optional: `source_name`, `load_cache_memory`. Internally: chunking, memory cache update, entity/relation extraction, graph alignment, versioned write.

### Find — retrieve

- **Recommended:** `POST /api/find` — semantic recall, graph expansion, and time filtering in one call; required: `query`; rest optional.  
- **Atomic endpoints:** Entity search (`/api/find/entities/search`, etc.), relations, memory cache, subgraph create/expand/filter, stats (`/api/find/stats`).  

Full paths and parameters: see `skills/tmg-memory-graph/reference.md` and `service_api.py`.

### Response format

- Success: `{"success": true, "data": ..., "elapsed_ms": 123.45}`
- Error: `{"success": false, "error": "message", "elapsed_ms": 12.34}`

---

## Data model (brief)

- **Entity:** Concept entity; `entity_id` (logical), `id` (version absolute ID), `name`, `content` (natural language), `physical_time`; versions form a chain.  
- **Relation:** Concept relation; natural-language description (no fixed relation types); `entity1/2_absolute_id`, `physical_time`, version chain.  
- **MemoryCache:** Internal context summary chain for alignment and reasoning.  

All content is natural language + time; no predefined tag schema.

---

## Configuration

See `service_config.example.json`; configure `service_config.json` with:

- **Service:** `host`, `port`, `storage_path`  
- **LLM:** `api_key`, `model`, `base_url`, `think`  
- **Embedding:** `embedding.model` (local path or HuggingFace id), `embedding.device`  
- **Chunking:** `chunking.window_size`, `chunking.overlap`  
- **Subgraph:** `subgraph_max_count`, `subgraph_ttl_seconds`  

---

## License

See the [LICENSE](LICENSE) file in the repository root, if present.
