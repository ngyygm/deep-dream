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
  <b>为 Agent 设计的长期记忆系统</b> —— 像人类一样存、取、回溯。
</p>

<p align="center">
  <a href="README.md">中文</a> · <a href="README.en.md">English</a> · <a href="README.ja.md">日本語</a>
</p>

---

## 简介

TMG 让 AI Agent 拥有**带时间的自然语言记忆**：专门为 Agent 提供**长期存取记忆**能力，**像人类一样**用自然语言记忆与回忆，并**将时间作为一等公民**——每条记忆可追溯，实体与关系带版本链。经历被写入一张统一知识图，用自然语言提问即可唤醒相关片段，并支持「那时发生了什么」式的时间回溯。

| 定位 | 说明 |
|------|------|
| **面向 Agent** | 为智能体提供长期记忆存储与检索，而非面向人类的笔记或知识库。 |
| **像人类一样** | 以自然语言写入与查询，不依赖预定义标签；由系统完成概念抽取与关系构建。 |
| **时间是一等公民** | 记忆带时间戳，实体/关系具备版本链，支持按时间范围或时间点回溯。 |
| **统一记忆图** | 所有记忆写入同一张图，通过语义检索与图谱扩展召回「一片相关记忆」。 |

系统职责边界：仅提供 **Remember**（写入）与 **Find**（检索）；**Select**（筛选与决策）由调用方完成。

### 与传统知识图谱的对比

| 维度 | 传统知识图谱 | TMG |
|------|--------------|-----|
| 关系表示 | 固定关系类型（如 is_a, located_in） | 自然语言描述（概念边） |
| 写入方式 | 需结构化输入与 schema | 直接输入文本/文档，系统自动抽取与对齐 |
| 时间模型 | 多为静态或简单时间戳 | 版本链 + 时间戳，支持按时间回溯 |
| 更新策略 | 多为覆盖更新 | 追加式更新，保留完整历史 |
| 检索方式 | 结构化查询、标签过滤 | 语义检索 + 图谱邻域扩展 |

---

## 系统架构

```mermaid
flowchart TB
    subgraph Input["输入层"]
        T[文本 / 文档]
        F[文件上传]
    end

    subgraph Pipeline["记忆流水线"]
        W[滑窗切片]
        M[Memory Agent]
        M --> M1[更新记忆缓存]
        M --> M2[概念实体抽取]
        M --> M3[概念关系抽取]
        M --> M4[图谱语义对齐]
        M --> M5[版本化写入]
    end

    subgraph Storage["统一记忆图"]
        E[(Entity 版本链)]
        R[(Relation 版本链)]
        C[(MemoryCache)]
    end

    subgraph Find["检索层"]
        Q[自然语言查询]
        S[语义召回]
        G[图谱扩展]
        Tf[时间过滤]
        Out[局部记忆区域]
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

## 快速开始

```bash
cp service_config.example.json service_config.json
# 编辑 service_config.json：配置 LLM 与 embedding
python service_api.py --config service_config.json
```

**写入记忆：**

```bash
curl -s -X POST http://localhost:16200/api/remember \
  -H "Content-Type: application/json" \
  -d '{"text": "林嘿嘿是考古学博士，在山洞遇见了会说话的白狐。白狐说已守护山洞三百年。"}' | jq
```

**检索记忆：**

```bash
curl -s -X POST http://localhost:16200/api/find \
  -H "Content-Type: application/json" \
  -d '{"query": "林嘿嘿和白狐之间发生了什么"}' | jq
```

---

## 使用 Skill（Agent 集成）

TMG 提供 **Skill**，使 Cursor、Claude 等 Agent 能够按文档完成部署、配置、启动及 API 调用，无需手写 HTTP 客户端。

### Skill 位置与内容

- **路径**：`Temporal_Memory_Graph/skills/tmg-memory-graph/`
- **文件**：`SKILL.md`（Agent 行为说明）、`reference.md`（接口速查）
- **作用**：支持「按文档执行」的 Agent 在阅读 SKILL 后即可完成何时调用 TMG、如何部署、如何调用 API。

### 三步让 Agent 使用 TMG

1. **暴露 Skill 给 Agent**  
   - **Cursor**：在规则中注明「使用 TMG 记忆时，请阅读并遵循 `Temporal_Memory_Graph/skills/tmg-memory-graph/SKILL.md`」，或将要点写入 `.cursor/rules`。  
   - **Claude / 其他**：将 `skills/tmg-memory-graph/` 加入该 Agent 的技能目录或知识库。

2. **通过自然语言触发**  
   当用户表达「把这件事记下来」「查一下之前关于某某的记忆」「对接 TMG 记忆服务」时，Agent 会读取 SKILL 并执行相应流程（检查服务状态 → 执行 remember/find）。

3. **Agent 将执行的操作**  
   - 若服务未就绪：克隆仓库 → 配置 `service_config.json` → 启动 `python service_api.py` → 使用 `GET /health` 确认。  
   - 写入：`POST /api/remember`（文本用 `text`，文件用 `file_path` 或 multipart 上传）。  
   - 检索：`POST /api/find` 传入自然语言 `query`；需要时可使用实体/关系/版本/子图等原子接口。

---

## API 概览

### Remember — 记忆写入

| 方式 | 说明 |
|------|------|
| 文本 | JSON body：`{"text": "..."}` |
| 本地文件 | JSON body：`{"file_path": "/path/to/file"}`（服务端路径） |
| 上传文件 | multipart：`file=@/path/to/file`（支持 txt / md / pdf / docx） |

可选参数：`source_name`、`load_cache_memory`。内部完成切片、记忆缓存更新、实体/关系抽取、图谱对齐与版本化写入。

### Find — 语义检索

- **推荐**：`POST /api/find`，单请求完成语义召回、图谱扩展与时间过滤；必填参数为 `query`，其余可选。  
- **原子接口**：实体检索（`/api/find/entities/search` 等）、关系检索、记忆缓存、子图创建/扩展/过滤、统计（`/api/find/stats`）等。  

完整路径与参数见 `skills/tmg-memory-graph/reference.md` 及 `service_api.py`。

### 响应格式

- 成功：`{"success": true, "data": ..., "elapsed_ms": 123.45}`
- 失败：`{"success": false, "error": "错误信息", "elapsed_ms": 12.34}`

---

## 数据模型简述

- **Entity**：概念实体；含 `entity_id`（逻辑 ID）、`id`（版本绝对 ID）、`name`、`content`（自然语言）、`physical_time`；多版本形成版本链。  
- **Relation**：概念关系；以自然语言描述（非固定关系类型），含 `entity1/2_absolute_id`、`physical_time` 及版本链。  
- **MemoryCache**：系统内部上下文摘要链，用于对齐与推理。  

全量内容为自然语言 + 时间；无预定义标签体系。

---

## 配置

参考 `service_config.example.json` 配置 `service_config.json`：

- **服务**：`host`、`port`、`storage_path`  
- **LLM**：`api_key`、`model`、`base_url`、`think`  
- **Embedding**：`embedding.model`（本地路径或 HuggingFace 模型名）、`embedding.device`  
- **分块**：`chunking.window_size`、`chunking.overlap`  
- **子图**：`subgraph_max_count`、`subgraph_ttl_seconds`  

---

## License

见仓库根目录 [LICENSE](LICENSE) 文件（如有）。
