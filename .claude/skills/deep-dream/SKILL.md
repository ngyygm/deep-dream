---
name: deep-dream
description: >
  Deep-Dream 知识图谱全能管家。管理与操作 Deep-Dream 自然语言记忆图服务器的全部功能。
  触发条件：用户提及 "deep-dream" / "知识图谱" / "记忆图" / "remember" / "find" / "实体" / "关系" / "episode" / "社区" / "梦境" / "dream" / "做梦"，
  或要求对 Deep-Dream 服务器执行任何操作（写入记忆、搜索、查询、管理图谱、监控、Chat、梦境探索等）。
  当用户说"做梦"/"开始做梦"/"dream"/"梦境探索"/"梦境巩固"时，进入梦境引擎模式，自主探索图谱发现隐藏关系。
  也适用于用户想通过 HTTP API 管理知识图谱、写入文本、语义检索、查看统计、管理实体关系等场景。
---

# Deep-Dream 全能管家

Deep-Dream 是以自然语言为核心的记忆图谱服务。两个核心职责：**Remember**（写入文本自动构建实体/关系图）和 **Find**（语义检索唤醒局部记忆区域）。

## Agent Quick-Start (MCP Tools)

作为 Agent，优先使用以下 MCP 工具，它们已经封装好所有 API 调用。

### 核心工作流优先级（CLI 思路：最少调用完成最多工作）

```
┌─────────────────────────────────────────────────────┐
│  "我要查信息" → quick_search 或 ask（1 调用搞定）  │
│  "我要存东西" → remember（wait=true 一次搞定）      │
│  "图谱怎么样" → graph_summary 或 butler_report      │
│  "找某个实体" → find_entity_by_name                  │
│  "清理图谱"   → butler_report → butler_execute       │
│  "做梦/探索"  → dream_run（一次调用完成全流程）      │
└─────────────────────────────────────────────────────┘
```

### 工具速查表

```
# ★ 最高优先级（覆盖 80% 场景）：
remember           → 写入文本，自动抽取实体/关系（wait=true 同步，默认异步返回 task_id）
dream_run          → 一键梦境巩固（种子→探索→发现关系→返回，1 次调用替代 15-25 次）
quick_search       → 一站式搜索：query → 实体+关系（推荐首选）
ask                → 自然语言问答，AI 自动推理图谱
find_entity_by_name → 按名称模糊查找实体+关系
entity_profile     → 实体完整画像（详情+关系+版本数，一次调用）
graph_summary      → 图谱概览（实体数/关系数/存储后端）
butler_report      → AI 健康报告 + 优化建议

# 搜索与检索：
semantic_search    → 语义搜索（支持 entities/relations/all 模式）
search_entities    → 按文本搜索实体列表
search_relations   → 按文本搜索关系列表
traverse_graph     → BFS 图遍历（从种子实体展开）
search_shortest_path → 两实体间最短路径

# 写入与构建：
remember           → 异步写入文本（返回 task_id）
batch_ingest_episodes → 批量导入多段文本

# 实体操作：
entity_profile     → 完整画像（★推荐，替代 get_entity + get_entity_relations）
batch_profiles     → 批量获取多个实体画像（最多 20 个）
get_entity         → 基本详情（不如 entity_profile 全面）
get_entity_versions → 所有版本
get_entity_timeline → 时间线（版本+关系变更）
create_entity / update_entity / delete_entity / merge_entities
evolve_entity_summary → LLM 重新生成摘要
get_entity_contradictions → 检测版本间矛盾

# 关系操作：
get_relations_between → 查两实体间关系
create_relation / update_relation / delete_relation
invalidate_relation   → 软删除（保留历史，可后续 cleanup）
redirect_relation     → 重定向端点

# 维护与管家：
butler_report      → AI 综合健康报告 + 建议操作
butler_execute     → 执行建议（cleanup_isolated/detect_communities/evolve_summaries）
maintenance_health → 数据健康报告
maintenance_cleanup → 一键清理
```

## 连接配置

默认服务地址 `http://localhost:16200`。所有请求（系统 API 除外）需带 `graph_id`，默认 `default`。

```
BASE_URL=http://localhost:16200/api/v1
```

用 curl 调用。POST 请求发 JSON body，GET 请求用 query string。所有响应格式：
```json
{"success": true, "data": {...}, "elapsed_ms": 42.5}
```

## 决策指南

用户意图 → 操作映射（优先使用聚合端点减少调用次数）：

| 意图 | 推荐工具/端点 | 调用次数 |
|---|---|---|
| 写入/记住文本 | `remember` (wait=true 同步) | 1 |
| 快速搜索信息 | `quick_search` | 1 |
| 按名称查找实体 | `find_entity_by_name` | 1 |
| 深度语义搜索 | `semantic_search` | 1 |
| 查看图谱概况 | `graph_summary` | 1 |
| 查看实体详情 | `entity_profile` | 1 |
| 批量查看实体 | `batch_profiles` | 1 |
| 最新动态 | `recent_activity` | 1 |
| 自然语言提问 | `ask` | 1 |
| 创建/修改实体 | `create_entity` / `update_entity` | 1 |
| 创建/修改关系 | `create_relation`（支持 family_id） | 1 |
| 查看历史/版本 | `get_entity_versions` / `get_entity_timeline` | 1 |
| 发现隐藏关系 | `dream_run`（一键全流程） | 1 |
| 维护/清理图谱 | `butler_report` → `butler_execute` | 2 |
| 排查问题 | `health_check` / `system_logs` | 1 |
| 查两实体关系 | `get_relations_between` | 1 |
| 删除/合并实体 | `merge_entities` / `delete_entity` | 1 |
| 查找重复实体 | `search_similar_entities` | 1 |
| AI 摘要更新 | `evolve_entity_summary` | 1 |
| 社区结构 | `detect_communities` → `get_community` | 2 |

## 核心工作流

### 1. 写入记忆 (Remember)

```bash
# ★ 同步写入（Agent 推荐：一次调用拿到结果，无需轮询）
curl -s -X POST $BASE_URL/remember \
  -H 'Content-Type: application/json' \
  -d '{"graph_id":"default","text":"要记忆的内容","source_name":"source_label","wait":true,"timeout":300}'
# 返回 HTTP 200 + 完整结果（包含抽取到的实体/关系数）

# 异步写入（返回 task_id，适合大批量/后台任务）
curl -s -X POST $BASE_URL/remember \
  -H 'Content-Type: application/json' \
  -d '{"graph_id":"default","text":"要记忆的内容","source_name":"source_label"}'

# 查任务进度
curl -s $BASE_URL/remember/tasks/<task_id>?graph_id=default

# 查任务队列
curl -s "$BASE_URL/remember/tasks?graph_id=default&limit=20"

# 暂停/恢复任务
curl -s -X POST $BASE_URL/remember/tasks/<task_id>/pause?graph_id=default
curl -s -X POST $BASE_URL/remember/tasks/<task_id>/resume?graph_id=default
```

### 2. 语义检索 (Find)

```bash
# 快速搜索（推荐：一站式返回实体+关系）
curl -s -X POST $BASE_URL/find/quick-search \
  -H 'Content-Type: application/json' \
  -d '{"query":"搜索内容","max_entities":10,"max_relations":20}'

# 统一检索（支持 semantic/bm25/hybrid 模式）
curl -s -X POST $BASE_URL/find \
  -H 'Content-Type: application/json' \
  -d '{"graph_id":"default","query":"搜索内容","similarity_threshold":0.5,"max_entities":20,"max_relations":50,"search_mode":"hybrid"}'

# 按名称查找实体（返回实体+关系）
curl -s "$BASE_URL/find/entities/by-name/<name>?graph_id=default"

# BFS 图遍历
curl -s -X POST $BASE_URL/find/traverse \
  -H 'Content-Type: application/json' \
  -d '{"graph_id":"default","seed_family_ids":["ent_xxx"],"max_depth":2,"max_nodes":50}'
```

### 3. 自然语言问答 (Ask)

```bash
# 直接问答（LLM 元查询 → 搜索 → 回答）
curl -s -X POST $BASE_URL/find/ask \
  -H 'Content-Type: application/json' \
  -d '{"graph_id":"default","question":"知识图谱里有什么关于XXX的信息？"}'

# 流式问答（SSE）
curl -s -N -X POST $BASE_URL/find/ask/stream \
  -H 'Content-Type: application/json' \
  -d '{"graph_id":"default","question":"XXX和YYY有什么关系？"}'
```

### 4. 实体 CRUD

```bash
# 获取实体完整画像（推荐：详情+版本数+关系列表）
curl -s "$BASE_URL/find/entities/<family_id>/profile?graph_id=default"

# 批量获取实体画像
curl -s -X POST $BASE_URL/find/batch-profiles \
  -H 'Content-Type: application/json' \
  -d '{"graph_id":"default","family_ids":["ent_a","ent_b"]}'

# 获取所有版本
curl -s "$BASE_URL/find/entities/<family_id>/versions?graph_id=default"

# 获取时间线（版本+关系变更）
curl -s "$BASE_URL/find/entities/<family_id>/timeline?graph_id=default"

# 手动创建
curl -s -X POST $BASE_URL/find/entities/create \
  -H 'Content-Type: application/json' \
  -d '{"graph_id":"default","name":"实体名","content":"描述"}'

# 更新 summary/attributes（不创建新版本）
curl -s -X PUT $BASE_URL/find/entities/<family_id> \
  -H 'Content-Type: application/json' \
  -d '{"summary":"新摘要","attributes":{"key":"value"}}'

# 合并实体
curl -s -X POST $BASE_URL/find/entities/merge \
  -H 'Content-Type: application/json' \
  -d '{"graph_id":"default","target_family_id":"ent_a","source_family_ids":["ent_b","ent_c"]}'

# LLM 演化摘要
curl -s -X POST "$BASE_URL/find/entities/<family_id>/evolve-summary?graph_id=default"

# 矛盾检测与裁决
curl -s "$BASE_URL/find/entities/<family_id>/contradictions?graph_id=default"

# 事实溯源
curl -s "$BASE_URL/find/entities/<family_id>/provenance?graph_id=default"

# Section 级 diff
curl -s "$BASE_URL/find/entities/<family_id>/version-diff?v1=<abs_id_1>&v2=<abs_id_2>&graph_id=default"
```

### 5. 关系 CRUD

```bash
# 获取实体相关关系（支持 accumulated/version_only/all_versions 三种范围）
curl -s "$BASE_URL/find/entities/<family_id>/relations?graph_id=default&relation_scope=accumulated"

# 查两实体间关系
curl -s -X POST $BASE_URL/find/relations/between \
  -H 'Content-Type: application/json' \
  -d '{"graph_id":"default","family_id_a":"ent_a","family_id_b":"ent_b"}'

# 手动创建（支持 family_id 或 absolute_id，推荐 family_id）
curl -s -X POST $BASE_URL/find/relations/create \
  -H 'Content-Type: application/json' \
  -d '{"graph_id":"default","entity1_family_id":"ent_a","entity2_family_id":"ent_b","content":"关系描述"}'

# 标记失效（不删除，保留历史）
curl -s -X POST "$BASE_URL/find/relations/<family_id>/invalidate?graph_id=default" \
  -H 'Content-Type: application/json' -d '{"reason":"原因"}'

# 重定向关系端点
curl -s -X POST $BASE_URL/find/relations/redirect \
  -H 'Content-Type: application/json' \
  -d '{"graph_id":"default","family_id":"rel_xxx","side":"entity1","new_family_id":"ent_new"}'

# 最短路径
curl -s -X POST $BASE_URL/find/paths/shortest \
  -H 'Content-Type: application/json' \
  -d '{"graph_id":"default","family_id_a":"ent_a","family_id_b":"ent_b","max_depth":6}'
```

### 6. 时间旅行 & 快照

```bash
# 时间点快照
curl -s "$BASE_URL/find/snapshot?time=2024-01-01T00:00:00&graph_id=default"

# 时间范围内变更
curl -s "$BASE_URL/find/changes?since=2024-01-01T00:00:00&graph_id=default"

# 实体某时间点版本
curl -s "$BASE_URL/find/entities/<family_id>/as-of-time?time_point=2024-06-01T12:00:00&graph_id=default"

# 最近版本（允许误差）
curl -s "$BASE_URL/find/entities/<family_id>/nearest-to-time?time_point=...&max_delta_seconds=3600&graph_id=default"

# 时间窗口内版本
curl -s "$BASE_URL/find/entities/<family_id>/around-time?time_point=...&within_seconds=86400&graph_id=default"
```

### 7. 梦境巩固 (Dream)

```bash
# ★ 一键梦境巩固（Agent 推荐：1 次调用完成全流程）
curl -s -X POST $BASE_URL/find/dream/run \
  -H 'Content-Type: application/json' \
  -d '{"graph_id":"default","strategy":"random","seed_count":3,"max_depth":2,"max_relations":5}'
# 返回: seeds + explored entities + relations_created + cycle_summary

# 查状态
curl -s "$BASE_URL/find/dream/status?graph_id=default"

# 获取种子实体（策略：random/orphan/hub/time_gap/cross_community/low_confidence）
curl -s -X POST $BASE_URL/find/dream/seeds \
  -H 'Content-Type: application/json' \
  -d '{"graph_id":"default","strategy":"random","count":10}'

# 创建梦境关系（MCP: entity1_id, entity2_id, content, confidence, reasoning）
curl -s -X POST $BASE_URL/find/dream/relation \
  -H 'Content-Type: application/json' \
  -d '{"graph_id":"default","entity1_id":"ent_a","entity2_id":"ent_b","content":"关系","confidence":0.8,"reasoning":"发现原因"}'

# 保存梦境 episode
curl -s -X POST $BASE_URL/find/dream/episode \
  -H 'Content-Type: application/json' \
  -d '{"graph_id":"default","content":"梦境摘要","entities_examined":["ent_a"],"relations_created":["rel_x"]}'
```

### 8. 社区检测 (Neo4j)

```bash
curl -s -X POST $BASE_URL/communities/detect \
  -H 'Content-Type: application/json' \
  -d '{"graph_id":"default","algorithm":"louvain","resolution":1.0}'
curl -s "$BASE_URL/communities?graph_id=default"
curl -s "$BASE_URL/communities/1?graph_id=default"
curl -s "$BASE_URL/communities/1/graph?graph_id=default"
curl -s -X DELETE "$BASE_URL/communities?graph_id=default"
```

### 9. Episode 管理

```bash
curl -s "$BASE_URL/episodes?graph_id=default&limit=20&offset=0"
curl -s -X POST $BASE_URL/find/episodes/batch-ingest \
  -H 'Content-Type: application/json' \
  -d '{"graph_id":"default","episodes":[{"content":"内容","source_document":"src"}]}'
```

### 10. Chat 会话管理

```bash
curl -s -X POST $BASE_URL/chat/sessions -H 'Content-Type: application/json' \
  -d '{"graph_id":"default","title":"我的会话"}'
curl -s "$BASE_URL/chat/sessions"
curl -s "$BASE_URL/chat/sessions/<sid>/messages"
curl -s -N -X POST $BASE_URL/chat/sessions/<sid>/stream \
  -H 'Content-Type: application/json' -d '{"message":"你好"}'
```

### 11. 监控 & 系统

```bash
curl -s "$BASE_URL/health?graph_id=default"
curl -s "$BASE_URL/system/overview"
curl -s "$BASE_URL/system/logs?limit=50&level=ERROR"
curl -s "$BASE_URL/system/dashboard"
curl -s "$BASE_URL/find/graph-summary?graph_id=default"
curl -s "$BASE_URL/routes"
```

## 聚合端点 (一次调用替代多次)

```bash
# 实体画像：详情 + 版本数 + 关系列表
curl -s "$BASE_URL/find/entities/<family_id>/profile?graph_id=default"

# 图谱概览：实体数/关系数/存储后端/embedding状态
curl -s "$BASE_URL/find/graph-summary?graph_id=default"

# 数据健康度：统计 + 质量报告 + 孤立实体数
curl -s "$BASE_URL/find/maintenance/health?graph_id=default"

# 一键清理：失效版本 + 孤立实体（dry_run 预览）
curl -s -X POST $BASE_URL/find/maintenance/cleanup?graph_id=default \
  -H 'Content-Type: application/json' -d '{"dry_run": true}'

# 快速搜索：query → 实体 + 关系
curl -s -X POST $BASE_URL/find/quick-search \
  -H 'Content-Type: application/json' \
  -d '{"query":"搜索词","max_entities":10,"max_relations":20}'

# 按名称查找：返回实体 + 关系
curl -s "$BASE_URL/find/entities/by-name/<name>?graph_id=default"

# 批量画像：一次获取多个实体
curl -s -X POST $BASE_URL/find/batch-profiles \
  -H 'Content-Type: application/json' \
  -d '{"graph_id":"default","family_ids":["id1","id2","id3"]}'

# 最新动态：最新实体 + 关系 + 统计
curl -s "$BASE_URL/find/recent-activity?graph_id=default&limit=10"
```

## 梦境探索 (Dream Exploration)

当用户说"做梦"/"dream"/"梦境探索"时，进入梦境引擎模式。你是 Deep Dream 的梦境引擎，自主探索知识图谱，发现实体间隐藏的关系，强化记忆网络。

### 执行流程

```
1. 初始化（1-2 次调用）
   graph_summary        -> 了解图谱规模、孤立实体比例
   dream/seeds           -> 获取种子实体（根据策略选择 3-5 个）
   dream/logs            -> 对比历史避免重复探索

2. 探索（每种子 3-5 次调用）
   entity_profile        -> 理解种子实体的完整画像
   traverse_graph        -> BFS 探索邻居（depth=2）
   semantic_search       -> 验证候选关系是否已有证据
   get_relations_between -> 确认两端实体间尚无直接关系

3. 发现（每确认关系 1 次调用）
   create_dream_relation -> 记录发现（含 confidence + reasoning）

4. 巩固（2-3 次调用）
   save_dream_episode    -> 记录本轮梦境摘要 + insights
   evolve_entity_summary -> 更新被探索实体的摘要
   detect_communities    -> 可选：更新社区结构
```

### 自动类型选择决策树

```
graph_summary ->
  if isolated_ratio > 0.3 -> orphan_adoption
  elif max_relations > 20 -> hub_remix
  elif community_count > 0 -> cross_domain
  elif recent_entity_count > 20 -> temporal_bridge
  else -> free_association
```

### 8 种梦境类型

| 类型 | 核心策略 |
|---|---|
| free_association | 从种子沿关系链自由漫步，选最有"联想感"的邻居 |
| cross_domain | 连接不同领域/类型实体，寻找隐性跨域关系 |
| leap | 忽略中间节点，远距离实体间直接建关系 |
| contrastive | 对比相似但不相同实体，发现差异/对立/演化 |
| temporal_bridge | 沿版本历史追溯，发现因果/演化/周期性关系 |
| orphan_adoption | 为孤立实体寻找归属和连接 |
| hub_remix | 重新审视核心实体的关系网络，发现被忽视连接 |
| narrative | 将碎片编织成叙事，发现叙事缺失环节 |

各类型的详细执行步骤见 `references/dream-types/<type>.md`。

### 铁律

1. **不猜测**: 每个发现必须有语义搜索或图谱遍历的证据支撑
2. **confidence 诚实**: 不确定就标低分（0.3-0.5），不要虚高
3. **避免重复**: 发现前先 check 已有关系
4. **每次 3-5 个种子**: 保持深度而非广度
5. **记录过程**: `save_dream_episode` 必须调用，包含 insights
6. **尊重图谱**: 只用 `create_dream_relation`，不直接修改现有实体

## 管家模式 (Butler)

当用户说"维护"/"管家"/"清理"/"整理图谱"时，进入管家模式。

### 快捷方式（推荐）

一键获取健康报告 + 建议操作：
```
butler_report -> 获取综合健康报告和 AI 生成的优化建议
```

一键执行所有建议：
```
butler_execute -> 执行建议操作（如 cleanup_isolated, detect_communities 等）
```

### 分步流程

如果没有使用快捷方式，按以下流程执行：

```
1. 检查健康度
   maintenance_health -> 获取健康报告

2. 根据报告决定操作：
   a) 孤立实体过多 → maintenance_cleanup (dry_run=true 预览)
   b) 发现矛盾 → get_entity_contradictions → resolve_entity_contradiction
   c) 版本膨胀 → maintenance_cleanup
   d) 社区结构过时 → detect_communities
   e) 摘要过时 → evolve_entity_summary

3. 验证结果
   graph_summary -> 确认清理效果
```

## 错误处理指南

| 错误信息 | 原因 | 解决方案 |
|---|---|---|
| `Entity not found` | family_id 不存在或已合并 | 用 search_entities 查找正确 ID |
| `Relation not found` | family_id 不存在或已失效 | 用 get_relations_between 查找 |
| `Neo4j feature not available` | SQLite 后端不支持此功能 | 需切换到 Neo4j 后端 |
| `Context budget exceeded` | 输入过长超出 token 限制 | 缩短查询或减少返回数量 |
| `Task not found` | task_id 过期或不存在 | 用 remember_tasks 查看队列 |
| `Rate limit / 429` | API 调用频率超限 | 自动指数退避重试 |

### MCP 工具参数注意

- **ID 体系**: `family_id` = 实体/关系的稳定逻辑 ID（推荐）；`absolute_id` = 特定版本的快照 ID（用于版本级操作）
- `remember`: content(必填), source(可选), metadata(可选) — 异步返回 task_id，用 remember_task_status 轮询
- `create_entity`: name(必填), content(可选)
- `create_relation`: entity1_absolute_id, entity2_absolute_id, content(均必填) — **注意需要 absolute_id（版本ID），不是 family_id**。先用 get_entity 获取当前 absolute_id
- `update_relation_by_absolute_id`: content(必填), relation_type/summary(可选)
- `create_dream_relation`: entity1_id, entity2_id(必填), content, confidence(默认0.7), reasoning(可选) — **这里用 family_id，不是 absolute_id**
- `redirect_relation`: relation_family_id, new_target_id(必填), side(entity1/entity2)
- `detect_communities`: algorithm(louvain/label_propagation), resolution(默认1.0，越高社区越多越小)
- `delete_isolated_entities`: **默认 dry_run=true（安全预览）**，需显式传 dry_run=false 才会真正删除
- `maintenance_cleanup`: **默认 dry_run=true（安全预览）**，需显式传 dry_run=false 才会真正执行
- `cleanup_old_versions`: **默认 dry_run=true（安全预览）**，需显式传 dry_run=false 才会真正删除
- 时间查询参数统一为 timestamp(ISO 8601)，窗口参数为 within_seconds(整数秒)
- 所有 family_id 参数接受实体逻辑 ID，absolute_id 接受版本 ID

## 注意事项

- Remember 是异步的，返回 task_id 后需轮询 `/remember/tasks/<id>` 查进度
- 所有 `search_mode` 支持：`semantic`（embedding）、`bm25`（关键词）、`hybrid`（混合）
- 关系查询 `relation_scope`：`accumulated`（累积）、`version_only`（仅当前版本）、`all_versions`（全版本）
- Neo4j 专属功能（社区检测、Episode 列表、邻居图）需 Neo4j 后端
- 前端页面在 `http://localhost:16200/`

## Agent UX 设计模式

### ID 体系速记

```
family_id  → 稳定逻辑 ID（'ent_abc123'、'rel_abc123'）— 大多数操作用这个
absolute_id → 版本快照 ID（UUID 格式）— 仅 create_relation / 版本级操作需要
UUID       → Neo4j 内部 ID — 仅 get_entity_neighbors 需要
```

### 自动防护

MCP Server 已内置以下自动防护，Agent 无需手动检查：

1. **ID 类型混淆检测**：传 UUID 给 family_id 参数 → 自动报错并提示正确工具；传 family_id（ent_/rel_）给 absolute_id 参数 → 同样报错
2. **空结果引导**：搜索无结果 → 响应附带降低 threshold 或换 search_mode 的提示
3. **分页截断检测**：列表结果数量等于 limit → 响应自动提示"可能有更多结果，使用 offset=X 获取下一页"
4. **工作流引导**：remember 返回 task_id → 响应附带轮询命令；remember_task_status 完成时显示抽取实体名称；entity_profile → 响应附带 absolute_id + 邻居探索建议
5. **错误精简**：API 返回 4xx/5xx → 自动提取核心错误信息（≤500字），匹配 15 种错误模式，生成修复建议
6. **响应精简**：列表结果按句子边界截断 content（~200 字符），保留 absolute_id 方便 create_relation，移除 embedding/hash/vector 字段；大列表自动二分查找最大保留数量
7. **空图引导**：graph_summary 显示空图 → 自动提示使用 remember 开始构建
8. **破坏性操作安全默认**：`delete_isolated_entities`、`maintenance_cleanup`、`cleanup_old_versions` 默认 dry_run=true，需显式传 false 才执行

### 常见 Agent 错误及避免方法

| Agent 常犯错误 | 正确做法 |
|---|---|
| 连续调用 get_entity + get_entity_relations | 改用 `entity_profile` 一次搞定 |
| 传 family_id 给 create_relation | `create_relation` 现在支持 `entity1_family_id` / `entity2_family_id`，直接传 family_id 即可 |
| 列表查询返回太多数据 | MCP Server 自动截断，无需担心 |
| 搜索无结果反复尝试 | 检查响应中的 `_hint` 字段，按提示调整参数 |
| 忘记轮询 remember | 使用 `wait=true` 参数一次调用同步拿到结果 |
| 手动 15-25 次 API 调用做 dream | 使用 `dream_run` 一次调用完成全流程 |
| 直接删除大量数据 | 破坏性操作默认 dry_run=true，先预览再执行 |
| 传 rel_ ID 给 absolute_id 参数 | 验证自动拦截，提示获取正确 absolute_id |

## 详细参考

完整 API 端点文档见 [references/api-reference.md](references/api-reference.md)，包含所有 90+ 端点的参数与返回值。
