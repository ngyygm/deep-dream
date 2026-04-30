# Deep-Dream API 完整参考

所有端点前缀 `/api/v1/`，旧路径 `/api/...` 自动 308 重定向。除系统 API 外需带 `graph_id`（默认 `default`）。

响应格式：`{"success": true/false, "data": {...}, "elapsed_ms": 42.5}`

---

## 目录

1. [健康检查](#1-健康检查)
2. [Remember — 记忆写入](#2-remember--记忆写入)
3. [Find — 统一检索](#3-find--统一检索)
4. [实体接口](#4-实体接口)
5. [关系接口](#5-关系接口)
6. [路径搜索](#6-路径搜索)
7. [记忆缓存](#7-记忆缓存)
8. [Episode 管理](#8-episode-管理)
9. [时间旅行](#9-时间旅行)
10. [梦境巩固](#10-梦境巩固)
11. [Agent 智能接口](#11-agent-智能接口)
12. [社区检测 (Neo4j)](#12-社区检测-neo4j)
13. [Episode (Neo4j)](#13-episode-neo4j)
14. [实体邻居 (Neo4j)](#14-实体邻居-neo4j)
15. [Concepts 统一概念查询](#15-concepts-统一概念查询)
16. [图谱管理](#16-图谱管理)
17. [文档管理](#17-文档管理)
18. [Chat 会话](#18-chat-会话)
19. [系统监控](#19-系统监控)

---

## 1. 健康检查

### GET /health
服务健康检查。

**Query**: `graph_id` (必填)

**返回**: `{graph_id, storage_backend, storage_path, embedding_available}`

### GET /health/llm
LLM 连通性检查（实际发推理请求）。

**Query**: `graph_id` (必填)

**返回**: `{graph_id, llm_available, message, response_preview}`

---

## 2. Remember — 记忆写入

### POST /remember
提交异步记忆写入任务。返回 202 + task_id。

**Body**:
| 字段 | 类型 | 必填 | 说明 |
|---|---|---|---|
| graph_id | string | 是 | 目标图谱 |
| text | string | 二选一 | 正文文本 |
| file | multipart | 二选一 | 上传文件 |
| file_path | string | 否 | 服务端文件路径 |
| source_name | string | 否 | 来源名，默认 api_input |
| doc_name | string | 否 | 兼容旧字段 |
| source_document | string | 否 | 新字段（优先） |
| load_cache_memory | bool | 否 | true=接续缓存链（串行），false=独立（可并行） |
| event_time | ISO8601 | 否 | 事件时间 |

**返回**: `{task_id, status:"queued", message, original_path}`

### GET /remember/tasks/<task_id>
查询任务状态。

**返回**: `{task_id, status, result?, error?, original_path}`

status: queued / running / completed / failed / paused

### DELETE /remember/tasks/<task_id>
删除任务。

### POST /remember/tasks/<task_id>/pause
暂停任务。

### POST /remember/tasks/<task_id>/resume
恢复任务。

### GET /remember/tasks
列出任务队列。

**Query**: `graph_id, limit` (默认50)

**返回**: `{tasks: [...], count}`

### GET /remember/monitor
实时监控快照。

**Query**: `graph_id, limit` (默认6)

**返回**: `{graph_id, storage, queue, threads}`

---

## 3. Find — 统一检索

### POST /find
统一语义检索入口（最常用）。

**Body**:
| 字段 | 类型 | 默认 | 说明 |
|---|---|---|---|
| graph_id | string | 必填 | |
| query | string | 必填 | 自然语言查询 |
| similarity_threshold | float | 0.5 | 语义相似度阈值 |
| max_entities | int | 20 | 最大实体数 |
| max_relations | int | 50 | 最大关系数 |
| expand | bool | true | 向外扩展邻域 |
| search_mode | string | semantic | semantic/bm25/hybrid |
| reranker | string | rrf | 重排序策略（rrf/node_degree） |
| time_before | ISO8601 | - | 时间上界 |
| time_after | ISO8601 | - | 时间下界 |

**返回**: `{query, entities: [...], relations: [...], entity_count, relation_count}`

### POST /find/candidates
按条件返回候选实体与关系。

**Body**: `graph_id, query_text, entity_name, similarity_threshold, max_entities, max_relations, time_before, time_after, include_entities, include_relations`

**返回**: `{entities: [...], relations: [...]}`

### GET /find/stats
图谱统计。

**返回**: `{total_entities, total_relations, total_memory_caches, total_episodes, total_communities}`

### GET /stats/counts
简化计数。

**返回**: `{entity_count, relation_count}`

### GET /find/graph-stats
图谱结构统计。

---

## 4. 实体接口

### GET /find/entities
列出实体。

**Query**: `graph_id, limit, offset`

### GET /find/entities/as-of-time
指定时间点的最新版本。

**Query**: `graph_id, time_point(必填), limit`

### GET/POST /find/entities/search
搜索实体。

**参数**:
| 字段 | 默认 | 说明 |
|---|---|---|
| query_name | 必填 | 搜索文本 |
| query_content | - | 内容搜索 |
| similarity_threshold | 0.5 | |
| max_results | 10 | |
| text_mode | name_and_content | name_only/content_only/name_and_content |
| similarity_method | embedding | embedding/text/jaccard/bleu |
| search_mode | semantic | semantic/bm25/hybrid |
| content_snippet_length | 50 | |

### GET /find/entities/<family_id>
按 family_id 获取最新版本。

### PUT /find/entities/<family_id>
更新实体。支持两种模式：
- `{summary, attributes}` → 直接更新（不创建版本）
- `{name, content}` → 创建新版本

### DELETE /find/entities/<family_id>
删除所有版本。

**Query**: `cascade` (bool, 默认 false)

### GET /find/entities/<family_id>/versions
所有版本列表。

### GET /find/entities/<family_id>/version-count
版本数量。

### GET /find/entities/<family_id>/as-of-time
某时间点最近过去版本。

**Query**: `time_point` (必填)

### GET /find/entities/<family_id>/nearest-to-time
距时间点最近版本。

**Query**: `time_point` (必填), `max_delta_seconds` (可选)

**返回**: `{family_id, query_time, matched, delta_seconds}`

### GET /find/entities/<family_id>/around-time
时间窗口内版本。

**Query**: `time_point` (必填), `within_seconds` (必填)

**返回**: `{family_id, query_time, within_seconds, count, matches: [...]}`

### GET /find/entities/<family_id>/timeline
实体时间线（版本+关系变更）。

**返回**: `{family_id, versions: [...], relations_timeline: [...]}`

### GET /find/entities/<family_id>/relations
实体相关关系。

**Query**:
| 参数 | 默认 | 说明 |
|---|---|---|
| limit | - | |
| time_point | - | ISO8601 |
| max_version_absolute_id | - | 焦点版本 |
| relation_scope | accumulated | accumulated/version_only/all_versions |

### POST /find/entities/<family_id>/evolve-summary
LLM 演化摘要（Phase A）。

**返回**: `{family_id, summary}`

### GET /find/entities/<family_id>/contradictions
矛盾检测。

**返回**: 矛盾列表

### POST /find/entities/<family_id>/resolve-contradiction
LLM 裁决矛盾。

**Body**: `{contradiction: {...}}`

### GET /find/entities/<family_id>/provenance
事实溯源 — 提及该实体的 Episode。

### GET /find/entities/<family_id>/section-history
Section 级版本历史。

**Query**: `section` (必填)

### GET /find/entities/<family_id>/version-diff
两版本 Section 级 diff。

**Query**: `v1, v2` (必填，absolute_id)

### GET /find/entities/<family_id>/patches
ContentPatch 记录。

**Query**: `section` (可选)

### GET /find/entities/absolute/<absolute_id>
按 absolute_id 读取。

### PUT /find/entities/absolute/<absolute_id>
更新指定版本字段。

**Body**: `{name?, content?, summary?, attributes?, confidence?}`

### DELETE /find/entities/absolute/<absolute_id>
删除指定版本（带关联保护）。

### GET /find/entities/absolute/<id>/embedding-preview
Embedding 向量预览。

**Query**: `num_values` (默认5)

### GET /find/entities/absolute/<id>/relations
按 absolute_id 查关联关系。

**Query**: `limit, time_point`

### POST /find/entities/create
手动创建实体。

**Body**: `{graph_id, name(必填), content, memory_cache_id, source_document}`

### POST /find/entities/merge
合并实体。

**Body**: `{graph_id, target_family_id(必填), source_family_ids(必填)}`

### POST /find/entities/batch-delete
批量删除。

**Body**: `{graph_id, family_ids(必填), cascade}`

### POST /find/entities/batch-delete-versions
批量删除版本（带保护）。

**Body**: `{graph_id, absolute_ids(必填)}`

**返回**: `{deleted: [...], blocked: {...}, summary}`

### POST /find/entities/split-version
拆分版本到新 family_id。

**Body**: `{graph_id, absolute_id(必填), new_family_id(可选)}`

### POST /find/entities/version-counts
批量查询版本数量。

**Body**: `{family_ids: [...]}`

### PUT /find/entities/<family_id>/confidence
手动设置实体置信度。

**Body**: `{confidence(必填, 0.0-1.0)}`

---

## 5. 关系接口

### GET /find/relations
列出关系。

**Query**: `graph_id, limit, offset`

### GET/POST /find/relations/search
搜索关系。

**参数**: `query_text(必填), similarity_threshold, max_results, search_mode`

### GET/POST /find/relations/between
查两实体间关系。

**参数**: `family_id_a(必填), family_id_b(必填)`

### GET /find/relations/absolute/<absolute_id>
按 absolute_id 读取。

### PUT /find/relations/absolute/<absolute_id>
更新指定版本。

**Body**: `{content?, summary?, attributes?, confidence?}`

### DELETE /find/relations/absolute/<absolute_id>
删除指定版本。

### GET /find/relations/absolute/<id>/embedding-preview
Embedding 预览。

### GET /find/relations/<family_id>/versions
所有版本。

### PUT /find/relations/<family_id>
创建新版本。

**Body**: `{content(必填)}`

### DELETE /find/relations/<family_id>
删除所有版本。

### POST /find/relations/<family_id>/invalidate
标记失效。

**Body**: `{reason}`

### GET /find/relations/invalidated
列出已失效关系。

### POST /find/relations/create
手动创建关系。

**Body**: `{graph_id, entity1_absolute_id(必填), entity2_absolute_id(必填), content(必填)}`

### POST /find/relations/batch-delete
批量删除。

**Body**: `{graph_id, family_ids(必填)}`

### POST /find/relations/batch-delete-versions
批量删除版本。

**Body**: `{graph_id, absolute_ids(必填)}`

### POST /find/relations/redirect
重定向关系端点。

**Body**: `{graph_id, family_id(必填), side(必填:"entity1"/"entity2"), new_family_id(必填)}`

### PUT /find/relations/<family_id>/confidence
手动设置关系置信度。

**Body**: `{confidence(必填, 0.0-1.0)}`

---

## 6. 路径搜索

### GET/POST /find/paths/shortest
最短路径（BFS）。

**参数**: `family_id_a(必填), family_id_b(必填), max_depth(默认6), max_paths(默认10)`

**返回**: `{source_entity, target_entity, path_length, total_shortest_paths, paths: [{entities, relations, length}]}`

### POST /find/paths/shortest-cypher
Neo4j Cypher shortestPath。

**Body**: `{graph_id, family_id_a, family_id_b, max_depth(默认6)}`

### POST /find/traverse
BFS 图遍历。

**Body**: `{graph_id, seed_family_ids(必填), max_depth(默认2), max_nodes(默认50)}`

---

## 7. 记忆缓存

### GET /find/memory-caches/latest
最新缓存。

**Query**: `graph_id, activity_type`

### GET /find/memory-caches/latest/metadata
最新缓存元数据。

### GET /find/memory-caches/<cache_id>
按 ID 读取。

### GET /find/memory-caches/<cache_id>/text
缓存原文。

### GET /find/memory-caches/<cache_id>/doc
对应完整文档（原文+缓存摘要）。

---

## 8. Episode 管理

### GET /find/episodes/<cache_id>
Episode 详情。

### DELETE /find/episodes/<cache_id>
删除 Episode。

### POST /find/episodes/search
搜索 Episode（BM25）。

**Body**: `{query(必填), limit}`

### POST /find/episodes/batch-ingest
批量导入。

**Body**: `{episodes: [{content, source_document, episode_type}]}`

---

## 9. 时间旅行

### GET /find/snapshot
时间点快照。

**Query**: `time(必填), limit`

**返回**: `{time, entities, relations, entity_count, relation_count}`

### GET /find/changes
时间范围变更。

**Query**: `since(必填), until, limit`

**返回**: `{since, until, entities, relations, entity_count, relation_count}`

---

## 10. 梦境巩固

### GET /find/dream/status
最近梦境状态。

### GET /find/dream/logs
历史日志。

**Query**: `limit` (默认20)

### GET /find/dream/logs/<cycle_id>
单条详情。

### POST /find/dream/seeds
获取种子实体。

**Body**:
| 字段 | 说明 |
|---|---|
| strategy | random/orphan/hub/time_gap/cross_community/low_confidence |
| count | 数量 (上限100) |
| exclude_family_ids | 排除列表 |
| community_id | 指定社区 |

**返回**: `{seeds: [...], strategy, count}`

### POST /find/dream/relation
创建梦境关系。

**Body**: `{entity1_id(必填), entity2_id(必填), content(必填), confidence(必填,0-1), reasoning(必填), dream_cycle_id}`

### POST /find/dream/episode
保存梦境 episode。

**Body**: `{content(必填), entities_examined, relations_created, strategy_used, dream_cycle_id}`

### GET /dream/candidates
列出 Dream 候选层关系。

**Query**: `limit(默认50), offset, status(hypothesized/verified/rejected)`

**返回**: `{relations: [...], total, offset, limit}`

### POST /dream/candidates/<family_id>/promote
将候选关系提升为已验证状态。

**Body**: `{evidence_source(默认"manual"), confidence(可选)}`

### POST /dream/candidates/<family_id>/demote
将候选关系降级为已拒绝状态。

**Body**: `{reason(可选)}`

### POST /dream/candidates/corroborate
佐证检查：验证两实体间是否有证据支撑候选关系。

**Body**: `{entity1_family_id(必填), entity2_family_id(必填)}`

---

## 11. Agent 智能接口

### POST /find/ask
自然语言问答。

**Body**: `{graph_id, question(必填)}`

**返回**: `{thought, query_plan, answer, results: {entities, relations}}`

### POST /find/ask/stream
SSE 流式问答。

### POST /find/explain
LLM 解释实体。

**Body**: `{graph_id, family_id(必填), aspect(默认summary)}`

aspect: summary / relations / timeline / contradictions

### GET /find/suggestions
智能建议。

---

## 12. 社区检测 (Neo4j)

### POST /communities/detect
运行社区检测。

**Body**: `{graph_id, algorithm(louvain/label_propagation), resolution(默认1.0, 0.1-10.0)}`

### GET /communities
列出社区。

**Query**: `graph_id, min_size(默认3), limit(默认50), offset`

### GET /communities/<cid>
社区详情。

### GET /communities/<cid>/graph
社区子图。

### DELETE /communities
清除社区标记。

---

## 13. Episode (Neo4j)

### GET /episodes
分页列出。

**Query**: `graph_id, limit(默认20,上限100), offset`

**返回**: `{episodes, total, limit, offset}`

### GET /episodes/<uuid>
详情。

### GET /episodes/<uuid>/entities
关联实体。

### POST /episodes/search
搜索。

**Body**: `{query(必填), limit}`

### DELETE /episodes/<uuid>
删除。

---

## 14. 实体邻居 (Neo4j)

### GET /find/entities/<uuid>/neighbors
获取邻居图。

**Query**: `graph_id, depth(默认1,上限5)`

---

## 15. Concepts 统一概念查询

### POST /concepts/search
统一概念搜索（可选 role 过滤，支持 semantic/bm25/hybrid 模式）。需要 Neo4j 后端。

**Body**:
| 字段 | 类型 | 必填 | 默认 | 说明 |
|---|---|---|---|---|
| query | string | 是 | - | 搜索文本 |
| role | string | 否 | - | entity/relation/observation |
| limit | int | 否 | 20 | 上限100 |
| threshold | float | 否 | 0.5 | 相似度阈值 |
| search_mode | string | 否 | bm25 | semantic/bm25/hybrid |
| time_point | ISO8601 | 否 | - | 时间点过滤 |

### GET /concepts
列出概念（分页 + 可选 role 过滤）。

**Query**: `role, limit(默认50,上限100), offset, time_point`

### GET /concepts/<family_id>
获取概念（任意角色，按 family_id）。

**Query**: `time_point(可选)`

### GET /concepts/<family_id>/neighbors
获取概念邻居（跨角色图遍历）。

**Query**: `max_depth(默认1,上限3), time_point(可选)`

### GET /concepts/<family_id>/provenance
概念溯源：返回所有提及此概念的 observation。

### POST /concepts/traverse
BFS 遍历概念图。

**Body**:
| 字段 | 类型 | 必填 | 默认 | 说明 |
|---|---|---|---|---|
| start_family_ids | array | 是 | - | 起始概念 ID 列表 |
| max_depth | int | 否 | 2 | 上限5 |
| time_point | ISO8601 | 否 | - | 时间点过滤 |

### GET /concepts/<family_id>/mentions
获取提及此概念的所有 Episode。

### GET /concepts/duplicates
检测潜在重复实体（按名称规范化分组）。

**Query**: `limit(默认500,上限2000)`

---

## 16. 图谱管理

### GET /graphs
列出所有图谱。

**返回**: `{graphs: [...], count}`

### POST /graphs
创建图谱。

**Body**: `{graph_id(必填)}`

---

## 17. 文档管理

### GET /docs
列出文档。

**返回**: `{docs: [{doc_hash, source_document, event_time, text_length, filename, ...}], count}`

### GET /docs/<filename>
获取文档内容。

**返回**: `{meta, original, cache}`

---

## 18. Chat 会话

### GET /chat/sessions
列出会话。

**Query**: `include_closed` (默认0)

### POST /chat/sessions
创建会话。

**Body**: `{title, graph_id}`

### GET /chat/sessions/<sid>
获取详情。

### PUT /chat/sessions/<sid>
更新元数据。

**Body**: `{graph_id?, title?}`

### DELETE /chat/sessions/<sid>
删除。

### POST /chat/sessions/<sid>/close
关闭（保留历史，终止进程）。

### POST /chat/sessions/<sid>/stream
发送消息（SSE 流式）。

**Body**: `{message}`

---

## 19. 系统监控

### GET /system/dashboard
仪表盘（合并端点）。

**Query**: `task_limit, log_limit, log_level, log_source, access_since`

### GET /system/overview
系统总览。

### GET /system/graphs
所有图谱摘要。

### GET /system/graphs/<graph_id>
单图谱详细状态。

### GET /system/tasks
全局任务列表。

**Query**: `limit` (默认50)

### GET /system/logs
系统日志。

**Query**: `limit, level(INFO/WARN/ERROR), source`

### GET /system/access-stats
访问统计。

**Query**: `since_seconds` (默认300)

### GET /routes
接口自描述索引。
