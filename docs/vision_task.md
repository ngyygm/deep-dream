# Vision Task Tracker

> 每轮迭代记录实际完成的功能和改进。按时间倒序排列。

## 2026-04-11

### [已完成] feat: confidence engine — Bayesian-inspired evidence-driven confidence evolution
- commit: 8a07c7d
- Storage 4个新方法: `update_entity_confidence`, `update_relation_confidence`, `adjust_confidence_on_corroboration`, `adjust_confidence_on_contradiction`
  - 印证: 每次独立来源 +0.05, Dream来源减半 (+0.025), 上限1.0
  - 矛盾: 每次冲突 -0.1, 下限0.0
  - 手动覆盖: 直接设置 [0.0, 1.0] 区间, 超出自动截断
- Pipeline集成: `entity._create_entity_version` 和 `relation._create_relation_version` 自动调用 `adjust_confidence_on_corroboration`
- API端点: `PUT /api/v1/find/entities/<fid>/confidence` + `PUT /api/v1/find/relations/<fid>/confidence`
- Fix: `adjust_confidence_on_corroboration` source_type与is_dream参数解耦（原先source_type="dream"会查错表）
- 测试: 35项测试覆盖4维度（印证增长、矛盾降低、手动覆盖、Dream权重），92项总测试全部通过
- 影响: `processor/storage/manager.py`, `processor/pipeline/entity.py`, `processor/pipeline/relation.py`, `server/blueprints/entities.py`, `server/blueprints/relations.py`

### [已完成] perf: save_episode_mentions N+1 → executemany batch INSERT
- `save_episode_mentions` 从逐行 INSERT 循环改为 `cursor.executemany()` 批量插入
- 影响：`processor/storage/manager.py`
- 测试：57 项新测试全部通过（含并发批量插入验证）

### [已完成] perf: merge_entity_families N+1 → batch SQL
- 版本计数查询：逐个 `SELECT COUNT(*)` → 单次 `GROUP BY family_id`
- 实体更新：逐个 `UPDATE` → 单次 `UPDATE ... IN (?)`
- 验证查询：逐个检查 → 单次 `GROUP BY`
- 重定向插入：逐个 `INSERT` → `executemany()`
- 影响：`processor/storage/manager.py`

### [已完成] fix: get_relations_by_entity_pairs 硬编码列 → 完整15列（含embedding）
- 原先硬编码14列缺少 `embedding`，导致 `_row_to_relation` 依赖 `len(row)` 守卫
- 改为显式15列 SELECT（与 `_RELATION_SELECT` 列顺序一致）
- 影响：`processor/storage/manager.py`

### [已完成] perf: lru_cache 纯函数缓存
- `_is_valid_entity_name`（extraction.py）：添加 `@lru_cache(maxsize=4096)`
- `_normalize_entity_name_for_matching`（entity.py）：添加 `@lru_cache(maxsize=4096)`
- 重复实体名称的候选匹配阶段性能提升

### [已完成] fix: 旧实体对齐路径 LLM 调用错误处理
- `_process_single_entity` 中 `analyze_entity_candidates_preliminary` 和 `analyze_entity_pair_detailed` 调用添加 try/except
- LLM 失败时 fallback 到 no_action / skip candidate，不再导致整个实体处理管道中断
- 影响：`processor/pipeline/entity.py`

### [已完成] test: 57项综合测试 — 4维度全覆盖
- 新增 `tests/test_improvements.py`
- 维度1：batch SQL 正确性（7 提及 + 6 合并测试，含并发和Unicode）
- 维度2：LRU缓存行为（8 测试，含中英文特殊字符）
- 维度3：LLM错误处理（3 测试，fallback验证）
- 维度4：列一致性（4 测试，embedding字段验证）
- 维度5：实体名称验证（30+ 测试，含参数化中英文混合）

## 2026-04-10

### [已完成] fix: over-extraction 三连修复 — strict prompt + proportional cap + dedup
- commit: 72ac369
- 问题：古文530字→64实体（10.2/100字），Step6 LLM输出截断→GLM-4-flash死循环→771s
- 修复1：user prompt 对齐 strict system prompt（"宁缺毋滥" vs 之前的"越多越好"）
- 修复2：Step 2.9 按比例限制实体数量（max 5/100字，下限15），优先保留有关系连接的
- 修复3：Step6 preliminary analysis 结果去重（GLM-4-flash 截断时重复同一ID数百次）
- 验证：古文524字→27实体→99.7s（7.7x加速）
- 影响：`processor/llm/entity_extraction.py`, `processor/pipeline/extraction.py`, `processor/pipeline/entity.py`

### [已完成] fix: 实体候选匹配添加核心名称规范化
- commit: 395e523
- 问题：同一实体因LLM返回不同名称变体（"张伟"/"张伟教授"/"张伟（北京大学教授）"）创建多个family_id
- 原因：Jaccard bigram相似度低于0.6阈值，候选搜索无法找到匹配
- 修复：`_normalize_entity_name_for_matching` 去除括号注释+称谓后缀
- 核心名称精确匹配时 lexical_score=0.85 + merge_safe=True
- GLM-4-flash测试验证：三次变体调用成功合并为单个实体

### [已完成] fix: remember_text() return dict 添加 entities/relations 计数
- commit: 37d618d
- `remember_text()` 返回值原先缺少实体/关系计数（始终为0）
- 新增 `step7_results` 数组捕获 `_align_relations` 返回值
- 聚合 `align_results` 和 `step7_results` 到返回dict

### [已完成] test: GLM-4-flash remember质量测试
- 8种文本类型×不同长度实测：短句(17字)→长文(379字)，中英文，对话，列表
- 结果：每100字提取~4-5实体、3-4关系，对话和列表格式提取更多
- 长文本295字耗时67s（step6实体对齐是瓶颈）
- 发现并修复dedup问题（见上方commit）

### [已完成] fix: Blueprint `request.app` → `current_app` — Flask测试兼容性
- Blueprint拆分后所有模块使用 `request.app.config[...]` 访问 app config
- Flask test client 中 `Request` 对象无 `.app` 属性，导致全部 API 测试 500
- 修复：5个Blueprint模块 + helpers.py 统一改用 `current_app.config[...]`
- 受影响文件：entities.py, system.py, remember.py, concepts.py, helpers.py
- 101项集成测试全部通过

### [已完成] perf: Neo4j get_entities_by_family_ids N+1 embedding → 批量get_batch
- `get_entities_by_family_ids` 中逐个 `_vector_store.get()` → 批量 `_vector_store.get_batch()`
- 一次查询获取所有 embedding，消除 R 次独立向量查找

### [已完成] perf: SQLite batch_get_entity_profiles N+1 resolve → 批量resolve_family_ids
- `batch_get_entity_profiles` 中逐个 `resolve_family_id()` → 批量 `resolve_family_ids()`
- 消除 N 次独立 SQL 查询

### [已完成] fix: entity.py 移除NEW_ENTITY无效DB调用
- `_process_single_entity` 中对未保存实体("NEW_ENTITY")调用 `get_relations_by_entities("NEW_ENTITY", cid)`
- 此查询始终返回空列表（实体未入库），是无效DB调用
- 移除该调用及相关 dead code（skipped_relations_count, skipped_entity_names）

### [已完成] fix: Neo4j get_relations_for_entities 模板字面量bug
- 第22处遗漏的 `RETURN {_RELATION_RETURN_FIELDS}` 未使用 `_expand_cypher()` 展开
- `_q()` 重命名为 `_expand_cypher()`，同时支持 `__ENT_FIELDS__` 和 `__REL_FIELDS__` 占位符

### [已完成] fix: Neo4j _RELATION_RETURN_FIELDS 模板字面量bug — 20处Cypher查询字段未展开
- `_RELATION_RETURN_FIELDS` 是模块级常量字符串，包含完整的 Relation 字段列表
- 20处 Cypher 查询中 `RETURN {_RELATION_RETURN_FIELDS}` 写在普通三引号字符串内
  - Python 不会展开普通字符串中的 `{...}`，导致字面文本发送给 Neo4j
  - 仅 1 处正确使用 f-string（`update_relation_by_absolute_id`）
- 修复方案：引入 `_q()` 辅助函数 + `__REL_FIELDS__` 占位符
  - 所有查询中改为 `RETURN __REL_FIELDS__`，用 `_q("""...""")` 包装
  - 无需转义 Cypher 自身的 `{...}`（如 `{uuid: $uuid}`），避免 f-string 逃逸风险
- 21 个方法受影响：get_relation_by_absolute_id, get_relation_versions, get_all_relations 等
- 现有 18 项集成测试全部通过

### [已完成] feat: Concept语义搜索 — embedding余弦相似度替代BM25 stub
- `search_concepts_by_similarity` 从纯BM25回退改为真正的embedding向量搜索
- 新增 `_get_latest_concepts_with_embeddings`：带TTL缓存的Concept embedding批量加载器
  - ROW_NUMBER()窗口函数取每个family_id最新版本
  - embedding BLOB → numpy数组解码
- 搜索流程：encode查询文本 → 构建归一化存储矩阵 → 矩阵乘法批量余弦相似度 → 阈值过滤 + 排序
- 无embedding客户端或无结果时自动回退BM25
- 结果附带 `_similarity_score` 字段
- 新增 `_concept_emb_cache` / `_concept_emb_cache_ts` 缓存，随 `_invalidate_emb_cache()` 统一失效
- 对齐 vision.md "语义是一等公民" 原则

### [已完成] perf: get_concept_neighbors relation角色 N+1→批量查询
- relation角色邻居解析：逐个 `SELECT family_id FROM concepts WHERE id = ?` → 单次 `IN (?)` 批量查询
- 与entity角色、observation角色的批量模式对齐

### [已完成] feat: extraction pipeline填充summary/confidence/content_format
- commit: f12fe40
- Entity: _build_new_entity/_build_entity_version 添加 summary(内容首行)、confidence=0.7、content_format="markdown"
- Relation: _build_new_relation/_build_relation_version 添加 summary(content截断200字符)、confidence=0.7、content_format="markdown"
- 新增 EntityProcessor._extract_summary: 跳过标题行取首行正文，回退到name

### [已完成] fix: _get_entities_with_embeddings 使用_ENTITY_SELECT替代硬编码列
- commit: 2f9e28a
- 之前仅加载9列(id~embedding)，summary/attributes/confidence等全部为None
- 与_get_relations_with_embeddings(已使用_RELATION_SELECT)对齐

### [已完成] fix: FTS rowid整数 + SELECT列常量化 + exclude_embedding偏移
- commit: deb73eb
- FTS5 rowid必须为整数，但entity/relation的id列是TEXT PRIMARY KEY
  - save_entity/save_relation: cursor.lastrowid替代absolute_id
  - bulk_save: 先查整数rowid再写入FTS
  - BM25 JOIN: e.id=fts.rowid → e.rowid=fts.rowid
  - FTS DELETE: 子查询SELECT rowid替代文本ID
- _ENTITY_SELECT/_RELATION_SELECT常量 + _row_to_entity/_row_to_relation辅助方法
  - 20+处SELECT列名和Entity/Relation构造消除重复
  - 单一数据源保证列顺序一致
- exclude_embedding列偏移: 移除embedding列导致summary等字段索引错位
  - 改为NULL占位替代列移除，保持列顺序
- Neo4j bulk_save_relations补齐summary/attributes/confidence等6个字段
- 18项集成测试全部通过

### [已完成] refactor: api.py 4980行→Blueprint 7模块 + 350行工厂
- server/api.py 从 4980 行单体拆分为 7 个 Blueprint 模块 + 350 行应用工厂
- 新增文件:
  - server/blueprints/__init__.py — 包标记
  - server/blueprints/helpers.py — 共享工具 (序列化、响应、时间解析)
  - server/blueprints/system.py — 健康检查、统计、系统监控 (12 routes)
  - server/blueprints/remember.py — Remember 任务管理 (6 routes)
  - server/blueprints/entities.py — 实体 CRUD、搜索、智能 (37 routes)
  - server/blueprints/relations.py — 关系 CRUD、路径、统一搜索 (26 routes)
  - server/blueprints/episodes.py — Episode 管理、快照、变更 (17 routes)
  - server/blueprints/dream.py — Dream 探索、Ask、Butler (17 routes)
  - server/blueprints/concepts.py — 概念、社区、图谱管理、聊天 (21 routes)
- api.py 保留: create_app() 中间件/CORS/限流/graph_id解析、build_processor()、main()、SPA 路由
- 126 个唯一路径、148 个路由方法，无重复注册
- 所有路由路径完全兼容原有 API

### [已完成] feat: Phase 4.1 — MCP concept_* 统一工具
- commit: 8e72559
- 新增7个MCP工具: search_concepts, list_concepts, get_concept, get_concept_neighbors, get_concept_provenance, traverse_concepts, get_concept_mentions

### [已完成] feat: Phase 3-4 — Neo4j统一查询 + API端点
- commit: e42005f
- Neo4j: 9个统一概念查询方法（get_concept_by_family_id, search_concepts_by_bm25, get_concept_neighbors 等）
- API: 7个 /api/v1/concepts/* 端点（search, list, get, neighbors, provenance, traverse, mentions）

### [已完成] feat: Phase 3 (SQLite) — 统一概念查询接口
- commit: a404b43
- 11个基于Concept的统一查询方法（get_concept_*, search_concepts_*, traverse_concepts 等）

### [已完成] feat: Phase 2 — concepts统一表 + 双写适配
- commit: 6abcc1b
- SQLite: concepts表 + concept_fts + 双写 + 启动迁移
- Neo4j: 所有写入路径添加 :Concept 标签 + role 属性

### [已完成] perf: search_episodes_by_bm25 文件遍历→SQLite LIKE过滤
- commit: ce40b7a
- episodes表有数据时SQL LIKE过滤候选集→Python评分→只加载top-N完整Episode
- episodes表为空时自动回退到旧的文件遍历逻辑

### [已完成] feat: Phase 1 — MENTIONS补全 + Episode入库SQLite
- commit: 3a7059b
- extraction.py: Entity MENTIONS无条件建立（含已存在的实体），新增Relation MENTIONS
- manager.py: 新增episodes表 + episode_mentions schema升级（target_type列）
  - 启动时从docs/目录迁移已有Episode元数据到SQLite（幂等）
  - save_episode同步写SQLite，get_episode/list_episodes兼容Neo4j接口
  - get_episode_entities支持relation目标（LEFT JOIN entities + relations）
  - 旧episode_mentions表自动迁移到新schema（rename→create→insert→drop）
- neo4j_store.py: save_episode_mentions支持target_type="relation"
  - get_entity_provenance扩展间接MENTIONS查询（通过Relation反查Episode）
  - get_episode_entities同时返回entity + relation目标
- api.py: episode端点兼容SQLite后端

### [已完成] docs: Concept统一设计文档
- 文件: docs/design/concept-unification.md
- 分析vision.md与现有实现的4大差距，规划4-Phase渐进式迁移方案
- Phase 1: MENTIONS补全 + Episode入库
- Phase 2: concepts统一表 + 双写适配
- Phase 3: 统一查询接口
- Phase 4: API统一 + 清理

### [已完成] perf: get_dream_seeds排除ID N+1→批量
- commit: b22de57
- exclude_ids逐个resolve_family_id+get_entity_by_family_id改为resolve_family_ids+get_entities_by_family_ids

### [已完成] perf: Neo4j get_graph_statistics 9次串行Cypher→3次
- commit: c2b541b
- 基础计数+度数统计(6次)合并为单次UNWIND聚合；修复变量名遮蔽(r→rec)

### [已完成] refactor: SQLite schema初始化去重
- commit: fffb8dd
- _init_database委托_ensure_tables，消除两处CREATE TABLE的drift

### [已完成] refactor: Neo4j _RELATION_RETURN_FIELDS 常量提取
- commit: 6d7d0da
- 22处重复字段列表提取为_RELATION_RETURN_FIELDS常量

### [已完成] perf: storage manager O(R*F)→O(R) + 关系查询轻量化
- commit: ea3d611
- get_relations_by_entity_pairs/batch_get_entity_profiles: 建reverse lookup dict替代嵌套循环
- get_entity_relations_by_family_id: 轻量SELECT仅取ID，避免加载全量BLOB

### [已完成] fix: FTS单版本删除不应清除整个family索引
- commit: ec7d6dd
- delete_entity_by_absolute_id等4方法: DELETE entity_fts WHERE family_id → WHERE rowid
- Neo4j post-delete: 先收集absolute_ids再DETACH DELETE

---

## 待改进项（按优先级）

### P0 正确性
- [x] ~~**FTS删除bug**: delete_entity_by_absolute_id 删单个版本时清除整个family的FTS索引~~ (ec7d6dd)
- [x] ~~**Neo4j post-delete bug**: delete_relation_by_id/delete_entity_all_versions 先DETACH DELETE再查版本ID~~ (ec7d6dd)

### P1 性能
- [x] ~~**get_relations_by_entity_pairs O(R*F)→O(R)**: 建reverse lookup dict替代嵌套循环~~ (ea3d611)
- [x] ~~**get_entity_relations_by_family_id**: 加载全量BLOB仅取ID→轻量SELECT~~ (ea3d611)
- [x] ~~**get_graph_statistics 9次串行Cypher**: 合并为3个查询~~ (c2b541b)
- [x] ~~**get_dream_seeds N+1**: 排除ID逐个resolve→批量~~ (b22de57)
- [x] ~~**search_episodes_by_bm25 2N文件读取**: SQL LIKE过滤→Python评分→top-N加载~~ (ce40b7a)

### P2 架构对齐（Concept统一）
- [x] ~~**Phase 1: MENTIONS补全 + Episode入库**: extraction.py MENTIONS无条件建立 + episodes SQLite表~~ (3a7059b)
- [x] ~~**Phase 2: concepts统一表 + 双写**: 新增concepts表 + concept_fts + 双写适配~~ (6abcc1b)
- [x] ~~**Phase 3: 统一查询接口**: get_concept_* / search_concepts_* / traverse_concepts~~ (a404b43, e42005f)
- [x] ~~**Phase 4: API统一**: /concepts/* 端点 + Neo4j统一查询~~ (e42005f)
- [x] ~~**Phase 4.1: MCP工具**: 7个 concept_* MCP 工具~~ (8e72559)

### P3 代码质量
- [x] ~~**Schema初始化去重**: _init_database与_ensure_tables重复~~ (fffb8dd)
- [x] ~~**Neo4j _RELATION_RETURN_FIELDS**: 22个方法重复字段列表~~ (6d7d0da)
- [x] ~~**api.py分模块**: ~5000行单文件，需按领域拆分~~ (7 Blueprint 模块 + 350 行工厂)

### P4 性能（2026-04-11）
- [x] ~~**save_episode_mentions N+1**: 逐行INSERT → executemany批量~~
- [x] ~~**merge_entity_families N+1**: 逐个SQL → batch SQL~~
- [x] ~~**get_relations_by_entity_pairs 硬编码列**: 缺少embedding → 完整15列~~
- [x] ~~**lru_cache纯函数**: _is_valid_entity_name + _normalize_entity_name_for_matching~~

### P5 健壮性（2026-04-11）
- [x] ~~**旧实体对齐路径LLM错误处理**: 无try/except → fallback no_action~~

### 待改进项

#### P1 架构
- [ ] **manager.py 分模块**: 4884行单类 → mixin 模式拆分（entity_store, relation_store, episode_store, concept_store）

#### P2 功能对齐（vision.md）
- [x] ~~**置信度引擎**: 置信度随证据增减动态调整~~ (8a07c7d)
- [ ] **矛盾检测**: 版本间语义冲突检测与修正标注
- [ ] **Dream 候选层**: Dream产物默认写入候选层，需证据/复核提升为事实
