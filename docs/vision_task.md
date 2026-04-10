# Vision Task Tracker

> 每轮迭代记录实际完成的功能和改进。按时间倒序排列。

## 2026-04-10

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
- [ ] **search_episodes_by_bm25 2N文件读取**: 需FTS5虚拟表或内存索引（manager.py ~914-941）

### P2 架构对齐（Concept统一）
- [ ] **Phase 1: MENTIONS补全 + Episode入库**: extraction.py MENTIONS无条件建立 + episodes SQLite表
- [ ] **Phase 2: concepts统一表 + 双写**: 新增concepts表 + concept_fts + 双写适配
- [ ] **Phase 3: 统一查询接口**: get_concept_* / search_concepts_* / traverse_concepts
- [ ] **Phase 4: API统一**: /concepts/* 端点 + MCP工具 + 旧端点兼容

### P3 代码质量
- [x] ~~**Schema初始化去重**: _init_database与_ensure_tables重复~~ (fffb8dd)
- [x] ~~**Neo4j _RELATION_RETURN_FIELDS**: 22个方法重复字段列表~~ (6d7d0da)
- [ ] **api.py分模块**: 4867行单文件，需按领域拆分（server/api.py）
