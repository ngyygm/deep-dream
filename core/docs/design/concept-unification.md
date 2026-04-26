# Concept 统一设计文档

> 目标：将当前 Entity/Relation/Episode 三类分离架构，迁移为 vision.md 描述的"万物皆 Concept"统一模型。

## 一、现状分析

### 1.1 数据模型层（models.py）

**现状**：`Concept` 和 `ConceptVersion` 已定义，但零引用。

```
Entity      — 14 个字段，含 family_id + absolute_id 版本体系
Relation    — 16 个字段，含 entity1_absolute_id / entity2_absolute_id
Episode     — 5 个字段，仅 absolute_id，无 family_id（不可版本化）
Concept     — 已定义，role/connects/versions/sources 字段完整，但从未被 import
```

**差距**：
- Episode 无 family_id，无法版本化（vision 明确要求 observation 也遵循版本规则）
- Relation 有 connects 语义（entity1/2），但存储为 absolute_id 而非 family_id
- Concept 的 `sources` 字段（溯源）在 Entity/Relation 中仅通过 `episode_id` 单值字段表达

### 1.2 存储层（SQLite manager.py）

**现状**：

| 表 | 行数 | 关键字段 |
|---|---|---|
| entities | ~4200行代码管理 | family_id, name, content, embedding, summary... |
| relations | ~4200行代码管理 | family_id, entity1_absolute_id, entity2_absolute_id, content... |
| entity_fts | FTS5 虚拟表 | name, content |
| relation_fts | FTS5 虚拟表 | content |
| episode_mentions | Episode→Entity 多对多 | episode_id, entity_absolute_id |
| content_patches | Section 级变更记录 | target_family_id, section_key, diff... |

Episode 存储为文件：`docs/{YYYYMMDD_HHMMSS}_{hash}/` 目录，含 original.txt、cache.md、meta.json。

**差距**：
- Episode 不在 SQLite 表中，无法通过图遍历访问
- episode_mentions 仅 Episode→Entity，无 Episode→Relation
- Entity 和 Relation 分表存储，共享大量相同字段但无法联合查询
- FTS 表按类型分（entity_fts / relation_fts），vision 要求统一搜索空间

### 1.3 存储层（Neo4j neo4j_store.py）

**现状**：

```
(:Entity {uuid, family_id, name, content, ...})
(:Relation {uuid, family_id, content, entity1, entity2, ...})
(:Episode {uuid, content, event_time, ...})
(ep:Episode)-[:MENTIONS]->(e:Entity)    ← 仅此方向
```

**差距**：
- Episode 是独立标签，非 Concept 的一种 role
- MENTIONS 仅 Episode→Entity，无 Episode→Relation
- Relation 作为边而非节点，无法被 MENTION、无法有自己的邻居
- Entity 和 Relation 字段结构高度重复

### 1.4 管线层（pipeline/extraction.py）

**现状**：
```
Step 1: text → LLM → Episode (Markdown summary)
Step 2-5: Episode context → entity/relation extraction
Step 6: entity alignment → save_episode_mentions(仅新创建的实体)
Step 7: relation alignment
```

**关键缺陷**（extraction.py:1781）：
```python
# 只为 unique_entities（新创建的版本）记录 MENTIONS
# 已存在且内容未变的实体，不记录 MENTIONS
if unique_entities:
    self.storage.save_episode_mentions(new_episode.absolute_id, abs_ids)
```

vision 明确要求：**关联建立是无条件的**。即使内容没变，也必须建立 Episode→Concept 关联。

### 1.5 API 层（server/api.py, ~4868 行）

**现状**：~90+ 端点，按类型分组。

| 分组 | 端点数 | 路径前缀 |
|---|---|---|
| Entity CRUD | 25+ | /api/v1/find/entities/ |
| Relation CRUD | 16+ | /api/v1/find/relations/ |
| Episode | 7 | /api/v1/find/episodes/ |
| 统一操作 | 5 | quick-search, traverse, snapshot, changes |
| Dream | 8+ | /api/v1/dream/ |
| 维护 | 10+ | health, stats, butler... |

已存在的"统一"端点（quick-search、traverse、snapshot、changes）已同时返回 entities 和 relations。

---

## 二、迁移策略

### 核心原则

1. **向后兼容**：Entity/Relation/Episode 数据类和旧 API 在过渡期保留
2. **渐进式**：分 4 个 Phase，每个 Phase 可独立部署和验证
3. **先底层后应用**：先完成存储层和管线层，再改 API 层
4. **零数据丢失**：所有迁移脚本必须幂等，可回滚

### Phase 1：MENTIONS 补全 + Episode 入库

**目标**：修复溯源完整性，让 Episode 成为可查询的图节点。

#### 1.1 SQLite：Episode 入库

新增 `episodes` 表：

```sql
CREATE TABLE IF NOT EXISTS episodes (
    id TEXT PRIMARY KEY,               -- absolute_id
    family_id TEXT NOT NULL,           -- 逻辑身份（可版本化）
    content TEXT NOT NULL,             -- Markdown 摘要
    event_time TEXT NOT NULL,
    processed_time TEXT NOT NOT,
    source_document TEXT DEFAULT '',
    activity_type TEXT DEFAULT '',
    episode_type TEXT DEFAULT '',       -- narrative | fact | conversation | dream
    doc_hash TEXT DEFAULT '',           -- 文件系统目录 hash
    embedding BLOB
);
```

- 启动时自动从 `docs/` 目录迁移已有 Episode 的 meta.json 到此表
- 文件存储保留（original.txt 仍在文件系统），SQLite 仅存元数据
- 后续 Episode 写入同时写 SQLite + 文件

#### 1.2 SQLite：MENTIONS 表扩展

```sql
-- 现有表结构已够用，只需扩展写入逻辑
-- episode_mentions(episode_id, entity_absolute_id, mention_context)
-- 改为：episode_mentions(episode_id, target_absolute_id, target_type, mention_context)
-- target_type: "entity" | "relation" | "episode"
```

添加 `target_type` 列（幂等 ALTER）。

#### 1.3 管线：MENTIONS 无条件建立

修改 `extraction.py` Step 6：

```python
# 修改前：仅记录新创建的实体
if unique_entities:
    self.storage.save_episode_mentions(new_episode.absolute_id, abs_ids)

# 修改后：记录所有被提及的实体和关系
all_mentioned_entity_ids = []
for entity_name, fid in entity_name_to_id.items():
    entity = self.storage.get_entity_by_family_id(fid)
    if entity:
        all_mentioned_entity_ids.append(entity.absolute_id)

all_mentioned_relation_ids = []
for rel in aligned_relations:
    all_mentioned_relation_ids.append(rel.absolute_id)

self.storage.save_episode_mentions(
    new_episode.absolute_id,
    all_mentioned_entity_ids,
    target_type="entity"
)
self.storage.save_episode_mentions(
    new_episode.absolute_id,
    all_mentioned_relation_ids,
    target_type="relation"
)
```

#### 1.4 Neo4j：同步变更

- Episode 入库：已有 `:Episode` 节点，无需变更
- MENTIONS 扩展：新增 `(ep)-[:MENTIONS]->(r:Relation)` 边
- `save_episode_mentions` 同时写 Entity 和 Relation 的 MENTIONS

#### 1.5 验证标准

- [ ] 每个 Episode 的 MENTIONS 包含所有提及的实体和关系
- [ ] 已存在的实体被提及时，MENTIONS 边也被创建
- [ ] Episode 可通过 SQLite 查询（不在仅依赖文件遍历）
- [ ] 现有 API 功能不退化

---

### Phase 2：统一 concepts 表 + 适配层

**目标**：新增统一的 `concepts` 表，通过适配层同时写新旧表。

#### 2.1 SQLite：新增 concepts 表

```sql
CREATE TABLE IF NOT EXISTS concepts (
    id TEXT PRIMARY KEY,               -- 版本唯一标识 (absolute_id)
    family_id TEXT NOT NULL,           -- 逻辑身份
    role TEXT NOT NULL,                -- "entity" | "relation" | "observation"
    name TEXT DEFAULT '',              -- 显示名称（entity 用）
    content TEXT NOT NULL,
    event_time TEXT NOT NULL,
    processed_time TEXT NOT NULL,
    source_document TEXT DEFAULT '',
    episode_id TEXT DEFAULT '',
    embedding BLOB,

    -- 角色专用字段
    connects TEXT DEFAULT '',          -- JSON: ["fid1", "fid2"] (relation role)
    activity_type TEXT DEFAULT '',     -- observation role
    episode_type TEXT DEFAULT '',      -- observation role

    -- 通用字段
    valid_at TEXT,
    invalid_at TEXT,
    summary TEXT,
    attributes TEXT,
    confidence REAL,
    content_format TEXT DEFAULT 'plain',
    provenance TEXT DEFAULT ''         -- JSON
);

CREATE INDEX idx_concepts_family ON concepts(family_id);
CREATE INDEX idx_concepts_role ON concepts(role);
CREATE INDEX idx_concepts_name ON concepts(name);
CREATE UNIQUE INDEX idx_concepts_unique ON concepts(family_id, processed_time);

-- 统一 FTS
CREATE VIRTUAL TABLE IF NOT EXISTS concept_fts USING fts5(name, content, family_id UNINDEXED, role UNINDEXED);
```

#### 2.2 双写适配

在 `StorageManager` 中增加双写逻辑：

```python
def save_entity(self, entity: Entity):
    """保存到旧 entities 表 + 新 concepts 表"""
    # ... 现有逻辑写入 entities 表 ...
    # 新增：同步写入 concepts 表
    self._write_concept_from_entity(entity)

def save_relation(self, relation: Relation):
    """保存到旧 relations 表 + 新 concepts 表"""
    # ... 现有逻辑写入 relations 表 ...
    # 新增：同步写入 concepts 表
    self._write_concept_from_relation(relation)

def save_episode(self, cache: Episode, ...):
    """保存到文件 + 新 concepts 表"""
    # ... 现有文件写入逻辑 ...
    # 新增：写入 concepts 表
    self._write_concept_from_episode(cache)
```

#### 2.3 ConceptMapper 工具

```python
class ConceptMapper:
    """旧 Entity/Relation/Episode ↔ Concept 的双向映射"""

    @staticmethod
    def entity_to_concept(entity: Entity) -> dict:
        return {
            "id": entity.absolute_id,
            "family_id": entity.family_id,
            "role": ROLE_ENTITY,
            "name": entity.name,
            "content": entity.content,
            "embedding": entity.embedding,
            ...
        }

    @staticmethod
    def relation_to_concept(relation: Relation) -> dict:
        return {
            "id": relation.absolute_id,
            "family_id": relation.family_id,
            "role": ROLE_RELATION,
            "name": "",  # 关系无名称
            "content": relation.content,
            "connects": json.dumps([relation.entity1_absolute_id, relation.entity2_absolute_id]),
            ...
        }

    @staticmethod
    def concept_to_entity(row: dict) -> Entity:
        """从 concepts 表行还原 Entity"""
        ...

    @staticmethod
    def concept_to_relation(row: dict) -> Relation:
        """从 concepts 表行还原 Relation"""
        ...
```

#### 2.4 Neo4j：标签统一

```cypher
-- 给所有 :Entity 节点加上 :Concept 标签和 role 属性
MATCH (e:Entity) SET e:Concept, e.role = 'entity'
-- 给所有 :Relation 节点加上 :Concept 标签和 role 属性
MATCH (r:Relation) SET r:Concept, r.role = 'relation'
-- 给所有 :Episode 节点加上 :Concept 标签和 role 属性
MATCH (ep:Episode) SET ep:Concept, ep.role = 'observation'
```

Neo4j 保留旧标签（`:Entity`, `:Relation`, `:Episode`）以保证查询兼容，新增 `:Concept` 标签用于统一查询。

#### 2.5 验证标准

- [ ] concepts 表包含与 entities + relations + episodes 等量的记录
- [ ] 双写不导致性能显著退化（<5% 写入延迟增加可接受）
- [ ] concept_fts 搜索结果覆盖 entity_fts + relation_fts 的并集
- [ ] Neo4j 所有节点都有 :Concept 标签和 role 属性

---

### Phase 3：统一查询接口

**目标**：新增基于 Concept 的统一查询方法，旧方法保持不变。

#### 3.1 统一读接口

```python
class StorageManager:
    # --- 新增：基于 Concept 的统一查询 ---

    def get_concept_by_family_id(self, family_id: str) -> Optional[dict]:
        """获取任意 role 的概念最新版本"""
        ...

    def get_concepts_by_family_ids(self, family_ids: List[str]) -> Dict[str, dict]:
        """批量获取概念"""
        ...

    def search_concepts_by_bm25(self, query: str, role: str = None, limit: int = 20) -> List[dict]:
        """BM25 搜索，可选按 role 过滤"""
        ...

    def search_concepts_by_similarity(self, query_text: str, role: str = None,
                                       threshold: float = 0.5, max_results: int = 10) -> List[dict]:
        """语义相似度搜索，所有 role 在同一语义空间"""
        ...

    def get_concept_neighbors(self, family_id: str, max_depth: int = 1) -> List[dict]:
        """获取概念的邻居（无论 role）"""
        # entity: 返回关联的 relation + 这些 relation 连接的其他 entity
        # relation: 返回它连接的 entity + 通过这些 entity 关联的其他 relation
        # observation: 返回它 MENTIONS 的所有 concept
        ...

    def get_concept_provenance(self, family_id: str) -> List[dict]:
        """溯源：返回所有提及此概念的 observation"""
        ...

    def traverse_concepts(self, start_family_ids: List[str], max_depth: int = 2) -> dict:
        """BFS 遍历概念图"""
        ...
```

#### 3.2 统一 MENTIONS 查询

```python
def get_concept_mentions(self, family_id: str) -> List[dict]:
    """获取提及此概念的所有 Episode"""
    # 从 concepts 表查所有 absolute_id
    # 从 episode_mentions 表查所有引用这些 absolute_id 的 Episode
    ...

def get_episode_concepts(self, episode_id: str) -> List[dict]:
    """获取 Episode 提及的所有概念（entity + relation）"""
    ...
```

#### 3.3 验证标准

- [ ] `search_concepts_by_bm25("Python")` 同时返回 Python 实体和 "Python 用于 AI" 关系
- [ ] `get_concept_neighbors` 对 entity/relation/observation 三种 role 均可工作
- [ ] `get_concept_provenance` 对 relation role 也可溯源到 Episode
- [ ] 旧 API 端点继续正常工作

---

### Phase 4：API 统一 + 清理

**目标**：在 API 层暴露 Concept 统一接口，逐步废弃旧分类型端点。

#### 4.1 新增统一端点

```
POST /api/v1/concepts/search          — 统一搜索（可选 role 过滤）
GET  /api/v1/concepts/{family_id}     — 获取概念（任意 role）
GET  /api/v1/concepts/{family_id}/neighbors — 概念邻居
GET  /api/v1/concepts/{family_id}/provenance — 溯源
POST /api/v1/concepts/traverse        — 图遍历
GET  /api/v1/concepts                 — 列表（分页 + role 过滤）
```

#### 4.2 旧端点兼容

- 旧端点 `/api/v1/find/entities/*` 和 `/api/v1/find/relations/*` 保留
- 内部逐步迁移到读 concepts 表
- 返回格式不变（Entity/Relation 结构）

#### 4.3 清理计划（远期）

当所有调用方迁移到统一接口后：

1. 移除 `entities` / `relations` 表的双写
2. 移除 `entity_fts` / `relation_fts`，统一使用 `concept_fts`
3. Entity/Relation dataclass 标记 `@deprecated`
4. API 旧端点标记为 deprecated（返回 Header: `Deprecation: true`）

---

## 三、数据迁移方案

### SQLite 迁移脚本

```python
def migrate_to_concepts(self):
    """将现有 entities + relations + episodes 数据迁移到 concepts 表"""

    # 1. entities → concepts (role=entity)
    self._get_conn().execute("""
        INSERT OR IGNORE INTO concepts
        (id, family_id, role, name, content, event_time, processed_time,
         source_document, episode_id, embedding, valid_at, invalid_at,
         summary, attributes, confidence, content_format)
        SELECT id, family_id, 'entity', name, content, event_time, processed_time,
               source_document, episode_id, embedding, valid_at, invalid_at,
               summary, attributes, confidence, content_format
        FROM entities
    """)

    # 2. relations → concepts (role=relation)
    self._get_conn().execute("""
        INSERT OR IGNORE INTO concepts
        (id, family_id, role, name, content, event_time, processed_time,
         source_document, episode_id, embedding, valid_at, invalid_at,
         summary, attributes, confidence, content_format, connects)
        SELECT id, family_id, 'relation', '', content, event_time, processed_time,
               source_document, episode_id, embedding, valid_at, invalid_at,
               summary, attributes, confidence, content_format,
               json_array(entity1_absolute_id, entity2_absolute_id)
        FROM relations
    """)

    # 3. episodes → concepts (role=observation)
    # 从文件系统读取 meta.json，插入 concepts 表
    for doc_dir in self.docs_dir.iterdir():
        ...

    # 4. 重建 FTS
    self._get_conn().execute("""
        INSERT OR IGNORE INTO concept_fts(rowid, name, content, family_id, role)
        SELECT id, name, content, family_id, role FROM concepts
    """)
```

### Neo4j 迁移脚本

```cypher
// 1. 添加 :Concept 标签和 role 属性
MATCH (e:Entity)  SET e:Concept, e.role = 'entity'
MATCH (r:Relation) SET r:Concept, r.role = 'relation'
MATCH (ep:Episode) SET ep:Concept, ep.role = 'observation'

// 2. 添加 Episode→Relation MENTIONS（根据 episode_id 反推）
MATCH (ep:Episode)-[:MENTIONS]->(e:Entity)<-[:RELATES_TO]-(r:Relation)
WHERE r.episode_id = ep.uuid
MERGE (ep)-[:MENTIONS {inferred: true}]->(r)

// 3. 补全 Episode→Entity MENTIONS（已存在但仅对新实体的）
// 需要根据 episode_id 字段回填
```

---

## 四、影响范围

### 不受影响

- **MCP Server**：映射 API 端点，API 兼容则 MCP 兼容
- **CLI / chat**：通过 API Client 调用，API 兼容则 CLI 兼容
- **Dream 系统**：使用 storage 层查询，适配层透明
- **前端可视化**：使用 REST API，API 兼容则前端兼容

### 需要修改

| 层 | 文件 | 变更范围 |
|---|---|---|
| models | processor/models.py | Episode 增加 family_id 字段 |
| storage | processor/storage/manager.py | 新增 concepts 表 + 双写 + 统一查询方法 |
| storage | processor/storage/neo4j_store.py | 新增 :Concept 标签 + 统一查询 |
| pipeline | processor/pipeline/extraction.py | MENTIONS 无条件建立 |
| pipeline | processor/pipeline/entity.py | 适配新 MENTIONS 逻辑 |
| pipeline | processor/pipeline/relation.py | 关系也记录 MENTIONS |
| api | server/api.py | 新增 /concepts/* 端点 |
| api | server/mcp/deep_dream_server.py | 新增 concept_* MCP 工具 |

---

## 五、实施顺序

```
Phase 1 (MENTIONS 补全 + Episode 入库)
  ├─ 1.1 episodes 表 + 文件迁移
  ├─ 1.2 episode_mentions 加 target_type 列
  ├─ 1.3 管线 MENTIONS 无条件建立
  ├─ 1.4 Neo4j MENTIONS 扩展
  └─ 1.5 验证 + 提交

Phase 2 (concepts 表 + 双写)
  ├─ 2.1 concepts 表 + concept_fts
  ├─ 2.2 双写适配层
  ├─ 2.3 数据迁移脚本
  ├─ 2.4 Neo4j :Concept 标签
  └─ 2.5 验证 + 提交

Phase 3 (统一查询接口)
  ├─ 3.1 get_concept_* 方法
  ├─ 3.2 search_concepts_* 方法
  ├─ 3.3 traverse_concepts 方法
  └─ 3.4 验证 + 提交

Phase 4 (API 统一)
  ├─ 4.1 /concepts/* 端点
  ├─ 4.2 MCP 工具
  ├─ 4.3 旧端点兼容层
  └─ 4.4 验证 + 提交
```

每个 Phase 完成后更新 `docs/vision_task.md`，提交并推送到 GitHub。

---

## 六、风险与回退

### 风险

1. **双写性能**：每次写入多一个 concepts 表操作。缓解：双写在同一个事务内，单次 commit。
2. **数据不一致**：双写期间如果中间失败，新旧表数据可能不一致。缓解：定期一致性检查脚本。
3. **FTS 索引膨胀**：concept_fts 覆盖全量数据，索引体积增大。缓解：FTS5 只索引 name+content，不索引 embedding。
4. **Neo4j 标签膨胀**：多标签不影响查询性能（Neo4j 标签是稀疏索引）。

### 回退方案

每个 Phase 独立，可单独回退：
- Phase 1 回退：删除 episodes 表 + episode_mentions.target_type 列
- Phase 2 回退：停止双写，删除 concepts 表
- Phase 3 回退：删除统一查询方法
- Phase 4 回退：删除 /concepts/* 端点

旧 API 和旧存储表在 Phase 4 清理之前始终保持可用。
