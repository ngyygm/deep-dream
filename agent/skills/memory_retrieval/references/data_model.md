# 时序记忆图谱数据模型详解

本文档详细说明记忆库中各数据类型的字段含义和使用方法。

## 核心概念

### 版本管理
记忆库采用类似 Git 的版本管理：
- 同一实体/关系可能有多个版本
- `entity_id` / `relation_id` 是逻辑标识，同一实体/关系的所有版本共享
- `id` 是物理标识，每个版本唯一

### 时间轴
- `physical_time`: 记录创建的物理时间
- 可用于追溯信息的时间顺序
- 判断"第几次"等时序问题的关键

### 场景关联
- `memory_cache_id`: 指向产生该记录的原始文档/场景
- 同一 `memory_cache_id` 的记录来自同一场景
- 可用于判断多个记录是否描述同一事件

---

## Entity（实体）

实体代表记忆库中的人、物、地点、概念等。

### 字段详解

| 字段 | 类型 | 说明 |
|------|------|------|
| `id` | string | 版本唯一标识。格式：`entity_时间戳_随机串`。每次更新产生新 id。 |
| `entity_id` | string | 实体唯一标识。格式：`ent_随机串`。同一实体的所有版本共享此 ID。 |
| `name` | string | 实体名称。**注意**：同一个人可能有多种称呼（别名、昵称、简称），可能对应不同的 entity_id。 |
| `content` | string | 实体的自然语言描述。包含实体的属性、特征、背景等信息。 |
| `physical_time` | datetime | 该版本创建的物理时间。ISO 格式。 |
| `memory_cache_id` | string | 来源场景ID。指向产生该记录的 MemoryCache。 |
| `embedding` | bytes | 向量表示（内部使用）。 |

### 使用要点

1. **搜索实体时**：
   - 使用 `search_entity` 工具
   - 可能返回多个结果（别名、同名不同人）
   - 需要根据 `content` 判断是否为目标实体

2. **获取实体详情时**：
   - 使用 `entity_id` 而非 `id`
   - `entity_id` 可以获取该实体的所有版本

3. **追溯历史时**：
   - 使用 `get_entity_versions` 获取所有版本
   - 按 `physical_time` 排序可以看到演变过程

---

## Relation（关系）

关系代表两个实体之间的关联，是无向的。

### 字段详解

| 字段 | 类型 | 说明 |
|------|------|------|
| `id` | string | 版本唯一标识。格式：`relation_时间戳_随机串`。 |
| `relation_id` | string | 关系唯一标识。格式：`rel_随机串`。同一关系的所有版本共享此 ID。 |
| `entity1_absolute_id` | string | 第一个实体的版本ID（指向具体版本的 `id`）。 |
| `entity2_absolute_id` | string | 第二个实体的版本ID（指向具体版本的 `id`）。 |
| `content` | string | 关系的自然语言描述。描述两个实体之间发生了什么。 |
| `physical_time` | datetime | 该记录的物理时间。**关键**：可用于时间排序。 |
| `memory_cache_id` | string | 来源场景ID。**关键**：同一 ID 表示来自同一场景/文档。 |
| `embedding` | bytes | 向量表示（内部使用）。 |

### 使用要点

1. **查询关系时**：
   - 必须先获取实体的 `entity_id`
   - 使用 `get_entity_relations` 获取直接关系
   - 无直接关系时使用 `get_relation_paths` 获取多跳路径

2. **判断时间顺序时**：
   - 按 `physical_time` 排序
   - 时间早的排在前面

3. **判断是否同一事件时**：
   - 检查 `memory_cache_id` 是否相同
   - 相同 = 来自同一场景/文档 = 可能是同一事件
   - 不同 = 来自不同场景 = 可能是不同事件

4. **获取更多上下文时**：
   - 使用 `memory_cache_id` 调用 `get_memory_cache`
   - 可以获取完整的场景描述

---

## MemoryCache（记忆缓存/场景）

MemoryCache 存储原始文档/场景的信息，是实体和关系的来源。

### 字段详解

| 字段 | 类型 | 说明 |
|------|------|------|
| `id` | string | 缓存唯一标识。格式：`cache_时间戳_随机串`。 |
| `content` | string | 场景的完整上下文。Markdown 格式，包含摘要、状态等信息。 |
| `physical_time` | datetime | 场景/文档的时间。 |
| `activity_type` | string | 活动类型。如"阅读小说"、"处理文档"等，描述该场景的类型。 |

### 使用要点

1. **获取场景详情时**：
   - 使用 `get_memory_cache` 工具
   - 传入实体/关系的 `memory_cache_id`

2. **判断事件上下文时**：
   - 场景内容可以提供完整的上下文
   - 帮助理解实体/关系记录的背景

3. **关联分析时**：
   - 多个记录的 `memory_cache_id` 相同表示来自同一场景
   - 可以用于判断多个记录是否描述同一事件

---

## 常见使用模式

### 模式1：查找实体
```
search_entity(query="名称") 
→ 获取 entity_id 
→ 后续使用 entity_id 进行其他查询
```

### 模式2：查询关系
```
search_entity(A) + search_entity(B)
→ get_entity_relations(entity_id_A, entity_id_B)
→ 若无直接关系，get_relation_paths(A, B)
```

### 模式3：时间排序
```
获取相关关系
→ 按 physical_time 排序
→ 用 memory_cache_id 分组（同组为同一场景）
→ 不同场景按时间排序确定顺序
```

### 模式4：获取上下文
```
找到关系/实体
→ 获取其 memory_cache_id
→ get_memory_cache(cache_id)
→ 阅读完整场景内容
```

---

## 重要提示

1. **entity_id vs id**
   - `entity_id` / `relation_id`：逻辑标识，用于查询
   - `id`：版本标识，用于追溯

2. **physical_time 的意义**
   - 是记录创建的时间，不一定是故事中事件发生的时间
   - 但可以作为时间排序的依据

3. **memory_cache_id 的妙用**
   - 是判断"同一事件"的关键
   - 同一场景的记录共享相同的 memory_cache_id
   - 可以用来分组和关联分析

4. **别名处理**
   - 同一个人可能有多种称呼
   - 搜索时要考虑可能的别名
   - 可能需要多次搜索来覆盖所有可能
