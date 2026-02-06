---
name: memory-retrieval
description: >-
  从时序记忆图谱中检索信息并进行推理。
  适用于任何需要查询实体、关系、时间线、场景的问题。
  支持复杂的多步推理、时序分析、场景判断等任务。
trigger_keywords:
  - 查询
  - 记忆
  - 实体
  - 关系
  - 什么时候
  - 第几次
  - 是谁
  - 什么关系
  - 历史
  - 变化
version: 2.0
author: TMG-Agent
---

# 记忆检索技能

从时序记忆图谱中检索信息并进行推理的通用技能。

## 数据模型

### Entity（实体）
| 字段 | 说明 |
|------|------|
| `entity_id` | 实体唯一标识。**注意**：同一个人/物可能有多个不同的 entity_id（如"史强"和"大史"可能是不同记录） |
| `id` | 版本唯一标识（同一实体的不同版本有不同的 id，但共享 entity_id） |
| `name` | 名称。**注意**：可能有别名、简称、昵称 |
| `content` | 自然语言描述 |
| `physical_time` | 该记录的物理时间（可用于时间排序） |
| `memory_cache_id` | 来源场景ID（指向产生该记录的原始文档/场景） |

### Relation（关系）
| 字段 | 说明 |
|------|------|
| `relation_id` | 关系唯一标识（同一关系的不同版本共享此ID） |
| `id` | 版本唯一标识 |
| `content` | 关系的自然语言描述 |
| `physical_time` | 该记录的物理时间（可用于时间排序） |
| `memory_cache_id` | 来源场景ID。**关键**：同一 memory_cache_id 表示来自同一场景/文档 |
| `entity1_absolute_id`, `entity2_absolute_id` | 关联的两个实体的版本ID |

### MemoryCache（场景/来源文档）
| 字段 | 说明 |
|------|------|
| `id` | 场景唯一标识 |
| `content` | 场景的完整上下文（Markdown格式） |
| `physical_time` | 场景/文档的时间 |
| `activity_type` | 活动类型（如"阅读小说"、"处理文档"） |

## 可用工具

### 1. search_entity
搜索实体。支持多种搜索方式。
- `query`: 搜索文本
- `search_mode`: "name_only" / "content_only" / "name_and_content"
- `similarity_method`: "embedding" / "text" / "jaccard"
- `threshold`: 相似度阈值（0-1）

### 2. get_entity_relations
获取实体间的直接关系。
- `entity_id`: 实体ID（必须先通过 search_entity 获取）
- `entity2_id`: 第二个实体ID（可选，指定则只返回两者间的关系）

### 3. get_relation_paths
获取两个实体间的多跳间接路径。当直接关系不存在时使用。
- `entity1_id`, `entity2_id`: 两个实体的 entity_id
- `max_hops`: 最大跳数（1-5）
- `include_relation_content`: 是否包含关系详情

### 4. get_entity_versions
获取实体或关系的历史版本。
- `target_type`: "entity" / "relation"
- `target_id`: entity_id 或 relation_id
- `include_cache_text`: 是否包含原始文本

### 5. get_memory_cache
获取场景/来源文档的完整内容。
- `cache_id`: memory_cache_id

### 6. search_relations
按内容搜索关系。
- `query`: 搜索文本
- `threshold`: 相似度阈值

## 通用原则

### 原则1：搜索要全面
- 一个实体可能有多种名称（别名、简称、昵称、绰号）
- 搜索时考虑用不同方式：精确名称、模糊匹配、语义搜索
- 保留所有可能相关的 entity_id，不要过早排除

### 原则2：理解数据结构
- `physical_time` 是记录时间，可用于确定事件的时间顺序
- `memory_cache_id` 相同 = 来自同一场景/文档 = 可能是同一事件
- 版本历史可以追溯信息的变化过程
- 通过 `get_memory_cache` 可以获取完整的上下文来辅助判断

### 原则3：推理要灵活
- 直接搜索找不到时，尝试间接路径（get_relation_paths）
- 信息不够判断时，获取场景上下文（get_memory_cache）
- 根据返回的数据调整策略，不要死板地按固定流程

### 原则4：按需深入
- 先广度搜索，确认关键实体和关系存在
- 再针对性深入，获取需要的详细信息
- 信息不足时扩大搜索范围或降低阈值

### 原则5：利用上下文
- 每条记录都有 memory_cache_id 指向原始场景
- 当需要判断"是否同一事件"、"具体发生了什么"时，获取场景内容
- 场景内容可以提供更完整的上下文信息

## 示例（启发思路，不限制方法）

### 示例1：简单实体查询
**问**：谁是叶文洁？
**思路**：search_entity("叶文洁") → 返回实体信息即可

### 示例2：关系查询
**问**：A和B是什么关系？
**思路**：
1. 搜索A的所有可能实体
2. 搜索B的所有可能实体
3. 对每对(A_id, B_id)查询直接关系
4. 若无直接关系，尝试多跳路径

### 示例3：时序/顺序问题
**问**：某事件的顺序？第N次？
**思路**：
1. 获取所有相关的关系记录
2. 按 physical_time 排序
3. 用 memory_cache_id 判断哪些是同一场景（同一次事件）
4. 不同场景按时间排序，确定第1次、第2次...

### 示例4：场景判断
**问**：当时发生了什么？
**思路**：
1. 找到相关的实体/关系
2. 获取其 memory_cache_id
3. 调用 get_memory_cache 获取完整场景内容

**这些只是示例。实际推理应根据具体问题和返回数据灵活调整，不必拘泥于固定流程。**

## Guidelines

- 优先使用 embedding 相似度搜索，阈值建议 0.5
- 找不到时尝试降低阈值或换搜索方式
- 关系查询前必须先获取有效的 entity_id
- 注意区分 entity_id（实体唯一）和 id（版本唯一）
- 充分利用 memory_cache_id 进行场景关联分析
