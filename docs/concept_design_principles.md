# Deep-Dream 概念体系设计守则

## Context

Deep-Dream 的核心哲学是 **"Everything is a Concept"**。当前系统已经实现了双标签架构（Entity/Relation 节点同时拥有 `:Concept` 标签），但概念层的认知模型还不够清晰。这份守则将用户的深度思考形式化为可执行的设计原则，覆盖概念定义、提取范围、版本迭代、MENTIONS 链接和查找体验。

---

## 第一条：万物皆概念

**定义**：从文本中提取的一切内容——实体、关系、Episode——都是概念（Concept）。概念是知识图谱的最小语义单元。

| 概念类型 | 标签 | 角色 | 说明 |
|----------|------|------|------|
| 实体概念 | `:Entity :Concept` | `entity` | 一个可独立存在的名词性概念（人、地、物、概念…） |
| 关系概念 | `:Relation :Concept` | `relation` | 两个实体之间的语义关联 |
| 观测概念 | `:Episode :Concept` | `observation` | 一段原始文本 + 状态记忆（记忆缓存） |

**如何应用**：
- Neo4j 双标签已实现，无需改动存储结构
- 所有面向"概念"的逻辑（embedding、搜索、版本迭代）应对三种类型一视同仁
- 当前缺失：Observation 概念没有 embedding，需要补齐

---

## 第二条：实体概念 = 记忆锚点（Memory Anchors）

**规则**：实体提取不只是传统 NER（人名/地名/组织名），而应提取**人类回忆这段文本时会立刻想到的一切**。

**提取范围**（比当前 prompt 更广）：

| 类别 | 当前 | 应有 | 示例 |
|------|------|------|------|
| 具名实体 | ✅ | ✅ | 贾宝玉、长安城 |
| 核心概念 | ✅ | ✅ | 红楼梦、概念对齐 |
| 物品场所 | ✅ | ✅ | 通灵宝玉、荣国府 |
| 事件制度 | ✅ | ✅ | 科举制度、元妃省亲 |
| 技术术语 | ✅ | ✅ | embedding、余弦相似度 |
| 抽象概念 | ✅ | ✅ | 爱情、轮回 |
| **引言/语录** | ❌ | ✅ | "好了歌"、"假作真时真亦假" |
| **URL/路径** | ❌ | ✅ | `core/llm/prompts.py`、`https://...` |
| **核心思想** | ❌ | ✅ | "色空观"、"无我" |
| **特殊事件** | ❌ | ✅ | "黛玉葬花"、"三顾茅庐" |
| **易混淆点** | ❌ | ✅ | "甄英莲/香菱"、"贾政/甄应嘉" |
| **新奇内容** | ❌ | ✅ | 文本中首次出现的独特概念 |
| **错误/异常** | ❌ | ✅ | 文本中提及的错误信息、异常现象 |

**如何应用**：
- 修改 `ENTITY_EXTRACT_SYSTEM` 和 `ENTITY_EXTRACT_USER`，在提取范围中加入上述类别
- 保持"不硬编码具体类型枚举"原则——用通用语言描述"人类会记住的"而非列出具体类别
- 实体命名规则不变：使用文本中的原始名称

---

## 第三条：关系概念 = 自然语言关联

**规则**：关系提取不只是7大类谓语，而应捕获**两个概念之间人类会自然联想到的一切联系**。

**提取范围**（比当前 prompt 更广）：

| 类别 | 当前 | 应有 | 示例 |
|------|------|------|------|
| 归属/组成 | ✅ | ✅ | A属于B |
| 因果/逻辑 | ✅ | ✅ | A导致B |
| 使用/依赖 | ✅ | ✅ | A使用B |
| 对比/对立 | ✅ | ✅ | A vs B |
| 层级关系 | ✅ | ✅ | 父概念-子概念 |
| 举例/说明 | ✅ | ✅ | A是B的例子 |
| 属性关系 | ✅ | ✅ | A是B的属性 |
| **函数调用** | ❌ | ✅ | `A()`调用了`B()` |
| **人物关系** | ❌ | ✅ | "A是B的父亲"（不仅是归属） |
| **交互行为** | ❌ | ✅ | "A与B对话"、"A赠B以物" |
| **行为连接** | ❌ | ✅ | "A做了X后B做了Y" |
| **时序关系** | ❌ | ✅ | "A发生在B之前" |
| **空间关系** | ❌ | ✅ | "A在B的旁边" |

**如何应用**：
- 修改 `RELATION_DISCOVER_SYSTEM` 和 `RELATION_DISCOVER_USER`
- 关系描述（content）应具体到"人类一句话能说清"的程度
- 关系 embedding 已修复为 `"{entity1_name} {content} {entity2_name}"`，无需再改

---

## 第四条：Episode 是特殊的观测概念

**规则**：Episode（观测概念）是所有提取概念的**溯源锚点**。每个 Episode 包含：
1. 原始文本片段（`content`）
2. 状态记忆（记忆缓存 summary + 自我思考）

**核心约束**：**所有从该 Episode 提取的实体和关系，必须通过 `MENTIONS` 边链接回该 Episode。**

```
(Episode) -[:MENTIONS]-> (Entity)
(Episode) -[:MENTIONS]-> (Relation)
```

**当前状态**：
- Entity MENTIONS：✅ 正常工作（`_record_entity_mentions` 在 `_align_entities` 末尾调用）
- Relation MENTIONS：✅ 已修复（并行路径 `orchestrator.py` + 串行路径 `alignment.py`）

**如何应用**：
- 两条代码路径都已覆盖 Relation MENTIONS
- 验证方式：查询 `MATCH (e:Episode)-[:MENTIONS]->(r:Relation) RETURN count(*)` 应 > 0

---

## 第五条：窗口内去重（Intra-Episode Dedup）

**规则**：同一个 Episode（窗口）内，同一概念被多次提取时，只保留一个代表，合并其内容。

**当前实现**：
- 实体：3-pass 核心名称去重 + `already_versioned_family_ids` set 防止同窗口重复版本化
- 关系：`unique_pending_relations` 按实体对去重
- **正确且有效，无需改动**

**如何应用**：
- 保持现有去重逻辑不变
- 去重后只创建一个版本（absolute_id），MENTIONS 只链接一次

---

## 第六条：跨 Episode 对齐（Cross-Episode Alignment）

**规则**：不同 Episode 提取的概念，通过名称+内容对比判断是否为同一概念。

**判断流程**：
1. 类型不同 → 绝不合并（不同类型的概念是不同概念）
2. 名称相同/相似 → 对比 content 是否描述同一对象
3. 三值判断：`same` / `different` / `uncertain`，`uncertain` 当 `different`
4. 确认同一概念 → 分配同一个 `family_id`
5. 确认不同概念 → 保持各自的 `family_id`

**当前实现**：`_align_entities` 和 `_align_relations` 已实现此逻辑，无需改动。

---

## 第七条：版本迭代 = Git 语义

**规则**：概念的版本迭代遵循 Git 的 fast-forward 合并语义。

```
family_id = 稳定逻辑 ID（类似 Git branch）
absolute_id = 版本快照（类似 Git commit hash）
valid_at / invalid_at = 时间窗口
```

### 7.1 新版本创建条件

**每次 Episode 提及 = 新版本**，无论内容是否有变化。版本数应等于 Episode 提及数。

**例外**：同一 Episode 内去重后的概念只创建一个版本（见第五条）。

### 7.2 内容合并规则

1. 新信息是旧内容子集 → 直接复用旧版本（不创建新版本？**不——仍然创建新版本**，但 content 指向旧版本）
2. 新信息有实质增量 → fast-forward merge（最小插入，不改变已有表述）
3. 新版本修正事实错误 → 替换旧版本对应表述
4. 不丢信息

**关键区别**：
- "版本数" = "Episode 提及数"（每次提及都记一笔，即使内容没变）
- "内容变更数" ≤ "版本数"（有些版本只是"又被提及了"，content 没变）

### 7.3 当前实现分析

- ✅ `already_versioned_family_ids` 防止同窗口重复版本化
- ✅ 内容合并使用 fast-forward 语义
- ❌ **版本数 ≠ Episode 提及数**：当新内容是旧内容子集时，当前实现可能跳过版本创建
- ❌ **Find 没有区分"所有版本"和"内容变更版本"**

### 7.4 需要的改动

1. **确保每次跨 Episode 对齐都创建新 absolute_id**，即使内容完全相同
   - 当前代码路径：别名合并（`_try_alias_merge`）和批量对齐（`_process_entity_with_batch_candidates`）需要检查
   - 当 `need_update=false` 时，仍然需要创建新版本（新 absolute_id），只是 content 复用旧版本
2. **Find API 增加版本过滤参数**
   - `include_unchanged: true/false`（默认 false：只返回内容有变更的版本）
   - `include_all_versions: true/false`（默认 false：只返回每个 family 的最新版本）

---

## 第八条：查找（Find）体验

**规则**：Find 应该像人类回忆一样工作——先找到概念，再追溯它的"记忆"。

### 8.1 基础查找

- 搜索概念 → 返回匹配的概念列表（实体 + 关系）
- 每个概念显示：名称、最新内容、版本数、内容变更次数

### 8.2 版本追溯

- 查看概念的所有版本（时间线）
- 过滤：只看内容有变更的版本 vs 所有版本（含"又被提及"的）
- 每个版本显示：content、valid_at、来源 Episode

### 8.3 Episode 追溯

- 通过 MENTIONS 边，从概念追溯到所有提及它的 Episode
- 从 Episode 可以看到原始文本和当时的状态记忆
- **这要求 Relation MENTIONS 先修复**

### 8.4 当前实现

- ✅ 混合搜索（BM25 + 向量）
- ✅ 实体/关系搜索 API
- ❌ 没有版本追溯 API
- ❌ 没有 Episode 追溯 API
- ✅ Relation MENTIONS 已修复

---

## 当前实现 Gap 总结

| Gap | 严重性 | 修复位置 |
|-----|--------|----------|
| ~~Relation MENTIONS 完全缺失~~ | ~~🔴 高~~ | ~~`orchestrator.py` ~L893~~ ✅ 已修复 |
| Observation 概念无 embedding | 🟡 中 | `neo4j/_episodes.py` |
| Prompt 不覆盖记忆锚点 | 🟡 中 | `prompts.py` ENTITY_EXTRACT |
| Prompt 不覆盖行为关联 | 🟡 中 | `prompts.py` RELATION_DISCOVER |
| 版本数 ≠ Episode 提及数 | 🟡 中 | `entity.py`, `relation.py` |
| Find 无版本过滤 | 🟢 低 | `find/hybrid.py`, API 层 |

---

## 验证方式

1. **MENTIONS 完整性**：`MATCH (e:Episode)-[:MENTIONS]->(c) RETURN labels(c), count(*)` — Entity 和 Relation 都应 > 0
2. **版本数 = 提及数**：选一个实体，数它的 absolute_id 数量，和提及它的 Episode 数量，应相等
3. **Embedding 差异化**：两个 content 相同但连接不同实体的 Relation，embedding 应不同
4. **提取覆盖率**：红楼梦节选中"好了歌"等引言应被提取为实体
