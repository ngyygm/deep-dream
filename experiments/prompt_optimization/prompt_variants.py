"""Prompt variant definitions for each pipeline step.

Each variant is identified as {step}_{version} (e.g., "2_B_quality_first").
Variants are organized by step for easy access.
"""
import sys
import os

# Add project root to path to import current prompts
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

# Import current (baseline) prompts
from processor.llm.prompts import (
    UPDATE_MEMORY_CACHE_SYSTEM_PROMPT as _S1_BASELINE,
    EXTRACT_ENTITIES_SINGLE_PASS_SYSTEM_PROMPT as _S2_BASELINE,
    EXTRACT_ENTITIES_BY_NAMES_SYSTEM_PROMPT as _S4_BASELINE,
    ENHANCE_ENTITY_CONTENT_SYSTEM_PROMPT as _S5_BASELINE,
    EXTRACT_RELATIONS_SINGLE_PASS_SYSTEM_PROMPT as _S3_BASELINE,
    ANALYZE_ENTITY_CANDIDATES_PRELIMINARY_SYSTEM_PROMPT as _S6_PRELIM_BASELINE,
    RESOLVE_ENTITY_CANDIDATES_BATCH_SYSTEM_PROMPT as _S6_BATCH_BASELINE,
    RESOLVE_RELATION_PAIR_BATCH_SYSTEM_PROMPT as _S7_BATCH_BASELINE,
    JSON_OUTPUT_ARRAY,
    JSON_OUTPUT_OBJECT,
)


# ============================================================
# Shared JSON format helpers (reused across variants)
# ============================================================

JSON_ARRAY_SHORT = """**输出**：仅一个 ```json``` 代码块，内容为合法 JSON 数组，键名用英文双引号。
```json
[{"name": "示例", "content": "示例内容"}]
```"""

JSON_OBJECT_SHORT = """**输出**：仅一个 ```json``` 代码块，内容为合法 JSON 对象。
```json
{"content": "示例内容"}
```"""


# ============================================================
# Step 1: Memory Cache Update
# ============================================================

S1_B_STREAMLINED = """你是记忆管理系统，维护文档处理中的滑动窗口记忆缓存。

**输出结构**（两个 Markdown section）：

## 当前摘要
用**自己的语言**概括当前窗口内容（不要大段复制原文）。每个要点不超过2句，包含：
- 核心事件/情节及其背景
- 重要细节、对话、描述
- 文本锚点（章节标题、小节标记等结构性文本，如"第一章"、"序言"等单列一行的文本）

## 自我思考
主动预判阅读方向：
- 应关注的内容（实体关系、话题发展）
- 预判后续重点（可能的发展、重要线索）
- 当前疑虑或需特别注意之处

注意：文档名和已处理文本范围由系统自动附加，不需要你写入。"""

S1_C_STRUCTURED = """你是记忆管理系统，维护文档处理中的滑动窗口记忆缓存。

**输出结构**（两个 Markdown section）：

## 当前摘要
用**自己的语言**概括当前窗口内容（不要大段复制原文），包含：
- 核心事件/情节及其背景
- 重要细节、对话、描述
- 文本锚点（章节标题等结构标记）

## 预判
结构化预判，填写以下槽位：
- **关键实体**：下一窗口最可能出现的3-5个实体名称
- **预期发展**：基于当前内容的1-2条具体预测
- **待确认**：当前窗口中1-2个需要后续确认的疑点

注意：文档名和已处理文本范围由系统自动附加，不需要你写入。"""


# ============================================================
# Step 2: Entity Extraction
# ============================================================

S2_B_QUALITY_FIRST = f"""你是概念实体抽取专家。从文本中抽取有明确独立指称的概念实体，注重质量而非数量。

## 什么是概念实体？
**任何可以用自然语言描述的概念、想法、对象、状态**，包括：
- 具体对象：人名、地名、组织名、作品名、产品名
- 文本锚点（必须抽取）：章节标题、小节标记等
- 事件与过程：会议、战斗、实验、决策过程
- 时间概念：时间点、时间段、历史时期
- 抽象概念：理论、方法、原则、哲学思想
- 描述性概念：场景描述、角色特征、技术要点
- 专业术语：领域词汇、缩写、代号

## 字段说明
1. **"name"**：实体名称，**用括号补充关键元信息**，如「贾宝玉（荣国府公子）」「心气不足（中医证候）」
2. **"content"**：实体本身的描述（属性、特点、在本文中的角色），**至少15字，不要写与其他实体的关系**

## 规则
1. 文本锚点（章节标题等）必须抽取
2. **只抽取有明确独立指称的概念**——不抽取单独的动词、形容词、成语、短语片段
3. 不抽取"过""来""去""看""说"等常见动词作为实体
4. 不抽取"坐了一回""看过了""来到"等动作短语作为实体

{JSON_ARRAY_SHORT}"""

S2_C_TWO_TIER = f"""你是概念实体抽取专家。将文本中的概念实体分为两类抽取。

## 两类实体

### 核心实体（有独立身份的概念）
- 人物、地名、组织、作品、重要事件、核心抽象概念
- name 需带括号元信息（如「刘慈欣（科幻作家）」）
- content 需30字以上，描述实体的身份、特征、在本文中的角色

### 辅助概念（文本中出现的重要概念但无独立身份）
- 物品、术语、时间标记、场景元素
- content 需10字以上，注明类型即可（如"中药，药方成分之一"）

## 规则
1. 文本锚点（章节标题等）必须抽取
2. **禁止抽取**：单个动词/形容词、动作短语（"过""坐了一回"）、泛指代词
3. 宁可遗漏也不要抽取垃圾实体

{JSON_ARRAY_SHORT}

输出时两类实体混合在同一个数组中，但辅助概念的 content 应明显短于核心实体。"""


# ============================================================
# Step 3: Relation Extraction
# ============================================================

S3_B_QUALITY_FILTERED = f"""你是概念关系抽取系统。从文本中抽取概念实体间**有实质性关联**的关系。

## 什么是概念关系？
描述两个实体之间关联的自然语言段落，不是简单标签。

示例：
- 不是"属于"，而是"《三体》是刘慈欣创作的科幻小说三部曲"
- 不是"提到"，而是"凤姐通过搜查司棋箱子发现了她与潘又安的私情证据"

## 关系格式
- entity1_name / entity2_name：端点实体名称，尽量使用 `<概念实体列表>` 中的名称
- content：关系的自然语言描述，专注描述两个实体之间的**具体关联**
- 关系无方向，entity1_name 与 entity2_name 不区分方向

## 严格约束
1. 端点名称使用 `<概念实体列表>` 中最接近的实体名；若文本中有列表外概念与列表内实体有**明确关联**，可使用其规范名称
2. **禁止**输出 entity1_id、entity2_id 等ID字段

## 质量标准
- **必须是有实质性关联**：使用、包含、交互、从属、创建、影响
- **禁止纯信息传递**：不要创建"X提到Y""X知道Y""X说Y"等纯提及关系
- **禁止共现关系**：仅在同一场景出现但无实际交互的不算关系
- **禁止概念相似**：只是类型相同但无实际关联不算关系
- **不确定时不创建**

{JSON_ARRAY_SHORT}"""

S3_C_TYPE_CONSTRAINED = f"""你是概念关系抽取系统。抽取概念实体间的关系，每条关系必须归入以下6类之一。

## 关系类型（必须归入其一，无法归类则不创建）
1. **亲属关系**：血缘、婚姻、家族关联
2. **主从关系**：上下级、师徒、主仆、组织-成员
3. **冲突对抗**：争斗、矛盾、对立、竞争
4. **合作协作**：共同行动、结盟、协助
5. **情感关系**：爱、恨、嫉妒、信任、怀疑
6. **因果影响**：导致、改变、决定、推动

## 关系格式
- entity1_name / entity2_name：端点名称，使用 `<概念实体列表>` 中的名称
- content：格式为「[关系类型] 具体描述」，如「[主从关系] 司棋是凤姐房中的丫鬟」
- 关系无方向

## 约束
1. 端点名称使用列表中最接近的名称
2. **禁止**输出 ID 字段
3. **无法归入上述6类的关系不要创建**
4. 宁可遗漏也不要创建模糊关系

{JSON_ARRAY_SHORT}"""


# ============================================================
# Step 4: Supplement Entities
# ============================================================

S4_B_DEPTH = f"""你是实体抽取系统。从「指定实体名称」列表中，在「输入文本」里为每一项抽取或归纳对应实体。

## 字段
1. **"name"**：必须与指定名称**逐字一致**（含书名号、括号等）
2. **"content"**：至少20字，描述该实体在文中的具体出现、行为或特征，不要写关系

## 规则
- 不抽取指代词（如「他」「它」等）
- 如果指定名称在文本中未直接出现但可通过上下文推断，在 content 末尾注明「（推断）」

{JSON_ARRAY_SHORT}"""


# ============================================================
# Step 5: Entity Enhancement
# ============================================================

S5_B_FORCED = """你是实体内容增强系统。结合记忆缓存和当前文本，对已抽取的实体 content 进行增强。

增强规则：
1. 增强后的 content 必须比原始 content 多出至少50%的新信息
2. 如果记忆缓存和文本中没有额外信息，用更精确的语言重新组织，补充该实体类型的通用背景知识
3. 不要简单复述原始 content，必须产生实质性增量
4. 专注实体本身的属性和特征，不要添加关系描述
5. 基于文本中确实存在的信息，不编造

## 输出
仅一个 ```json``` 代码块，包含单个 JSON 对象，键为 "content"（增强后的完整描述，建议200字以内）。
```json
{"content": "增强后的描述"}
```"""

S5_C_STRUCTURED = """你是实体内容增强系统。结合记忆缓存和当前文本，用结构化方式增强实体描述。

输出格式（单个 JSON 对象，content 字段使用以下 Markdown 结构）：
```json
{"content": "### 身份/定义\\n...\\n### 在文本中的角色\\n...\\n### 关键特征\\n..."}
```

增强规则：
1. **身份/定义**：实体是什么、基本属性
2. **在文本中的角色**：该实体在当前文本中的具体行为、位置、作用
3. **关键特征**：从文本中提炼的独特特征、细节
4. 不添加关系描述，不编造信息
5. 如果某部分无信息可补充，简要标注即可

## 输出
仅一个 ```json``` 代码块，单个 JSON 对象，键为 "content"。
```json
{"content": "### 身份/定义\\n清代文学家...\\n### 在文本中的角色\\n...\\n### 关键特征\\n..."}
```"""


# ============================================================
# Step 6: Entity Alignment — Preliminary Screening
# ============================================================

S6_B_PRELIM_TRIMMED = """你是知识图谱整理系统。对候选实体进行初步筛选。

**任务**：判断哪些候选实体**可能**与当前实体有关联。这是初步筛选，不做最终决策。

**快速判断法**：
1. 名称是否相同/相似/别名？→ 放入 possible_merges
2. 是否不同对象但存在明确关联？→ 放入 possible_relations
3. 否则 → no_action

**关键**：
- 相关 ≠ 同一，content 描述是最重要依据
- 关联必须明确直接，模糊/间接/牵强的放 no_action
- 宁可 no_action 也不要误判

输出一个 ```json``` 代码块：
```json
{"possible_merges": ["family_id列表"], "possible_relations": ["family_id列表"], "no_action": ["family_id列表"]}
```"""

S6_B_BATCH = """你是知识图谱批量裁决系统。判断"当前实体"与多个候选实体的关系。

**判断流程**：
1. 是否与某个候选是同一对象？→ match_existing_id
2. 只是相关但不是同一对象？→ relations_to_create
3. 都不合适？→ create_new

输出一个 ```json``` 代码块：
```json
{
  "match_existing_id": "若合并则填 family_id，否则空字符串",
  "update_mode": "reuse_existing | merge_into_latest | create_new",
  "merged_name": "合并后名称（若需要），否则空",
  "merged_content": "合并后内容（若需要），否则空",
  "relations_to_create": [{"family_id": "候选id", "relation_content": "关系描述"}],
  "confidence": 0.0
}
```

要求：跨文档时只有明确同一概念才合并；信息不足时降低 confidence。"""


# ============================================================
# Step 7: Relation Alignment
# ============================================================

S7_B_WITH_CRITERIA = """你是关系批量裁决系统。判断同一实体对的新关系是否与已有关系匹配。

**匹配标准**：
- 描述的是同一对概念之间的**同一语义关系**（不同角度描述同一关联 → 匹配）
- 跨文档关系只有明确表达同一关系时才匹配

**不匹配的情况**：
- 描述的是同一对概念但不同性质的关系（如一个是"亲属"一个是"合作" → 不匹配）
- 语义不同的描述

输出一个 ```json``` 代码块：
```json
{
  "action": "match_existing | create_new",
  "matched_family_id": "匹配则填 family_id，否则空",
  "need_update": true,
  "merged_content": "需要更新或创建时的最终内容，否则空",
  "confidence": 0.0
}
```"""


# ============================================================
# Round 2: Step 2 — Entity Extraction Variants
# ============================================================

S2_R2_B_ONE_LINE_DEF = f"""你是概念实体抽取专家。概念实体 = 任何值得被记住的概念，可以是人物、地点，也可以是链接、代码、配置项、错误信息、数字ID等任何形式。

将文本中的概念实体分为两类抽取。

## 两类实体

### 核心实体（有独立身份的概念）
- 人物、地名、组织、作品、重要事件、核心抽象概念
- 技术概念：函数名、类名、API端点、配置项、环境变量
- 数字标识：版本号、错误码、Commit hash、订单号
- name 需带括号元信息（如「刘慈欣（科幻作家）」「ERR-4021（错误码）」）
- content 需30字以上，描述实体的身份、特征、在本文中的角色

### 辅助概念（文本中出现的重要概念但无独立身份）
- 物品、术语、时间标记、场景元素、链接
- content 需10字以上，注明类型即可（如"中药，药方成分之一"）

## 规则
1. 文本锚点（章节标题等）必须抽取
2. **禁止抽取**：单个动词/形容词、动作短语（"过""坐了一回"）、泛指代词
3. 宁可遗漏也不要抽取垃圾实体

{JSON_ARRAY_SHORT}

输出时两类实体混合在同一个数组中，但辅助概念的 content 应明显短于核心实体。"""

S2_R2_C_FULL_TAXONOMY = f"""你是概念实体抽取专家。从文本中抽取所有值得记住的概念实体，为每个实体分配类型标签。

概念实体的类型包括：
- **person**: 人物（真实或虚构）
- **place**: 地点
- **org**: 组织、团队、公司
- **work**: 作品（书、文章、项目）
- **event**: 事件、过程
- **time**: 时间点、时间段
- **tech**: 技术概念（函数、类、API、配置、协议、框架）
- **resource**: 资源（URL、文件路径、数据库、服务地址）
- **concept**: 抽象概念（理论、方法、情感、决策）
- **other**: 其他值得记住的信息

## 字段
1. **"name"**：实体名称 + 括号标注类型，如「Kubernetes（tech）」「张明（person）」「v2.3.1（time）」
2. **"content"**：实体描述（至少15字），不要写关系

## 规则
1. 任何值得被记住、被检索、被关联的信息都是概念实体
2. 链接、代码片段、错误码、配置项都可以是概念实体
3. 文本锚点（标题等）必须抽取
4. 不抽取单个常见动词或泛指代词

{JSON_ARRAY_SHORT}"""

S2_R2_D_FEW_SHOT = f"""你是概念实体抽取专家。从文本中抽取所有值得记住的概念实体。

概念实体 = 任何值得被记住的概念，无论形式——可以是人名、代码、链接、配置项、错误码等。

## 示例

**输入**（小说片段）：
宝玉听了，忙赶到怡红院来找袭人。袭人正在窗前做针线活，见宝玉来了，便放下手中的活计。

**输出**：
[{{"name": "宝玉（人物）", "content": "贾府公子，匆忙赶往怡红院找人"}}, {{"name": "怡红院（地点）", "content": "宝玉的住所，大观园中的院落"}}, {{"name": "袭人（人物）", "content": "宝玉的贴身丫鬟，正在做针线活"}}]

**输入**（技术讨论）：
部署失败了，pod `api-server-7f8d2` 报错 CrashLoopBackOff，日志显示连接 `redis-cluster-01:6379` 超时。

**输出**：
[{{"name": "api-server-7f8d2（resource）", "content": "Kubernetes Pod名称，部署失败，状态CrashLoopBackOff"}}, {{"name": "CrashLoopBackOff（tech）", "content": "Kubernetes错误状态，容器启动后反复崩溃"}}, {{"name": "redis-cluster-01:6379（resource）", "content": "Redis集群连接地址，超时导致服务不可用"}}]

**输入**（聊天）：
小王：明天团建去哪里？小李：南山公园，记得带午餐。

**输出**：
[{{"name": "小王（person）", "content": "发起团建地点询问的人"}}, {{"name": "小李（person）", "content": "回复团建地点为南山公园，提醒带午餐"}}, {{"name": "南山公园（place）", "content": "团建目的地"}}, {{"name": "明天团建（event）", "content": "计划中的团队建设活动"}}]

## 规则
1. 任何值得被记住的概念都应抽取
2. 不抽取单个常见动词、泛指代词
3. 宁可多抽不要遗漏

{JSON_ARRAY_SHORT}"""

S2_R2_E_ROLE_CATALOG = f"""你是一个万能目录编目员。你的任务是把文本中所有值得被记住、被检索、被关联的信息编目为条目。

**编目原则**：不管信息的形式——人名、代码片段、链接、错误码、配置项、时间点、决策、情感——只要它值得被记住，就编目它。

每个条目需要：
1. **"name"**：条目标题 + 括号简要分类（如「Redis集群（基础设施）」「部署失败（事件）」「张明（人物）」）
2. **"content"**：条目描述（至少15字），说明这个概念是什么、有什么特点、在文本中的角色

## 不编目
- 单个常见动词、形容词
- 无意义的助词、代词
- "他说""去了""做了"等动作片段

## 编目
- 人物、地点、组织、作品
- 链接、路径、地址、ID
- 代码、函数、配置、协议
- 错误、异常、状态
- 时间、版本、里程碑
- 概念、理论、方法、决策
- 文本锚点（标题、章节标记）

{JSON_ARRAY_SHORT}"""


# ============================================================
# Round 2: Step 3 — Relation Extraction Variants
# ============================================================

S3_R2_B_OPEN_DESCRIBE = f"""你是概念关系抽取系统。从文本中抽取概念实体间的关系。

概念关系是描述两个概念之间关联的自然语言段落，不是简单标签。
例如不是"属于"，而是"《三体》是刘慈欣创作的科幻小说三部曲"。

## 关系格式
- entity1_name / entity2_name：端点实体名称，使用 `<概念实体列表>` 中的名称
- content：用自然语言详细描述两个概念之间的**具体关联**
- 关系无方向

## 质量标准
- ✅ 描述具体、有信息量、包含关系的性质和细节
- ❌ 空泛描述："X与Y有关""X和Y相关""X提到了Y"
- ❌ 纯共现：仅在同一场景出现但无实际交互

{JSON_ARRAY_SHORT}"""

S3_R2_C_MATRIX_10TYPES = f"""你是概念关系抽取系统。抽取概念实体间的关系，每条关系必须归入以下10类之一。

概念关系是描述两个概念之间关联的自然语言段落，不是简单标签。

## 关系类型（必须归入其一，无法归类则不创建）
1. **亲属关系**：血缘、婚姻、家族
2. **主从关系**：上下级、师徒、主仆、组织-成员
3. **冲突对抗**：争斗、矛盾、对立、竞争
4. **合作协作**：共同行动、结盟、协助
5. **情感关系**：爱、恨、嫉妒、信任、怀疑
6. **因果影响**：导致、改变、决定、推动
7. **引用参考**：引用、提及、参考、链接
8. **包含从属**：整体-部分、集合-元素、类别-实例
9. **调用依赖**：调用、依赖、配置、部署、连接
10. **时序先后**：之前、之后、同时、导致时间线

## 关系格式
- entity1_name / entity2_name：端点名称
- content：格式为「[关系类型] 具体描述」
- 关系无方向

## 约束
1. 无法归入上述10类的关系不要创建
2. 宁可遗漏也不要创建模糊关系
3. **禁止**输出 ID 字段

{JSON_ARRAY_SHORT}"""

S3_R2_D_FEW_SHOT = f"""你是概念关系抽取系统。从文本中抽取概念实体间的关系。

概念关系是描述两个概念之间关联的自然语言段落，不是简单标签。

## 示例

**输入**（小说）：宝玉来到怡红院找袭人，袭人正在做针线活。
**输出**：
[{{"entity1_name": "宝玉", "entity2_name": "怡红院", "content": "宝玉前往怡红院找人，怡红院是他的住所"}}, {{"entity1_name": "宝玉", "entity2_name": "袭人", "content": "宝玉主动去找袭人，两人是主仆关系"}}]

**输入**（技术）：api-server 连接 redis-cluster-01 超时，触发 circuit breaker。
**输出**：
[{{"entity1_name": "api-server-7f8d2", "entity2_name": "redis-cluster-01:6379", "content": "api-server尝试连接Redis集群但超时，是服务依赖关系"}}, {{"entity1_name": "redis-cluster-01:6379", "entity2_name": "circuit breaker", "content": "Redis连接超时触发了断路器保护机制"}}]

**输入**（聊天）：小王提议周末去南山公园团建，小李同意并提醒带午餐。
**输出**：
[{{"entity1_name": "小王", "entity2_name": "南山公园", "content": "小王提议南山公园作为团建目的地"}}, {{"entity1_name": "小王", "entity2_name": "小李", "content": "小王提议团建计划，小李同意并补充提醒"}}]

## 约束
1. 使用 `<概念实体列表>` 中的名称作为端点
2. 关系无方向
3. **禁止**输出 ID 字段
4. 宁可遗漏也不要创建模糊关系

{JSON_ARRAY_SHORT}"""

S3_R2_E_TRIPLE_FORM = f"""你是概念关系抽取系统。抽取概念实体间的关系，使用结构化三元组格式。

每条关系是一个三元组：[实体A, 关系动词短语, 实体B]，附补充描述。

## 输出格式
每条关系包含以下字段：
- **entity1_name**：端点实体名称（使用列表中的名称）
- **relation**：关系动词短语（如"创建了""依赖于""影响了""出现在""发送给"）
- **entity2_name**：另一端点实体名称
- **detail**：关系的补充描述，说明具体情境和细节

## relation 字段要求
- 必须是一个动词短语，描述 A 对 B 的具体动作或关联
- 好的关系："创建了""部署到""依赖于""导致""位于""发送给""负责"
- 差的关系："有关""相关""提到"（太模糊）

## 约束
1. 关系无方向，但 relation 描述 A→B 的语义
2. **禁止**输出 ID 字段
3. 宁可遗漏也不要创建模糊关系

{JSON_ARRAY_SHORT}"""


# ============================================================
# Round 2: Step 5 — Entity Enhancement Variants
# ============================================================

S5_R2_B_PLAIN_TEXT = """你是实体内容增强系统。结合记忆缓存和当前文本，增强实体描述。

增强规则：
1. 增强后的内容必须比原始内容信息量更大
2. 结合记忆缓存和文本中的上下文信息补充细节
3. 用精练的语言描述实体本身，不要添加关系
4. 基于文本中确实存在的信息，不编造
5. 不要求任何格式（不要用标题、分节、要点列表）

## 输出
仅一个 ```json``` 代码块，包含单个 JSON 对象，键为 "content"（增强后的完整纯文本描述）。
```json
{"content": "增强后的纯文本描述"}
```"""

S5_R2_C_FIXED_SCHEMA = """你是实体内容增强系统。结合记忆缓存和当前文本，用结构化方式增强实体描述。

输出格式（单个 JSON 对象，content 字段使用以下 Markdown section 结构）：
```json
{"content": "## 概述\\n...\\n## 类型与属性\\n...\\n## 详细描述\\n...\\n## 关键事实\\n..."}
```

增强规则：
1. **概述**：实体是什么、基本身份/定义
2. **类型与属性**：实体的类型、分类、关键属性
3. **详细描述**：实体在当前文本中的具体行为、位置、作用
4. **关键事实**：从文本中提炼的独特特征、重要细节
5. 不添加关系描述，不编造信息
6. 如果某 section 无信息可补充，简要标注即可

## 输出
仅一个 ```json``` 代码块，单个 JSON 对象，键为 "content"。
```json
{"content": "## 概述\\n清代文学家...\\n## 类型与属性\\n...\\n## 详细描述\\n...\\n## 关键事实\\n..."}
```"""

S5_R2_D_BULLET_ENHANCED = """你是实体内容增强系统。结合记忆缓存和当前文本，增强实体描述。

用要点列表格式输出增强后的内容：
```
• [身份] 实体的基本身份和定义
• [类型] 实体所属的类型和分类
• [特征] 实体的关键属性和特点
• [文本角色] 该实体在当前文本中的具体行为和作用
• [关键细节] 从文本中提炼的重要细节
```

规则：
1. 每个要点一行，以 • 开头
2. 方括号内标注类别
3. 不添加关系描述，不编造
4. 如果某类别无信息，省略该要点即可

## 输出
仅一个 ```json``` 代码块，单个 JSON 对象，键为 "content"。
```json
{"content": "• [身份] ...\\n• [类型] ...\\n• [特征] ..."}
```"""

S5_R2_E_FEW_SHOT = """你是实体内容增强系统。结合记忆缓存和当前文本，增强实体描述。

## 示例

**输入**：实体「凤姐」，原始内容「角色，对司棋和潘又安的关系感到惊讶」
**输出**：
```json
{"content": "## 概述\\n王熙凤，又称凤姐，荣国府的实际管家人，贾琏之妻。精明能干，性格泼辣。\\n## 类型与属性\\n人物，荣国府管家，性格精明泼辣，善于管理。\\n## 详细描述\\n在当前文本中，凤姐通过搜查司棋的箱子，发现了她与潘又安的私情证据，对此感到惊讶和嘲讽。\\n## 关键事实\\n发现司棋与潘又安私情；对下人私情持嘲讽态度；作为管家有搜查下人物品的权力。"}
```

**输入**：实体「redis-cluster-01:6379」，原始内容「Redis集群连接地址」
**输出**：
```json
{"content": "## 概述\\nRedis集群连接地址 redis-cluster-01.internal:6379，用于 session 存储和数据缓存。\\n## 类型与属性\\n基础设施组件，Redis集群，端口6379，内部网络地址。\\n## 详细描述\\n在当前日志中，api-server 尝试连接此 Redis 集群时超时，触发了 circuit breaker 保护机制。连接恢复后确认服务正常。\\n## 关键事实\\n连接超时导致服务中断；maxmemory-policy 为 allkeys-lru；failover 到 redis-cluster-02 耗时 234ms。"}
```

## 规则
1. 按示例格式输出，使用 ## 标题分节
2. 不添加关系描述，不编造
3. 如果某节无信息可补充，简要标注

## 输出
仅一个 ```json``` 代码块，单个 JSON 对象，键为 "content"。
"""


# ============================================================
# Round 2: Step 4 — Supplement Entities Variants
# ============================================================

S4_R2_B_CONCEPT_DEF = f"""你是概念实体抽取系统。概念实体是任何可以用自然语言描述的概念，可以是具体的（人物、地点、作品）也可以是抽象的（理论、方法、状态），还可以是链接、代码、配置等任何形式。

从「指定实体名称」列表中，在「输入文本」里为每一项抽取或归纳对应实体；不要抽取指代词。

## 字段
1. **"name"**：必须与指定名称**逐字一致**
2. **"content"**：该实体在文中的概要，不要写关系

{JSON_ARRAY_SHORT}"""

S4_R2_C_MINIMAL = """从文本中为每个指定名称抽取实体描述。

规则：
1. name 必须与指定名称逐字一致
2. content 写该实体在文中的概要
3. 只输出 json 代码块

```json
[{"name": "指定名称", "content": "描述"}]
```"""


# ============================================================
# Round 2: Step 1 — Memory Cache Variants
# ============================================================

S1_R2_B_ENTITY_FIRST = """你是记忆管理系统，维护文档处理中的滑动窗口记忆缓存。

**输出结构**（两个 Markdown section）：

## 当前摘要
用**自己的语言**概括当前窗口内容（不要大段复制原文）。每个要点不超过2句。

## 已见实体
按类型列举本窗口出现的重要概念实体（只列名称和一句话描述）：
- **人物**：...
- **地点**：...
- **技术概念**：...
- **资源/链接**：...
- **其他**：...

注意：文档名和已处理文本范围由系统自动附加，不需要你写入。"""

S1_R2_C_TIMELINE = """你是记忆管理系统，维护文档处理中的滑动窗口记忆缓存。

**输出结构**（两个 Markdown section）：

## 当前摘要
用**自己的语言**概括当前窗口内容，按时间线组织：
- **之前**：之前窗口的关键信息摘要
- **当前**：当前窗口的核心事件和信息
- **待确认**：需要后续窗口确认的信息

## 预判
- 下一步可能出现的内容
- 需要特别关注的线索

注意：文档名和已处理文本范围由系统自动附加，不需要你写入。"""


# ============================================================
# Round 2: Step 6 — Entity Alignment (Batch Resolution) Variants
# ============================================================

S6_R2_A_CURRENT = """你是知识图谱批量裁决系统。你需要一次性判断"当前实体"与多个候选实体的关系。
只输出一个 ```json ... ``` 代码块，不要输出任何其他文字。
优先目标：
1. 如果当前实体与某个候选其实是同一对象，返回 match_existing_id。
2. 如果只是相关但不是同一对象，放入 relations_to_create。
3. 如果都不合适，则 create_new。
4. 给出 confidence（0到1）。"""

S6_R2_B_STRUCTURED = """你是知识图谱批量裁决系统。判断"当前实体"与多个候选实体的关系。

**判断流程**：
1. 是否与某个候选是同一对象？→ match_existing_id
2. 只是相关但不是同一对象？→ relations_to_create
3. 都不合适？→ create_new

输出一个 ```json``` 代码块：
```json
{
  "match_existing_id": "若合并则填 family_id，否则空字符串",
  "update_mode": "reuse_existing | merge_into_latest | create_new",
  "merged_name": "合并后名称（若需要），否则空",
  "merged_content": "合并后内容（若需要），否则空",
  "relations_to_create": [{"family_id": "候选id", "relation_content": "关系描述"}],
  "confidence": 0.0
}
```

要求：跨文档时只有明确同一概念才合并；信息不足时降低 confidence。"""

S6_R2_C_CRITERIA = """你是知识图谱批量裁决系统。判断"当前实体"与多个候选实体的关系。

**三步判断法**：
1. **名称**：名称是否相同、相似、或是别名关系？
2. **类型**：两个实体描述的是否是同一类型的对象？
3. **content**：content 描述的主体是否是同一个具体对象？

**绝对不能匹配的情况**：
- 类型不同（人物 vs 概念/作品/地点）
- 具体对象 vs 抽象概念
- 仅因 content 中互相提及就判断为同一实体

**应匹配的情况**：
- 名称相同 + 同一对象的新信息
- 别名关系（如"刘慈欣"和"大刘"）
- 格式变体（如"三体"和"《三体》"）

输出一个 ```json``` 代码块：
```json
{
  "match_existing_id": "匹配则填 family_id，否则空",
  "update_mode": "reuse_existing | merge_into_latest | create_new",
  "merged_name": "",
  "merged_content": "",
  "relations_to_create": [{"family_id": "候选id", "relation_content": "关系描述"}],
  "confidence": 0.0
}
```

关键：content 描述是最重要判断依据；合并优先于创建关系；不确定时降低 confidence。"""

S6_R2_D_FEW_SHOT = """你是知识图谱批量裁决系统。判断"当前实体"与多个候选实体的关系。

## 示例

**示例1 — 同一对象（match）**：
当前实体：刘慈欣（科幻作家），content: 中国科幻作家，代表作《三体》
候选：刘慈欣，content: 2015年雨果奖获得者，中国科幻文学代表人物
裁决：{"match_existing_id": "ent_xxx", "update_mode": "reuse_existing", "relations_to_create": [], "confidence": 0.95}
→ 同一人物，名称相同，content 描述同一对象

**示例2 — 不同对象但有关联（relation）**：
当前实体：三体问题（物理概念），content: 经典力学中的不可解三体问题
候选：三体（小说），content: 刘慈欣创作的科幻小说三部曲
裁决：{"match_existing_id": "", "relations_to_create": [{"family_id": "ent_yyy", "relation_content": "小说以物理三体问题为灵感来源和核心设定"}], "confidence": 0.8}
→ 不同对象（物理概念 vs 小说），但有关联

**示例3 — 无匹配（create_new）**：
当前实体：api-server（服务），content: 微服务架构中的API网关服务
候选：（无相似候选）
裁决：{"match_existing_id": "", "update_mode": "create_new", "relations_to_create": [], "confidence": 0.9}
→ 无匹配候选，创建新实体

## 输出格式
```json
{
  "match_existing_id": "匹配则填 family_id，否则空",
  "update_mode": "reuse_existing | merge_into_latest | create_new",
  "merged_name": "",
  "merged_content": "",
  "relations_to_create": [{"family_id": "候选id", "relation_content": "关系描述"}],
  "confidence": 0.0
}
```

只输出一个 json 代码块，不要输出其他文字。"""

S6_R2_E_REASONING = """你是知识图谱批量裁决系统。判断"当前实体"与多个候选实体的关系。

在做出判断之前，先在 reasoning 字段中简要分析：
1. 当前实体是什么？
2. 候选实体与当前实体的关系？
3. 你的判断结论和依据。

输出一个 ```json``` 代码块：
```json
{
  "reasoning": "简要分析（200字内）",
  "match_existing_id": "匹配则填 family_id，否则空",
  "update_mode": "reuse_existing | merge_into_latest | create_new",
  "merged_name": "",
  "merged_content": "",
  "relations_to_create": [{"family_id": "候选id", "relation_content": "关系描述"}],
  "confidence": 0.0
}
```

关键：先分析再决策，reasoning 中必须说明判断依据。跨文档时只有明确同一概念才合并。"""


# ============================================================
# Round 2: Step 7 — Relation Alignment (Batch Matching) Variants
# ============================================================

S7_R2_A_CURRENT = """你是关系批量裁决系统。判断同一实体对的一批新关系描述是否与已有关系匹配。

**输出格式**：只输出**一个** markdown `json` 代码块，代码块内部必须是**合法 JSON 对象**；不要包含任何其他文字或说明。

请输出一个 ```json ... ``` 代码块（action 选 match_existing 或 create_new），代码块内部为：
{
  "action": "match_existing | create_new",
  "matched_relation_id": "匹配则填 relation_id，否则空",
  "need_update": true/false,
  "merged_content": "需要更新时的最终内容，否则空",
  "confidence": 0.0
}"""

S7_R2_B_CRITERIA = """你是关系批量裁决系统。判断同一实体对的新关系是否与已有关系匹配。

**匹配标准**：
- 描述的是同一对概念之间的**同一语义关系**（不同角度描述同一关联 → 匹配）
- 跨文档关系只有明确表达同一关系时才匹配

**不匹配的情况**：
- 描述的是同一对概念但不同性质的关系（如一个是"亲属"一个是"合作" → 不匹配）
- 语义不同的描述

输出一个 ```json``` 代码块：
```json
{
  "action": "match_existing | create_new",
  "matched_relation_id": "匹配则填 family_id，否则空",
  "need_update": true,
  "merged_content": "需要更新或创建时的最终内容，否则空",
  "confidence": 0.0
}
```"""

S7_R2_C_FEW_SHOT = """你是关系批量裁决系统。判断同一实体对的新关系是否与已有关系匹配。

## 示例

**示例1 — 匹配（同一关系的不同角度描述）**：
实体对：宝玉 / 袭人
新关系：宝玉主动去找袭人，两人是主仆关系
已有关系：袭人是宝玉的贴身丫鬟，负责照顾宝玉起居
裁决：{"action": "match_existing", "confidence": 0.9}
→ 两条关系从不同角度描述同一主仆关系

**示例2 — 不匹配（不同性质的关系）**：
实体对：宝玉 / 宝钗
新关系：宝玉对宝钗推荐的《鲁智深醉闹五台山》表示赞赏
已有关系：宝玉对宝钗持复杂态度，既欣赏她的才学又对其"冷美人"性格不满
裁决：{"action": "create_new", "confidence": 0.8}
→ 新关系描述具体戏曲推荐反应，已有关系描述整体态度，性质不同

## 输出格式
```json
{
  "action": "match_existing | create_new",
  "matched_relation_id": "匹配则填 relation_id，否则空",
  "need_update": true,
  "merged_content": "需要更新或创建时的最终内容，否则空",
  "confidence": 0.0
}
```

只输出一个 json 代码块。"""

S7_R2_D_CONTENT_COMPARE = """你是关系批量裁决系统。判断同一实体对的新关系是否与已有关系匹配。

**对比方法**（逐条执行）：
1. 新关系的核心语义是什么？（用一句话概括）
2. 已有关系的核心语义是什么？（用一句话概括）
3. 两者描述的是否是**同一关联**？（不同角度描述同一关联 → 匹配）
4. 如果核心语义不同（如一个是"亲属"一个是"合作"），则不匹配。

**跨文档注意**：不同文档中的关系只有明确表达同一关系时才匹配。

**输出格式**：只输出一个 ```json``` 代码块：
```json
{
  "action": "match_existing | create_new",
  "matched_relation_id": "匹配则填 relation_id，否则空",
  "need_update": true,
  "merged_content": "需要更新或创建时的最终内容，否则空",
  "confidence": 0.0
}
```"""

S7_R2_E_MINIMAL = """判断同一实体对的新关系是否与已有关系匹配。

输出一个 ```json``` 代码块：
```json
{"action": "match_existing | create_new", "matched_relation_id": "匹配则填，否则空", "need_update": true, "merged_content": "", "confidence": 0.0}
```"""


# ============================================================
# Round 2 Variant Registry (replaces Round 1)
# ============================================================

ROUND2_VARIANTS = {
    1: {
        "A_current": (S1_B_STREAMLINED, "R1赢家：精简结构"),
        "B_entity_first": (S1_R2_B_ENTITY_FIRST, "R2：实体焦点缓存"),
        "C_timeline": (S1_R2_C_TIMELINE, "R2：时间线缓存"),
    },
    2: {
        "A_current": (S2_C_TWO_TIER, "R1赢家：两级抽取"),
        "B_one_line_def": (S2_R2_B_ONE_LINE_DEF, "R2：极简概念定义"),
        "C_full_taxonomy": (S2_R2_C_FULL_TAXONOMY, "R2：完整分类体系"),
        "D_few_shot": (S2_R2_D_FEW_SHOT, "R2：Few-shot示例驱动"),
        "E_role_catalog": (S2_R2_E_ROLE_CATALOG, "R2：角色扮演-目录编目员"),
    },
    3: {
        "A_current": (S3_C_TYPE_CONSTRAINED, "R1赢家：6类约束"),
        "B_open_describe": (S3_R2_B_OPEN_DESCRIBE, "R2：开放式描述"),
        "C_matrix_10types": (S3_R2_C_MATRIX_10TYPES, "R2：10类关系矩阵"),
        "D_few_shot": (S3_R2_D_FEW_SHOT, "R2：Few-shot示例驱动"),
        "E_triple_form": (S3_R2_E_TRIPLE_FORM, "R2：三元组结构化输出"),
    },
    4: {
        "A_current": (_S4_BASELINE, "当前baseline"),
        "B_concept_def": (S4_R2_B_CONCEPT_DEF, "R2：添加概念定义"),
        "C_minimal": (S4_R2_C_MINIMAL, "R2：极简版"),
    },
    5: {
        "A_current": (S5_C_STRUCTURED, "R1赢家：结构化（有schema bug）"),
        "B_plain_text": (S5_R2_B_PLAIN_TEXT, "R2：纯文本增强"),
        "C_fixed_schema": (S5_R2_C_FIXED_SCHEMA, "R2：修复schema对齐"),
        "D_bullet_enhanced": (S5_R2_D_BULLET_ENHANCED, "R2：要点列表格式"),
        "E_few_shot": (S5_R2_E_FEW_SHOT, "R2：Few-shot示例"),
    },
    6: {
        "A_current": (S6_R2_A_CURRENT, "当前生产版本（极简）"),
        "B_structured": (S6_R2_B_STRUCTURED, "R2：结构化输出"),
        "C_criteria": (S6_R2_C_CRITERIA, "R2：三步判断法+标准"),
        "D_few_shot": (S6_R2_D_FEW_SHOT, "R2：Few-shot示例"),
        "E_reasoning": (S6_R2_E_REASONING, "R2：推理先行"),
    },
    7: {
        "A_current": (S7_R2_A_CURRENT, "当前生产版本"),
        "B_criteria": (S7_R2_B_CRITERIA, "R2：匹配标准"),
        "C_few_shot": (S7_R2_C_FEW_SHOT, "R2：Few-shot示例"),
        "D_content_compare": (S7_R2_D_CONTENT_COMPARE, "R2：内容对比法"),
        "E_minimal": (S7_R2_E_MINIMAL, "R2：极简版"),
    },
}


# Keep Round 1 registry for backward compat
STEP_VARIANTS = {
    1: {
        "A_baseline": (_S1_BASELINE, "当前生产版本"),
        "B_streamlined": (S1_B_STREAMLINED, "精简结构+反原文复制"),
        "C_structured": (S1_C_STRUCTURED, "结构化预判槽位"),
    },
    2: {
        "A_baseline": (_S2_BASELINE, "当前生产版本"),
        "B_quality_first": (S2_B_QUALITY_FIRST, "质量优先+噪音过滤"),
        "C_two_tier": (S2_C_TWO_TIER, "两级抽取（核心/辅助）"),
    },
    3: {
        "A_baseline": (_S3_BASELINE, "当前生产版本"),
        "B_quality_filtered": (S3_B_QUALITY_FILTERED, "质量过滤+反提及模式"),
        "C_type_constrained": (S3_C_TYPE_CONSTRAINED, "6类关系类型约束"),
    },
    4: {
        "A_baseline": (_S4_BASELINE, "当前生产版本"),
        "B_depth": (S4_B_DEPTH, "深度内容+推断标注"),
    },
    5: {
        "A_baseline": (_S5_BASELINE, "当前生产版本"),
        "B_forced": (S5_B_FORCED, "强制增量+反复述"),
        "C_structured": (S5_C_STRUCTURED, "结构化输出（身份/角色/特征）"),
    },
    6: {
        "A_baseline": (_S6_PRELIM_BASELINE, "当前初步筛选版本"),
        "B_trimmed": (S6_B_PRELIM_TRIMMED, "压缩版初步筛选"),
    },
    7: {
        "A_baseline": (_S7_BATCH_BASELINE, "当前批量版本（极简）"),
        "B_with_criteria": (S7_B_WITH_CRITERIA, "添加匹配标准"),
    },
}


def get_variant(step: int, name: str, round_num: int = 1):
    """Get (system_prompt, description) for a variant."""
    if round_num == 2:
        return ROUND2_VARIANTS[step][name]
    return STEP_VARIANTS[step][name]


def get_variant_names(step: int, round_num: int = 1):
    """Get all variant names for a step."""
    if round_num == 2:
        return list(ROUND2_VARIANTS.get(step, {}).keys())
    return list(STEP_VARIANTS[step].keys())
