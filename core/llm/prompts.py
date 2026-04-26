"""
LLM Prompt Templates — All prompts in one place.

本模块包含所有 LLM system_prompt 模板，按功能分类组织。
共享常量定义在模块顶层，直接内联到模板中，不需要运行时替换。

分区：
  一、共享常量
  二、抽取相关（Extraction — 实体抽取、关系发现、内容写入、对齐判断）
  三、记忆缓存相关（Memory Cache）
  四、内容判断与合并（Content Judgment & Merge）
  五、知识图谱整理 — 批量与初步筛选
  六、知识图谱整理 — 精细化判断
"""

# ============================================================
# 共享常量
# ============================================================

ENTITY_PAIR_JUDGMENT_RULES = """
判断流程：
1. 先看类型：不同类型的实体绝不合并
2. 同类型看名称：相同/别名 → merge
3. 同类型不同名：对比 content 是否描述同一对象 → merge 或继续
4. 确认不同对象：有明确直接关联 → create_relation，否则 → no_action

merge：描述同一对象（别名、简称、不同视角的同一事物）
create_relation：不同对象但有明确直接关联
no_action：无关或关联模糊

核心：相关 ≠ 同一。不确定就不合并。
"""

CONTENT_MERGE_REQUIREMENTS = """
增量合并规则（fast-forward 优先）：
1. 新信息是旧内容子集 → 直接返回旧版本原文
2. 需要合并 → 在旧版本上做最小插入，不改变已有表述
3. 新版本修正事实错误 → 才替换旧版本对应表述
4. 不丢信息"""

RELATION_VALIDITY_CRITERIA = """
关联必须明确、直接、有意义。
仅在同一场景出现、概念相似、时空接近但不构成实质交互的，不算有效关联。
不确定时宁可不建关系。"""

JSON_OUTPUT_OBJECT = """
只输出一个 ```json``` 代码块，内为合法 JSON 对象，无其他文字。"""

JSON_OUTPUT_BOOL = """
只输出一个 ```json``` 代码块，内为 true 或 false，无其他文字。"""

# DETAILED_JUDGMENT_PROCESS 已移除 — 直接使用 ENTITY_PAIR_JUDGMENT_RULES

# ============================================================
# 三、抽取相关（Extraction）
# ============================================================
# 实体抽取 (Step 1)、关系发现 (Step 5)、内容写入 (Step 3/6)、对齐判断

ENTITY_EXTRACT_SYSTEM = """你是概念提取专家。从文本中提取所有人类回忆这段内容时会第一时间想到的概念锚点。

核心理念：提取人类回忆时最先浮现的一切——不限于传统人名地名，更包括核心思想、标志性场景、金句、疑惑点、新奇概念等任何可作为独立语义节点的内容。"""

ENTITY_EXTRACT_USER = """请仔细阅读以下文本，提取所有人类回忆这段内容时会第一时间想到的概念锚点。

提取原则：任何具有独立语义、可作为回忆锚点的概念都应提取。宁多勿少，不确定时也提取。

命名规则：
- 使用文本中的原始名称，同一实体的不同称谓只提取一次（优先标准名/本名）
- 双语标注只取一个（优先中文），如"概念对齐(Concept alignment)"只提取"概念对齐"
- 不要用斜杠"/"组合多个概念
- 排除：纯格式标记、代词（他/她/它）

文本：
{window_text}

只输出一个```json```代码块，内部是字符串数组（去重）：
```json
["概念A", "概念B"]
```"""

ENTITY_REFINE_USER = """请再确认是否有遗漏的概念实体。只补充名词性概念。
如果没有遗漏，输出空数组。"""

RELATION_DISCOVER_SYSTEM = """你是关系发现专家。从文本中找出概念之间人类会自然联想到的一切联系。
核心理念：任何两个概念在文本中有交互、关联或共现因果，都应发现。"""

RELATION_DISCOVER_USER = """给定概念列表，从文本中找出有人类可感知关联的概念对。

关联范围：任何人类会自然联想到的联系。宁多勿少。关系内容必须具体（一句话能说清），泛泛描述无效。

每个概念对只需出现一次（A→B 和 B→A 视为同一对）。

概念列表：{entity_names}

文本：
{window_text}

只输出一个```json```代码块，内部是概念对数组（每对只需出现一次）：
```json
[["概念A", "概念B"], ["概念C", "概念D"]]
```"""

RELATION_REFINE_USER = """请检查是否有遗漏的关系，特别关注之前未出现在任何关系对中的概念。如果没有，返回空数组。"""

ORPHAN_RECOVERY_USER = """以下实体在文本中出现，但未与任何其他实体建立关系。
请仔细分析文本，为每个孤立实体找到与之有关系的其他实体。

孤立实体：{orphan_names}
其他实体：{other_entity_names}

文本：
{window_text}

规则：
1. 只建立确实存在于文本中的关系
2. 如果某个孤立实体确实与文本中任何其他实体没有关系，不要强行建立
3. 每对只需出现一次（A→B 和 B→A 视为同一对）

只输出一个```json```代码块，内部是概念对数组：
```json
[["概念A", "概念B"]]
```

如果没有任何关系可以建立，返回空数组：
```json
[]
```"""

ENTITY_CONTENT_WRITE_SYSTEM = """你是知识描述专家。根据文本为指定实体撰写简洁准确的描述。
只输出JSON格式。"""

ENTITY_CONTENT_WRITE_USER = """根据以下文本，描述实体"{entity_name}"。

要求：
1. 只描述该实体本身，不描述与其他实体的关系
2. 包含：它是什么、有什么特征、在文本中的角色
3. 如有时地数量等具体信息，要包含
4. 30到100字，用自己的语言概括

禁止：模板化开头（"该实体是…"）、Markdown标题、推诿内容。

示例："曹操（155-220），字孟德，东汉末年政治家、军事家，魏国奠基人"

文本：
{window_text}

只输出一个```json```代码块：
```json
{{"content": "描述内容"}}
```"""

ENTITY_BATCH_CONTENT_WRITE_SYSTEM = """你是知识描述专家。根据文本为多个实体批量撰写简洁准确的描述。
为每个实体独立描述，只描述该实体本身，不描述与其他实体的关系。
30到100字，用自己的语言概括。禁止模板化开头。"""

ENTITY_BATCH_CONTENT_WRITE_USER = """根据以下文本，为每个实体撰写描述。

实体列表：{entity_names}

文本：
{window_text}

只输出一个```json```代码块，内为数组，每个元素含 name 和 content：
```json
[{{"name": "实体名", "content": "描述内容"}}]
```"""

RELATION_CONTENT_WRITE_SYSTEM = """你是关系描述专家。用自然语言描述两个实体间的具体关联。
只输出JSON格式。"""

RELATION_CONTENT_WRITE_USER = """根据以下文本，描述"{entity_a}"和"{entity_b}"之间的关系。

要求：
1. 具体描述关联内容（10到50字）
2. 包含具体关联动作或关系性质

禁止泛泛描述（"有关联""存在关系""合作关系"等纯标签）。
示例："曹雪芹创作了红楼梦，以自身家族兴衰为蓝本"

文本：
{window_text}

只输出一个```json```代码块：
```json
{{"content": "关系描述"}}
```"""

RELATION_BATCH_CONTENT_WRITE_SYSTEM = """批量描述实体对间的具体关联。每对10到50字，包含具体关联动作。禁止泛泛描述。只输出JSON。"""

RELATION_BATCH_CONTENT_WRITE_USER = """根据文本描述每对实体的关系。关系对：{pair_list}
文本：{window_text}
只输出```json```数组：[{{"entity1":"A","entity2":"B","content":"关系"}}]"""

ENTITY_ALIGNMENT_JUDGE_SYSTEM = """判断两个实体是否同一对象。同一对象常有多种称谓（字号、官职、尊称、简称）。

类型不同 → 直接判 different，无需分析内容。
名称完全不同时，只有 content 明确描述同一对象才判 same。
不确定时选 uncertain。"""

ENTITY_ALIGNMENT_JUDGE_USER = """实体A（新抽取）: "{name_a}"
内容摘要: {content_a}

实体B（已有）: "{name_b}"
内容摘要: {content_b}

{name_relationship}
- same: 同一对象（别名、字号、简称、content角色重合）
- different: 不同对象（类型不同、相似但不同概念）
- uncertain: 无法确定

输出 ```json``` 代码块：
```json
{{"verdict": "same|different|uncertain", "confidence": 0.0-1.0}}
```"""


# ============================================================
# 四、记忆缓存相关（Memory Cache）
# ============================================================

UPDATE_MEMORY_CACHE_SYSTEM_PROMPT = """你是记忆管理器。根据<记忆缓存>和<输入文本>，更新记忆缓存。

**只输出以下两个 Markdown section，不要输出其他 section。**

## 当前摘要
用**自己的语言**改写当前窗口内容（禁止复制原文句子）。要求：
- 每个要点1-2句，涵盖：核心事件/情节、重要对话/细节、背景情境
- 如果是首个窗口（无缓存），直接概括输入文本
- 如果输入文本包含章节号或回目（如"第七十八回""第一章""序言"），在摘要末尾用一行标注，例如：
  当前位置：第七十八回
  （只标注这种结构性的章节/回目标题，不要标注人物名、地点名或事件描述）

## 自我思考
直接写出你的分析。要求：
- 提到具体人物/实体名称，分析它们之间的可能关系
- 预判1-2个具体的后续发展方向
- 标注当前最值得关注的疑点"""

CREATE_DOCUMENT_OVERALL_MEMORY_SYSTEM_PROMPT = """你是一个记忆管理系统。为即将处理的文档生成简短的「文档整体记忆」，供后续文档作为上下文衔接。
输出 Markdown 格式，一段到两段，包含：文档名、主题/类型、关键内容预告。不要写成长篇摘要。"""





# ============================================================
# 五、内容判断与合并（Content Judgment & Merge）
# ============================================================

JUDGE_CONTENT_NEED_UPDATE_SYSTEM_PROMPT = """你是内容比较系统。判断新版本内容是否已被旧版本包含。
- 新内容是旧内容子集或重复 → false
- 新内容有新信息、修正旧内容、有实质性差异 → true
- 参考每条的 source_document。跨文档时更谨慎：只有明确同一概念且新内容仅为补充时才返回 false。

只输出 true 或 false，无需解释。"""

MERGE_ENTITY_NAME_SYSTEM_PROMPT = f"""将两个名称合并为规范名称。选择最常用/规范的为主名称，别称用括号附加。

示例："科幻世界"+"科幻世界出版机构"→"科幻世界（出版机构）"，"北京"+"北京市"→"北京"

{JSON_OUTPUT_OBJECT}
{{{{"name": "规范名称"}}}}"""

JUDGE_RELATION_MATCH_SYSTEM_PROMPT = f"""判断新关系是否与已有关系相同或非常相似。参考 source_document，跨文档时只有明确同一语义关系才匹配。

{JSON_OUTPUT_OBJECT}
匹配：{{"family_id": "...", "need_update": true/false}}
不匹配：null"""

def _make_merge_contents_prompt(concept_type: str, scope_desc: str) -> str:
    """Factory for incremental merge prompts. Reduces entity/relation duplication."""
    return f"""增量合并多个{concept_type}描述。第一个是基础版本，后续是新信息。
只有{scope_desc}时才融合。
{CONTENT_MERGE_REQUIREMENTS}
直接输出合并后的文字，不要 JSON 包装。"""

MERGE_MULTIPLE_ENTITY_CONTENTS_SYSTEM_PROMPT = _make_merge_contents_prompt("实体", "描述同一概念实体")
MERGE_MULTIPLE_RELATION_CONTENTS_SYSTEM_PROMPT = _make_merge_contents_prompt("关系", "描述同一对概念间同一关系")


# ============================================================
# 六、知识图谱整理 - 批量与初步筛选（Knowledge Graph Organization）
# ============================================================

ANALYZE_ENTITY_CANDIDATES_PRELIMINARY_SYSTEM_PROMPT = """初步筛选候选实体。只选出与当前实体高度可能同一概念的候选（名称相似/别名/content描述同一对象）。

排除标准：类型明显不同的候选直接排除。
不确定的不要选。后续会详细判断。

输出 ```json``` 代码块：
```json
{"candidates": ["family_id列表"]}
```"""

RESOLVE_ENTITY_CANDIDATES_BATCH_SYSTEM_PROMPT = """你是知识图谱实体对齐系统。判断"当前实体"是否与某个候选是同一对象。

证据优先级：
1. 名称信号（最强）：name_match_type 含义——substring(子串包含)/exact(核心名相同)/within_batch_alias(同批别名) 为强合并信号；neighbor_expansion(邻居共享) 为辅助信号；无字段=名称无关
2. 角色指纹：对比两实体在原文中与谁互动、处于什么事件。高度重合→可能合并

裁决：名称匹配+类型相同→默认合并；类型不同→绝不合并；名称无关→不合并（除非极强角色指纹+类型一致）。
不合并但有明确关联时建 relations_to_create。
不确定时选 create_new，宁漏勿误。

输出 ```json``` 代码块：
```json
{"match_existing_id": "", "update_mode": "reuse_existing|merge_into_latest|create_new", "merged_name": "", "relations_to_create": [{"family_id": "", "relation_content": ""}], "confidence": 0.0}
```"""

JUDGE_AND_GENERATE_RELATION_SYSTEM_PROMPT = f"""判断两个实体间是否存在明确、有意义的关联，有则生成关系描述。

{RELATION_VALIDITY_CRITERIA}

关系描述：只专注两者关系，准确完整，至少10字。

{JSON_OUTPUT_OBJECT}
{{{{"need_create": true/false, "confidence": 0.0-1.0, "content": "关系描述（need_create=true时必填）"}}}}"""

# ============================================================
# 七、知识图谱整理 - 精细化判断（Detailed Judgment）
# ============================================================

def analyze_entity_pair_detailed_system_prompt(existing_relations_note: str = "") -> str:
    """生成 analyze_entity_pair_detailed 的 system_prompt"""
    return f"""你是知识图谱整理系统。对两个实体进行精细化判断。

{ENTITY_PAIR_JUDGMENT_RULES}
{existing_relations_note}
输出 ```json``` 代码块：
{{
  "action": "merge|create_relation|no_action",
  "relation_content": "create_relation时填写关系描述，否则空字符串"
}}"""

RESOLVE_RELATION_PAIR_BATCH_SYSTEM_PROMPT = """你是关系对齐系统。判断同一实体对的新关系是否与已有关系描述同一性质的关系。

提取核心谓语/动作，对比性质是否相同。
匹配："A是B的组成部分" ↔ "B由A等组成"（都是组成关系）
不匹配："A在酒店休息" vs "A在酒店喝酒"（休息≠喝酒）

不要因涉及同一实体对或相似场景就匹配，核心谓语必须同一性质。
参考 source_document，跨文档时只有明确表达同一语义关系才可匹配。

confidence: 确信匹配0.8-1.0，确信不匹配0.7-0.9，不确定0.3-0.6。

输出 ```json``` 代码块：
```json
{"action": "match_existing|create_new", "matched_relation_id": "", "need_update": false, "confidence": 0.0}
```"""

