"""
Reasoner 推理器的 Prompt 模板

设计原则：
- 强调数据分析能力
- 利用数据结构特征（physical_time, memory_cache_id）
- 通用推理而非特定流程
"""

# 问题分析提示词
QUESTION_ANALYSIS_PROMPT = """你是一个问题分析专家。请分析用户的问题，理解其本质需求。

## 问题类型判断指南

### direct（直接查询）
- 只需找到实体信息即可回答
- 例如："史强是谁？"、"汪淼的职业是什么？"
- **不涉及**：关系、时间顺序、多次事件

### reasoning（需要推理）
- 需要整合多个信息进行推理
- 例如："史强和汪淼是什么关系？"、"谁参与了XX事件？"
- **不涉及**：时间顺序、第几次

### temporal_reasoning（时序推理）⭐
- **涉及时间顺序、第几次、最早/最晚**
- 关键词：第一次、第二次、第N次、最早、最晚、先后、顺序、时间线
- 例如："他们第一次见面是什么时候？"、"最早发生的事件是什么？"
- **必须**：需要找到相关事件，按时间排序，确定顺序

## 用户问题

{question}

## 请分析

**特别注意**：如果问题包含"第一次"、"第二次"、"第几次"、"最早"、"最晚"等时序关键词，必须判断为 `temporal_reasoning`。

请以 JSON 格式输出分析结果：
```json
{{
    "question_type": "direct/reasoning/temporal_reasoning",
    "key_entities": ["需要查找的关键实体（考虑可能的别名）"],
    "key_relations": ["需要查找的关键关系"],
    "sub_goals": [
        {{
            "description": "子目标描述",
            "depends_on": []
        }}
    ],
    "reasoning_hints": "推理思路提示",
    "potential_challenges": ["可能遇到的挑战（如别名、间接关系等）"]
}}
```
"""

# 信息整合提示词
FACT_INTEGRATION_PROMPT = """你是一个数据分析专家。请从收集的信息中提取关键事实。

## 数据字段说明

| 字段 | 说明 |
|------|------|
| `entity_id` | 实体唯一标识，用于后续查询 |
| `relation_id` | 关系唯一标识，用于后续查询 |
| `physical_time` | **关键** - 记录时间，用于时间排序 |
| `memory_cache_id` | **关键** - 来源场景ID，同一ID表示同一场景/文档 |
| `content` | 自然语言描述 |

## 用户问题

{question}

## 已收集的信息

{collected_info}

## 当前已知事实

{known_facts}

## 请提取

分析收集的数据，提取关键信息：

1. **识别实体**：提取所有实体及其 entity_id
2. **识别关系**：提取关系及其 relation_id
3. **时间分析**：提取所有 physical_time，用于时间排序
4. **场景分组**：分析 memory_cache_id，判断哪些记录来自同一场景
5. **缺失信息**：判断还缺什么信息

请以 JSON 格式输出：
```json
{{
    "new_facts": {{
        "fact_key": "事实内容"
    }},
    "entity_facts": {{
        "entity_id": {{
            "name": "实体名称",
            "physical_time": "记录时间",
            "memory_cache_id": "来源场景"
        }}
    }},
    "relation_facts": {{
        "relation_id": {{
            "entity1": "实体1",
            "entity2": "实体2",
            "content_summary": "关系摘要",
            "physical_time": "记录时间",
            "memory_cache_id": "来源场景"
        }}
    }},
    "scene_groups": {{
        "memory_cache_id_1": ["记录1", "记录2"],
        "memory_cache_id_2": ["记录3"]
    }},
    "temporal_sequence": [
        {{
            "event": "事件描述",
            "physical_time": "时间",
            "memory_cache_id": "场景ID"
        }}
    ],
    "still_missing": ["缺失的信息"],
    "hypotheses": [
        {{
            "content": "假设",
            "confidence": 0.5,
            "needs_verification": "需要验证的内容"
        }}
    ]
}}
```
"""

# 推理结论提示词
CONCLUSION_PROMPT = """你是一个推理专家。请根据收集的信息，对用户问题进行推理。

## 数据分析要点

1. **时间分析**：使用 `physical_time` 确定时间顺序
2. **场景分析**：使用 `memory_cache_id` 判断是否同一场景/事件
   - 同一 memory_cache_id = 来自同一文档 = 可能是同一事件
   - 不同 memory_cache_id = 来自不同文档 = 可能是不同事件
3. **推理链**：从已知事实逐步推导结论

## 用户问题

{question}

## 问题类型

{question_type}

## 已知事实

{known_facts}

## 实体信息

{entity_facts}

## 关系信息

{relation_facts}

## 假设

{hypotheses}

## 请推理

基于以上信息，进行推理并得出结论：

```json
{{
    "can_conclude": true/false,
    "conclusion": "最终结论",
    "reasoning_chain": [
        "推理步骤1：...",
        "推理步骤2：...",
        "..."
    ],
    "evidence": ["支持结论的证据"],
    "confidence": 0.0-1.0,
    "limitations": ["结论的局限性"]
}}
```

如果无法得出结论：
```json
{{
    "can_conclude": false,
    "reason": "原因",
    "still_needed": ["还需要的信息"],
    "suggested_actions": ["建议的下一步操作"]
}}
```
"""

# 时序推理专用提示词
TEMPORAL_REASONING_PROMPT = """你是一个时序推理专家。请分析时间相关的信息。

## 时序分析方法

### 1. 利用 physical_time
- 每条记录都有 `physical_time` 字段
- 按 physical_time 升序排列可确定时间顺序

### 2. 利用 memory_cache_id 判断场景
- **同一 memory_cache_id** = 来自同一文档/场景 = 可能是同一事件
- **不同 memory_cache_id** = 来自不同文档/场景 = 可能是不同事件
- 这对判断"第几次"等问题至关重要

### 3. 确定顺序
- 将记录按 memory_cache_id 分组（同组为同一场景）
- 每组取最早的 physical_time 作为该场景的时间
- 按场景时间排序，确定第1次、第2次...

## 用户问题

{question}

## 相关事件

{events}

## 时间信息

{time_info}

## 请分析

1. 按 memory_cache_id 分组记录
2. 计算每组的时间
3. 按时间排序确定顺序
4. 识别用户询问的目标事件

输出 JSON：
```json
{{
    "scene_analysis": [
        {{
            "memory_cache_id": "场景ID",
            "events": ["该场景包含的事件"],
            "physical_time": "场景时间",
            "order": 1
        }}
    ],
    "target_event": {{
        "description": "目标事件",
        "order": "第几个",
        "physical_time": "时间",
        "memory_cache_id": "所属场景"
    }},
    "conclusion": "结论",
    "confidence": 0.0-1.0,
    "reasoning": "推理过程"
}}
```
"""


def format_known_facts(known_facts: dict) -> str:
    """格式化已知事实"""
    if not known_facts:
        return "暂无已知事实"
    
    lines = []
    for key, value in known_facts.items():
        if isinstance(value, dict):
            lines.append(f"- **{key}**:")
            for k, v in value.items():
                lines.append(f"  - {k}: {v}")
        else:
            lines.append(f"- **{key}**: {value}")
    
    return "\n".join(lines)


def format_entity_facts(entity_facts: dict) -> str:
    """格式化实体事实"""
    if not entity_facts:
        return "暂无实体信息"
    
    lines = []
    for eid, facts in entity_facts.items():
        name = facts.get("name", eid)
        ptime = facts.get("physical_time", "未知")
        cache_id = facts.get("memory_cache_id", "未知")
        lines.append(f"### 实体: {name}")
        lines.append(f"- entity_id: {eid}")
        lines.append(f"- physical_time: {ptime}")
        lines.append(f"- memory_cache_id: {cache_id}")
        for key, value in facts.items():
            if key not in ("name", "physical_time", "memory_cache_id"):
                lines.append(f"- {key}: {str(value)[:200]}")
    
    return "\n".join(lines)


def format_relation_facts(relation_facts: dict) -> str:
    """格式化关系事实"""
    if not relation_facts:
        return "暂无关系信息"
    
    lines = []
    for rid, facts in relation_facts.items():
        e1 = facts.get("entity1_name", facts.get("entity1", "?"))
        e2 = facts.get("entity2_name", facts.get("entity2", "?"))
        content = facts.get("content", facts.get("content_summary", ""))[:200]
        ptime = facts.get("physical_time", "未知")
        cache_id = facts.get("memory_cache_id", "未知")
        
        lines.append(f"### 关系: [{e1}] -- [{e2}]")
        lines.append(f"- relation_id: {rid}")
        lines.append(f"- physical_time: {ptime}")
        lines.append(f"- memory_cache_id: {cache_id}")
        lines.append(f"- 内容: {content}")
    
    return "\n".join(lines)


def format_hypotheses(hypotheses: list) -> str:
    """格式化假设列表"""
    if not hypotheses:
        return "暂无假设"
    
    lines = []
    for hyp in hypotheses:
        if isinstance(hyp, dict):
            content = hyp.get("content", str(hyp))
            confidence = hyp.get("confidence", 0.5)
            verified = hyp.get("verified")
            status = "✓" if verified == True else ("✗" if verified == False else "?")
            lines.append(f"- [{status}] [{confidence:.0%}] {content}")
        else:
            lines.append(f"- {hyp.content} (置信度: {hyp.confidence:.0%})")
    
    return "\n".join(lines)
