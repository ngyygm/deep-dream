"""
Evaluator 评估器的 Prompt 模板
"""

# 系统提示词
EVALUATOR_SYSTEM_PROMPT = """你是一个记忆检索评估专家。你的任务是评估当前收集的记忆信息是否足够回答用户的问题。

## 评估标准

1. **信息完整性**：是否包含回答问题所需的所有关键信息
2. **信息相关性**：收集的信息是否与问题直接相关
3. **信息可靠性**：信息来源是否可追溯，是否有多个来源佐证
4. **推理可行性**：基于现有信息是否可以进行合理推理

## 评估原则

- 如果直接找到了答案，标记为信息充足
- 如果可以通过合理推理得出答案，标记为信息充足
- 如果关键信息缺失，说明还需要查询什么
- 如果查询多轮仍无法找到相关信息，可能记忆库中确实没有相关记忆

## 输出格式

请以 JSON 格式输出评估结果：
```json
{
    "is_sufficient": true/false,
    "reasoning": "评估理由",
    "memories_to_keep": ["保留的关键记忆ID或描述"],
    "memories_to_discard": ["可以丢弃的非关键记忆"],
    "next_action": "如果不充足，建议的下一步动作",
    "answer_hint": "如果充足，基于收集信息的答案提示"
}
```
"""

# 评估请求模板
EVALUATOR_REQUEST_TEMPLATE = """## 用户问题

{question}

## 已收集的记忆信息

{collected_memories}

## 迭代信息

当前已进行 {iteration} 轮查询。

## 请评估

请评估当前收集的信息是否足够回答用户问题。考虑：
1. 是否找到了问题涉及的实体？
2. 是否找到了相关的关系？
3. 是否有足够的信息进行推理？
4. 是否需要继续查询？查询什么？
"""

# 带推理状态的评估请求模板
EVALUATOR_REQUEST_WITH_REASONING = """## 用户问题

{question}

## 当前问题类型

{question_type}

## 推理状态

### 子目标完成情况
{sub_goals}

### 已知事实
{known_facts}

### 缺失信息
{missing_info}

### 待验证假设
{hypotheses}

## 已收集的记忆信息

{collected_memories}

## 迭代信息

当前已进行 {iteration} 轮查询。

## 请评估

请从**推理可行性**角度评估：

1. **问题类型评估** ⭐：当前问题类型是否合适？
   - 如果当前类型是 `direct`，但只找到实体信息无法回答问题（如需要关系、时间顺序），应建议调整为 `reasoning` 或 `temporal_reasoning`
   - 如果当前类型是 `reasoning`，但发现需要时间顺序，应建议调整为 `temporal_reasoning`
   - 如果问题包含"第一次"、"第二次"等时序关键词，但类型不是 `temporal_reasoning`，必须调整

2. **子目标评估**：各子目标的完成情况如何？还有哪些未完成？

3. **事实评估**：已知事实是否足以支持推理？

4. **假设评估**：待验证的假设是否可以验证或否定？

5. **推理可行性**：基于当前信息，是否可以进行推理得出结论？

6. **下一步建议**：如果不充足，应该查询什么来补充推理所需的信息？

请以 JSON 格式输出：
```json
{{
    "is_sufficient": true/false,
    "reasoning": "评估理由",
    "question_type_adjustment": {{
        "should_adjust": true/false,
        "new_type": "direct/reasoning/temporal_reasoning",
        "reason": "调整原因（如果不调整则为空）"
    }},
    "sub_goal_status": {{
        "completed": ["已完成的子目标"],
        "pending": ["待完成的子目标"],
        "blocked": ["被阻塞的子目标及原因"]
    }},
    "reasoning_feasibility": {{
        "can_reason": true/false,
        "missing_for_reasoning": ["推理还需要的信息"],
        "confidence": 0.0-1.0
    }},
    "hypothesis_updates": [
        {{"hypothesis_id": "hyp_id", "action": "verify/reject/needs_more_info"}}
    ],
    "next_action": "建议的下一步动作",
    "answer_hint": "如果可以推理，给出答案提示"
}}
```
"""

# 推理评估专用系统提示词
REASONING_EVALUATOR_SYSTEM_PROMPT = """你是一个推理评估专家。你的任务是评估当前收集的信息是否足够支持推理，以及推理是否可行。

## 评估维度

1. **信息完整性**：推理所需的关键信息是否齐全
2. **推理链可行性**：能否从已知事实推导出结论
3. **假设验证状态**：待验证假设是否得到确认或否定
4. **置信度评估**：如果可以推理，结论的可信度有多高

## 问题类型特点

- **direct**（直接查询）：只需找到相关实体信息即可
- **reasoning**（需要推理）：需要整合多个信息进行推理
- **temporal_reasoning**（时序推理）：需要确定事件的时间顺序

## 评估原则

- 对于直接查询，找到实体信息即可标记为充足
- 对于推理问题，需要确保推理链的每一步都有支撑
- 对于时序推理，需要有足够的时间信息来确定顺序
- 如果多轮查询后仍无法获取关键信息，应建议改变策略或接受不确定性

## 输出格式

请以 JSON 格式输出评估结果。
"""


def format_collected_memories(memories: list) -> str:
    """格式化已收集的记忆"""
    if not memories:
        return "尚未收集任何记忆信息。"
    
    lines = []
    for i, memory in enumerate(memories, 1):
        if isinstance(memory, dict):
            tool_name = memory.get("tool_name", "unknown")
            result = memory.get("result", {})
            
            lines.append(f"### 查询 {i}: {tool_name}")
            
            if isinstance(result, dict):
                # 处理实体
                if "entities" in result and result["entities"]:
                    lines.append("**找到的实体：**")
                    for entity in result["entities"][:10]:
                        name = entity.get("name", "Unknown")
                        content = entity.get("content", "")[:200]
                        lines.append(f"- {name}: {content}")
                
                # 处理关系
                if "relations" in result and result["relations"]:
                    lines.append("**找到的关系：**")
                    for rel in result["relations"][:10]:
                        content = rel.get("content", "")[:200]
                        e1 = rel.get("entity1_name", "?")
                        e2 = rel.get("entity2_name", "?")
                        lines.append(f"- [{e1}] -- [{e2}]: {content}")
                
                # 处理版本
                if "versions" in result and result["versions"]:
                    lines.append(f"**版本历史：** 共 {len(result['versions'])} 个版本")
                    if result.get("earliest_time"):
                        lines.append(f"  - 最早时间: {result['earliest_time']}")
                    if result.get("latest_time"):
                        lines.append(f"  - 最新时间: {result['latest_time']}")
                
                # 处理缓存
                if "cache" in result and result["cache"]:
                    cache = result["cache"]
                    content = cache.get("content", "")[:300]
                    lines.append(f"**记忆缓存：** {content}")
                
                # 处理时间查询的单个实体
                if "entity" in result and result["entity"]:
                    entity = result["entity"]
                    lines.append(f"**时间点实体：** {entity.get('name', 'Unknown')}")
                    lines.append(f"  内容: {entity.get('content', '')[:200]}")
                
                # 处理消息
                if result.get("message"):
                    lines.append(f"*{result['message']}*")
            
            lines.append("")
        else:
            lines.append(f"{i}. {str(memory)[:300]}")
    
    return "\n".join(lines)
