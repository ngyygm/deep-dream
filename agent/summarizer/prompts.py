"""
Summarizer 总结器的 Prompt 模板
"""

# 信息筛选提示词
FILTER_INFO_PROMPT = """你是一个信息筛选专家。请根据用户问题，从收集的信息中筛选出最相关、最有用的内容。

## 用户问题

{question}

## 收集的实体信息

{entity_info}

## 收集的关系信息

{relation_info}

## 其他已知事实

{other_facts}

## 请筛选

请筛选出对回答问题最有帮助的信息，以 JSON 格式输出：

```json
{{
    "relevant_entities": [
        {{
            "entity_id": "实体ID",
            "name": "实体名称",
            "relevance": "与问题的关联说明",
            "key_info": "关键信息摘要"
        }}
    ],
    "relevant_relations": [
        {{
            "relation_id": "关系ID", 
            "entities": ["实体1", "实体2"],
            "relevance": "与问题的关联说明",
            "key_info": "关键信息摘要"
        }}
    ],
    "irrelevant_items": ["不相关的项目ID列表"],
    "filter_reasoning": "筛选的理由说明"
}}
```
"""

# 推理总结提示词
SUMMARY_PROMPT = """你是一个推理总结专家。请根据收集的信息和推理过程，生成一个完整的推理总结。

## 用户问题

{question}

## 问题类型

{question_type}

## 推理过程

### 子目标完成情况
{sub_goals}

### 关键事实
{key_facts}

### 相关实体
{entities}

### 相关关系
{relations}

### 推理假设
{hypotheses}

### 推理结论
{conclusion}

## 请生成总结

请以 JSON 格式生成一个结构化的推理总结：

```json
{{
    "summary": {{
        "question": "用户问题",
        "answer": "最终答案",
        "confidence": 0.0-1.0,
        "answer_type": "direct/inferred/uncertain"
    }},
    "reasoning_chain": [
        {{
            "step": 1,
            "action": "执行的动作",
            "result": "得到的结果",
            "insight": "获得的洞察"
        }}
    ],
    "evidence": {{
        "supporting": ["支持答案的证据"],
        "entities_used": ["使用的实体名称"],
        "relations_used": ["使用的关系描述"]
    }},
    "limitations": ["答案的局限性或不确定性"],
    "context_for_llm": "可供外部LLM使用的上下文文本"
}}
```
"""

# 上下文生成提示词
CONTEXT_GENERATION_PROMPT = """请根据以下信息，生成一段简洁、信息密集的上下文文本，供其他 LLM 使用来回答用户问题。

## 用户问题

{question}

## 筛选后的关键信息

### 实体
{entities}

### 关系
{relations}

### 推理结论
{conclusion}

## 要求

1. 只包含与问题直接相关的信息
2. 使用清晰的结构组织信息
3. 突出关键事实和推理链路
4. 控制在 500-1000 字以内

请直接输出上下文文本，不需要 JSON 格式。
"""


def format_entity_for_filter(entities: dict) -> str:
    """格式化实体信息用于筛选"""
    if not entities:
        return "暂无实体信息"
    
    lines = []
    for eid, facts in entities.items():
        name = facts.get("name", eid)
        content = facts.get("content", "")[:200]
        lines.append(f"- [{eid}] {name}: {content}")
    
    return "\n".join(lines)


def format_relation_for_filter(relations: dict) -> str:
    """格式化关系信息用于筛选"""
    if not relations:
        return "暂无关系信息"
    
    lines = []
    for rid, facts in relations.items():
        e1 = facts.get("entity1_name", "?")
        e2 = facts.get("entity2_name", "?")
        content = facts.get("content", "")[:200]
        lines.append(f"- [{rid}] {e1} -- {e2}: {content}")
    
    return "\n".join(lines)


def format_sub_goals(sub_goals: list) -> str:
    """格式化子目标"""
    if not sub_goals:
        return "无子目标"
    
    lines = []
    for goal in sub_goals:
        status = goal.get("status", "unknown")
        desc = goal.get("description", "")
        result = goal.get("result", "")
        
        status_icon = {
            "pending": "⏳",
            "in_progress": "🔄", 
            "completed": "✅",
            "failed": "❌"
        }.get(status, "?")
        
        lines.append(f"{status_icon} {desc}")
        if result:
            lines.append(f"   结果: {str(result)[:100]}")
    
    return "\n".join(lines)


def format_hypotheses_for_summary(hypotheses: list) -> str:
    """格式化假设用于总结"""
    if not hypotheses:
        return "无假设"
    
    lines = []
    for hyp in hypotheses:
        content = hyp.get("content", str(hyp))
        confidence = hyp.get("confidence", 0.5)
        verified = hyp.get("verified")
        
        if verified is True:
            status = "✓ 已验证"
        elif verified is False:
            status = "✗ 已否定"
        else:
            status = f"? 待验证 ({confidence:.0%})"
        
        lines.append(f"- {content} [{status}]")
    
    return "\n".join(lines)
