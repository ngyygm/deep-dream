"""
Planner 规划器的 Prompt 模板

设计原则：
- 提供清晰的数据模型说明
- 给出通用原则而非特定流程
- 让 LLM 有完全的规划自由度
"""

PLANNER_SYSTEM_PROMPT = """你是一个智能记忆检索助手。你的任务是分析用户的问题，自主规划如何从记忆图谱中检索相关信息。

## 数据模型

### Entity（实体）
| 字段 | 说明 |
|------|------|
| `entity_id` | 实体唯一标识。**注意**：同一个人/物可能有多个名称对应不同 entity_id |
| `name` | 名称。可能有别名、简称、昵称 |
| `content` | 自然语言描述 |
| `physical_time` | 记录时间（可用于时间排序） |
| `memory_cache_id` | 来源场景ID |

### Relation（关系）
| 字段 | 说明 |
|------|------|
| `relation_id` | 关系唯一标识 |
| `content` | 关系描述 |
| `physical_time` | 记录时间（可用于时间排序） |
| `memory_cache_id` | 来源场景ID。**关键**：同一 ID = 同一场景/文档 |

### MemoryCache（场景）
| 字段 | 说明 |
|------|------|
| `id` | 场景唯一标识 |
| `content` | 完整的场景上下文 |

## 可用工具

{tools_description}

## 通用原则

### 1. 搜索要全面
- 一个名字可能有多种写法（别名、简称、绰号）
- 可以用不同方式搜索：精确匹配、模糊搜索、语义搜索
- 保留所有可能相关的 entity_id，不要过早排除

### 2. 理解数据结构
- `physical_time` 可用于确定时间顺序
- `memory_cache_id` 相同 = 来自同一场景 = 可能是同一事件
- 版本历史可追溯变化过程
- 通过 `get_memory_cache` 可获取完整上下文

### 3. 灵活应对
- 直接搜索找不到时，尝试间接路径
- 信息不够时，获取场景上下文补充
- 根据返回的数据调整策略

### 4. 工具依赖
- 大多数工具需要 `entity_id`，必须先通过 `search_entity` 获取
- 不能用名称直接调用需要 entity_id 的工具

## 输出格式

请以 JSON 格式输出你的规划：
```json
{{
    "analysis": "对问题的分析，识别关键实体、关系、时间等",
    "tool_calls": [
        {{
            "tool_name": "工具名称",
            "parameters": {{}},
            "reason": "调用原因"
        }}
    ],
    "next_steps": "如果这些调用不够，下一步可能做什么",
    "is_complete": false
}}
```

如果已有足够信息回答问题，设置 `is_complete: true`。
"""

PLANNER_REQUEST_TEMPLATE = """## 用户问题

{question}

## 当前已收集的信息

{collected_info}

## 推理状态

{reasoning_state}

## 请规划下一步

根据问题和已收集的信息，自主决定：
1. 需要调用哪些工具？
2. 用什么参数？
3. 为什么这样做？

请输出 JSON 格式的规划。
"""

PLANNER_REQUEST_TEMPLATE_SIMPLE = """## 用户问题

{question}

## 当前已收集的信息

{collected_info}

## 请规划下一步

根据问题和已收集的信息，自主决定需要调用哪些工具。
请输出 JSON 格式的规划。
"""

TOOL_DESCRIPTION_TEMPLATE = """### {name}

{description}

**参数：**
{parameters}
"""

NO_TOOL_NEEDED_PROMPT = """根据已收集的信息：
1. 是否足够回答用户问题？
2. 如果不够，还需要什么？

如果已经足够，请设置 is_complete 为 true。
"""


def format_tools_description(tools: dict) -> str:
    """格式化工具描述"""
    descriptions = []
    for name, definition in tools.items():
        params_lines = []
        for param in definition.parameters:
            required = "必填" if param.required else "可选"
            params_lines.append(f"  - `{param.name}` ({param.type}, {required}): {param.description}")
        params_str = "\n".join(params_lines) if params_lines else "  无参数"
        descriptions.append(TOOL_DESCRIPTION_TEMPLATE.format(
            name=name,
            description=definition.description,
            parameters=params_str
        ))
    return "\n".join(descriptions)


def _format_time(time_str: str) -> str:
    """格式化时间字符串，只保留日期部分"""
    if not time_str:
        return "未知"
    # 处理 ISO 格式时间
    if 'T' in str(time_str):
        return str(time_str).split('T')[0]
    return str(time_str)[:10]


def format_collected_info(info: list) -> str:
    """格式化已收集的信息，突出显示关键字段"""
    if not info:
        return "尚未收集任何信息。"
    
    lines = []
    for i, item in enumerate(info, 1):
        if isinstance(item, dict):
            tool_name = item.get("tool_name", "unknown")
            result = item.get("result", {})
            
            if "entities" in result:
                entities = result['entities']
                summary = f"找到 {len(entities)} 个实体"
                if entities:
                    entity_details = []
                    for e in entities[:5]:
                        name = e.get('name', 'Unknown')
                        eid = e.get('entity_id', 'unknown')
                        ptime = _format_time(e.get('physical_time'))
                        cache_id = e.get('memory_cache_id', '')[:20] if e.get('memory_cache_id') else ''
                        detail = f"{name} (entity_id='{eid}', time='{ptime}'"
                        if cache_id:
                            detail += f", cache='{cache_id}...'"
                        detail += ")"
                        entity_details.append(detail)
                    summary += ":\n    " + "\n    ".join(entity_details)
                lines.append(f"{i}. [{tool_name}] {summary}")
                
            elif "relations" in result:
                relations = result['relations']
                summary = f"找到 {len(relations)} 个关系"
                if relations:
                    rel_details = []
                    for r in relations[:5]:
                        e1 = r.get('entity1_name', 'Unknown')
                        e2 = r.get('entity2_name', 'Unknown')
                        rid = r.get('relation_id', 'unknown')
                        ptime = _format_time(r.get('physical_time'))
                        cache_id = r.get('memory_cache_id', '')[:15] if r.get('memory_cache_id') else ''
                        content = r.get('content', '')[:40]
                        detail = f"[{e1}] -- [{e2}] (relation_id='{rid}', time='{ptime}'"
                        if cache_id:
                            detail += f", cache='{cache_id}...'"
                        detail += f"): {content}..."
                        rel_details.append(detail)
                    summary += ":\n    " + "\n    ".join(rel_details)
                lines.append(f"{i}. [{tool_name}] {summary}")
                
            elif "paths" in result:
                paths = result['paths']
                summary = f"找到 {len(paths)} 条路径"
                if paths:
                    path_details = []
                    for p in paths[:3]:
                        desc = p.get('path_description', '')[:80]
                        hops = p.get('hop_count', 0)
                        edges = p.get('edges', [])
                        if edges:
                            times = [_format_time(e.get('physical_time')) for e in edges if e.get('physical_time')]
                            cache_ids = list(set(e.get('memory_cache_id', '')[:10] for e in edges if e.get('memory_cache_id')))
                            time_info = f", 时间: {', '.join(times)}" if times else ""
                            cache_info = f", 场景数: {len(cache_ids)}" if cache_ids else ""
                        else:
                            time_info = ""
                            cache_info = ""
                        path_details.append(f"({hops}跳{time_info}{cache_info}) {desc}...")
                    summary += ":\n    " + "\n    ".join(path_details)
                lines.append(f"{i}. [{tool_name}] {summary}")
                
            elif "versions" in result:
                versions = result['versions']
                earliest = result.get('earliest_time', '')
                latest = result.get('latest_time', '')
                time_range = ""
                if earliest and latest:
                    time_range = f"，时间范围: {_format_time(earliest)} ~ {_format_time(latest)}"
                elif earliest:
                    time_range = f"，最早: {_format_time(earliest)}"
                summary = f"找到 {len(versions)} 个版本{time_range}"
                if versions:
                    ver_details = []
                    for v in versions[:5]:
                        ptime = _format_time(v.get('physical_time'))
                        cache_id = v.get('memory_cache_id', '')[:15] if v.get('memory_cache_id') else ''
                        content = v.get('content', '')[:30]
                        detail = f"[{ptime}]"
                        if cache_id:
                            detail += f" (cache='{cache_id}...')"
                        detail += f" {content}..."
                        ver_details.append(detail)
                    summary += ":\n    " + "\n    ".join(ver_details)
                lines.append(f"{i}. [{tool_name}] {summary}")
                
            elif "cache" in result:
                cache = result.get('cache', {})
                ptime = _format_time(cache.get('physical_time'))
                activity = cache.get('activity_type', '')
                cache_id = cache.get('id', '')[:20]
                content_preview = cache.get('content', '')[:100]
                summary = f"获取了场景 (id='{cache_id}...', time='{ptime}', activity='{activity}')"
                if content_preview:
                    summary += f"\n    内容预览: {content_preview}..."
                lines.append(f"{i}. [{tool_name}] {summary}")
                
            elif "entity" in result:
                entity = result.get('entity', {})
                name = entity.get('name', 'Unknown')
                eid = entity.get('entity_id', 'unknown')
                ptime = _format_time(entity.get('physical_time'))
                summary = f"找到实体: {name} (entity_id='{eid}', time='{ptime}')"
                lines.append(f"{i}. [{tool_name}] {summary}")
                
            else:
                summary = str(result)[:200]
                lines.append(f"{i}. [{tool_name}] {summary}")
        else:
            lines.append(f"{i}. {str(item)[:200]}")
    
    return "\n".join(lines)
