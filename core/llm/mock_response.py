"""
Mock LLM response utilities extracted from client.py.

Used when no API endpoint is available (testing / offline mode).
"""
import json
import re
from typing import Any

from .json_repair import (
    _CURRENT_ENTITY_NAME_RE,
    _ENTRY_NAME_RE,
    _FAMILY_ID_RE,
)


def _mock_json_fence(payload: Any) -> str:
    """将可 JSON 序列化的值包在单个 ```json 代码块内，与线上 prompt 约定一致。"""
    body = json.dumps(payload, ensure_ascii=False)
    return f"```json\n{body}\n```"


def mock_llm_response(prompt: str) -> str:
    """模拟LLM响应（用于测试）"""
    prompt_lower = prompt.lower()

    def _extract_tag_block(tag: str) -> str:
        match = re.search(rf"<{tag}>\s*(.*?)\s*</{tag}>", prompt, re.DOTALL)
        return match.group(1).strip() if match else ""

    def _extract_bullet_names(*tags: str) -> list[str]:
        for tag in tags:
            block = _extract_tag_block(tag)
            if not block:
                continue
            names = []
            for line in block.splitlines():
                line = line.strip()
                if not line.startswith("-"):
                    continue
                item = line[1:].strip()
                if "|" in item:
                    item = item.split("|", 1)[0].strip()
                if "<->" in item:
                    continue
                if item:
                    names.append(item)
            if names:
                return names
        return []

    def _extract_candidate_pairs() -> list[dict]:
        block = _extract_tag_block("候选概念对")
        pairs = []
        for line in block.splitlines():
            line = line.strip()
            if not line.startswith("-"):
                continue
            item = line[1:].strip()
            hint = ""
            if "|" in item:
                pair_part, hint_part = item.split("|", 1)
                item = pair_part.strip()
                hint = hint_part.replace("线索:", "").strip()
            if "<->" not in item:
                continue
            left, right = [part.strip() for part in item.split("<->", 1)]
            if not left or not right or left == right:
                continue
            entity1_name, entity2_name = sorted((left, right))
            pairs.append({
                "entity1_name": entity1_name,
                "entity2_name": entity2_name,
                "content": hint,
            })
        return pairs

    if ("更新记忆缓存" in prompt or "memory_cache" in prompt_lower
            or "创建初始记忆缓存" in prompt or "创建初始的记忆缓存" in prompt):
        return """当前摘要：正在处理文档内容。当前阅读的是文档的开头部分，介绍了故事的基本背景和主要人物。重要细节包括主要人物的基本信息和故事的初始情境。

自我思考：
- 应该关注：主要人物的身份、性格特点、故事发生的背景环境
- 预判重点：后续情节可能围绕这些主要人物展开，需要留意人物之间的关系和故事的发展方向
- 疑虑：暂无特别疑虑，需要继续阅读以了解故事的发展

系统状态：
- 已处理文本范围：处理到"文档开始"结束
- 当前文档名：示例文档.txt"""
    elif "候选实体列表" in prompt and "match_existing_id" in prompt:
        _candidate_block = prompt.split("</当前实体>")[1] if "</当前实体>" in prompt else ""
        _current_name_match = _CURRENT_ENTITY_NAME_RE.search(prompt)
        _current_name = _current_name_match.group(1) if _current_name_match else ""
        _candidate_entries = _candidate_block.split("候选")[1:] if _candidate_block else []
        _match_id = ""
        _update_mode = "create_new"
        for _entry in _candidate_entries:
            _cid_m = _FAMILY_ID_RE.search(_entry)
            _cname_m = _ENTRY_NAME_RE.search(_entry)
            if _cid_m and _cname_m and _cname_m.group(1) == _current_name:
                _match_id = _cid_m.group(1)
                _update_mode = "reuse_existing"
                break
        return _mock_json_fence({
            "match_existing_id": _match_id,
            "update_mode": _update_mode,
            "merged_name": "",
            "merged_content": "",
            "relations_to_create": [],
            "confidence": 0.9 if _match_id else 0.3,
        })
    elif ("判断.*实体.*匹配" in prompt or "judge.*entity.*match" in prompt_lower or
          "判断新抽取的实体是否与已有实体" in prompt):
        return _mock_json_fence({
            "family_id": "ent_001",
            "need_update": False
        })
    elif "<指定实体名称>" in prompt:
        names = _extract_bullet_names("指定实体名称")
        if not names:
            names = ["示例实体1"]
        return _mock_json_fence([
            {
                "name": name,
                "content": f"{name}在当前文本中被提及，并有一段稳定的结构化描述。"
            }
            for name in names
        ])
    elif "请召回所有结构性文本锚点概念候选" in prompt or "结构性文本锚点概念候选" in prompt:
        return _mock_json_fence([
            {"name": "第一章", "content": "文本中的结构性章节标题。"},
            {"name": "需求分析阶段", "content": "文本中明确出现的阶段性锚点。"},
        ])
    elif "请召回所有具体/具名概念候选" in prompt or "具名概念候选" in prompt:
        return _mock_json_fence([
            {"name": "示例实体1", "content": "文本中明确出现的具体概念。"},
            {"name": "示例实体2", "content": "文本中明确出现的另一具体概念。"},
        ])
    elif "请召回所有具体/具名概念候选" in prompt or "具体/具名概念候选" in prompt:
        return _mock_json_fence([
            {"name": "示例实体1", "content": "文本中明确出现的具体概念。"},
            {"name": "示例实体2", "content": "文本中明确出现的另一具体概念。"},
        ])
    elif "请召回所有抽象/过程/时间/文本锚点类概念候选" in prompt or "抽象/过程/时间/文本锚点类概念候选" in prompt:
        return _mock_json_fence([
            {"name": "示例主题", "content": "文本中的抽象主题或过程概念。"}
        ])
    elif "<已召回概念列表>" in prompt or "请只补充上面列表中明显遗漏" in prompt:
        known_names = set(_extract_bullet_names("已召回概念列表"))
        candidate = "补充概念"
        if candidate in known_names:
            return _mock_json_fence([])
        return _mock_json_fence([
            {"name": candidate, "content": "对已召回概念的补充概念。"}
        ])
    elif "<候选概念对>" in prompt and "只为候选概念对写出具体关系内容" in prompt:
        pairs = _extract_candidate_pairs()
        return _mock_json_fence([
            {
                "entity1_name": pair["entity1_name"],
                "entity2_name": pair["entity2_name"],
                "content": pair.get("content") or f"{pair['entity1_name']}与{pair['entity2_name']}在文本中存在明确关联。"
            }
            for pair in pairs
        ])
    elif "<稳定概念实体列表>" in prompt and "值得建立关系的概念对" in prompt:
        names = _extract_bullet_names("稳定概念实体列表")
        if len(names) < 2:
            return _mock_json_fence([])
        pairs = []
        for left, right in zip(names, names[1:]):
            entity1_name, entity2_name = sorted((left, right))
            pairs.append({
                "entity1_name": entity1_name,
                "entity2_name": entity2_name,
                "content": f"{entity1_name}与{entity2_name}之间存在关系线索"
            })
            if len(pairs) >= 2:
                break
        return _mock_json_fence(pairs)
    elif "继续生成" in prompt or "继续补充" in prompt:
        return _mock_json_fence([])
    elif "输出格式纠错" in prompt or "json 代码块" in prompt_lower:
        return _mock_json_fence([])
    elif ("抽取关系" in prompt or "抽取所有概念实体间的关系" in prompt or
          "relation" in prompt_lower or "从输入文本中抽取实体之间的关系" in prompt or
          "关系抽取" in prompt or "实体间的关系" in prompt):
        names = _extract_bullet_names("概念实体列表", "稳定概念实体列表")
        if not names and "已抽取的实体：" in prompt:
            entities_section = prompt.split("已抽取的实体：", 1)[1].split("</已抽取实体>", 1)[0].strip()
            if not entities_section:
                return _mock_json_fence([])
        if len(names) >= 2:
            entity1_name, entity2_name = sorted((names[0], names[1]))
        else:
            entity1_name, entity2_name = "示例实体1", "示例实体2"
        return _mock_json_fence([
            {
                "entity1_name": entity1_name,
                "entity2_name": entity2_name,
                "content": f"{entity1_name}与{entity2_name}之间存在稳定关系。"
            }
        ])
    elif ("实体后验增强" in prompt or "enhance.*entity.*content" in prompt_lower or
          "对该实体的content进行更细致的补全和挖掘" in prompt or "增强后的完整实体content" in prompt):
        if "当前content：" in prompt:
            original_content = prompt.split("当前content：", 1)[1].split("</已抽取实体>", 1)[0].strip()
            enhanced_content = f"{original_content}\n\n[增强信息]：基于记忆缓存和当前文本的补充细节和上下文信息。"
        else:
            enhanced_content = "这是一个示例实体的描述\n\n[增强信息]：基于记忆缓存和当前文本的补充细节和上下文信息。"
        return _mock_json_fence({"content": enhanced_content})
    elif ("抽取实体" in prompt or "抽取所有概念实体" in prompt or "entity" in prompt_lower or
          "从输入文本中抽取所有实体" in prompt or "实体抽取" in prompt or
          "概念实体" in prompt):
        return _mock_json_fence([
            {
                "name": "示例实体1",
                "content": "这是一个示例实体的描述，包含足够的结构化信息。"
            }
        ])
    elif ("判断" in prompt and "合并" in prompt and "实体" in prompt) or "merge_entity_name" in prompt_lower:
        return _mock_json_fence({"merged_name": "示例实体1", "merged_content": "合并后的描述"})
    elif ("判断" in prompt and "更新" in prompt and ("content" in prompt_lower or "内容" in prompt)):
        return _mock_json_fence({"need_update": False})
    elif ("关系" in prompt and "匹配" in prompt) or "relation_match" in prompt_lower:
        return _mock_json_fence({"family_id": None})
    elif ("生成关系" in prompt or "relation_content" in prompt_lower or "关系的content" in prompt):
        return _mock_json_fence({"content": "这是一个示例关系描述"})
    elif "知识图谱整理" in prompt or "consolidation" in prompt_lower:
        return "知识图谱整理完成，未发现需要处理的重复实体。"
    elif ("整体记忆" in prompt or "document_overall" in prompt_lower or "文档整体" in prompt):
        return "# 文档整体记忆\n\n这是一份示例文档的整体描述。"
    return "默认响应"
