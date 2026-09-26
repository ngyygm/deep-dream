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

    def _names_from_concept_list() -> list[str]:
        if "概念列表：" not in prompt:
            return []
        block = prompt.split("概念列表：", 1)[1].split("\n", 1)[0]
        if "、" in block:
            return [name.strip(" \"'[]") for name in block.split("、") if name.strip(" \"'[]")]
        try:
            value = json.loads(block)
            if isinstance(value, list):
                return [str(item) for item in value]
        except (TypeError, json.JSONDecodeError):
            pass
        return [name.strip() for name in block.split(",") if name.strip()]

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
    elif "<窗口文本>" in prompt and "请一次性抽取" in prompt:
        window_block = _extract_tag_block("窗口文本")
        names = []
        for name in re.findall(r"\b[A-Z][A-Za-z]*(?:\s+[A-Z][A-Za-z]*)?\b", window_block):
            if name not in {"JSON", "Markdown"} and name not in names:
                names.append(name)
        names = names[:6] or ["示例实体1", "示例实体2"]
        return _mock_json_fence({
            "entities": [
                {"name": name, "content": f"{name}是当前文本中具有明确事实信息的核心概念。"}
                for name in names
            ],
            "relations": [
                {
                    "entity1_name": left,
                    "entity2_name": right,
                    "content": f"{left}与{right}在当前事件中直接相关。",
                }
                for left, right in zip(names, names[1:])
            ],
        })
    elif "<待对齐实体列表>" in prompt and "逐实体判断" in prompt:
        results = []
        for block in re.findall(r'<待对齐实体 name="([^"]+)">(.*?)</待对齐实体>', prompt, re.DOTALL):
            name, body = block
            match_id = ""
            for fid, cand_name in re.findall(
                    r"候选\d+: family_id=(\S*) \| name=([^|（]+)", body):
                if cand_name.strip() == name:
                    match_id = fid.strip()
                    break
            results.append({
                "name": name,
                "match_existing_id": match_id,
                "update_mode": "reuse_existing" if match_id else "create_new",
                "merged_name": "",
                "relations_to_create": [],
                "confidence": 0.9 if match_id else 0.6,
            })
        return _mock_json_fence({"results": results})
    elif "<待裁决关系对列表>" in prompt and "逐对判断" in prompt:
        results = []
        for e1, e2, body in re.findall(
                r'<待裁决关系对 entity1="([^"]+)" entity2="([^"]+)">(.*?)</待裁决关系对>',
                prompt, re.DOTALL):
            results.append({
                "entity1_name": e1,
                "entity2_name": e2,
                "action": "create_new",
                "matched_relation_id": "",
                "need_update": True,
                "confidence": 0.8,
            })
        return _mock_json_fence({"results": results})
    elif "关系对数组" in prompt or ("给定概念列表" in prompt and "概念对" in prompt):
        names = _names_from_concept_list()
        return _mock_json_fence([[left, right] for left, right in zip(names, names[1:])][:3])
    elif "根据文本描述每对概念的关系" in prompt:
        quoted = re.findall(r"['\"]([^'\"]+)['\"]", prompt.split("文本：", 1)[0])
        pairs = list(zip(quoted[::2], quoted[1::2]))
        return _mock_json_fence([
            {"entity1": left, "entity2": right, "content": f"{left}与{right}在当前事件中直接相关。"}
            for left, right in pairs[:6]
        ])
    elif "之间的关系" in prompt and "关系描述" in prompt and '"content"' in prompt:
        names = re.findall(r'描述"([^"]+)"和"([^"]+)"', prompt)
        left, right = names[0] if names else ("示例实体1", "示例实体2")
        return _mock_json_fence({"content": f"{left}与{right}在当前事件中直接相关。"})
    elif '"verdict": "same|different|uncertain"' in prompt:
        return _mock_json_fence({"verdict": "different", "confidence": 0.9})
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
    elif "输出格式纠错" in prompt or "json 代码块" in prompt_lower:
        return _mock_json_fence([])
    elif ("判断" in prompt and "合并" in prompt and "实体" in prompt) or "merge_entity_name" in prompt_lower:
        return _mock_json_fence({"merged_name": "示例实体1", "merged_content": "合并后的描述"})
    elif ("关系" in prompt and "匹配" in prompt) or "relation_match" in prompt_lower:
        return _mock_json_fence({"family_id": None})
    elif ("生成关系" in prompt or "relation_content" in prompt_lower or "关系的content" in prompt):
        return _mock_json_fence({"content": "这是一个示例关系描述"})
    elif "知识图谱整理" in prompt or "consolidation" in prompt_lower:
        return "知识图谱整理完成，未发现需要处理的重复实体。"
    return "默认响应"
