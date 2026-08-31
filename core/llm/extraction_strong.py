"""强模型单遍抽取：一次 LLM 调用同时完成实体发现、内容写作、关系发现与关系内容。

适用 strong-v1 profile（强模型、大窗口）。设计约束：
- 绝不落入弱模型的多层回退阶梯——一次调用 + 一次 JSON 解析重试，再失败即抛出，
  由窗口级 failed_window 机制自动补跑兜底
- 输出 schema 与现管线步骤2-8 的产物一致：
  {"entities": [{"name", "content"}],
   "relations": [{"entity1_name", "entity2_name", "content"}]}
"""
from __future__ import annotations

import json

from core.utils import entity_match_key
from typing import Any, Dict, List, Optional

from .prompts import STRUCTURED_WINDOW_EXTRACTION_SYSTEM_PROMPT


class _StrongExtractionMixin:
    """LLMClient mixin：单遍结构化窗口抽取。"""

    def extract_window_structured(
        self,
        window_text: str,
        *,
        max_entities: int = 24,
        max_relations: int = 32,
        known_entity_names: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """一次调用抽取窗口内的实体（含内容）与关系（含内容）。

        Args:
            window_text: 当前窗口文本
            max_entities / max_relations: 输出上限（prompt 内约束 + 解析后硬截断）
            known_entity_names: 可选，已有库中的高频实体名，提示模型优先对齐命名

        Returns:
            {"entities": [{"name","content"}],
             "relations": [{"entity1_name","entity2_name","content"}]}

        Raises:
            ValueError: 两次尝试均无法解析出有效 JSON 时
        """
        max_entities = max(1, int(max_entities))
        max_relations = max(1, int(max_relations))

        known_note = ""
        if known_entity_names:
            _shown = [str(n) for n in known_entity_names[:40] if n]
            if _shown:
                known_note = (
                    "\n\n<库中已有实体名（如窗口中出现同一对象，请直接复用这些名称）>\n"
                    + "; ".join(_shown) + "\n</库中已有实体名>"
                )

        user_prompt = f"""<窗口文本>
{window_text}
</窗口文本>{known_note}

请一次性抽取上述窗口文本中的概念实体与实体间关系。要求：
1. 实体：文本中出现的、值得长期记忆的概念对象（人物/组织/项目/物品/地点/抽象概念等）。
   每个实体给出规范简短名称（≤12字）与完整内容描述。
   目标抽取约 {max_entities} 个实体：把窗口内所有可检索的事实网罗完整，
   包括只出现一次的次要事实（日期/时长/数量/归属/偏好/计划/变动）。
   对这类原子事实，实体名直接取原文中承载该事实的短语 span
   （例："I've known these friends for 4 years" → 实体名 "known these friends for 4 years"；
    "started her PhD in 2019" → 实体名 "started her PhD in 2019"），
   让该事实可被逐字检索命中；span 名可放宽到 40 字符，不受 ≤12 字限制。
2. 【命名硬约束】实体名必须逐字取自窗口文本中出现的名称 span：
   保留原文语言与拼写，禁止翻译、改写或自创名称。
   （例：英文文本中的 "LGBTQ support group" 不得写成 "LGBTQ互助小组"；
   人名/地名/机构名一律使用原文形式。）
3. 关系：只连接已抽取的实体，描述两个实体之间在文本中体现的具体关系。
   目标抽取约 {max_relations} 条关系，覆盖事实之间的连接。
4. content 描述要结合窗口文本写出完整信息，不要空洞概括。

只输出一个 ```json``` 代码块：
```json
{{"entities": [{{"name": "", "content": ""}}],
 "relations": [{{"entity1_name": "", "entity2_name": "", "content": ""}}]}}
```"""

        messages = [
            {"role": "system", "content": STRUCTURED_WINDOW_EXTRACTION_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ]

        def _parse(response: str) -> Dict[str, Any]:
            data = self._parse_json_response(response)
            if not isinstance(data, dict):
                # 走 json.JSONDecodeError 让 call_llm_until_json_parses 重试一次
                raise json.JSONDecodeError(
                    "structured extraction: 需要 JSON 对象", response[:200], 0)
            raw_entities = data.get("entities")
            raw_relations = data.get("relations")
            if not isinstance(raw_entities, list):
                raw_entities = []
            if not isinstance(raw_relations, list):
                raw_relations = []

            entities: List[Dict[str, str]] = []
            seen_names = set()
            for item in raw_entities[: max_entities * 2]:
                if not isinstance(item, dict):
                    continue
                name = str(item.get("name", "") or "").strip()
                content = str(item.get("content", "") or "").strip()
                if not name or len(name) > 60:
                    continue
                key = entity_match_key(name)
                if key in seen_names:
                    continue
                seen_names.add(key)
                entities.append({"name": name, "content": content})

            name_set = {entity_match_key(e["name"]) for e in entities}
            relations: List[Dict[str, str]] = []
            seen_pairs = set()
            for item in raw_relations[: max_relations * 2]:
                if not isinstance(item, dict):
                    continue
                a = str(item.get("entity1_name", "") or "").strip()
                b = str(item.get("entity2_name", "") or "").strip()
                content = str(item.get("content", "") or "").strip()
                ka, kb = entity_match_key(a), entity_match_key(b)
                if not a or not b or ka == kb:
                    continue
                # 端点必须是已抽取实体（按统一匹配键回规范名）
                if ka not in name_set or kb not in name_set:
                    continue
                pk = tuple(sorted([ka, kb]))
                if pk in seen_pairs:
                    continue
                seen_pairs.add(pk)
                relations.append({
                    "entity1_name": a, "entity2_name": b, "content": content,
                })

            if not entities and not relations:
                # 空 JSON（如模型拒答）也值得重试一次，走统一的 JSONDecodeError 路径
                raise json.JSONDecodeError(
                    "structured extraction: 解析结果为空", response[:200], 0)
            return {
                "entities": entities[:max_entities],
                "relations": relations[:max_relations],
            }

        _prev_step = getattr(self, "_current_distill_step", None)
        self._current_distill_step = "02s_onepass_extract"
        try:
            # retries=5：thinking 模型（无法关思考的端点）偶发把整个输出预算烧在
            # 推理上（content 空 + finish_reason=length），3 次尝试有概率全空，
            # 窗口永久失败会级联成整 doc 重跑（benchmark 下一次循环 1h+）。
            # 每次重试是新的思考掷骰，5 次全空概率指数级低；截断扩容阶梯 1×..16×。
            result, _ = self.call_llm_until_json_parses(
                messages, parse_fn=_parse, json_parse_retries=5,
            )
        finally:
            self._current_distill_step = _prev_step
        return result
