"""LLM 实体消歧 - 跨 Episode 的实体匹配。"""
from __future__ import annotations

import json
from typing import List, Optional

from ..models import Entity
from ..utils import wprint

ENTITY_RESOLUTION_SYSTEM_PROMPT = """你是一个知识图谱实体消歧助手。你的任务是判断一个新抽取的实体是否与已有实体相同。

判断标准：
1. 名称相同、相似或为别名关系 → 匹配
2. 描述的是同一类型的同一具体对象 → 匹配
3. 类型不同（人物 vs 概念 vs 作品） → 不匹配
4. 仅因内容中互相提及就判断为同一实体 → 不匹配

请只输出一个 ```json ... ``` 代码块，包含键 "matched_family_id"（值为匹配到的 family_id 字符串，无匹配则为 null）。"""


class EntityResolutionMixin:
    """实体消歧 mixin，通过 LLMClient 多继承使用。"""

    async def resolve_entity(
        self,
        name: str,
        content: str,
        candidates: List[Entity],
    ) -> Optional[str]:
        """判断新实体是否与已有实体相同。

        Args:
            name: 新实体名称
            content: 新实体内容
            candidates: 候选已有实体列表

        Returns:
            匹配到的 family_id 或 None
        """
        if not candidates:
            return None

        candidates_text = "\n".join(
            f"- family_id: {e.family_id}, 名称: {e.name}, 内容: {e.content[:500]}"
            for e in candidates[:10]  # 限制候选数量
        )

        prompt = f"""<新实体>
名称: {name}
内容: {content[:500]}
</新实体>

<候选已有实体>
{candidates_text}
</候选已有实体>

请判断新实体是否与上述某个候选实体是同一个实体："""

        messages = [
            {"role": "system", "content": ENTITY_RESOLUTION_SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ]

        try:
            result, _ = self.call_llm_until_json_parses(
                messages,
                parse_fn=lambda r: self._parse_resolution_response(r),
                json_parse_retries=2,
            )
            return result
        except Exception as e:
            wprint(f"实体消歧失败，使用字符串匹配: {e}")
            return None

    async def resolve_entities_batch(
        self,
        new_entities: List[dict],
        existing: List[Entity],
    ) -> List[dict]:
        """批量消歧：为新实体列表中的每个实体判断是否与已有实体匹配。

        Args:
            new_entities: 新实体列表，每个包含 name 和 content
            existing: 已有实体列表

        Returns:
            更新后的 new_entities 列表，匹配到的实体会增加 matched_family_id 字段
        """
        if not existing or not new_entities:
            return new_entities

        # 建立名称索引用于快速预筛选
        existing_name_map: dict = {}
        for e in existing:
            existing_name_map.setdefault(e.name, []).append(e)

        results = []
        for ne in new_entities:
            name = ne.get("name", "").strip()
            if not name:
                results.append(ne)
                continue

            # 先尝试精确名称匹配（但需要验证内容语义一致性）
            if name in existing_name_map:
                matched_entity = existing_name_map[name][0]
                ne_content = (ne.get("content") or "").strip()
                # 精确名称匹配 + 有内容：只有当内容也明显相关时才自动合并
                if ne_content and matched_entity.content:
                    # 内容有实质性重叠才自动匹配，否则仍走 LLM 消歧
                    ne_words = set(ne_content[:200].split())
                    ex_words = set((matched_entity.content or "")[:200].split())
                    overlap = ne_words & ex_words
                    overlap_ratio = len(overlap) / max(len(ne_words | ex_words), 1)
                    if overlap_ratio >= 0.3:
                        ne["matched_family_id"] = matched_entity.family_id
                        results.append(ne)
                        continue
                    # overlap 不足，回退到 LLM 消歧
                else:
                    # 无内容的精确名称匹配，保留自动匹配
                    ne["matched_family_id"] = matched_entity.family_id
                    results.append(ne)
                    continue

            # 尝试 LLM 消歧
            candidates = existing[:20]  # 限制候选数量
            matched_id = await self.resolve_entity(name, ne.get("content", ""), candidates)
            if matched_id:
                ne["matched_family_id"] = matched_id

            results.append(ne)

        return results

    def _parse_resolution_response(self, response: str) -> Optional[str]:
        """解析实体消歧的 LLM 响应。"""
        result = self._parse_json_response(response)
        if not isinstance(result, dict):
            return None
        matched = result.get("matched_family_id")
        if matched and isinstance(matched, str) and matched.strip():
            return matched.strip()
        return None
